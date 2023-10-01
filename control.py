#!/usr/bin/env python3
import os, sys, time, json
from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)

# openpilot imports
import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.system.hardware import PC

# device gets images not through the compressor
if not PC:
  os.environ["IMAGE"] = "2"
  os.environ["FLOAT16"] = "1"
  import cv2
  from cereal.visionipc import VisionIpcClient, VisionStreamType

# tinygrad!
import tinygrad
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.helpers import Timing, dtypes

# repo local imports
from video import live_decode_frames
from train import TinyNet

# this is the yolo foundation model
def get_foundation():
  # add tinygrad and tinygrad examples to python path
  if PC:
    sys.path.append(str(Path(tinygrad.__path__[0]).parent))
    sys.path.append(str(Path(tinygrad.__path__[0]).parent / "examples"))
  else:
    sys.path.append("/data/openpilot/tinygrad_repo/examples")
    sys.path.append("/data/openpilot/tinygrad_repo")

  from yolov8 import YOLOv8, get_variant_multiples
  yolo_variant = "n"
  depth, width, ratio = get_variant_multiples(yolo_variant)
  yolo_infer = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)

  from extra.utils import download_file
  from tinygrad.nn.state import safe_load, load_state_dict
  weights_location = Path("/tmp") / f'yolov8{yolo_variant}.safetensors'
  download_file(f'https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8{yolo_variant}.safetensors', weights_location)
  state_dict = safe_load(weights_location)
  load_state_dict(yolo_infer, state_dict)
  def foundation(imgs:Tensor):
    x = yolo_infer.net(imgs.permute(0,3,1,2).float() / 255)
    x = yolo_infer.fpn(*x)
    return x[2]
  return foundation

def get_tinynet():
  net = TinyNet()
  load_state_dict(net, safe_load("tinynet.safetensors"))
  return net

def get_pred():
  # net runner
  Tensor.no_grad = True
  Tensor.training = False
  foundation = get_foundation()
  net = get_tinynet()
  @TinyJit
  def pred(x):
    return net(foundation(x)).exp()[0].realize()

  # warm up the net runner
  with Timing("building 1: "): pred(Tensor.rand(1,480,640,3, dtype=dtypes.uint8))
  with Timing("building 2: "): pred(Tensor.rand(1,480,640,3, dtype=dtypes.uint8))
  with Timing("testing rt: "): pred(Tensor.rand(1,480,640,3, dtype=dtypes.uint8)).numpy()
  return pred

if __name__ == "__main__":
  # controlling the body
  # REQUIRED on PC: run bridge on device back to YOUR ip
  # /data/openpilot/cereal/messaging/bridge 192.168.60.251 customReservedRawData1
  pm = messaging.PubMaster(['customReservedRawData1'])
  def control(x, y):
    dat = messaging.new_message()
    dat.customReservedRawData1 = json.dumps({'x': x, 'y': y}).encode()
    pm.send('customReservedRawData1', dat)
  print("pm connected")

  # getting frames
  if PC:
    # TODO: replace with DEVICE ip
    dcamData = messaging.sub_sock("driverEncodeData", None, addr="192.168.63.69", conflate=False)
  else:
    vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)
    while not vipc_client.connect(False): time.sleep(0.1)
  print("vipc connected")

  # get the net runner
  pred = get_pred()

  if PC:
    seen_iframe = False
    from openpilot.common.window import Window
    win = Window(640, 480)

  # run at max 5hz
  rk = Ratekeeper(5)
  while 1:
    # get frame
    if PC:
      frm = messaging.drain_sock(dcamData, wait_for_one=True)
      if not seen_iframe: frm = frm[-20:]
      frms, seen_iframe = live_decode_frames(frm, seen_iframe)
      if not frms: continue
      img = frms[-1]
    else:
      yuv_img_raw = vipc_client.recv()
      if yuv_img_raw is None or not yuv_img_raw.data.any(): continue
      imgff = yuv_img_raw.data.reshape(-1, vipc_client.stride)
      imgff = imgff[8:8+vipc_client.height * 3 // 2, :vipc_client.width]
      img = cv2.cvtColor(imgff, cv2.COLOR_YUV2RGB_NV12)
      img = cv2.resize(img, (640, 480))

    # run the model
    probs = pred(Tensor(img).reshape(1, 480, 640, 3)).numpy()

    # control policy, turn harder if we are more confident
    if probs[0] > 0.99: choice = 0
    elif probs[2] > 0.99: choice = 4
    elif probs[1] > 0.99: choice = 2
    elif probs[0] > probs[2]: choice = 1
    else: choice = 3

    # minus y is right, plus y is left
    x = -0.5 - (probs[1]*0.5)
    y = [-0.7, -0.2, 0, 0.2, 0.7][choice]
    control(x, y)
    print(f"{x:5.2f} {y:5.2f}", probs)

    # draw frame and run at 5hz
    if PC: win.draw(frms[-1])
    rk.keep_time()
