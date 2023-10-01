import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper

from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.nn.state import load_state_dict, safe_load

import sys
import tinygrad
from pathlib import Path

import json
import numpy as np
np.set_printoptions(suppress=True)

from video import live_decode_frames
from train import TinyNet

# this is the yolo foundation model
def get_foundation():
  # add tinygrad and tinygrad examples to python path
  sys.path.append(str(Path(tinygrad.__path__[0]).parent))
  sys.path.append(str(Path(tinygrad.__path__[0]).parent / "examples"))

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

if __name__ == "__main__":
  messaging.context = messaging.Context()

  # TODO: run bridge on device back to your ip
  pm = messaging.PubMaster(['customReservedRawData1'])
  def control(x,y):
    dat = messaging.new_message()
    dat.customReservedRawData1 = json.dumps({'x': x, 'y': y}).encode()
    pm.send('customReservedRawData1', dat)

  # TODO: replace with your ip
  socks = {x:messaging.sub_sock(x, None, addr="192.168.63.69", conflate=False) for x in ["driverEncodeData"]}

  # net runner
  foundation = get_foundation()
  net = get_tinynet()
  @TinyJit
  def pred(x):
    out = foundation(x)
    return [x.realize() for x in net(out)]

  Tensor.no_grad = True
  Tensor.training = False
  seen_iframe = False

  from openpilot.common.window import Window
  win = Window(640, 480)

  rk = Ratekeeper(5)
  while 1:
    # get frames
    frm = messaging.drain_sock(socks["driverEncodeData"], wait_for_one=True)
    if not seen_iframe: frm = frm[-20:]
    frms, seen_iframe = live_decode_frames(frm, seen_iframe)
    if not frms: continue

    # run the model
    probs = pred(Tensor(frms[-1].reshape(1, 480, 640, 3))).numpy()
    probs = np.exp(probs[0])

    # control policy, turn harder if we are more confident
    if probs[0] > 0.99: choice = 0
    elif probs[2] > 0.99: choice = 4
    elif probs[1] > 0.99: choice = 2
    elif probs[0] > probs[2]: choice = 1
    else: choice = 3

    # minus y is right, plus y is left
    x = -0.3 - (probs[1]*0.3)
    y = [-0.6, -0.2, 0, 0.2, 0.6][choice]
    control(x, y)
    print(f"{x:5.2f} {y:5.2f}", distance_in_course, probs)

    # draw frame and run at 5hz
    win.draw(frms[-1])
    rk.keep_time()
