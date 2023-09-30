import os
os.environ["IMAGE"] = "2"
import sys
import json
sys.path.append("/data/openpilot/tinygrad_repo/examples")
sys.path.append("/data/openpilot/tinygrad_repo")
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.helpers import Timing, dtypes
import cv2
import numpy as np

from control import get_foundation, get_tinynet
import cereal.messaging as messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType

if __name__ == "__main__":
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)
  while not vipc_client.connect(False): time.sleep(0.1)
  print("vipc connected")

  pm = messaging.PubMaster(['customReservedRawData1'])
  def control(x,y):
    dat = messaging.new_message()
    dat.customReservedRawData1 = json.dumps({'x': x, 'y': y}).encode()
    pm.send('customReservedRawData1', dat)
  print("pm connected")

  # net runner
  foundation = get_foundation()
  net = get_tinynet()
  @TinyJit
  def pred(x):
    out = foundation(x)
    return [x.realize() for x in net(out)]
  with Timing("building 1: "):
    pred(Tensor.rand(1,480,640,3, dtype=dtypes.uint8))
  with Timing("building 2: "):
    pred(Tensor.rand(1,480,640,3, dtype=dtypes.uint8))

  for i in range(3):
    with Timing():
      out = pred(Tensor.rand(1,480,640,3, dtype=dtypes.uint8))
      probs, distance_in_course = [x.numpy() for x in out]

  while 1:
    # get frame
    yuv_img_raw = vipc_client.recv()
    if yuv_img_raw is None or not yuv_img_raw.data.any(): continue
    imgff = yuv_img_raw.data.reshape(-1, vipc_client.stride)
    imgff = imgff[8:8+vipc_client.height * 3 // 2, :vipc_client.width]
    img = cv2.cvtColor(imgff, cv2.COLOR_YUV2RGB_NV12)
    img = cv2.resize(img, (640, 480))

    # run the model
    out = pred(Tensor(img).reshape(1, 480, 640, 3))
    probs, distance_in_course = [x.numpy() for x in out]
    probs = np.exp(probs[0])

    probs, distance_in_course = [x.numpy() for x in out]
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
