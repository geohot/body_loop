import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.window import Window

from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.nn.state import load_state_dict, safe_load

import json
import numpy as np
np.set_printoptions(suppress=True)

from helpers import get_foundation, live_decode_frames
from train import TinyNet

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

  win = Window(640, 480)

  rk = Ratekeeper(5)
  while 1:
    # get frames
    frm = messaging.drain_sock(socks["driverEncodeData"], wait_for_one=True)
    if not seen_iframe: frm = frm[-20:]
    frms, seen_iframe = live_decode_frames(frm, seen_iframe)
    if not frms: continue

    # run the model
    out = pred(Tensor(frms[-1].reshape(1, 480, 640, 3)))
    probs, distance_in_course = [x.numpy() for x in out]
    probs = np.exp(probs[0])

    # control policy, turn harder if we are more confident
    if probs[0] > 0.96: choice = 0
    elif probs[2] > 0.96: choice = 4
    elif probs[1] > 0.96: choice = 2
    elif probs[0] > probs[2]: choice = 1
    else: choice = 3

    # minus y is right, plus y is left
    y = [-0.5, -0.2, 0, 0.2, 0.5][choice]
    control(-0.35, y)
    print(f"{y:5.1f}", distance_in_course, probs)

    # draw frame and run at 5hz
    win.draw(frms[-1])
    rk.keep_time()
