#!/usr/bin/env python3
import random
from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Linear
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
from tinygrad.nn.optim import Adam
from tqdm import trange
from tinygrad.jit import TinyJit
import numpy as np

train_set = [
  ("left_1_dcamera", 0),
  ("left_3_dcamera", 0),
  ("right_1_dcamera", 2),
  ("right_3_dcamera", 2),
  ("straight_1_dcamera", 1),
  ("ljust_1_dcamera", 0),
  ("rjust_1_dcamera", 2),
  ("left_day_1_dcamera", 0),
  ("right_day_1_dcamera", 2),
  ("left_day_2_dcamera", 0),
  ("right_day_2_dcamera", 2),
  ("left_day_3_dcamera", 0),
  ("right_day_3_dcamera", 2),
  ("left_day_4_dcamera", 0),
  ("right_day_4_dcamera", 2),
  ("extra_left_1_dcamera", 0),
  ("extra_right_1_dcamera", 2),
  ("extra_left_2_dcamera", 0),
  ("extra_right_2_dcamera", 2),
  ("extra_straight_2_dcamera", 1),
  ("extra_left_3_dcamera", 0),
  ("extra_right_3_dcamera", 2),
  ("messup_left_1_dcamera", 0),
  ("messup_right_1_dcamera", 2),
  ("messup_left_2_dcamera", 0),
  ("messup_right_2_dcamera", 2),
  ("reverse_1_dcamera", 3),
  ("reverse_2_dcamera", 3),
  ("reverse_3_dcamera", 3),
]

test_set = [
  ("left_2_dcamera", 0),
  ("straight_2_dcamera", 1),
  ("right_2_dcamera", 2),
]

# this is the net used after the yolo foundation model
class TinyNet:
  def __init__(self):
    self.c1 = Conv2d(256,64,3)
    self.c2 = Conv2d(64,8,3)
    self.l = Linear(1408,4)
  def __call__(self, x):
    x = self.c1(x).gelu()
    x = self.c2(x).gelu().dropout(0.85)
    x = x.reshape(x.shape[0], -1)
    x = self.l(x)
    return x.log_softmax()

# TODO: fix dropout in the JIT
#@TinyJit
def train_step(x,y):
  Tensor.training = True
  optim.lr *= 0.999
  out = net(x)
  loss = out.sparse_categorical_crossentropy(y)
  optim.zero_grad()
  loss.backward()
  optim.step()
  cat = out.argmax(axis=-1)
  accuracy = (cat == y).mean()
  return loss.realize(), accuracy.realize()

@TinyJit
def test_step(tx,ty):
  Tensor.training = False
  out = net(tx)
  loss = out.sparse_categorical_crossentropy(ty)
  cat = out.argmax(axis=-1)
  return loss.realize(), (cat == ty).mean().realize()

def get_minibatch(sets, bs=32):
  xs, ys = [], []
  for _ in range(bs):
    src, val = random.choice(sets)
    sel = random.randint(0, src.shape[0]-1)
    xs.append(src[sel:sel+1])
    ys.append(val)
  return Tensor(np.concatenate(xs, axis=0)), Tensor(np.array(ys))

# add the flips to the training set with the inverse option
def get_flips(x):
  ret = []
  for t,y in x:
    if y == 2: y = 0
    elif y == 0: y = 2
    ret.append((t+"_flip", y))
  return ret

if __name__ == "__main__":
  train_set += get_flips(train_set)
  test_set += get_flips(test_set)

  train_sets = [(safe_load(f"data/{fn}.safetensors")["x"].numpy(),y) for fn,y in train_set]
  test_sets = [(safe_load(f"data/{fn}.safetensors")["x"].numpy(),y) for fn,y in test_set]

  # get test set
  tx,ty = get_minibatch(test_sets, 1024)

  Tensor.no_grad = False
  Tensor.training = True
  net = TinyNet()
  optim = Adam(get_parameters(net))

  acc, tacc, losses, tlosses = [], [], [], []
  for i in (t:=trange(600)):
    if i%10 == 0: test_loss, test_accuracy = test_step(tx, ty)
    x,y = get_minibatch(train_sets, 64)
    loss, accuracy = train_step(x, y)
    losses.append(float(loss.numpy()))
    tlosses.append(float(test_loss.numpy()))
    acc.append(float(accuracy.numpy()))
    tacc.append(float(test_accuracy.numpy()))
    t.set_description("loss %.2f accuracy %.3f test_accuracy %.3f test_loss %.4f lr: %f" % (loss.numpy(), accuracy.numpy(), test_accuracy.numpy(), test_loss.numpy(), optim.lr.numpy()))

  safe_save(get_state_dict(net), "tinynet.safetensors")

  import matplotlib.pyplot as plt
  plt.plot(losses)
  plt.plot(tlosses)
  plt.plot(acc)
  plt.plot(tacc)
  plt.show()
