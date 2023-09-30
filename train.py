import random
from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Linear
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
from tinygrad.nn.optim import Adam
from tqdm import trange
from tinygrad.jit import TinyJit
import numpy as np

# TODO: programmatic flip adding
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
]

test_set = [
  ("left_2_dcamera", 0),
  ("right_2_dcamera", 2),
  ("straight_2_dcamera", 1),
]

# this is the net used after the yolo foundation model
class TinyNet:
  def __init__(self):
    self.c1 = Conv2d(256,8,3)
    self.l = Linear(1872,4)
  def __call__(self, x):
    x = self.c1(x).gelu().dropout(0.6)
    x = x.reshape(x.shape[0], -1)
    x = self.l(x)
    return x[:, 0:3].log_softmax(), x[:, 3].sigmoid()

# TODO: fix dropout in the JIT
#@TinyJit
def train_step(x,y,z):
  Tensor.training = True
  optim.lr *= 0.999
  out,dist = net(x)
  loss = out.sparse_categorical_crossentropy(y)
  loss2 = (z-dist).square().mean()
  real_loss = loss+loss2
  optim.zero_grad()
  real_loss.backward()
  optim.step()
  cat = out.argmax(axis=-1)
  accuracy = (cat == y).mean()
  return loss.realize(), loss2.realize(), accuracy.realize()

@TinyJit
def test_step(tx,ty):
  Tensor.training = False
  out,_ = net(tx)
  loss = out.sparse_categorical_crossentropy(ty)
  cat = out.argmax(axis=-1)
  return loss.realize(), (cat == ty).mean().realize()

def get_minibatch(sets, bs=32):
  xs, ys, zs = [], [], []
  for _ in range(bs):
    src, val = random.choice(sets)
    sel = random.randint(0, src.shape[0]-1)
    xs.append(src[sel:sel+1])
    ys.append(val)
    zs.append(sel/src.shape[0])
  return Tensor(np.concatenate(xs, axis=0)), Tensor(np.array(ys)), Tensor(np.array(zs, dtype=np.float32))

def get_flips(x):
  ret = []
  for t,y in x:
    if y == 2: y = 0
    elif y == 0: y = 2
    ret.append((t+"_flip", y))
  return ret

if __name__ == "__main__":
  #train_set += get_flips(train_set)
  #test_set += get_flips(test_set)

  train_sets = [(safe_load(f"data/{fn}.safetensors")["x"].numpy(),y) for fn,y in train_set]
  test_sets = [(safe_load(f"data/{fn}.safetensors")["x"].numpy(),y) for fn,y in test_set]

  # get test set
  tx,ty,tz = get_minibatch(test_sets, 1024)

  Tensor.no_grad = False
  Tensor.training = True
  net = TinyNet()
  optim = Adam(get_parameters(net))

  acc, tacc, losses, tlosses = [], [], [], []
  for i in (t:=trange(500)):
    if i%10 == 0: test_loss, test_accuracy = test_step(tx, ty)
    x,y,z = get_minibatch(train_sets, 64)
    loss, loss2, accuracy = train_step(x,y,z)
    losses.append(float(loss.numpy()))
    tlosses.append(float(test_loss.numpy()))
    acc.append(float(accuracy.numpy()))
    tacc.append(float(test_accuracy.numpy()))
    t.set_description("loss %.2f mse %.2f accuracy %.3f test_accuracy %.3f test_loss %.4f lr: %f" % (loss.numpy(), loss2.numpy(), accuracy.numpy(), test_accuracy.numpy(), test_loss.numpy(), optim.lr.numpy()))

  safe_save(get_state_dict(net), "tinynet.safetensors")

  import matplotlib.pyplot as plt
  plt.plot(losses)
  plt.plot(tlosses)
  plt.plot(acc)
  plt.plot(tacc)
  plt.show()
