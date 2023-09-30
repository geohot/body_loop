import sys
from control import get_foundation, get_tinynet
from video import frames_from_file
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
import numpy as np

if __name__ == "__main__":
  foundation = get_foundation()
  net = get_tinynet()

  test_data = "data/straight_2_dcamera.hevc" if len(sys.argv) == 1 else sys.argv[1]

  Tensor.no_grad = True
  Tensor.training = False

  @TinyJit
  def pred(x):
    out = foundation(x)
    return [x.realize() for x in net(out)]

  dist = []
  all_probs = []
  for i,frm in enumerate(frames_from_file(test_data)):
    #if i == 100: break
    out = pred(Tensor(frm.reshape(1, 480, 640, 3)))
    probs, distance_in_course = [x.numpy() for x in out]
    probs = np.exp(probs[0])
    print(i, distance_in_course, probs)
    all_probs.append(probs)
    dist.append(float(distance_in_course))

  import matplotlib.pyplot as plt
  plt.figure(figsize=(16,16))
  plt.subplot(211)
  plt.plot(dist)
  plt.subplot(212)
  plt.plot(all_probs)
  plt.legend(["left", "straight", "right"])
  plt.show()
