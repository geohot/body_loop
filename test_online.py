#!/usr/bin/env python3
import sys
from control import get_pred
from video import frames_from_file
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
import numpy as np

if __name__ == "__main__":
  test_data = "data/straight_2_dcamera.hevc" if len(sys.argv) == 1 else sys.argv[1]
  pred = get_pred()

  all_probs = []
  for i,frm in enumerate(frames_from_file(test_data)):
    probs = pred(Tensor(frm.reshape(1, 480, 640, 3))).numpy().copy()
    print(i, probs)
    all_probs.append(probs)

  import matplotlib.pyplot as plt
  plt.figure(figsize=(16,16))
  plt.plot(all_probs)
  plt.legend(["left", "straight", "right"])
  plt.show()
