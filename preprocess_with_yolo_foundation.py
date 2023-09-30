import os
from tinygrad.nn.state import safe_save
from helpers import get_foundation, frames_from_file
from tinygrad.tensor import Tensor
import numpy as np
from tqdm import tqdm

def preprocess(foundation, fn):
  print("preprocessing", fn)
  frms = []
  xs = []
  for frm in tqdm(frames_from_file(fn)):
    frms.append(frm)
    if len(frms) == 256:
      xs.append(foundation(np.concatenate([x.reshape(1,480,640,3) for x in frms])).realize())
      frms = []
  if frms:
    xs.append(foundation(np.concatenate([x.reshape(1,480,640,3) for x in frms])).realize())
  return safe_save({"x": Tensor.cat(*xs, dim=0)}, fn.replace(".hevc", ".safetensors"))

if __name__ == "__main__":
  foundation = get_foundation()
  for fn in os.listdir("data"):
    if fn.endswith(".hevc"):
      preprocess(foundation, "data/"+fn)
