import os
from tinygrad.nn.state import safe_save
from helpers import get_foundation, frames_from_file
from tinygrad.tensor import Tensor
import numpy as np
from tqdm import tqdm

def preprocess(foundation, fn, fn_out, flip=False):
  print("preprocessing", fn, fn_out)
  frms = []
  xs = []
  for frm in tqdm(frames_from_file(fn)):
    frms.append(frm)
    if len(frms) == 256:
      data = np.concatenate([x.reshape(1,480,640,3) for x in frms])
      if flip: data = data[:, :, ::-1]
      xs.append(foundation(data).realize())
      frms = []
  if frms:
    data = np.concatenate([x.reshape(1,480,640,3) for x in frms])
    if flip: data = data[:, :, ::-1]
    xs.append(foundation(data).realize())
  return safe_save({"x": Tensor.cat(*xs, dim=0)}, fn_out)

if __name__ == "__main__":
  foundation = get_foundation()
  for fn in os.listdir("data"):
    if fn.endswith(".hevc"):
      preprocess(foundation, "data/"+fn, "data/"+fn.replace(".hevc", ".safetensors"))
      preprocess(foundation, "data/"+fn, "data/"+fn.replace(".hevc", "_flip.safetensors"), flip=True)
