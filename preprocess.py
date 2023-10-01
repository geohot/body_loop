import os
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from control import get_foundation
from video import frames_from_file
from tinygrad.tensor import Tensor
Tensor.training = False
Tensor.no_grad = True
import numpy as np
from tqdm import tqdm

def preprocess(foundation, fn, fn_out, flip=False):
  if not fn.endswith(".hevc"): return
  if 'online' in fn: return
  print("preprocessing", fn, fn_out)
  if os.path.isfile(fn_out): return
  frms = []
  xs = []
  for frm in tqdm(frames_from_file(fn)):
    frms.append(frm)
    if len(frms) == 256:
      data = Tensor(np.concatenate([x.reshape(1,480,640,3) for x in frms]))
      if flip: data = data[:, :, ::-1]
      xs.append(foundation(data).realize())
      frms = []
  if frms:
    data = Tensor(np.concatenate([x.reshape(1,480,640,3) for x in frms]))
    if flip: data = data[:, :, ::-1]
    xs.append(foundation(data).realize())
  big_x = Tensor.cat(*xs, dim=0)
  print(big_x.shape)
  return safe_save({"x": big_x}, fn_out)

if __name__ == "__main__":
  foundation = get_foundation()
  for fn in os.listdir("data"):
    preprocess(foundation, "data/"+fn, "data/"+fn.replace(".hevc", ".safetensors"))
    preprocess(foundation, "data/"+fn, "data/"+fn.replace(".hevc", "_flip.safetensors"), True)

