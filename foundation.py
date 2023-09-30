import os
import sys
import tinygrad
from tinygrad.tensor import Tensor
from pathlib import Path
from tinygrad.nn.state import safe_save
from video import frames_from_file
from tinygrad.tensor import Tensor
Tensor.training = False
Tensor.no_grad = True
import numpy as np
from tqdm import tqdm

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
  def foundation(imgs):
    imgs_tensor = imgs if isinstance(imgs, Tensor) else Tensor(imgs)
    x = yolo_infer.net(imgs_tensor.permute(0,3,1,2)) # / 255)
    x = yolo_infer.fpn(*x)
    return x[2]
  return foundation

def preprocess(foundation, fn, fn_out, flip=False):
  if not fn.endswith(".hevc"): return
  print("preprocessing", fn, fn_out)
  if os.path.isfile(fn_out): return
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
  big_x = Tensor.cat(*xs, dim=0)
  print(big_x.shape)
  return safe_save({"x": big_x}, fn_out)

if __name__ == "__main__":
  foundation = get_foundation()
  for fn in os.listdir("data"):
    preprocess(foundation, "data/"+fn, "data/"+fn.replace(".hevc", ".safetensors"))
    preprocess(foundation, "data/"+fn, "data/"+fn.replace(".hevc", "_flip.safetensors"), True)

