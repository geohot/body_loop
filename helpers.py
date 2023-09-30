import sys
import av
import tinygrad
from tinygrad.tensor import Tensor
from pathlib import Path

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
    x = yolo_infer.net((imgs if isinstance(imgs, Tensor) else Tensor(imgs)).permute(0,3,1,2))
    x = yolo_infer.fpn(*x)
    return x[2]
  return foundation

# pass in msgs to decode them to frames
codec = None
def live_decode_frames(msgs, seen_iframe=False):
  global codec
  if not seen_iframe: codec = av.CodecContext.create("hevc", "r")
  imgs = []
  V4L2_BUF_FLAG_KEYFRAME = 8
  for evt in msgs:
    evta = getattr(evt, evt.which())
    if not seen_iframe and not (evta.idx.flags & V4L2_BUF_FLAG_KEYFRAME): continue
    if not seen_iframe:
      codec.decode(av.packet.Packet(evta.header))
      seen_iframe = True
    frames = codec.decode(av.packet.Packet(evta.data))
    if not len(frames): continue
    imgs.append(frames[0].to_ndarray(format=av.video.format.VideoFormat('rgb24')))
  return imgs, seen_iframe

# get frames from dcamera files
def frames_from_file(fn):
  fmt = av.video.format.VideoFormat('rgb24')
  container = av.open(fn)
  for frame in container.decode(video=0): yield frame.to_ndarray(format=fmt)
