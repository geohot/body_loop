import av

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
