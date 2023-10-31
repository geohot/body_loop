#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from control import get_foundation
from video import frames_from_file

Tensor.training = False
Tensor.no_grad = True

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
NUM_CHANNELS = 3

def preprocess(foundation, fn, fn_out, flip=False):
    """
    Preprocesses video frames and saves the result to a file.

    Args:
        foundation (callable): The foundation function.
        fn (str): Input video file path.
        fn_out (str): Output file path.
        flip (bool, optional): Whether to flip the frames. Defaults to False.
    """
    if not fn.endswith(".hevc"):
        return
    if 'online' in fn:
        return
    print("Preprocessing", fn, fn_out)
    if os.path.isfile(fn_out):
        return
    
    frms = [frm for frm in frames_from_file(fn)]
    xs = [foundation(Tensor(np.concatenate([x.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS) for x in frms])))
          .realize() for frms in np.array_split(frms, len(frms) // 256)]
    
    big_x = Tensor.cat(*xs, dim=0)
    print(big_x.shape)
    safe_save({"x": big_x}, fn_out)

def main():
    foundation = get_foundation()
    data_dir = "data"
    
    for fn in os.listdir(data_dir):
        if fn.endswith(".hevc"):
            preprocess(foundation, os.path.join(data_dir, fn), os.path.join(data_dir, fn.replace(".hevc", ".safetensors")))
            preprocess(foundation, os.path.join(data_dir, fn), os.path.join(data_dir, fn.replace(".hevc", "_flip.safetensors")), True)

if __name__ == "__main__":
    main()

