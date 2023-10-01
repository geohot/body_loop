comma body loop driver
===

Can drive the body in the comma office in loops.

It uses a YOLOv8 foundational model to preprocess the images, then a policy model to determine left, right, or straight. The policy model is currently trained with hand labelled data (included), but can be extended with RL.

Preprocessing, training, and inference (both on PC and device) are all done with [tinygrad](https://github.com/tinygrad/tinygrad).

Dataset
------

We collected a dataset of carrying the body through the loop.

"left" means it's too far left and should go right
"right" means it's too far right and should go left
"straight" means you are good to go straight
"reverse" means you are against a wall and should reverse

Training the model
------

```bash
# First, process the data:
./preprocess.py

# Then, train the model (tinynet.safetensors):
./train.py
```

Running
------

You can either run on PC remotely controlling a body, or on a [comma 3X](https://comma.ai/shop/comma-3x).

```bash
./control.py
```

Dependencies
------

A few standard Python deps. `opencv-python` is only needed on device.

`pip install av tqdm opencv-python`

You need a recent version of tinygrad for everything. openpilot for the controls stuff, and cv2 on device for color transform and resizing (TODO: replace with tinygrad!)
