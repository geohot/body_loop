comma body loop driver
===

Can drive the body in the comma office in loops.

It uses a YOLOv8 foundational model to preprocess the images, then a policy model to determine left, right, or straight. The policy model is currently trained with hand labelled data (included), but can be extended with RL.

Dataset
------

We collected a dataset of carrying the body through the loop.

"left" means it's too far left and should go right
"right" means it's too far right and should go left
"straight" means you are good to go straight

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

You can either run on PC remotely controlling a body, or on a 3X.

```bash
./control.py
```
