# Yolo Models

This repository has different weights of yolo models before
fine tuning.

## Setup on Raspberry Pi

Create a virtual environment:
```
python3 -m create venv vision
```

Activate the virtual environment

```
source vision/bin/activate
```
Install the packages:

```
pip install requirements.txt
```

## Test

In `yolo.py`, ensure the camera you are using is accessible at 0
or change accordingly.

To ensure you setup is fine, run:

```
python3 yolo_test.py
```

To stop, press 'q'