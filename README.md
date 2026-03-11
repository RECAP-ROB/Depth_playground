# Depth_playground

Has different depth perception algorithm methods

## Monocular Methods

Use of one camera to estimate distance of an object.

### Triangular similarity

This needs the width of the object to calculate the distance the object is from the camera
while using the bounding boxes on the object after detection.

This proved to be accurate for fixed objects when you approach them from one direction and
not suitable for dynamically objects if you change orientation of object.

### MiDas Depth model

Use of a Depth perception model to create a depth map of surrounding.

MiDas has proved to be very good with h=the creation of depth maps.

It however provides relative depth:
how close an object is from the camera. It can be calibrated to provide distance in a specific metric.
This can enable it tell the exact distance. 

It small version is still however computationally heavy to run on the raspberry pi 4B.

## Stereo Vision

Use of two cameras to estimate distance of an object and create depth map.

On good calibration, one can get very accurate estimates of distanc of a specifci object.
It is the most promising depth perception method to use.

## Setup

To run these tests and examples, One needs to install the packages used first:

```
pip install requirements.txt
```

