# YOLOv5_foxy_d435

## Intro
This package utilizes pyrealsense2 to establish a connection with the RealSense D435i camera, enabling it to perform object detection with YOLOv5s. Additionally, it provides distance measurements between the camera and the detected objects.

## How to use it?
* Step1:Please connect your depth camera using a USB 3.1 interface to ensure sufficient bandwidth.
* Step2:Build and source your workspace.
* Step3:`ros2 launch detection detect.launch.py`