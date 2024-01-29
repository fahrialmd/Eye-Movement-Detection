# Project Name: Eye Gaze Navigation System

Tailored for wheelchair implementation, the Eye Navigation System facilitates hands-free interaction with electronic devices. Users with mobility impairments navigate their wheelchairs accurately and effortlessly through the system's advanced features, including Self-Calibrated Illumination (SCI), MobileNetV3L, and Facial Landmark Detection. The integration of SCI ensures enhanced accuracy in low-light conditions by dynamically adjusting illumination levels for optimal image clarity and recognition. This innovative solution serves as a bridge between assistive technology and ML-driven navigation, offering uninterrupted functionality and greater autonomy to users across diverse environments.

## System Components:

1.  **Computation Device:**
    The Jetson Nano is a compact, energy-efficient computing platform developed by NVIDIA, specifically designed for embedded AI and edge computing applications making it ideal for deployment in resource-constrained environments such as robotics, IoT devices, and embedded systems.

2.  **Video Capture Setup:**
    The system begins by capturing live video feed from a camera or webcam connected to Jetson Nano

3.  **SCI (Self Calibrated Illumination):**
    SCI is a machine learning model specifically designed to enhance image quality in low-light conditions. It operates quickly, flexibly, and robustly in real-world scenarios.

4.  **Facial Landmark Model:**
    The facial landmark model is tasked with detecting and isolating the user's left eye, which is subsequently defined as regions of interest (ROI). This model utilizes TensorRT to ensure optimized performance during real-time inference.

5.  **MobileNetV3s Model:**
    The MobileNetV3L model is deployed for image classification within defined ROI . It identifies predetermined classes of eye movements based on the features observed within the ROIs.

6.  **Command Transmission to Arduino:**
    Upon classifying the ROI, the system transmits corresponding commands to an Arduino device through a serial connection. These commands prompt the movement of wheelchair motors in alignment with the identified classes.
    
## Workflow:

1.  **Video Capture and Preprocessing:**
    The system initiates by capturing live video feed from a camera or webcam connected to the Jetson Nano. It then applies preprocessing techniques to enhance the quality and clarity of the images, ensuring accurate analysis in subsequent stages.

2.  **Facial Landmark Detection:**
    Utilizing the facial landmark model, the system detects and tracks key facial landmarks like the corners of the eyes, nose, and mouth. These landmarks serve as reference points for analyzing eye movements and determining the user's gaze direction.

3.  **Region of Interest (ROI) Identification:**
    Based on the detected facial landmarks, the system defines ROIs corresponding to specific areas of interest, typically centered around the user's eyes or focal point. These ROIs enable targeted analysis and classification of relevant visual stimuli.

4.  **Image Classification and Command Generation:**
    Within each ROI, the system applies the MobileNetV3L model for image classification. It identifies predetermined classes of eye movements based on the features observed within the ROIs. Upon classification, corresponding commands are generated and transmitted to the Arduino device through a serial connection.

5. **Real-Time Feedback and Display:**
    Throughout the process, the system provides real-time feedback to the user, displaying the interpreted commands or actions on the screen. This feedback loop ensures seamless interaction and enables users to navigate devices intuitively.

## Installation
1. Use Nvidia [TensorRT container 22.10](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt) for Nvidia GPU driver 520 version (use `nvidia-smi` to check your version)
2. Install all dependencies from requirements.txt `pip install -r requirements.txt`
3. For deploying in Jetson Nano 4GB use dusty-nv's onnxruntine [container](https://github.com/dusty-nv/jetson-containers/tree/master/packages/onnxruntime) and to avoid future error don't install dependencies via requirements.txt because that file is made from tensorrt container

##  Demonstration:
Please see the [deploy code](deploy.py) to run this project

##  Reference
SCI: https://github.com/vis-opt-group/SCI.
Facial Landmark: https://github.com/NobuoTsukamoto/tensorrt-examples/tree/main/python/face_landmark.



