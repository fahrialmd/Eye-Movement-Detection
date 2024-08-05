#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Disable access control xhost +

import cv2
import numpy as np
import tensorrt as trt
import common
import time
import serial
import time
from model import Finetunemodel
from PIL import Image
import torch
from torch.autograd import Variable

# Set up TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


# Initialize serial port - replace '/dev/ttyACM0' with the port Arduino is connected to
ser = serial.Serial('/dev/ttyACM0', 9600) 
time.sleep(2) # Wait for the connection to settle


def get_engine(engine_file_path):
    """Loads a TensorRT engine from a file"""
    print(f"Reading engine from file {engine_file_path}")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def find_center_of_line(coords):
    """Calculates the center of a line given by two points"""
    (x1, y1), (x2, y2) = coords
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def create_roi(image, center_point, roi_size = (100,100)):
    """Extracts a square ROI from the image with a given center and size"""
    cx, cy = center_point
    start_x = cx - roi_size[0] // 2
    start_y = cy - roi_size[1] // 2
    end_x = start_x + roi_size[0]
    end_y = start_y + roi_size[1]
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(end_x, image.shape[1])
    end_y = min(end_y, image.shape[0])
    roi = [(start_x, start_y), (end_x, end_y)]
    return roi


def classify_image(
    roi_frame,
    mobilenet_context,
    mobilenet_engine,
):
    # Preprocess the ROI for MobileNetV3s
    mobilenet_input_shape = (224, 224)  
    roi_frame_resized = cv2.resize(roi_frame, mobilenet_input_shape)
    roi_frame_rgb = cv2.cvtColor(roi_frame_resized, cv2.COLOR_BGR2RGB)
    roi_frame_normalized = roi_frame_rgb.astype("float32") / 255.0
    roi_frame_batch = np.expand_dims(roi_frame_normalized, axis=0)

    # Perform inference with MobileNetV3s
    (
        mobilenet_inputs,
        mobilenet_outputs,
        mobilenet_bindings,
        mobilenet_stream,
    ) = common.allocate_buffers(mobilenet_engine)
    mobilenet_inputs[0].host = roi_frame_batch
    mobilenet_outputs = common.do_inference_v2(
        mobilenet_context,
        bindings=mobilenet_bindings,
        inputs=mobilenet_inputs,
        outputs=mobilenet_outputs,
        stream=mobilenet_stream,
    )

    
    class_logits = np.array(mobilenet_outputs[0])

    # Convert logits to probabilities and apply softmax
    class_probabilities = np.exp(class_logits) / np.sum(np.exp(class_logits))

    # Get the index of the highest probability class after applying bias
    predicted_class = np.argmax(class_probabilities)
    return predicted_class


def navigation(predicted_class):
    ready_class = None
    # Static variables to keep the timestamp and class
    if not hasattr(navigation, "last_time"):
        navigation.last_time = time.time()
        navigation.last_class = predicted_class

    current_time = time.time()
    # Check if the current predicted class is the same as the last predicted class
    if predicted_class == navigation.last_class:
        # Check if it has been more than one second
        if (current_time - navigation.last_time) > 0.5:
            ready_class = predicted_class
            print(f">>>>>>> {predicted_class}")
            send_command(predicted_class)
            # Reset the timer for the new class detection
            navigation.last_time = current_time
    else:
        # Update the last predicted class and reset the timer
        navigation.last_class = predicted_class
        navigation.last_time = current_time
    return predicted_class

# send command to arduino
def send_command(command):
    label_map = {0: "q", 1: "w", 2: "a", 3: "d"}
    command = label_map.get(command)
    print(f"Command: {command}")
    # ser.write(command.encode())

def landmark_detection(frame, input_shape, keypoints_engine, keypoints_context, cap):
    im = cv2.cvtColor(cv2.resize(frame, input_shape), cv2.COLOR_BGR2RGB)
    im = np.expand_dims(np.asarray(im, dtype="float32"), axis=0)

    # Perform keypoints inference
    (
        keypoints_inputs,
        keypoints_outputs,
        keypoints_bindings,
        keypoints_stream,
    ) = common.allocate_buffers(keypoints_engine)
    keypoints_inputs[0].host = im
    keypoints_outputs = common.do_inference_v2(
        keypoints_context,
        bindings=keypoints_bindings,
        inputs=keypoints_inputs,
        outputs=keypoints_outputs,
        stream=keypoints_stream,
    )
    # Process and display keypoints results
    landmarks = np.array(keypoints_outputs[0]).reshape([-1, 2])
    centers = [
        tuple(
            map(
                int,
                landmark
                * np.array(
                    [
                        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    ],
                ),
            )
        )
        for landmark in landmarks[36:40:3]
    ]
    return centers

def tensor_to_cv2(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    image_numpy = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

def process_frame(frame, model):
    # Convert the frame to tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_pil = frame_pil.resize((224, 224))
    frame_tensor = torch.Tensor(np.array(frame_pil)).permute(2, 0, 1).float() / 255.0
    frame_tensor = frame_tensor.unsqueeze(0).cuda()

    # Inference
    with torch.no_grad():
        input = Variable(frame_tensor).cuda()
        _, r = model(input)
        enhanced_image = tensor_to_cv2(r)

    return enhanced_image

def main():
    # Video capture setup
    # gst_str = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)352, height=(int)288, framerate=(fraction)30/1 ! jpegdec ! videoconvert ! appsink"
    # cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture("/dev/video0")
    print(
        f"Input Video (height, width, fps): {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, {cap.get(cv2.CAP_PROP_FPS)}"
    )

    # Load SCI model
    modelSCI = Finetunemodel("models/sci.pt")
    modelSCI = modelSCI.cuda()
    modelSCI.eval()

    # Load keypoints model
    keypoints_model = "models/keypoints.trt"
    keypoints_engine = get_engine(keypoints_model)
    keypoints_context = keypoints_engine.create_execution_context()
    input_shape = (160, 160)

    # Load MobileNetV3s model
    mobilenet_engine_path = "models/model3.engine"
    mobilenet_engine = get_engine(mobilenet_engine_path)
    mobilenet_context = mobilenet_engine.create_execution_context()
    mobilenet_input_shape = (224, 224)  # Assuming MobileNetV3s expects 224x224 input

    while cap.isOpened():
        start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break
        frame = cv2.flip(frame, 1)
        frame = process_frame(frame, modelSCI)
        frame = cv2.resize(frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        centers = landmark_detection(frame, input_shape, keypoints_engine, keypoints_context, cap)
        roi = create_roi(frame, find_center_of_line(centers), (87, 87))
        cv2.rectangle(frame, (roi[0][0]-5, roi[0][1]-5), roi[1], (0, 0, 255), 1)

        # Get the ROI (Region of Interest) frame from your existing code
        roi_frame = frame[roi[0][1] : roi[1][1], roi[0][0] : roi[1][0]]

        # Call the classify_image function
        predicted_class = classify_image(roi_frame, mobilenet_context, mobilenet_engine)
        navigation(predicted_class)

        # Display the classification result
        label_map = {0: "close", 1: "forward", 2: "left", 3: "right"}
        class_text = f"{label_map.get(predicted_class)}"
        cv2.putText(
            frame, class_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        # Display inference time
        inference_time = (time.perf_counter() - start) * 1000
        fps_text = f"{inference_time:.2f}ms"
        cv2.putText(
            frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1
        )

        # Show the frame
        cv2.imshow("Eye Gaze Navigation", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()