#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorrt as trt
import common
import time

# Set up TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

WINDOW_NAME = "Eye Gaze Navigation"

def get_engine(engine_file_path):
    """Loads a TensorRT engine from a file"""
    print(f"Reading engine from file {engine_file_path}")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def find_center_of_line(coords):
    """Calculates the center of a line given by two points"""
    (x1, y1), (x2, y2) = coords
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def create_roi(image, center_point, roi_size=(100, 100), out_size=(100,100)):
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
    # roi = image[start_y:end_y, start_x:end_x]
    # roi = cv2.resize(roi, out_size)
    roi = [(start_x,start_y),(end_x,end_y)]
    return roi

def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # Video capture setup
    gst_str = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)352, height=(int)288, framerate=(fraction)30/1 ! jpegdec ! videoconvert ! appsink"
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    print(f"Input Video (height, width, fps): {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, {cap.get(cv2.CAP_PROP_FPS)}")

    # Load model
    model = "keypoints.trt"
    engine = get_engine(model)
    context = engine.create_execution_context()
    input_shape = (160, 160)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        im = cv2.cvtColor(cv2.resize(frame, input_shape), cv2.COLOR_BGR2RGB)
        im = np.expand_dims(np.asarray(im, dtype="float32"), axis=0)

        # Perform inference
        start = time.perf_counter()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = im
        outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        inference_time = (time.perf_counter() - start) * 1000

        # Process and display results
        landmarks = np.array(outputs[0]).reshape([-1, 2])
        centers = [tuple(map(int, landmark * np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]))) for landmark in landmarks[36:40:3]]
        roi = create_roi(frame, find_center_of_line(centers),(50,50))
        cv2.rectangle(frame,roi[0],roi[1],(0,0,255),1)

        fps_text = f"{inference_time:.2f}ms"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
