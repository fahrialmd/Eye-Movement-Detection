import cv2
import dlib
import numpy as np
import onnxruntime
import time


def send_command(command):
    if command == "forward":
        print(
            "##################################MAJU##################################"
        )
    elif command == "left":
        print(
            "##################################KIRI##################################"
        )
    elif command == "right":
        print(
            "##################################KANAN##################################"
        )
    elif command == "stop":
        print(
            "##################################BERHENTI##################################"
        )


def classify_frame(frame, session):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    frame_batch = frame_batch.astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = session.run([output_name], {input_name: frame_batch})[0]
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_probability = predictions[0][predicted_class_index]

    label_map = {0: "close", 1: "forward", 2: "left", 3: "right"}
    predicted_class = label_map.get(predicted_class_index, "Unknown")

    return predicted_class, predicted_class_probability


def get_intersection_point(A1, B1, C1, A2, B2, C2):
    det = A1 * B2 - A2 * B1
    x_center = (C1 * B2 - C2 * B1) / det
    y_center = (A1 * C2 - A2 * C1) / det
    return int(x_center), int(y_center)


def get_roi(frame, landmarks):
    coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in [37, 40, 38, 41]]
    A1, B1, C1 = (
        coords[1][1] - coords[0][1],
        coords[0][0] - coords[1][0],
        (coords[1][1] - coords[0][1]) * coords[0][0]
        + (coords[0][0] - coords[1][0]) * coords[0][1],
    )
    A2, B2, C2 = (
        coords[3][1] - coords[2][1],
        coords[2][0] - coords[3][0],
        (coords[3][1] - coords[2][1]) * coords[2][0]
        + (coords[2][0] - coords[3][0]) * coords[2][1],
    )
    x_center, y_center = get_intersection_point(A1, B1, C1, A2, B2, C2)
    buff2 = 43
    pt1, pt2 = (x_center - buff2, y_center - buff2), (
        x_center + buff2,
        y_center + buff2,
    )
    return cv2.resize(frame[pt1[1] : pt2[1], pt1[0] : pt2[0]], (224, 224))


def show_info(frame, output_text):
    cv2.putText(
        frame,
        output_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def main():
    onnx_model_path = "epoch50.onnx"
    providers = [
        ("TensorrtExecutionProvider", {
            'trt_max_workspace_size': 2147483648,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': 'trt_engine_cache/'
        }),
        'CUDAExecutionProvider', 'CPUExecutionProvider'
    ]
    session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    video_path = "/dev/video2"
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    prev_class_name = None
    same_class_start_time = None
    close_class_start_time = None

    output_text = ""
    command_name = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            roi = get_roi(frame, landmarks)
            class_name, class_probability = classify_frame(roi, session)

            cv2.putText(
                frame,
                f"Class: {class_name} Probability: {class_probability:.4f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                f"{fps}",
                (width - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            if prev_class_name == class_name:
                if same_class_start_time is None:
                    same_class_start_time = time.time()
                if class_name == "close":
                    if close_class_start_time is None:
                        close_class_start_time = time.time()
                    elif time.time() - close_class_start_time >= 1:
                        output_text = "Override"
                        send_command(command_name)
                        if time.time() - close_class_start_time >= 1.5:
                            output_text = "STOP"
                            send_command("stop")
                        show_info(frame, output_text)
                        command_name = ""
                        output_text = ""
                else:
                    close_class_start_time = None
                if time.time() - same_class_start_time >= 0.3:
                    if class_name != "close":
                        if output_text == "":
                            output_text = f"Available: {class_name}"
                            command_name = class_name
                    else:
                        if time.time() - close_class_start_time >= 0.3:
                            show_info(frame, output_text)
                            output_text = ""
                            break
            else:
                same_class_start_time = None

            prev_class_name = class_name
            cv2.imshow("ROI", roi)

        show_info(frame, output_text)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
