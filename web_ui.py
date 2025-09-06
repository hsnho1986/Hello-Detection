import cv2
import gradio as gr
import numpy as np
from openvino.runtime import Core

core = Core()
model_path = "model/horizontal-text-detection-0001.xml"
model = core.read_model(model_path)
compiled_model = core.compile_model(model, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output("boxes")
N, C, H, W = input_layer.shape


def convert_result_to_image(bgr_image, resized_image, boxes, threshold=0.3):
    colors = {"red": (255, 0, 0)}
    (real_y, real_x), (resized_y, resized_x) = (
        bgr_image.shape[:2],
        resized_image.shape[:2],
    )
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    for box in boxes:
        conf = box[-1]
        if conf > threshold:
            xmin, ymin, xmax, ymax = box[:-1]
            xmin *= ratio_x
            xmax *= ratio_x
            ymin *= ratio_y
            ymax *= ratio_y
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            cv2.rectangle(rgb_image, pt1, pt2, colors["red"], 2)
    return rgb_image


def detect(image):
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(bgr_image, (W, H))
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    boxes = compiled_model([input_image])[output_layer]
    boxes = boxes[~np.all(boxes == 0, axis=1)]
    return convert_result_to_image(bgr_image, resized_image, boxes)


demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy", label="Input image"),
    outputs=gr.Image(type="numpy", label="Detected text"),
    title="Horizontal Text Detection",
    description="Upload an image to detect text using OpenVINO model.",
)


if __name__ == "__main__":
    demo.launch()
