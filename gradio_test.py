import numpy as np
import gradio as gr
import cv2
from PageNet.infer import Detector
# from inference.colorization_pipline import ImageColorizationPipeline
import os
import zipfile
import tempfile
import shutil

detector = Detector()

with gr.Blocks(title="图像智能处理") as demo:
    with gr.Tab("手写体检测识别"):
        with gr.Row():
            input = gr.Image(type='filepath')
            output = gr.Image()
        btn_det_image = gr.Button("手写体检测识别")
        btn_det_image.click(fn=detector.detect,inputs=input,outputs=output)



demo.launch(height=800, share=True)