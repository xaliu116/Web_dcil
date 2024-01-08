import numpy as np
import gradio as gr
import cv2 
from inference.colorization_pipline import run_image_colorization, load_colorizer_model, colorize_image, colorize_files
import os
import zipfile
import tempfile
import shutil
import gradio as gr
from AesUST.style_transfer_pipeline import *
from PIL import Image

# 图像上色部分
load_colorizer_model()
run_image_colorization(image_path="DDColor/test_imgs/000000050145.jpg")

# 图像风格迁移部分
load_models()
run_style_transfer(Image.open("AesUST/inputs/content/0.jpg"), 
                   Image.open("AesUST/inputs/style/0.jpg"))


with gr.Blocks(title="图像智能处理") as demo:
    with gr.Tab("图像上色"):
        with gr.Tab("单张图像上色"):
            with gr.Row():
                input = gr.Image(label="原始图像")
                output = gr.Image(label="上色图像")
            with gr.Row():
                btn_clear_image = gr.ClearButton(value="清空",components=[input, output])
                btn_colorization_image = gr.Button("图像上色")
        with gr.Tab("批量处理"):
            with gr.Row():
                input_files =  gr.File(label="上传图像列表", file_count="multiple")
                output_files = gr.File(label="上色压缩包")
            with gr.Row():
                btn_clear_files = gr.ClearButton(value="清空", components=[input_files, output_files])
                btn_colorization_files = gr.Button("图像上色")
            
    btn_colorization_image.click(fn=colorize_image,inputs=input, outputs=output)
    btn_colorization_files.click(fn=colorize_files,inputs=input_files, outputs=output_files)

    with gr.Tab("图像风格迁移"):
        with gr.Tab("一组图像风格迁移"):
            with gr.Row():
                input_content_image = gr.Image(label="内容图像")
                input_style_image = gr.Image(label="风格图像")
                output = gr.Image()
            with gr.Row():
                btn_stylize_image = gr.Button("风格迁移")
                btn_clear_image = gr.ClearButton(value="清空", components=[input_content_image, input_style_image, output])
        with gr.Tab("多组图像风格迁移"):
            with gr.Row():
                input_content_files =  gr.File(label="上传内容图像列表", file_count="multiple")
                input_style_files =  gr.File(label="上传风格图像列表", file_count="multiple")
                output_files = gr.File(label="风格迁移压缩包")   
            with gr.Row():
                btn_stylize_files = gr.Button("风格迁移")
                btn_clear_files = gr.ClearButton(value="清空", components=[input_content_files, input_style_files, output_files])


    btn_stylize_image.click(run_style_transfer, inputs=[input_content_image, input_style_image], outputs=output)
    btn_stylize_files.click(run_multiple_style_transfer, inputs=[input_content_files, input_style_files], outputs=output_files)
    

    with gr.Tab("图像动态化"):
        with gr.Row():
            input = gr.Image()
            output = gr.Image()
        
        btn_transfer_image = gr.Button("图像动态化")
    
demo.launch(height = 800, share=True)