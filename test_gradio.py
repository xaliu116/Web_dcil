import numpy as np
import gradio as gr
import cv2 
from inference.colorization_pipline import ImageColorizationPipeline

colorizer = ImageColorizationPipeline(model_path="DDColor/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt")
# 上色网络预热  慢启动
test_image = cv2.imread("DDColor/test_imgs/000000050145.jpg")
test = colorizer.process(test_image)
del test
del test_image


def colorize_image(input_img):
   
    image_out = colorizer.process(input_img)
    image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
    
    return image_out


with gr.Blocks(title="图像智能处理") as demo:
    with gr.Tab("Image Colorization"):
        with gr.Row():
            input = gr.Image()
            output = gr.Image()

        btn1 = gr.Button("Image Colorization")
    
    btn1.click(fn=colorize_image,inputs=input, outputs=output)
    with gr.Tab("style transfer"):
        with gr.Row():
            input = gr.Image()
            output = gr.Image()
        
        btn2 = gr.Button("style transfer")

    

demo.launch(height = 600)