import gradio as gr
from AesUST.style_transfer_pipeline import *
from PIL import Image
import cv2 
from DDColor.inference.colorization_pipline import ImageColorizationPipeline
import os
import zipfile
import tempfile
import shutil


colorizer = ImageColorizationPipeline(model_path="DDColor/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt")
# 上色网络预热  慢启动
test_image = cv2.imread("DDColor/test_imgs/000000050145.jpg")
colorizer.process(test_image)

def colorize_image(input_img):
   
    image_out = colorizer.process(input_img)
    image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
    
    return image_out

def coloriza_files(input_files):
    # Create a temporary directory to save the colorized images
    temp_dir = tempfile.mkdtemp(prefix="colorized_images_")


    # Process each file and save the colorized images
    for file_path in input_files:
        img = cv2.imread(file_path)
        colorized_img = colorizer.process(img)
        
        save_path = os.path.join(temp_dir, os.path.basename(file_path))
        cv2.imwrite(save_path, colorized_img)

    # Create a zip file containing the colorized images
    zip_filename = "colorized_images.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname=arcname)

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    return zip_filename

# Define the function to handle file uploads for colorizing multiple files
def colorize_files(input_files):
    zip_filename = coloriza_files(input_files)
    return zip_filename

# 提前加载模型
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


    btn_colorization_image.click(fn=colorize_image,inputs=input, outputs=output)
    btn_colorization_files.click(fn=colorize_files,inputs=input_files, outputs=output_files)
    btn_stylize_image.click(run_style_transfer, inputs=[input_content_image, input_style_image], outputs=output)
    btn_stylize_files.click(run_multiple_style_transfer, inputs=[input_content_files, input_style_files], outputs=output_files)


demo.launch()

