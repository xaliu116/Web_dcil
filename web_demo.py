import gradio as gr
from AesUST.style_transfer_pipeline import *
from PIL import Image


# 提前加载模型
load_models()
run_style_transfer(Image.open("AesUST/inputs/content/0.jpg"), 
                   Image.open("AesUST/inputs/style/0.jpg"))


with gr.Blocks(title="图像智能处理") as demo:
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


demo.launch()

