import sys
sys.path.append("./first_order_motion")
sys.path.append("./RapidLaTeXOCR")

from first_order_motion.animation_pipeline import save_image_and_run
from inference.colorization_pipline import run_image_colorization, load_colorizer_model, colorize_image, colorize_files
from AesUST.style_transfer_pipeline import load_models, run_style_transfer, run_multiple_style_transfer
from PageNet.infer import Detector
import TextRec.TextColor as TextColor
from TextRec.demo import Demo
from RapidLaTeXOCR.demo import LatexConverter, LatexOCR

import gradio as gr
from PIL import Image
from docx import Document

from config import ROOT_PATH


# 图像上色部分
load_colorizer_model()
run_image_colorization(image_path=ROOT_PATH + "./DDColor/test_imgs/000000050145.jpg")

# 图像风格迁移部分
load_models()
run_style_transfer(Image.open(ROOT_PATH + "./AesUST/inputs/content/0.jpg"), 
                   Image.open(ROOT_PATH + "./AesUST/inputs/style/0.jpg"))

# 图像动态化部分
def image_aninmation(video, image):
    return save_image_and_run(video, image)

# 手写体识别部分
detector = Detector()

# 多文本识别部分
def multiTex_Recognition(img):
    res = TextColor.Solve_Image(img)
    text = Demo(res)
    return text

# 公式识别部分
model = LatexOCR()

def save_docx_and_get_latex(img):
    latex_res, elapse = model(img)
    converter = LatexConverter()
    eqn = converter.convert(latex_res)

    doc = Document()
    paragraph = doc.add_paragraph(style=None)
    paragraph._element.append(eqn)
    
    docx_path = 'recognized_formula.docx'
    doc.save(docx_path)

    return latex_res, docx_path


# Gradio 接口部分
with gr.Blocks(title="图像智能处理") as demo:
    with gr.Tab("图像上色"):
        with gr.Tab("单张图像上色"):
            with gr.Row():
                input_colorization_image = gr.Image(label="原始图像")
                output_colorization_image = gr.Image(label="上色图像")
            with gr.Row():
                btn_clear_colorization_image = gr.ClearButton(value="清空",components=[input_colorization_image, output_colorization_image])
                btn_colorization_image = gr.Button("图像上色")
        with gr.Tab("批量处理"):
            with gr.Row():
                input_colorization_files =  gr.File(label="上传图像列表", file_count="multiple")
                output_colorization_files = gr.File(label="上色压缩包")
            with gr.Row():
                btn_clear_colorization_files = gr.ClearButton(value="清空", components=[input_colorization_files, output_colorization_files])
                btn_colorization_files = gr.Button("图像上色")

    btn_colorization_image.click(fn=colorize_image,inputs=input_colorization_image, outputs=output_colorization_image)
    btn_colorization_files.click(fn=colorize_files,inputs=input_colorization_files, outputs=output_colorization_files)

    with gr.Tab("图像风格迁移"):
        with gr.Tab("一组图像风格迁移"):
            with gr.Row():
                input_stylize_content_image = gr.Image(label="内容图像")
                input_stylize_style_image = gr.Image(label="风格图像")
                output_stylize_image = gr.Image()
            with gr.Row():
                btn_stylize_image = gr.Button("风格迁移")
                btn_clear_stylize_image = gr.ClearButton(value="清空", components=[input_stylize_content_image, input_stylize_style_image, output_stylize_image])
        with gr.Tab("多组图像风格迁移"):
            with gr.Row():
                input_stylize_content_files =  gr.File(label="上传内容图像列表", file_count="multiple")
                input_stylize_style_files =  gr.File(label="上传风格图像列表", file_count="multiple")
                output_stylize_files = gr.File(label="风格迁移压缩包")   
            with gr.Row():
                btn_stylize_files = gr.Button("风格迁移")
                btn_clear_stylize_files = gr.ClearButton(value="清空", components=[input_stylize_content_files, input_stylize_style_files, output_stylize_files])

    btn_stylize_image.click(run_style_transfer, inputs=[input_stylize_content_image, input_stylize_style_image], outputs=output_stylize_image)
    btn_stylize_files.click(run_multiple_style_transfer, inputs=[input_stylize_content_files, input_stylize_style_files], outputs=output_stylize_files)

    with gr.Tab("图像动态化"):
        with gr.Row():
            input_transfer_video = gr.Video(label="上传视频")
            input_transfer_image = gr.Image(label="上传图像")
            output_transfer_video = gr.Video(label="动态化视频")
        with gr.Row():
            btn_transfer_image = gr.Button("图像动态化")
            btn_clear_transfer_image = gr.ClearButton(value="清空", components=[input_transfer_video, input_transfer_image, output_transfer_video])

    btn_transfer_image.click(image_aninmation, inputs=[input_transfer_video, input_transfer_image], outputs=output_transfer_video)

    with gr.Tab("手写体检测识别"):
        with gr.Row():
            input_detect_image = gr.Image(type='filepath', label="上传手写体图像")
            output_detect_image = gr.Image(label="手写体检测结果")
        with gr.Row():
            btn_det_image = gr.Button("手写体检测识别")
            btn_clear_det_image = gr.ClearButton(value="清空", components=[input_detect_image, output_detect_image])

    btn_det_image.click(fn=detector.detect, inputs=input_detect_image, outputs=output_detect_image)
    
    with gr.Tab("多语言文本识别"):
        with gr.Row():
            input_detect_image = gr.Image(label="上传多语言文本图像")
            output_detect_text = gr.Textbox(label="多语言文本识别结果")
        with gr.Row():
            btn_mtr_image = gr.Button("多语言文本识别")
            btn_clear_mtr_image = gr.ClearButton(value="清空", components=[input_detect_image, output_detect_text])

    btn_mtr_image.click(fn=multiTex_Recognition, inputs=input_detect_image, outputs=output_detect_text)

    with gr.Tab("公式识别"):
        with gr.Row():
            input_detect_image = gr.Image(label="上传公式图像")
            output_detect_text = gr.Textbox(label="公式识别latex结果")
            output_detect_omml = gr.File(label="公式识别word结果")
        with gr.Row():
            btn_latex_image = gr.Button("公式识别")
            btn_clear_latex_image = gr.ClearButton(value="清空", components=[input_detect_image, output_detect_text])

    btn_latex_image.click(fn=save_docx_and_get_latex, inputs=input_detect_image, outputs=[output_detect_text, output_detect_omml])


# 启动 Gradio 接口
demo.launch(height=800, share=True, debug=True)
