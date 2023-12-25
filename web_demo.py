import gradio as gr
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
<<<<<<< HEAD
=======
import cv2
>>>>>>> 20838012b4d549e0366888bc62ff8ad0c24127a4

def dcnet_style_transfer_anime(content_img):
    img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                           model='damo/cv_unet_person-image-cartoon_compound-models')
<<<<<<< HEAD
    result = img_cartoon(content_img)
=======
    print(content_img.shape)
    result = img_cartoon(dict(content=content_img))
>>>>>>> 20838012b4d549e0366888bc62ff8ad0c24127a4
    output_img = result[OutputKeys.OUTPUT_IMG]
    return output_img

def dcnet_style_transfer_3d(content_img):
    img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                           model='damo/cv_unet_person-image-cartoon-3d_compound-models')
<<<<<<< HEAD
    result = img_cartoon(content_img)
=======
    result = img_cartoon(dict(content=content_img))
>>>>>>> 20838012b4d549e0366888bc62ff8ad0c24127a4
    output_img = result[OutputKeys.OUTPUT_IMG]
    return output_img

with gr.Blocks(title="Style Transfer Demo") as demo:
    with gr.Tab("Cartoon Style Transfer"):
        with gr.Row():
            dcnet_input_anime = gr.Image()
            dcnet_output_anime = gr.Image()
        dcnet_button_anime = gr.Button("Style Transfer")
    with gr.Tab("3d Style Transfer"):
        with gr.Row():
            dcnet_input_3d = gr.Image()
            dcnet_output_3d = gr.Image()
        dcnet_button_3d = gr.Button("Style Transfer")

    dcnet_button_anime.click(dcnet_style_transfer_anime, inputs=dcnet_input_anime, outputs=dcnet_output_anime)
    dcnet_button_3d.click(dcnet_style_transfer_3d, inputs=dcnet_input_3d, outputs=dcnet_output_3d)


demo.launch()

