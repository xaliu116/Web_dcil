import gradio as gr
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

def dcnet_style_transfer_anime(content_img):
    img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                           model='damo/cv_unet_person-image-cartoon_compound-models')
    result = img_cartoon(content_img)
    output_img = result[OutputKeys.OUTPUT_IMG]
    return output_img

def dcnet_style_transfer_3d(content_img):
    img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                           model='damo/cv_unet_person-image-cartoon-3d_compound-models')
    result = img_cartoon(content_img)
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

