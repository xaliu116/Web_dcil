import gradio as gr
import TextRec.TextColor as TextColor
from TextRec.demo import Demo
import TextRec.model.crnn as crnn


def demo_web(img):
    # test data
    res = TextColor.Solve_Image(img)
    text = Demo(res)
    return text


inputs = gr.inputs.Image()
outputs = gr.outputs.Textbox()
gr.Interface(fn=demo_web, inputs=inputs, outputs=outputs).launch()