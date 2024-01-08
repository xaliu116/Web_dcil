import gradio as gr
import torch
import TextRec.model.crnn as crnn
import TextRec.utils
import torch.utils.data
import cv2
from PIL import Image
import torchvision.transforms as transforms
import TextRec.TextColor as TextColor

# Define Alpha
alphabet = open('./TextRec/data/benchmark.txt', 'r').read()
converter = TextRec.utils.strLabelConverter(alphabet)

# load modal
nclass = len(alphabet) + 1
crnn = crnn.CRNN(3, 256, nclass, 32).cuda()
crnn = torch.nn.DataParallel(crnn)
crnn.load_state_dict(torch.load('./TextRec/history/exp_1124/best_model.pth'))


def demo(net, res, converter):
    print('Start Demo')
    for para in net.parameters():
        para.requires_grad = False
    net.eval()

    # get data
    all_img = []
    for img in res:
        # 缩放
        oriH = img.shape[0]
        oriW = img.shape[1]
        img = cv2.resize(img, (int(32 / oriH * oriW), 32), Image.BILINEAR)
        img = transforms.ToTensor()(img)
        all_img.append(img)
    num = len(all_img)

    text = ''

    for i in range(num):
        image = all_img[i]
        image = torch.unsqueeze(image, dim=0)
        batch_size = len(image)
        # images = torch.Tensor(image)

        preds = net(image)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        res_real = converter.decode(preds, preds_size, raw=False)
        text = text + str(res_real) +'\r'

    return text


def demo_web(img):
    # test data
    res = TextColor.Solve_Image(img)
    text = demo(crnn, res, converter)
    return text


# inputs = gr.inputs.Image()
# outputs = gr.outputs.Textbox()
# gr.Interface(fn=demo_web, inputs=inputs, outputs=outputs).launch()
