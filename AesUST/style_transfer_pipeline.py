import torch
import torch.nn as nn
from torchvision import transforms
import AesUST.net as net
import time
from PIL import Image
import tempfile
from torchvision.utils import save_image
import zipfile
import os
import shutil
from config import ROOT_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = net.Transform(in_planes = 512)
discriminator = net.AesDiscriminator()
decoder = net.decoder
vgg = net.vgg
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def load_models():
    decoder.eval()
    transform.eval()
    vgg.eval()
    discriminator.eval()
    decoder.load_state_dict(torch.load(ROOT_PATH + './AesUST/models/decoder.pth'))
    transform.load_state_dict(torch.load(ROOT_PATH + './AesUST/models/transformer.pth'))
    vgg.load_state_dict(torch.load(ROOT_PATH + './AesUST/models/vgg_normalised.pth'))
    discriminator.load_state_dict(torch.load(ROOT_PATH + './AesUST/models/discriminator.pth'))

    enc_1.to(device)
    enc_2.to(device)
    enc_3.to(device)
    enc_4.to(device)
    enc_5.to(device)

    transform.to(device)
    decoder.to(device)
    discriminator.to(device)


def style_transfer(enc_1, enc_2, enc_3, enc_4, enc_5, content, style, alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    
    Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
    Content5_1 = enc_5(Content4_1)
    Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
    Style5_1 = enc_5(Style4_1)
    aesthetic_s_feats, _ = discriminator(style)
            
    if interpolation_weights:
        _, C, H, W = Content4_1.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = transform(Content4_1, Style4_1, Content5_1, Style5_1, aesthetic_s_feats)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
            
        if alpha < 1.0:
            aesthetic_c_feats, _ = discriminator(content)
            feat_cc = transform(Content4_1, Content4_1, Content5_1, Content5_1, aesthetic_c_feats)
            feat = feat * alpha + feat_cc[0:1] * (1 - alpha)
    
    else: 
        feat = transform(Content4_1, Style4_1, Content5_1, Style5_1, aesthetic_s_feats)
        
        if alpha < 1.0:
            aesthetic_c_feats, _ = discriminator(content)
            feat_cc = transform(Content4_1, Content4_1, Content5_1, Content5_1, aesthetic_c_feats)
            feat = feat * alpha + feat_cc * (1 - alpha)
        
    return decoder(feat)


def run_style_transfer(content_image, style_image):
    content_tf = test_transform(0, False)
    style_tf = test_transform(0, False)

    content = content_tf(content_image)
    style = style_tf(style_image)

    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    alpha = 1.0

    torch.cuda.synchronize() 
    start_time = time.time()

    with torch.no_grad():
        output = style_transfer(enc_1, enc_2, enc_3, enc_4, enc_5, content, style, alpha)

    torch.cuda.synchronize()
    end_time = time.time()
    print('Elapsed time: %.4f seconds' % (end_time - start_time))

    output.clamp(0, 255)
    output = output.cpu()

    output_pil = transforms.ToPILImage()(output[0])

    return output_pil

def run_multiple_style_transfer(content_paths, style_paths):
    content_tf = test_transform(0, False)
    style_tf = test_transform(0, False)
    temp_dir = tempfile.mkdtemp(prefix="transfered_images_")

    for content_path in content_paths:
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            alpha = 1.0

            torch.cuda.synchronize() 
            start_time = time.time()
            
            with torch.no_grad():
                output = style_transfer(enc_1, enc_2, enc_3, enc_4, enc_5, content, style, alpha)

            torch.cuda.synchronize()
            end_time = time.time()
            print('Elapsed time: %.4f seconds' % (end_time - start_time))

            output.clamp(0, 255)
            output = output.cpu()

            output_name = '{:s}_stylized_{:s}{:s}'.format(
                os.path.basename(content_path).split('.')[0], 
                os.path.basename(style_path).split('.')[0],  
                os.path.splitext(content_path)[1])
            save_path = os.path.join(temp_dir, output_name)
            save_image(output, save_path)

    zip_filename = "风格迁移后的图像.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname=arcname)

    shutil.rmtree(temp_dir)
    return zip_filename