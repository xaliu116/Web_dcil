import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F
import shutil
import tempfile
import zipfile

from config import ROOT_PATH


class ImageColorizationPipeline(object):

    def __init__(self, input_size=512):
        
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.decoder_type = "MultiScaleColorDecoder"

        if self.decoder_type == 'MultiScaleColorDecoder':
            self.model = DDColor(
                'convnext-l',
                decoder_name='MultiScaleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ).to(self.device)
        else:
            self.model = DDColor(
                'convnext-l',
                decoder_name='SingleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=256,
            ).to(self.device)

        
    @torch.no_grad()
    def load_state_dict(self, model_path):
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()

    @torch.no_grad()
    def process(self, img):
        self.height, self.width = img.shape[:2]
        # print(self.width, self.height)
        # if self.width * self.height < 100000:
        #     self.input_size = 256

        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # resize rgb image -> lab -> get grey -> rgb
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor_gray_rgb).cpu()  # (1, 2, self.height, self.width)

        # resize ab -> concat original l -> rgb
        output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)    

        return output_img


def colorize_image(img,model_path=ROOT_PATH + "./DDColor/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"):
    colorizer = ImageColorizationPipeline(model_path=model_path)
   
    image_out = colorizer.process(img)
    
    return image_out


colorizer = ImageColorizationPipeline() 
def load_colorizer_model(model_path=ROOT_PATH + "./DDColor/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"):
    
    colorizer.load_state_dict(model_path)   

def run_image_colorization(image_path, image_out_path=None):
    # Load the image
    img = cv2.imread(image_path)
    # Process the image
    image_out = colorizer.process(img)

    # Save the image
    if image_out_path  is not None:
        cv2.imwrite(image_out_path, image_out)

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pretrain/net_g_200000.pth')
    parser.add_argument('--input', type=str, default='figure/', help='input test image folder or video path')
    parser.add_argument('--output', type=str, default='results', help='output folder or video path')
    parser.add_argument('--input_size', type=int, default=512, help='input size for model')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    img_list = os.listdir(args.input)
    assert len(img_list) > 0

    colorizer = ImageColorizationPipeline(model_path=args.model_path, input_size=args.input_size)

    for name in tqdm(img_list):
        img = cv2.imread(os.path.join(args.input, name))
        image_out = colorizer.process(img)
        cv2.imwrite(os.path.join(args.output, name), image_out)


if __name__ == '__main__':
    main()
