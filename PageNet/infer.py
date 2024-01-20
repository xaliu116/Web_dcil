import os

import numpy as np

from PageNet.model import build_model
from PageNet.utils.converter import Converter
import yaml
from PageNet.utils.parser import default_parser
import torch

from PageNet.utils.decode import det_rec_nms, PageDecoder
from torchvision.transforms import Compose
from PageNet.data.transforms_test import RandomResize, SizeAjust, ToTensor

import cv2
from PIL import Image, ImageDraw, ImageFont

from config import ROOT_PATH


class Detector:
    def __init__(self):
        parser = default_parser()
        args = parser.parse_args()
        args.config = ROOT_PATH + './PageNet/configs/casia-hwdb.yaml'
        self.cfg = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
        self.init_model()

    def init_model(self):
        # dict
        self.cfg['DATA']['DICT'] = ROOT_PATH + './PageNet/dicts/casia-hwdb.txt'
        self.converter = Converter(self.cfg['DATA']['DICT'])

        # build model
        self.model = build_model(self.cfg)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        os.makedirs(self.cfg['OUTPUT_FOLDER'], exist_ok=True)

        self.model.eval()

        # build processor
        layout = self.cfg['POST_PROCESS']['LAYOUT'] if 'LAYOUT' in self.cfg['POST_PROCESS'] else 'generic'
        self.page_decoder = PageDecoder(
            se_thres=self.cfg['POST_PROCESS']['SOL_EOL_CONF_THRES'],
            max_steps=self.cfg['POST_PROCESS']['READ_ORDER_MAX_STEP'],
            layout=layout
        )

        self.image_mode = self.cfg['DATA']['VAL']['IMAGE_MODE']

        tfm_cfgs = self.cfg['DATA']['VAL']

        transforms = []
        force_resize = tfm_cfgs['FORCE_RESIZE'] if 'FORCE_RESIZE' in tfm_cfgs else True
        transforms.append(RandomResize(tfm_cfgs['WIDTHS'], tfm_cfgs['MAX_HEIGHT'], force_resize))
        transforms.append(SizeAjust(tfm_cfgs['SIZE_STRIDE']))
        transforms.append(ToTensor())
        if len(transforms) == 0:
            return None
        self.transforms = Compose(transforms)

    def detect(self,image):
        oriImage = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image = self.transforms(oriImage).unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            pred_det_rec, pred_read_order, pred_sol, pred_eol = self.model(image)
        pred_det_rec = pred_det_rec[0].cpu().numpy()
        pred_read_order = pred_read_order[0].cpu().numpy()
        pred_sol = pred_sol[0].cpu().numpy()
        pred_eol = pred_eol[0].cpu().numpy()

        pred_det_rec = det_rec_nms(
            pred_det_rec=pred_det_rec,
            img_shape=image.shape[-2:],
            dis_weight=self.cfg['POST_PROCESS']['DIS_WEIGHT'],
            conf_thres=self.cfg['POST_PROCESS']['CONF_THRES'],
            nms_thres=self.cfg['POST_PROCESS']['NMS_THRES']
        )

        img = self.paint_det(oriImage,pred_det_rec)
        # cv2.imwrite('res.jpg',img)
        line_results, _ = self.page_decoder.decode(
            output=pred_det_rec,
            pred_read=pred_read_order,
            pred_start=pred_sol,
            pred_end=pred_eol,
            img_shape=image.shape[-2:],
        )

        line_words = []
        for line_result in line_results:
            line_word = [np.argmax(pred_det_rec[id][5:]) for id in line_result]
            line_word = self.converter.decode(line_word)
            line_words.append(line_word)

        return img

    def paint_det(self,oriImage, det_rec):
       image = cv2.cvtColor(oriImage, cv2.COLOR_BGR2RGB)
       image = Image.fromarray(image)
       draw = ImageDraw.Draw(image)
       det_size = det_rec.shape[0]
       font = ImageFont.truetype(ROOT_PATH + "./PageNet/SIMSUN.TTC",40)
       for i in range(det_size):
           box_xywh = det_rec[i][:4]
           word = self.converter.decode([np.argmax(det_rec[i][5:])])
           draw.rectangle([(int)(box_xywh[0] - box_xywh[2] / 2),
                           (int)(box_xywh[1] - box_xywh[3] / 2),
                           (int)(box_xywh[0] + box_xywh[2] / 2),
                           (int)(box_xywh[1] + box_xywh[3] / 2)],
                          outline=(255,0,0),
                          width=2)
           # cv2.rectangle(image,
           #               ((int)(box_xywh[0] - box_xywh[2] / 2), (int)(box_xywh[1] - box_xywh[3] / 2)),
           #               ((int)(box_xywh[0] + box_xywh[2] / 2), (int)(box_xywh[1] + box_xywh[3] / 2)),
           #               (0,255,0,0),
           #               2)

           draw.text(((int)(box_xywh[0] + box_xywh[2] / 2), (int)(box_xywh[1] - box_xywh[3] / 2)),word,font=font,fill=(0,0,0),stroke_width=1)
           # cv2.putText(image, word, ((int)(box_xywh[0] + box_xywh[2] / 2), (int)(box_xywh[1] - box_xywh[3] / 2)),cv2.FONT_HERSHEY_COMPLEX, 2.0, (0,0,0), 5)
       image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
       return image

# def main(cfg):
#     val_dataset = build_dataset(cfg, 'val')
#     val_dataloader = build_dataloader(val_dataset, 'val', cfg)
#     converter = Converter(cfg['DATA']['DICT'])
#
#     model = build_model(cfg)
#     model = model.cuda()
#
#     os.makedirs(cfg['OUTPUT_FOLDER'], exist_ok=True)
#
#     # validate(model, val_dataloader, converter, cfg)
#     # validate
#     model.eval()
#
#     layout = cfg['POST_PROCESS']['LAYOUT'] if 'LAYOUT' in cfg['POST_PROCESS'] else 'generic'
#     page_decoder = PageDecoder(
#         se_thres=cfg['POST_PROCESS']['SOL_EOL_CONF_THRES'],
#         max_steps=cfg['POST_PROCESS']['READ_ORDER_MAX_STEP'],
#         layout=layout
#     )
#
#     total_De = 0
#     total_Se = 0
#     total_Ie = 0
#     total_Len = 0
#     to_log = ''
#     for sample in tqdm(val_dataloader):
#         images = sample['image'].cuda()
#         labels = sample['label']
#         num_chars = sample['num_char_per_line']
#         filename = sample['filename']
#
#         with torch.no_grad():
#             pred_det_rec, pred_read_order, pred_sol, pred_eol = model(images)
#         pred_det_rec = pred_det_rec[0].cpu().numpy()
#         pred_read_order = pred_read_order[0].cpu().numpy()
#         pred_sol = pred_sol[0].cpu().numpy()
#         pred_eol = pred_eol[0].cpu().numpy()
#
#         pred_det_rec = det_rec_nms(
#             pred_det_rec=pred_det_rec,
#             img_shape=images.shape[-2:],
#             dis_weight=cfg['POST_PROCESS']['DIS_WEIGHT'],
#             conf_thres=cfg['POST_PROCESS']['CONF_THRES'],
#             nms_thres=cfg['POST_PROCESS']['NMS_THRES']
#         )
#
#         line_results, _ = page_decoder.decode(
#             output=pred_det_rec,
#             pred_read=pred_read_order,
#             pred_start=pred_sol,
#             pred_end=pred_eol,
#             img_shape=images.shape[-2:],
#         )

if __name__ == '__main__':
    detector = Detector()
    detector.detect('1.jpg')

    #cfg = yaml.load(open(args.config, 'r'))