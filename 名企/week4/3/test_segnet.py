import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
# from model.fcn import VGGNet, FCNs
from dataset.lane_cls_data import LaneTestDataset
import os
from tqdm import tqdm
#from model.resnet import resnet50
#from model.resnet50_fcn import FCNs
from model.SegNet import SegNet

CKPT_PATH = "./ckpt/segnet/epoch_40.pth"
OUT_PATH = "./output/segnet/"
IMG_H = 288
IMG_W = 800


def main():
    os.makedirs(OUT_PATH, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seg_model = SegNet(3, 2)
    if device == 'cuda':
        seg_model.to(device)
    seg_model.load_state_dict(torch.load(CKPT_PATH))

    seg_model.eval()

    test_set = LaneTestDataset(list_path='./test.tsv',
                               dir_path='./data_road',
                               img_shape=(IMG_W, IMG_H))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():
        for image, image_path in tqdm(test_loader):
            image = image.to(device)
            output = seg_model(image)
            output = torch.sigmoid(output)
            mask = torch.argmax(output, dim=1).cpu().numpy().transpose((1, 2, 0))
            mask = mask.reshape(IMG_H, IMG_W)
            image = image.cpu().numpy().reshape(3, IMG_H, IMG_W).transpose((1, 2, 0)) * 255
            image[..., 2] = np.where(mask == 0, 255, image[..., 2])

            cv2.imwrite(os.path.join(OUT_PATH, os.path.basename(image_path[0])), image)


if __name__ == "__main__":
    main()
