import cv2
import os
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from dataset import KittiRoadTestDataset
from transform import Resize
from torchvision.transforms import ToTensor
from models import Vgg11, Fcn, ProxyNet

if __name__ == '__main__':
    ITERATION, BATCH_SIZE, CLASSES_NUM = 50, 1, 2
    HEIGHT, WIDTH = 288, 800
    vgg11 = Vgg11()
    fineTuringNet = ProxyNet('vgg11', vgg11.features)
    model = Fcn(scale=8, featureProxyNet=fineTuringNet, classesNum=CLASSES_NUM)

    # 加载模型
    state_dict = torch.load('fcn_road_segment.pth')
    model.load_state_dict(state_dict)

    testSet = KittiRoadTestDataset(path='./data_road/training', type='um', transforms=[
        Resize(HEIGHT, WIDTH, cv2.INTER_LINEAR),
        # HistogramNormalize(),
        # Mixup(shadowImg, random_translation=False, random_rotation=False),
        ToTensor(),
    ])
    testLoader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=0)
    total = len(testSet)
    for batch, (img, img_path) in enumerate(testLoader):
        with torch.no_grad():
            img = img
            output = model(img)
            output = torch.sigmoid(output)
            output = output.reshape((CLASSES_NUM, 288, 800)).numpy()
            segment = np.zeros((3, 288, 800))
            if CLASSES_NUM == 1:
                segment[2, output[0] >= 0.5] = 1
            else:
                segment[2, output[0] >= output[1]] = 1
            segment *= 255
            segment = segment.astype(np.uint8).transpose((1, 2, 0))

            source = img
            source = source.reshape((3, 288, 800)).numpy()
            source *= 255
            source = source.astype(np.uint8).transpose((1, 2, 0))

            # 合并道路图片和原图
            segment = cv2.addWeighted(source, 1, segment, 0.5, 0)
            name = 'output/segment_' + os.path.basename(img_path[0])
            cv2.imwrite(name, segment)
            print('[{}/{}] generate {}'.format(batch, total, name))