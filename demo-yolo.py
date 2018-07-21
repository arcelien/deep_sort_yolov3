#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')


def main(yolo):
    # open images
    image_name = "image_00016"
    frame = cv2.imread(image_name+".png") #fromarray(frame)
    # frame = frame[..., ::-1]
    image2 = Image.fromarray(frame) #Image.open(image_name+".png")

    print("raw image shape", frame.shape)

    boxs = yolo.detect_image(image2)

    print('number of detections:', len(boxs))
    print('output from yolo [x/y/w/h]', boxs)

    def tlbr(boxes_from_yolo):
        ret = np.asarray(boxes_from_yolo, dtype=np.float)
        ret[2:] += ret[:2]
        return ret

    # test ENCODER
    # Definition of the parameters
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    features = encoder(frame, boxs)
    print("encoder output, shape (N, 128)", features.shape, type(features))
    for i in range(len(features)):
        print(type(features))
        print("encoder feature %i" %i, features[i])
        # isaac = [0.17416, -0.0932972, -0.104568, -0.220437, -0.00695556, 0.0482332, -0.112266, -0.0695435, -0.0557783, 0.0213394, -0.0903171, -0.0502907, 0.0216275, -0.116078, 0.0536015, 0.0210424, 0.0249984, -0.123807, 0.133824, 0.0214012, -0.0744454, 0.0354529, 0.0187223, 0.0412712, -0.0624598, -0.0634697, -0.0875341, 0.0513688, 0.210238, -0.0843545, -0.0127525, -0.0301388, -0.00151958, 0.0585501, 0.129407, 0.221101, 0.00389636, -0.0926077, 0.19958, -0.0793905, 0.0685829, -0.0912193, 0.0345996, 0.0547468, 0.0360889, 0.0742928, 0.0578631, -0.0960154, 0.0123658, 0.0540992, -0.0755997, -0.0354944, -0.0292894, -0.0794065, 0.0667017, 0.279665, -0.051411, -0.0465481, -0.0336999, -0.0945843, -0.0606229, 0.187728, -0.0553175, -0.100959, -0.0450222, -0.0633078, -0.058316, -0.0276071, -0.0698739, -0.0335653, -0.0377984, 0.0185102, 0.0733323, -0.0709907, 0.0619849, 0.105787, 0.037568, 0.0650037, 0.0146009, 0.102158, 0.0346205, 0.0726852, -0.0955681, 0.0624858, -0.0795696, -0.0523344, -0.0866519, 0.0399679, -0.0736391, 0.0488118, -0.0225057, -0.0555216, 0.256203, 0.147901, 0.0710665, -0.0781941, -0.0962824, 0.0450298, 0.0344503, -0.17849, 0.0431207, -0.00694193, -0.0330423, 0.00139363, 0.183582, -0.0307825, -0.0634953, -0.142661, 0.0130336, 0.0165756, -0.0666203, -0.0778516, -0.0777053, 0.0768846, -0.0476742, -0.073179, 0.0110823, 0.00330162, -0.106379, -0.0972888, -0.0963217, 0.0546198, -0.0557374, -0.0114606, 0.0126591, -0.0290135, 0.128163, -0.103716]
        # mse = mse = ((features[i] - isaac) ** 2).mean()
        # print("features mse:", mse)

    # test YOLO model ONLY
    for bbox in boxs:
        bbox = tlbr(bbox)
        print('printed, postprocessed bbox [x1/y1/x2/y2]:', bbox)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    cv2.imshow('', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
