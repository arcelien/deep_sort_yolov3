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
    image2 = Image.open(image_name+".png")

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
