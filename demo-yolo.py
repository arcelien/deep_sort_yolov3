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
    image = cv2.imread("image_00016.png") #fromarray(frame)
    frame = image
    image2 = Image.open("image_00016.png")

    print("input to yolo", image.shape)

    boxs = yolo.detect_image(image2)

    print('number of detections:', len(boxs))
    print('output from yolo [x/y/w/h]', boxs)

    def tlbr(boxes_from_yolo):
        ret = np.asarray(boxes_from_yolo, dtype=np.float)
        ret[2:] += ret[:2]
        return ret

    for bbox in boxs:
        bbox = tlbr(bbox)
        print('printed, postprocessed bbox [x1/y1/x2/y2]:', bbox)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    cv2.imshow('', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
