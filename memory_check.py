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
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # model_filename = 'model_data/mars-small128.pb'
    # encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # tracker = Tracker(metric)

    fps = 0.0
    index = 0
    image_name = "isaac-dataset/image_"

    # save for testing tracker:
    # input: detections for each frame = [tlwh, confidence, feature]
    # output: list of bboxes for tracks

    saved_inputs = []
    saved_outputs = []
    while True:
        if index < 21:
            index += 1
            continue
        time.sleep(10)

        image_name_full = image_name + "%05d" % index + ".png"
        print(image_name_full)
        if index >= 81:
            break
        frame = cv2.imread(image_name_full)  # frame shape 640*480*3
        index += 1
        t1 = time.time()
        print('input image shape', frame.shape)

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)
        time.sleep(10)


if __name__ == '__main__':
    main(YOLO())

