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

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    fps = 0.0
    index = 0
    image_name = "isaac-dataset/image_"
    
    # save for testing tracker: 
    # input: detections for each frame = [tlwh, confidence, feature]
    # output: list of bboxes for tracks
    
    saved_inputs = []
    saved_outputs = []
    while True:
        if index < 3:
            index += 1
            continue
		
        image_name_full = image_name + "%05d" % index +".png"
        print(image_name_full)
        if index > 81:
            break
        frame = cv2.imread(image_name_full)  # frame shape 640*480*3
        index += 1
        t1 = time.time()
        print('input image shape', frame.shape)

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)
        print("box_num",len(boxs))
        print("boxes", boxs)

        features = encoder(frame,boxs)
        # # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # save inputs
        saved_inputs.append([(d.tlwh, d.confidence, d.feature) for d in detections])

        # # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        one_output = []

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue 
            bbox = track.to_tlbr()
            one_output.append(bbox)

        saved_outputs.append(one_output)
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        #     cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        # for det in detections:
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        # cv2.imshow('', frame)
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # time.sleep(0.1)

    print("end, showing saved in/out")
    print(saved_inputs[0])
    print(saved_outputs[0])
    
    

if __name__ == '__main__':
    main(YOLO())
