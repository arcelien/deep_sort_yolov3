#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

import tensorflow as tf

class YOLO(object):
    def __init__(self):
        self.do_freezing = False # whether to save a freezed model
        self.is_frozen = False # whether to load a frozen model or just normal weights
        self.freeze_quantized = False # whether to do quantization while freezing

        if self.do_freezing or not self.is_frozen:
            self.model_path = 'model_data/yolo_tiny.h5'
            self.anchors_path = 'model_data/tiny_yolo_anchors.txt'
            self.classes_path = 'model_data/coco_classes.txt'
            self.score = 0.5
            self.iou = 0.5
            self.class_names = self._get_class()
            self.anchors = self._get_anchors()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            K.set_session(self.sess)
            # self.sess = K.get_session()

            self.model_image_size = (512, 960) # fixed size or (None, None)
            self.is_fixed_size = self.model_image_size != (None, None)
            self.boxes, self.scores, self.classes = self.generate()

        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            K.set_session(self.sess)
            # self.sess = K.get_session()
            self.model_image_size = (512, 960) # fixed size or (None, None)
            self.is_fixed_size = self.model_image_size != (None, None)
            self.classes_path = 'model_data/coco_classes.txt'
            self.class_names = self._get_class()
            if self.is_frozen:
                # load the frozen model
                with self.sess.graph.as_default():
                    output_graph_def = tf.GraphDef()
                    frozen_model_name = './tiny_yolo_model_total'
                    if self.freeze_quantized:
                        frozen_model_name += '_quantized'
                    frozen_model_name += '.pb'
                    with open(frozen_model_name, "rb") as f:
                        output_graph_def.ParseFromString(f.read())
                        tf.import_graph_def(output_graph_def, name="")
                    # print(len(output_graph_def.node))
                    for node in output_graph_def.node:
                        assert "Variable" != node.op
                    #     if ("input" in node.op) or ("placeholder" in node.op):
                    #         print(node.op)

                    sess = self.sess
                    self.boxes = sess.graph.get_tensor_by_name("output_boxes:0")
                    self.scores = sess.graph.get_tensor_by_name("output_scores:0")
                    self.classes = sess.graph.get_tensor_by_name("output_classes:0")
                    self.input_image_shape = sess.graph.get_tensor_by_name("Placeholder_59:0")
                    self.yolo_input = sess.graph.get_tensor_by_name("input_1:0")

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path, compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        if self.do_freezing or not self.is_frozen:
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
        else:
            # print('input to yolo', image.size, image_data.shape)

            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

        # print(out_boxes, out_scores, out_classes)
        # print(out_boxes.shape, out_scores.shape, out_classes.shape)
        
        def model_saver():
            # save the model
            sess = self.sess

            # save to tensorboard to visualize
            summary_writer = tf.summary.FileWriter(logdir='./logs/')
            summary_writer.add_graph(graph=sess.graph)
            print('saved tb graph')

            pred_node_names = ["output_boxes", "output_scores", "output_classes"]
            output_fld = './'
            output_model_file = 'tiny_yolo_model_total'
            from tensorflow.python.framework import graph_util
            from tensorflow.python.framework import graph_io
            if self.freeze_quantized: # quantize
                from tensorflow.tools.graph_transforms import TransformGraph
                transforms = ["quantize_weights", "quantize_nodes"]
                transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
                constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
                output_model_file += "_quantized"
            else:
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
            output_model_file += ".pb"
            graph_io.write_graph(constant_graph, output_fld, output_model_file, as_text=False)
            print('saved the freezed graph (ready for inference) at: ', output_fld + output_model_file)
            assert False

        if self.do_freezing and not self.is_frozen:
            model_saver()

        return_boxs = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class != 'person' :
                continue
            box = out_boxes[i]
           # score = out_scores[i]  
            x = int(box[1])  
            y = int(box[0])  
            w = int(box[3]-box[1])
            h = int(box[2]-box[0])
            if x < 0 :
                w = w + x
                x = 0
            if y < 0 :
                h = h + y
                y = 0 
            return_boxs.append([x,y,w,h])

        return return_boxs

    def close_session(self):
        self.sess.close()
