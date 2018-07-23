# Files Needed

## Freezing yolo model
1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.

```
   wget https://pjreddie.com/media/files/yolov3.weights
   python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
   python demo.py
```

3. Run `demo.py` with `self.do_freezing=True` in `yolo.py`

## Getting mars-small128.pb
1. Download resources from https://owncloud.uni-koblenz.de/owncloud/s/f9JB0Jr7f3zzqs8
2. Freeze the model with `tools/freeze_model.py`

## Get the dataset
Put a dataset of pictures in the main directory

# Dependencies

The code is compatible with Python 2.7 and 3. The following dependencies are needed to run the tracker:

    NumPy
    sklean
    OpenCV

Additionally, feature generation requires TensorFlow-1.4.0.

# Note 
 file model_data/mars-small128.pb  had convert to tensorflow-1.4.0
 
 file model_data/yolo.h5 is to large to upload ,so you need convert it from Darknet Yolo model to a keras model by yourself
 
 yolo.h5 model can download from https://drive.google.com/file/d/1uvXFacPnrSMw6ldWTyLLjGLETlEsUvcE/view?usp=sharing , use tensorflow1.4.0
 
# Test
 use : 'video_capture = cv2.VideoCapture('path to video')' use a video file or 'video_capture = cv2.VideoCapture(0)' use camera
 
 speed : when only run yolo detection about 11-13 fps  , after add deep_sort about 11.5 fps
 
 test video : https://www.bilibili.com/video/av23500163/
 
 From the issue https://github.com/Qidian213/deep_sort_yolov3/issues/7 , it can tracks cars, birds and trucks too and performs well .



