import cv2
import torch
import numpy as np
import math
from super_gradients.training import models
from super_gradients.training.processing import (DetectionCenterPadding,StandardizeImage, 
                                                 ImagePermute, ComposeProcessing, 
                                                 DetectionLongestMaxSizeRescale)
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import streamlit as st 

file_path = 'coco-labels-paper.txt'
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

names = []
with open(file_path, 'r') as file:
    for line in file:
        names.append(line.strip())

st.header(":hand: Welcome To YoLo Nas Object Detection and Tracking : ")
st.info("""
This app uses the cutting-edge YOLO Nas algorithm to detect objects in real-time video streams. 
But that's not all it also employs the powerful DeepSort algorithm to track these objects, 
providing you with seamless tracking capabilities. Easily upload a video feed, 
and watch as our app identifies and tracks objects with precision.
It's simple, efficient, and ready to help you monitor and analyze moving objects effortlessly!
""")
with st.sidebar : 
    device_name =st.selectbox("Device : " , ["cpu" , "cuda"])
    if device_name == 'cuda' : 
        device = torch.device("cuda:0")
    else : 
        device = torch.device("cpu")
    source_name  = st.selectbox("select you source feed : " , ["File" , "Camera" ,"URL"])
    conf = st.slider("Select threshold confidence value : " , min_value=0.1 , max_value=1.0 , value=0.25)
    iou = st.slider("Select Intersection over union (iou) value : " , min_value=0.1 , max_value=1.0 , value=0.5)


if source_name == "File" : 
    uploadf = st.file_uploader("Upload your Video : " , type=["mp4" , "mkv" , "avi"])
    if uploadf : 
        cap = cv2.VideoCapture(uploadf.name)
elif source_name == "Camera" :
    cap = cv2.VideoCapture(0)
else : 
    source = st.text_input("Input your Url feed and press Entre")
    cap = cv2.VideoCapture(source)

#model=models.get('yolo_nas_s',num_classes=len(names) , 
#                 checkpoint_path="yolo_nas_s_coco.pth").to(device)

model=models.get('yolo_nas_s', pretrained_weights="coco").to(device)
model.set_dataset_processing_params(
    class_names=names,
    image_processor=ComposeProcessing(
                                    [DetectionLongestMaxSizeRescale(output_shape=(636, 636)),
                                     DetectionCenterPadding(output_shape=(640, 640), 
                                                            pad_value=114),
                                     StandardizeImage(max_value=255.0),
                                     ImagePermute(permutation=(2, 0, 1)),]),
    iou=iou ,conf=conf)

cfg_deep = get_config()
cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                    max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                    use_cuda=False)

def compute_color_for_labels(label):
  """
  Simple function that adds fixed color depending on the class
  """
  if label == 0: #person
      color = (85,45,255)
  elif label == 2: # Car
      color = (222,82,175)
  elif label == 3:  # Motobike
      color = (0, 204, 255)
  elif label == 5:  # Bus
      color = (0, 149, 255)
  else:
      color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
  return tuple(color)


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1,  x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[0]
        y2 += offset[0]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        cv2.rectangle(img, (x1, y1), (x2, y2), color= compute_color_for_labels(cat),thickness=2, lineType=cv2.LINE_AA)
        label = str(id) + ":" + names[cat]
        (w,h), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)
        t_size=cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)[0]
        c2=x1+t_size[0], y1-t_size[1]-3
        cv2.rectangle(img, (x1, y1), c2, color=compute_color_for_labels(cat), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(img, str(label), (x1, y1-2), 0, 1/2, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return img




if st.button("Start detection and Tracking") : 
    frame_window = st.image( [] )
    while True:
        xywh_bboxs = []
        confs = []
        oids = []
        ret, frame = cap.read()
        if ret:
            result = list(model.predict(frame))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels): 

                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((confidence*100))/100
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                bbox_width = abs(x1-x2)
                bbox_height = abs(y1-y2)
                xcycwh = [cx, cy, bbox_width, bbox_height]
                xywh_bboxs.append(xcycwh)
                confs.append(conf)
                oids.append(int(cls))
            xywhs = torch.tensor(xywh_bboxs)
            confss= torch.tensor(confs)
            outputs = deepsort.update(xywhs, confss, oids, frame)
            if len(outputs)>0:
                bbox_xyxy = outputs[:,:4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]
                draw_boxes(frame, bbox_xyxy, identities, object_id , names=names)
                #output.write(frame)
            #cv2.imshow('Video', frame)
            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #   break
            frame  = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB)
            frame_window.image(frame)
        else:
            break
