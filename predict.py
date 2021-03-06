######################### Necessary imports ##################

import cv2
import numpy as np
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
import imutils
from gautam_draw import draw_bounding_boxes
from object_detection import YOLO


fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = 1200
frame_height = 675https://github.com/GautamBharadwaj/Helmet-Detection-with-Tracking-YOLOv3/blob/main/predict.py
out = cv2.VideoWriter('C:/Users/Gautam/Desktop/final_folder/output/output_file2.avi', fourcc, 20.0,(frame_width, frame_height))

#################### Loading the Weights and configuration file into our system ##################

path = "C:/Users/Gautam/Desktop/final_folder/helmet1.mp4"
video = cv2.VideoCapture(path)
modelpath = 'C:/Users/Gautam/Desktop/final_folder/'
net = cv2.dnn.readNet('{}/yolov3-helmet.weights'.format(modelpath), '{}/yolov3-helmet.cfg'.format(modelpath))
classes = []

with open("weights/coco_classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(path)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

#############3 Object Tracking ################

max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

while True:
    ret, frame = cap.read()
    if ret != True:
        break
    height, width, _ = frame.shape        #### getting the frame height and width

    yolo_model = YOLO(net, frame, width, height, nms_max_overlap)
    features,detections,indexes = yolo_model.predict()

    #### performing the tracking

    tracker.predict()
    tracker.update(detections)

    #### drawing the bounding boxes
    draw_bounding_boxes(tracker,detections,frame)

    frame = imutils.resize(frame, width=1200)
    out.write(frame)
    cv2.imshow('Image', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
