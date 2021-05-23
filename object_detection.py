import numpy as np
import cv2
from tools import generate_detections as gdet
from deep_sort import preprocessing
from deep_sort.detection import Detection
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

class YOLO():
    def __init__(self,net, img, width, height, nms_max_overlap):
        self.net = net
        self.img = img
        self.width = width
        self.height = height
        self.nms_max_overlap = nms_max_overlap

    def predict(self):
        blob = cv2.dnn.blobFromImage(self.img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_layers_names = self.net.getUnconnectedOutLayersNames()
        layerOutputs = self.net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        features = encoder(self.img, boxes)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxs, self.nms_max_overlap, scores)

        return features, detections, indexes
