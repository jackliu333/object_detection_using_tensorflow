# import the necessary packages
from torchvision.models import detection, resnet50, ResNet50_Weights
from imutils.video import VideoStream, FPS
import numpy as np
import imutils
import pickle
import torch
import time
import cv2
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from collections import Counter
from torchvision import models
import os

CONFIGS = {
    # determine the current device and based on that set the pin memory
    # flag
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    # specify ImageNet mean and standard deviation
    "IMG_MEAN": [0.485, 0.456, 0.406],
    "IMG_STD": [0.229, 0.224, 0.225],
    "MC_DROPOUT_ENABLED": False,  # Switch to enable/disable MC Dropout for confidence score
    "NUM_DROPOUT_RUNS": 3,
    "CONFIDENCE_THRESHOLD": 0,
}

# define existing categories
annotations = pd.read_csv("master_list.csv")
all_categories = list(annotations['label'].unique())

# load label encoder 
def load_label_encoder():
    le_prdtype = pickle.loads(open("./model_weights/le_prdtype.pickle", "rb").read())
    le_weight = pickle.loads(open("./model_weights/le_weight.pickle", "rb").read())
    le_halal = pickle.loads(open("./model_weights/le_halal.pickle", "rb").read())
    le_healthy = pickle.loads(open("./model_weights/le_healthy.pickle", "rb").read())
    
    return le_prdtype, le_weight, le_halal, le_healthy

le_prdtype, le_weight, le_halal, le_healthy = load_label_encoder()

# print(le_prdtype.classes_)

class MultiHeadResNet(nn.Module):
    def __init__(self, num_classes_prdtype, num_classes_weight, num_classes_halal, num_classes_healthy):
        super(MultiHeadResNet, self).__init__()
        self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # Define custom fully connected layers for each prediction head
        self.fc_prdtype = nn.Linear(num_ftrs, num_classes_prdtype)
        self.fc_weight = nn.Linear(num_ftrs, num_classes_weight)
        self.fc_halal = nn.Linear(num_ftrs, num_classes_halal)
        self.fc_healthy = nn.Linear(num_ftrs, num_classes_healthy)
        self.fc_bbox = nn.Linear(num_ftrs, 4)

    def forward(self, x):
        x = self.base_model(x)
        prdtype = self.fc_prdtype(x)
        weight = self.fc_weight(x)
        halal = self.fc_halal(x)
        healthy = self.fc_healthy(x)
        box = self.fc_bbox(x)
        return prdtype, weight, halal, healthy, box


# class MultiHeadResNet(nn.Module):
#     def __init__(self, num_classes_prdtype, num_classes_weight, num_classes_halal, num_classes_healthy):
#         super(MultiHeadResNet, self).__init__()
#         self.base_model = models.resnet50(pretrained=True)
#         num_ftrs = self.base_model.fc.in_features
#         self.base_model.fc = nn.Identity()

#         self.fc_prdtype = self._make_head(num_ftrs, num_classes_prdtype)
#         self.fc_weight = self._make_head(num_ftrs, num_classes_weight)
#         self.fc_halal = self._make_head(num_ftrs, num_classes_halal)
#         self.fc_healthy = self._make_head(num_ftrs, num_classes_healthy)
#         self.fc_bbox = self._make_head(num_ftrs, 4, output_activation=None)

#     def _make_head(self, in_features, out_features, output_activation=nn.Softmax(dim=1)):
#         layers = [
#             nn.Linear(in_features, 32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, out_features)
#         ]
#         if output_activation:
#             layers.append(output_activation)
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.base_model(x)
#         prdtype = self.fc_prdtype(x)
#         weight = self.fc_weight(x)
#         halal = self.fc_halal(x)
#         healthy = self.fc_healthy(x)
#         bbox = self.fc_bbox(x)
#         return prdtype, weight, halal, healthy, bbox


# Load the trained MultiHeadResNet model
def load_model():
    # Verify the number of classes for each label
    num_classes_prdtype = len(le_prdtype.classes_)
    num_classes_weight = len(le_weight.classes_)
    num_classes_halal = len(le_halal.classes_)
    num_classes_healthy = len(le_healthy.classes_)
    # print(num_classes_prdtype)
    # print(num_classes_healthy)

    custom_resnet_model = MultiHeadResNet(
        num_classes_prdtype=num_classes_prdtype,
        num_classes_weight=num_classes_weight,
        num_classes_halal=num_classes_halal,
        num_classes_healthy=num_classes_healthy
    )

    model_path = './model_weights/multi_head_model.pth'
    # print("test1")
    if os.path.exists(model_path):
        custom_resnet_model.load_state_dict(torch.load(model_path, map_location=CONFIGS['DEVICE']))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # print("test2")
    custom_resnet_model.to(CONFIGS['DEVICE'])
    custom_resnet_model.eval()
    return custom_resnet_model

model = load_model()



transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIGS['IMG_MEAN'], std=CONFIGS['IMG_STD'])
])


# Function to process and predict using the model
def process_and_predict(frame, model):
    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (320, 320))
    frame = frame.transpose((2, 0, 1))
    frame = torch.from_numpy(frame).float()
    frame = transforms_test(frame).unsqueeze(0).to(CONFIGS['DEVICE'])

    # Perform prediction
    with torch.no_grad():
        prdtype, weight, halal, healthy, bbox = model(frame)

        # Calculate probabilities using softmax
        prdtype_prob = F.softmax(prdtype, dim=1).max(1)[0].item()
        weight_prob = F.softmax(weight, dim=1).max(1)[0].item()
        halal_prob = F.softmax(halal, dim=1).max(1)[0].item()
        healthy_prob = F.softmax(healthy, dim=1).max(1)[0].item()

        prdtype = prdtype.argmax(1)
        weight = weight.argmax(1)
        halal = halal.argmax(1)
        healthy = healthy.argmax(1)

        # Calculate joint probability
        joint_prob = prdtype_prob * weight_prob * halal_prob * healthy_prob
        

    # Decode the predictions
    prdtype_label = le_prdtype.inverse_transform(prdtype.cpu())[0]
    weight_label = le_weight.inverse_transform(weight.cpu())[0]
    halal_label = le_halal.inverse_transform(halal.cpu())[0]
    healthy_label = le_healthy.inverse_transform(healthy.cpu())[0]

    # Extract bounding box coordinates
    bbox = bbox.squeeze().cpu().numpy()
    (startX, startY, endX, endY) = bbox

    return (prdtype_label, weight_label, halal_label, healthy_label), (startX, startY, endX, endY), joint_prob

# Video stream loop
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    orig = frame.copy()

    # Get predictions and bounding box
    labels, bbox_coords, joint_prob = process_and_predict(frame, model)
    (startX, startY, endX, endY) = bbox_coords

    (h, w) = orig.shape[:2]

    # Scale the predicted bounding box coordinates based on the image dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    
    # Draw the bounding box and labels on the frame only if joint probability is above the threshold
    if joint_prob >= CONFIGS['CONFIDENCE_THRESHOLD']:
        # Draw bounding box
        # print(startX)
        cv2.rectangle(orig, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)

        # Prepare the label text
        # label_text = f"Prdtype: {labels[0]}, Weight: {labels[1]}, Halal: {labels[2]}, Healthy: {labels[3]}"

        # Coordinates for text (above the bounding box)
        text_x = int(startX)
        text_y = int(startY) - 10 if startY - 10 > 10 else int(startY) + 20

        # Draw text
        # cv2.putText(orig, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # Define starting position for the text
        text_x = 10
        text_y = 50
        line_offset = 50  # Offset between lines

        # Prepare the label texts
        label_texts = [
            f"Prdtype: {labels[0]}",
            f"Weight: {labels[1]}",
            f"Halal: {labels[2]}",
            f"Healthy: {labels[3]}"
        ]

        # Draw each label text in a new line
        for label_text in label_texts:
            cv2.putText(orig, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            text_y += line_offset  # Move to the next line



    cv2.imshow("Frame", orig)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
