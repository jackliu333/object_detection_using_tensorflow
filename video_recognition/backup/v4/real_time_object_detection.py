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
    "CONFIDENCE_THRESHOLD": 0.15,
    "SINGLE_FRAME_TOP_PRED": False, # showing top prediction for each label;
    "SINGLE_FRAME_TOP_TWO_PRED": False, # showing the best one based on combinations of top two predictions observed within training set
    "MULTIPLE_FRAME": False,
    "RECOGNITION_ENABLED": False
}

# define existing categories
annotations = pd.read_csv("../master_list.csv")
all_categories = list(annotations['label'].unique())

# load label encoder 
def load_label_encoder():
    le_prdtype = pickle.loads(open("../NN_model/model_weights/original/le_prdtype.pickle", "rb").read())
    le_weight = pickle.loads(open("../NN_model/model_weights/original/le_weight.pickle", "rb").read())
    le_halal = pickle.loads(open("../NN_model/model_weights/original/le_halal.pickle", "rb").read())
    le_healthy = pickle.loads(open("../NN_model/model_weights/original/le_healthy.pickle", "rb").read())
    
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

    model_path = '../NN_model/model_weights/original/multi_head_model.pth'
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

def get_top_two_probs_and_indices(tensor):
    # Convert logits to probabilities using softmax
    probs_tensor = F.softmax(tensor, dim=-1)
    # Get the values and indices of the top two probabilities
    top_two_probs, top_two_indices = torch.topk(probs_tensor, 2, dim=-1)
    return top_two_probs.detach().cpu().numpy(), top_two_indices.detach().cpu().numpy()  # Detach tensors before converting to numpy

def generate_labels_and_probs(top_two_probs, top_two_indices, label_encoders):
    # Flatten the input arrays if they are lists of single-element arrays
    top_two_probs = [np.array(probs).flatten() for probs in top_two_probs]
    top_two_indices = [np.array(indices).flatten() for indices in top_two_indices]
    
    # If there's only one probability, duplicate it
    top_two_probs = [np.repeat(probs, 2) if len(probs) == 1 else probs for probs in top_two_probs]
    top_two_indices = [np.repeat(indices, 2) if len(indices) == 1 else indices for indices in top_two_indices]
    
    # Generate all possible label combinations and their joint probabilities
    labels = []
    joint_probs = []
    for i in range(2):  # Loop over the top two probabilities for each sub-label
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    indices = [top_two_indices[0][i], 
                               top_two_indices[1][j], 
                               top_two_indices[2][k], 
                               top_two_indices[3][l]]
                    probs = [top_two_probs[0][i], 
                             top_two_probs[1][j], 
                             top_two_probs[2][k], 
                             top_two_probs[3][l]]
                    
                    # Inverse transform the indices to get the labels
                    label = "_".join(encoder.inverse_transform([idx])[0] for idx, encoder in zip(indices, label_encoders))
                    joint_prob = np.prod(probs)  # Compute the joint probability
                    labels.append(label)
                    joint_probs.append(joint_prob)
    
    # Normalize the joint probabilities so that they sum to one
    total_prob = sum(joint_probs)
    normalized_joint_probs = [prob / total_prob for prob in joint_probs]
    
    return labels, normalized_joint_probs

# Function to process and predict using the model
def process_and_predict(frame, model):
    model.eval()

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

        top_prdtype = prdtype.argmax(1)
        top_weight = weight.argmax(1)
        top_halal = halal.argmax(1)
        top_healthy = healthy.argmax(1)

        # # Calculate joint probability
        # joint_prob = prdtype_prob * weight_prob * halal_prob * healthy_prob
        

    # Decode the predictions
    prdtype_label = le_prdtype.inverse_transform(top_prdtype.cpu())[0]
    weight_label = le_weight.inverse_transform(top_weight.cpu())[0]
    halal_label = le_halal.inverse_transform(top_halal.cpu())[0]
    healthy_label = le_healthy.inverse_transform(top_healthy.cpu())[0]

    # Extract bounding box coordinates
    bbox = bbox.squeeze().cpu().numpy()
    (startX, startY, endX, endY) = bbox

    # return (prdtype_label, weight_label, halal_label, healthy_label), (startX, startY, endX, endY), joint_prob
    return (prdtype_label, weight_label, halal_label, healthy_label), (startX, startY, endX, endY), \
     (prdtype_prob, weight_prob, halal_prob, healthy_prob), (prdtype, weight, halal, healthy)

def get_top_two_predictions_and_frequencies(sublabel_list):
    # print(sublabel_list)
    # Count the frequency of each label in the sublabel list
    label_freq = Counter(sublabel_list)
    
    # Find the two most common labels and their counts
    top_two_labels_and_freq = label_freq.most_common(2)
    
    # Calculate the total number of labels to compute relative frequencies
    total_labels = len(sublabel_list)
    
    # Compute relative frequencies and corresponding labels
    top_two_labels = [label for label, _ in top_two_labels_and_freq]
    top_two_probs = [count / total_labels for _, count in top_two_labels_and_freq]

    # print(top_two_labels)
    
    # Check if there is only one element in the list, if so add 0 or 1 depending on the existing value
    if len(top_two_labels) < 2:
        additional_label = 0 if top_two_labels[0] != 0 else 1
        top_two_labels.append(additional_label)
        top_two_probs.append(0)
    
    return top_two_labels, top_two_probs


# Video stream loop
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


def reset_multi_frame():
    global sublabel_list_1, sublabel_list_2, sublabel_list_3, sublabel_list_4
    global countdown_active, countdown_start_time, countdown_duration, show_result

    # Lists to hold label predictions for each sublabel in multi-frame mode
    sublabel_list_1 = []
    sublabel_list_2 = []
    sublabel_list_3 = []
    sublabel_list_4 = []

    # Initially, countdown is not active
    countdown_active = False
    countdown_start_time = 0
    countdown_duration = 3  # Countdown from 3 seconds
    show_result = False

reset_multi_frame()

while True:
    frame = vs.read()
    orig = frame.copy()

    # Define the instructions text
    instructions = [
        "1: Real time simple",
        "2: Real time complex",
        "3: Multiple frames simple",
        "*4: Multiple frames GPT",
        "*5: Multiple frames Human",
        "0: Show video only",
        "q: Quit"
    ]

    # Calculate the starting position for the instructions text on the top right corner
    # Adjust these values as needed for your video frame size and desired positioning
    start_x = orig.shape[1] - 530  # Start x pixels from the right edge
    start_y = 30  # Start 20 pixels from the top
    line_offset = 50  # Space between lines

    # Font size
    font_scale = 1.2  # Increase this value for larger text
    thickness = 3  # Increase for bolder text
    # Color in BGR
    color = (0, 0, 255)  # Red color

    # Draw each instruction line on the frame
    for instruction in instructions:
        cv2.putText(orig, instruction, (start_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        start_y += line_offset

    if CONFIGS['RECOGNITION_ENABLED']:
        # Get predictions and bounding box
        labels, bbox_coords, top_pred_probs, pred_logits = process_and_predict(frame, model)
        (startX, startY, endX, endY) = bbox_coords
        (prdtype_prob, weight_prob, halal_prob, healthy_prob) = top_pred_probs
        (labelPreds_prdtype, labelPreds_weight, labelPreds_halal, labelPreds_healthy) = pred_logits

        (h, w) = orig.shape[:2]

        # Scale the predicted bounding box coordinates based on the image dimensions
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)

        if CONFIGS['SINGLE_FRAME_TOP_PRED']:
            # Draw the bounding box and labels on the frame only if joint probability is above the threshold
            if prdtype_prob >= CONFIGS['CONFIDENCE_THRESHOLD'] and weight_prob >= CONFIGS['CONFIDENCE_THRESHOLD'] \
            and halal_prob >= CONFIGS['CONFIDENCE_THRESHOLD'] and healthy_prob >= CONFIGS['CONFIDENCE_THRESHOLD']:
                # Draw bounding box
                # print(startX)
                cv2.rectangle(orig, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)

                # Prepare the label text
                # label_text = f"Prdtype: {labels[0]}, Weight: {labels[1]}, Halal: {labels[2]}, Healthy: {labels[3]}"

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

        if CONFIGS['SINGLE_FRAME_TOP_TWO_PRED']:
            # Extract top two probabilities and indices for each sub-label
            top_two_probs_and_indices = [get_top_two_probs_and_indices(tensor) for tensor in 
                                         [labelPreds_prdtype, labelPreds_weight, labelPreds_halal, labelPreds_healthy]]
            # Separate the probabilities and indices
            top_two_probs, top_two_indices = zip(*top_two_probs_and_indices)
            # Generate all possible label combinations and their joint probabilities
            all_possible_labels, all_joint_probs = generate_labels_and_probs(top_two_probs, top_two_indices, 
                                          [le_prdtype, le_weight, le_halal, le_healthy])
            # Filter out the valid labels and their joint probabilities
            valid_labels_and_probs = [(label, prob) for label, prob in zip(all_possible_labels, all_joint_probs) if label in all_categories]
            # If there are no valid labels, set label to 'Nonfood'
            if not valid_labels_and_probs:
                label = "Nonfood"
                prob = 0
            else:
                # Otherwise, pick the valid label with the highest joint probability
                label, prob = max(valid_labels_and_probs, key=lambda x: x[1])
                prob *= 100
            confidence_text = f"Prediction confidence: {float(prob):.2f}%"

            # Only draw bounding box and text if the prediction confidence is above 30%
            if float(prob) >= CONFIGS['CONFIDENCE_THRESHOLD'] * 100:
                # Draw bounding box
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Draw text
                # cv2.putText(orig, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                # Define starting position for the text
                text_x = 10
                text_y = 50
                line_offset = 50  # Offset between lines
                # Prepare the label texts
                label_texts = [
                    f"Prediction: {label}",
                    f"Confidence: {float(prob):.2f}%"
                ]

                # Draw each label text in a new line
                for label_text in label_texts:
                    cv2.putText(orig, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    text_y += line_offset  # Move to the next line

        if CONFIGS['MULTIPLE_FRAME']:
            # print("starting scanning...")
            first_scanning = True
            countdown_trigger = True

            # cv2.imshow("Frame", orig)

            # if first_scanning:
            #     cv2.imshow("Frame", orig)
            #     first_scanning = False

            # Display countdown if triggered
            # Countdown display logic
            if countdown_active:
                current_time = time.time()
                elapsed_time = current_time - countdown_start_time
                countdown_time = countdown_duration - int(elapsed_time)

                # Display the countdown time if within the countdown period
                if 0 < countdown_time <= countdown_duration:
                    text = f"{countdown_time}"  # Display numbers 3 to 1
                    (frame_height, frame_width) = orig.shape[:2]
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 2)[0]
                    text_x = (frame_width - text_size[0]) // 2
                    text_y = (frame_height + text_size[1]) // 2
                    cv2.putText(orig, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

                    # Extract top two probabilities and indices for each sub-label
                    top_two_probs_and_indices = [get_top_two_probs_and_indices(tensor) for tensor in 
                                                 [labelPreds_prdtype, labelPreds_weight, labelPreds_halal, labelPreds_healthy]]
                    # print(top_two_probs_and_indices)

                    # Extract the top predicted label index of the class labels
                    top_preds_indices = [class_labels[0][0] for _, class_labels in top_two_probs_and_indices]
                    
                    sublabel_list_1.append(top_preds_indices[0])
                    sublabel_list_2.append(top_preds_indices[1])
                    sublabel_list_3.append(top_preds_indices[2])
                    sublabel_list_4.append(top_preds_indices[3])

                    # print(top_preds_indices)
                    # print(sublabel_list_1)

                # Stop the countdown after reaching 0
                if countdown_time < 0:
                    countdown_active = False
                    show_result = True
                    # print("count down complete")

            if show_result:
                print("show results")
                first_time_process = True
                # print(sublabel_list_1)
                if first_time_process:
                    # Extract top two predictions and their relative frequencies for each sublabel list
                    top_two_labels_1, top_two_probs_1 = get_top_two_predictions_and_frequencies(sublabel_list_1)
                    top_two_labels_2, top_two_probs_2 = get_top_two_predictions_and_frequencies(sublabel_list_2)
                    top_two_labels_3, top_two_probs_3 = get_top_two_predictions_and_frequencies(sublabel_list_3)
                    top_two_labels_4, top_two_probs_4 = get_top_two_predictions_and_frequencies(sublabel_list_4)

                    top_probs_combined = np.array([
                        top_two_probs_1,
                        top_two_probs_2,
                        top_two_probs_3,
                        top_two_probs_4
                    ]).astype('float32')

                    top_labels_combined = np.array([
                        top_two_labels_1,
                        top_two_labels_2,
                        top_two_labels_3,
                        top_two_labels_4
                    ])

                    # print(top_probs_combined)
                    # print(top_labels_combined)

                    # Generate all possible label combinations and their joint probabilities
                    all_possible_labels_scan, all_joint_probs_scan = generate_labels_and_probs(top_probs_combined, top_labels_combined, 
                                                                                    [le_prdtype, le_weight, le_halal, le_healthy])

                    # Filter out the valid labels and their joint probabilities
                    valid_labels_and_probs_scan = [(label, prob) for label, prob in zip(all_possible_labels_scan, all_joint_probs_scan) if label in all_categories]

                    # If there are no valid labels, set label to 'Nonfood'
                    if not valid_labels_and_probs_scan:
                        label_scan = "Nonfood"
                        prob_scan = "NA"
                    else:
                        # Otherwise, pick the valid label with the highest joint probability
                        label_scan, prob_scan = max(valid_labels_and_probs_scan, key=lambda x: x[1])
                        prob_scan *= 100

                    first_time_process = False

                # Draw bounding box
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Draw text
                # cv2.putText(orig, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                # Define starting position for the text
                text_x = 10
                text_y = 50
                line_offset = 50  # Offset between lines
                # Prepare the label texts
                label_texts = [
                    f"Prediction: {label_scan}",
                    f"Confidence: {float(prob_scan):.2f}%"
                ]

                print(label_scan)
                print(prob_scan)

                # Draw each label text in a new line
                for label_text in label_texts:
                    cv2.putText(orig, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    text_y += line_offset  # Move to the next line


    # Show the frame
    cv2.imshow("Frame", orig)

    # key = cv2.waitKey(1) & 0xFF
    # if key == ord("q"):
    #     break
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # Mode selection based on key press
    if key == ord("1"):
        CONFIGS['RECOGNITION_ENABLED'] = True
        CONFIGS['SINGLE_FRAME_TOP_PRED'] = True
        CONFIGS['SINGLE_FRAME_TOP_TWO_PRED'] = False
        CONFIGS['MULTIPLE_FRAME'] = False
        reset_multi_frame()
    elif key == ord("2"):
        CONFIGS['RECOGNITION_ENABLED'] = True
        CONFIGS['SINGLE_FRAME_TOP_PRED'] = False
        CONFIGS['SINGLE_FRAME_TOP_TWO_PRED'] = True
        CONFIGS['MULTIPLE_FRAME'] = False
        reset_multi_frame()
    elif key == ord("3"):
        if not countdown_active and not show_result:
            CONFIGS['RECOGNITION_ENABLED'] = True
            CONFIGS['SINGLE_FRAME_TOP_PRED'] = False
            CONFIGS['SINGLE_FRAME_TOP_TWO_PRED'] = False
            CONFIGS['MULTIPLE_FRAME'] = True
            countdown_active = True
            countdown_start_time = time.time()  # Reset countdown start time
            show_result = False
        else:
            reset_multi_frame()
    elif key == ord("0"):
        CONFIGS['RECOGNITION_ENABLED'] = False
        reset_multi_frame()
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
