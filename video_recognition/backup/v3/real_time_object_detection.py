# import the necessary packages
from torchvision.models import detection, resnet50
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

CONFIGS = {
    # determine the current device and based on that set the pin memory
    # flag
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    # specify ImageNet mean and standard deviation
    "IMG_MEAN": [0.485, 0.456, 0.406],
    "IMG_STD": [0.229, 0.224, 0.225],
    "MC_DROPOUT_ENABLED": False,  # Switch to enable/disable MC Dropout for confidence score
    "NUM_DROPOUT_RUNS": 3,
    "CONFIDENCE_THRESHOLD": 1,
}

# define existing categories
annotations = pd.read_csv("master_list.csv")
all_categories = list(annotations['label'].unique())

# load label encoder 
def load_label_encoder():
    le_prdtype = pickle.loads(open("../NN_model/le_prdtype.pickle", "rb").read())
    le_weight = pickle.loads(open("../NN_model/le_weight.pickle", "rb").read())
    le_halal = pickle.loads(open("../NN_model/le_halal.pickle", "rb").read())
    le_healthy = pickle.loads(open("../NN_model/le_healthy.pickle", "rb").read())
    
    return le_prdtype, le_weight, le_halal, le_healthy

le_prdtype, le_weight, le_halal, le_healthy = load_label_encoder()


class CustomResNet(nn.Module):
    def __init__(self, base_model, num_categories1, num_categories2, num_categories3, num_categories4, original_in_features):
        super(CustomResNet, self).__init__()
        self.base_model = base_model
        self.hidden_dim1 = 256  # First hidden layer dimensions
        self.hidden_dim2 = 256  # Second hidden layer dimensions
        self.dropout = nn.Dropout(p=0.1)

        # Head for ProductType
        self.fc1_1 = nn.Linear(original_in_features, self.hidden_dim1)
        # self.fc1_1 = nn.Linear(base_model.fc.in_features, self.hidden_dim1)
        self.fc1_2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc1_3 = nn.Linear(self.hidden_dim2, num_categories1)

        # Head for Weight
        self.fc2_1 = nn.Linear(original_in_features + num_categories1, self.hidden_dim1)
        self.fc2_2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc2_3 = nn.Linear(self.hidden_dim2, num_categories2)

        # Head for HalalStatus
        self.fc3_1 = nn.Linear(original_in_features + num_categories1 + num_categories2, self.hidden_dim1)
        self.fc3_2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3_3 = nn.Linear(self.hidden_dim2, num_categories3)

        # Head for HealthStatus
        self.fc4_1 = nn.Linear(original_in_features + num_categories1 + num_categories2 + num_categories3, self.hidden_dim1)
        self.fc4_2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc4_3 = nn.Linear(self.hidden_dim2, num_categories4)

        # Head for BoundingBox
        self.fc_bbox = nn.Linear(original_in_features + num_categories1 + num_categories2 + num_categories3 + num_categories4, 4)

        
    def forward(self, x):
        x_base = self.base_model(x)

        # Head for ProductType
        x1 = nn.functional.relu(self.fc1_1(x_base))
        x1 = self.dropout(x1)
        x1 = nn.functional.relu(self.fc1_2(x1))
        x1 = self.dropout(x1)
        out1 = self.fc1_3(x1)

        # Head for Weight
        x2 = nn.functional.relu(self.fc2_1(torch.cat((x_base, out1), dim=1)))
        x2 = self.dropout(x2)
        x2 = nn.functional.relu(self.fc2_2(x2))
        x2 = self.dropout(x2)
        out2 = self.fc2_3(x2)

        # Head for HalalStatus
        x3 = nn.functional.relu(self.fc3_1(torch.cat((x_base, out1, out2), dim=1)))
        x3 = self.dropout(x3)
        x3 = nn.functional.relu(self.fc3_2(x3))
        x3 = self.dropout(x3)
        out3 = self.fc3_3(x3)

        # Head for HealthStatus
        x4 = nn.functional.relu(self.fc4_1(torch.cat((x_base, out1, out2, out3), dim=1)))
        x4 = self.dropout(x4)
        x4 = nn.functional.relu(self.fc4_2(x4))
        x4 = self.dropout(x4)
        out4 = self.fc4_3(x4)

        # Head for BoundingBox
        bbox = self.fc_bbox(torch.cat((x_base, out1, out2, out3, out4), dim=1))

        return out1, out2, out3, out4, bbox



# Load the best model
def load_model():
    base_model = resnet50(pretrained=True)
    original_in_features = base_model.fc.in_features
    base_model.fc = nn.Identity()
    custom_resnet_model = CustomResNet(base_model, len(le_prdtype.classes_), len(le_weight.classes_), len(le_halal.classes_), len(le_healthy.classes_), original_in_features)
    custom_resnet_model.load_state_dict(torch.load('../NN_model/model_state.pt', map_location=torch.device('cpu')))
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



from collections import Counter

def get_top_two_predictions_and_frequencies(sublabel_list):
    # Count the frequency of each label in the sublabel list
    label_freq = Counter(sublabel_list)
    
    # Find the two most common labels and their counts
    top_two_labels_and_freq = label_freq.most_common(2)
    
    # Calculate the total number of labels to compute relative frequencies
    total_labels = len(sublabel_list)
    
    # Compute relative frequencies and corresponding labels
    top_two_labels = [label for label, _ in top_two_labels_and_freq]
    top_two_probs = [count / total_labels for _, count in top_two_labels_and_freq]
    
    # Check if there is only one element in the list, if so add 0 or 1 depending on the existing value
    if len(top_two_labels) < 2:
        additional_label = 0 if top_two_labels[0] != 0 else 1
        top_two_labels.append(additional_label)
        top_two_probs.append(0)
    
    return top_two_labels, top_two_probs



# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Text settings for user instruction
font = cv2.FONT_HERSHEY_SIMPLEX
start_text = "press 's' to start scanning"
start_text_font = cv2.FONT_HERSHEY_SIMPLEX
start_text_color = (0, 0, 255)  # red color
start_text_scale = 0.6
start_text_thickness = 2

# Get the width and height of the text box
text_size = cv2.getTextSize(start_text, start_text_font, start_text_scale, start_text_thickness)[0]


# Variable to keep track of whether scanning has started
start_scanning = False
scan_start_time = None
first_scan_flag = False
label_scan = "Undefined"
prob_scan = 0
confidence_text_scan = ""

# Define the countdown trigger and start time outside of the main loop
countdown_trigger = False
countdown_start_time = 0

# Main loop for processing video frames
while True:
    # Grab the frame from the threaded video stream and resize it
    frame = vs.read()
    # frame = imutils.resize(frame, width=400)
    orig = frame.copy()

    ################# SINGEL FRAME PROCESSING ################
    # Convert the frame from BGR to RGB channel ordering and change
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, (224, 224))
    frame = cv2.resize(frame, (320, 320))
    frame = frame.transpose((2, 0, 1))

    frame = torch.from_numpy(frame)
    frame = transforms_test(frame).to(CONFIGS['DEVICE'])
    frame = frame.unsqueeze(0)

    # Perform inference with deterministic model
    model.eval()  # Make sure to set the model to evaluation mode
    (labelPreds_prdtype, labelPreds_weight, labelPreds_halal, labelPreds_healthy, boxPreds) = model(frame)
    deterministic_labels = [labelPreds_prdtype, labelPreds_weight, labelPreds_halal, labelPreds_healthy]

    # MC Dropout Inference
    if CONFIGS['MC_DROPOUT_ENABLED']:
        n_dropout_runs = CONFIGS['NUM_DROPOUT_RUNS']  # Number of Dropout runs
        total_distance = 0  # Variable to store total distance

        model.train()  # Activate dropout layers for inference

        for _ in range(n_dropout_runs):
            # Single forward pass with dropout enabled
            dropout_labels_prdtype, dropout_labels_weight, dropout_labels_halal, dropout_labels_healthy, _ = model(frame)
            dropout_labels = [dropout_labels_prdtype, dropout_labels_weight, dropout_labels_halal, dropout_labels_healthy]
          
            # Compute distance based on sub-labels
            d = sum([int((dropout_labels[i].argmax(1) != deterministic_labels[i].argmax(1)).float().item())
                     for i in range(4)])
            total_distance += d

        # Compute average distance across MC Dropout runs
        avg_distance = total_distance / n_dropout_runs

        # Convert average distance to a confidence percentage score
        confidence_score = (1 - (avg_distance / 4)) * 100
        # confidence_text = f"Confidence: {confidence_score:.2f}%"
        model_uncertainty = 1 - confidence_score
        model_uncertainty_text = f"Model uncertainty: {model_uncertainty:.2f}%"
    else:
        # confidence_text = "Confidence: N/A"
        model_uncertainty_text = "Model uncertainty: NA"

    model.eval()

    # Extract bounding box coordinates
    (startX, startY, endX, endY) = boxPreds[0].detach().cpu().numpy()


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
    # print(valid_labels_and_probs)

    # If there are no valid labels, set label to 'Nonfood'
    if not valid_labels_and_probs:
        label = "Nonfood"
        prob = 0
    else:
        # Otherwise, pick the valid label with the highest joint probability
        label, prob = max(valid_labels_and_probs, key=lambda x: x[1])
        prob *= 100

    # # Additional check for 'Nonfood' sub-label
    # if any(sub_label == "Nonfood" for sub_label in label.split('_')):
    #     label = "Nonfood"

    confidence_text = f"Prediction confidence: {float(prob):.2f}%"


    # Resize the original image such that it fits on our screen, and grab its dimensions
    # orig = imutils.resize(orig, width=600)
    (h, w) = orig.shape[:2]

    # Scale the predicted bounding box coordinates based on the image dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    # Compute positions
    y_label = startY - 20 if startY - 20 > 20 else startY + 20
    y_label -= 15 
    y_confidence = y_label + 15  # Initialize, then adjust as needed to avoid overlap
    y_uncertainty = y_confidence + 15

    # Check if text will fit within the image boundary
    if y_label < 0:
        y_label = 20
    if y_confidence < 0:
        y_confidence = 35
    if y_confidence < 0:
        y_uncertainty = 50
        
    # Only draw bounding box and text if the prediction confidence is above 30%
    if float(prob) >= CONFIGS['CONFIDENCE_THRESHOLD']:
        # Draw bounding box
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Draw label and confidence score
        font_scale = 0.5
        cv2.putText(orig, label, (startX, y_label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        cv2.putText(orig, confidence_text, (startX, y_confidence), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        cv2.putText(orig, model_uncertainty_text, (startX, y_uncertainty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Output", orig)

    ################# SCANNING PROCESSING ################
    # # Set the text's x and y coordinates so that it's in the upper right corner
    # text_x = orig.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
    # text_y = text_size[1] + 10  # 10 pixels from the top edge

    # # print(text_x)
    # # print(text_y)

    # # print(type(frame))

    # # Display start scanning text
    # cv2.putText(orig, start_text, (text_x, text_y), start_text_font, start_text_scale, start_text_color, start_text_thickness)
    text_x = 10  # 10 pixels from the left edge
    text_y = text_size[1] + 10  # 10 pixels from the top edge
    cv2.putText(orig, start_text, (text_x, text_y), start_text_font, start_text_scale, start_text_color, start_text_thickness)


    # Check for user input to start scanning
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        print("starting scanning...")
        start_scanning = True
        scan_start_time = time.time()
        first_scan_flag = True
        countdown_trigger = True
        countdown_start_time = time.time()

        # Lists to hold label predictions for each sublabel
        sublabel_list_1 = []
        sublabel_list_2 = []
        sublabel_list_3 = []
        sublabel_list_4 = []

    # Display countdown if triggered
    if countdown_trigger:
        time_passed = time.time() - countdown_start_time
        countdown_time = 3 - int(time_passed)
        
        # If countdown is over, reset the trigger
        if countdown_time < 0:
            countdown_trigger = False
        
        # Otherwise, display the countdown time
        else:
            text = f"{countdown_time}"  # The countdown text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = orig.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
            text_y = text_size[1] + 10  # 10 pixels from the top edge
            cv2.putText(orig, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Display most frequent label text if scanning has started
    if start_scanning:
        label_scan = ""  # Default value
        prob_scan = 0  # Default value
        confidence_text_scan = ""  # Default value

        # Extract the top predicted label index of the class labels
        top_preds_indices = [class_labels[0][0] for _, class_labels in top_two_probs_and_indices]
        
        # # A list of your label encoders in the corresponding order
        # label_encoders = [le_prdtype, le_weight, le_halal, le_healthy]
        # # Obtain the label name for each element using inverse_transform for each label encoder
        # top_label_names = [label_encoders[i].inverse_transform([pred])[0] for i, pred in enumerate(top_preds)]

        sublabel_list_1.append(top_preds_indices[0])
        sublabel_list_2.append(top_preds_indices[1])
        sublabel_list_3.append(top_preds_indices[2])
        sublabel_list_4.append(top_preds_indices[3])

        # If 3 seconds have passed since we started scanning, stop scanning and process the frames
        if time.time() - scan_start_time > 3:
            start_scanning = False
            scan_start_time = None

            print(len(sublabel_list_1))
            print(Counter(sublabel_list_1))

            # Extract top two predictions and their relative frequencies for each sublabel list
            top_two_labels_1, top_two_probs_1 = get_top_two_predictions_and_frequencies(sublabel_list_1)
            top_two_labels_2, top_two_probs_2 = get_top_two_predictions_and_frequencies(sublabel_list_2)
            top_two_labels_3, top_two_probs_3 = get_top_two_predictions_and_frequencies(sublabel_list_3)
            top_two_labels_4, top_two_probs_4 = get_top_two_predictions_and_frequencies(sublabel_list_4)

            print(top_two_probs_1)
            print(top_two_probs_2)
            print(top_two_probs_3)
            print(top_two_probs_4)

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

            print(top_probs_combined)
            print(top_labels_combined)

            # Generate all possible label combinations and their joint probabilities
            all_possible_labels_scan, all_joint_probs_scan = generate_labels_and_probs(top_probs_combined, top_labels_combined, 
                                                                            [le_prdtype, le_weight, le_halal, le_healthy])

            # Filter out the valid labels and their joint probabilities
            valid_labels_and_probs_scan = [(label, prob) for label, prob in zip(all_possible_labels_scan, all_joint_probs_scan) if label in all_categories]
            # print(valid_labels_and_probs)

            # If there are no valid labels, set label to 'Nonfood'
            if not valid_labels_and_probs:
                label_scan = "Nonfood"
                prob_scan = "NA"
            else:
                # Otherwise, pick the valid label with the highest joint probability
                label_scan, prob_scan = max(valid_labels_and_probs, key=lambda x: x[1])
                prob_scan *= 100

            confidence_text_scan = f"Prediction confidence: {float(prob_scan):.2f}%"

            print(label_scan)
            print(confidence_text_scan)

        
    if first_scan_flag:
        # Compute the y-coordinate offset based on the height of the 'start_text'
        start_text_offset_y = 20  # Some padding between the texts
        label_scan_position_y = text_y + start_text_offset_y

        # Put 'label_scan' text on the frame
        cv2.putText(orig, label_scan, (text_x, label_scan_position_y),
                    start_text_font, start_text_scale, start_text_color, start_text_thickness)

        confidence_text_scan_position_y = text_y + 2*start_text_offset_y

        # Put 'confidence_text_scan' text on the frame
        cv2.putText(orig, confidence_text_scan, (text_x, confidence_text_scan_position_y),
                    start_text_font, start_text_scale, start_text_color, start_text_thickness)


    # Show the output frame
    cv2.imshow("Frame", orig)
    


    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    fps.update()

# Display FPS information and cleanup
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
