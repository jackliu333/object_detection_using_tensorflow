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
import base64
import requests
import sys
import pandas as pd
from openai import OpenAI
import time
import subprocess
import json

# Set the R_HOME environment variable to the R home directory used by RStudio
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
os.environ['PATH'] = '/Library/Frameworks/R.framework/Resources/bin' + os.pathsep + os.environ['PATH']

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
    "MULTIPLE_FRAME3": False,
    "RECOGNITION_ENABLED": False,
    "MODEL_PATH": 'trainingdataAll'
}

# define existing categories
annotations = pd.read_csv("../master_list.csv")
all_categories = list(annotations['label'].unique())

# load label encoder 
def load_label_encoder():
    le_prdtype = pickle.loads(open(os.path.join('../NN_model/model_weights', CONFIGS['MODEL_PATH'], 'le_prdtype.pickle'), "rb").read())
    le_weight = pickle.loads(open(os.path.join('../NN_model/model_weights', CONFIGS['MODEL_PATH'], 'le_weight.pickle'), "rb").read())
    le_halal = pickle.loads(open(os.path.join('../NN_model/model_weights', CONFIGS['MODEL_PATH'], 'le_halal.pickle'), "rb").read())
    le_healthy = pickle.loads(open(os.path.join('../NN_model/model_weights', CONFIGS['MODEL_PATH'], 'le_healthy.pickle'), "rb").read())
    
    return le_prdtype, le_weight, le_halal, le_healthy

le_prdtype, le_weight, le_halal, le_healthy = load_label_encoder()
label_encoders = [le_prdtype, le_weight, le_halal, le_healthy]
# print(le_prdtype.classes_)

# get prediction label names for product type
tmp_df = pd.read_csv(os.path.join('../NN_model/model_weights', CONFIGS['MODEL_PATH'], 'main_imgs_results_big_model.csv'))
prdtype_colnames = tmp_df.filter(like='ProductType', axis=1).columns.tolist()
# print(len(prdtype_colnames))

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

    model_path = os.path.join('../NN_model/model_weights', CONFIGS['MODEL_PATH'], 'multi_head_model.pth')
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

def encode_image_array(image_array):
    """Encodes an image array to Base64."""
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

with open('/Users/liupeng/Desktop/Research/api.txt', 'r') as file:
    api_key = file.read()

def gpt_scoring(img, testflag=True):
    test_df = {
            'product_type': ['Unknown'],
            'weight': ['Unknown'],
            'halal': ['Unknown'],
            'health': ['Unknown'],
            'image_reflection': ['Unknown'],
            'image_clarity': ['Unknown'],
            'product_type_confidence': ['Unknown'],
            'weight_confidence': ['Unknown'],
            'halal_confidence': ['Unknown'],
            'health_confidence': ['Unknown']
        }

    # Creating the DataFrame
    test_df = pd.DataFrame(test_df)
    test_content = "Unkonwn"

    if testflag:
        time.sleep(2)

        return test_df, test_content

    base64_image = encode_image_array(img)

    headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {api_key}"
        }

    payload = {
              "model": "gpt-4-vision-preview",
              "messages": [
                {
                  "role": "user",
                  "content": [
                    {
                      "type": "text",
                      "text": """ 
                      For this image, can you make a prediction for the following four labels? 
                      The product type is based on the appearance of the product, 
                      the weight is to recognize how heavy the product is by identify the weight information on the appearance,
                      the halal status is to recognize if the product is halal food or not, 
                      and the healthy status is to recognize if the product is healthy if it contains a red triangle shape based on Singapore standard.
                      Note that you can only choose one from the given options in the bracket for each label, even if you are not sure. 
                      also there is no need to add an extra note to your answer.
                        product type (
                            Babyfood
                            BabyMilkPowder
                            BeehoonVermicelli
                            BiscuitsCrackersCookies
                            Book
                            BreakfastCereals
                            CannedBakedBeans
                            CannedBeefOtherMeats
                            CannedBraisedPeanuts
                            CannedChicken
                            CannedFruits
                            CannedMushrooms
                            CannedPacketCreamersSweet
                            CannedPickles
                            CannedPorkLunchronMeat
                            CannedSardinesMackerel
                            CannedSoup
                            CannedTunaDace
                            CannedVegetarianFood
                            ChocolateMaltPowder
                            ChocolateSpread
                            CoffeePowder
                            CoffeeTeaDrink
                            CookingCreamMilk
                            CookingPastePowder
                            CornChip
                            DarkSoySauce
                            DriedBeans
                            DriedFruits
                            DriedMeatSeafood
                            DriedVegetables
                            FlavoredMilkDrink
                            Flour
                            FruitJuiceDrink
                            HerbsSpices
                            InstantMeals
                            InstantNoodlesMultipack
                            InstantNoodlesSingle
                            Jam
                            Kaya
                            KetchupChilliSauce
                            LightSoySauce
                            MaternalMilkPowder
                            MilkDrink
                            MilkPowder
                            Nuts
                            Oil
                            OtherBakingNeeds
                            OtherCannedBeansPeasNuts
                            OtherCannedSeafood
                            OtherCannedVegetables
                            OtherDriedFood
                            OtherHotBeveragesPowder
                            OtherNoodles
                            OtherSauceDressing
                            OtherSpreads
                            Pasta
                            PastaSauce
                            PeanutButter
                            Potatochips
                            PotatoSticks
                            RiceBrownOthers
                            RiceWhite
                            RolledOatsInstantOatmeal
                            Salt
                            SoftDrinksOtherReadyToDrink
                            SoupStock
                            Sugar
                            SweetsChocolatesOthers
                            TeaPowderLeaves
                            WetWiper
                        ),
                        weight ('400-499g', '700-799g', '500-599g', '200-299g', '100-199g',
                           '1-99g', '300-399g', '600-699g', '800-899g', '1000-1999g',
                           '900-999g', '3000-3999g'
                        ),
                        halal status ('NonHalal', 'Halal'),
                        healthy status ('NonHealthy', 'Healthy'),
                        
                        Also, provide the following assessment
                        image reflection (High, Medium, Low)
                        image clarity (High, Medium, Low)
                        prediction confidence for the product type (High, Medium, Low)
                        prediction confidence for the weight (High, Medium, Low)
                        prediction confidence for the halal status (High, Medium, Low)
                        prediction confidence for the healthy status (High, Medium, Low)

                        format your answer in json format so that it could be easily converted to dataframe
                        """
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                      }
                    }
                  ]
                }
              ],
              "max_tokens": 300
            }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # content = response.json()['choices'][0]['message']['content']

    try:
        # Attempt to access the data that may cause a KeyError
        content = response.json()['choices'][0]['message']['content']
    except KeyError as e:
        # Handle the error: for example, log it, assign a default value, etc.
        print(f"KeyError occurred: {e}")
        return test_df, test_content

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Extracting the message content (the table with predictions)
    content = response.json()['choices'][0]['message']['content']
    # cat("GPT triggered")

    # Parsing the content to extract prediction values
    # Splitting the content string into lines and then parsing each line
    lines = content.strip().split('\n')[2:]  # Skipping the header

    # Parsing the strings to extract the relevant information
    data_parsed = [line.replace('"', '').strip() for line in lines if ':' in line]
    data_dict = {item.split(":")[0].strip(): item.split(":")[1].strip().strip(',') for item in data_parsed}

    # Converting the dictionary into a DataFrame
    df_from_strings = pd.DataFrame([data_dict])
    col_names = ['product_type', 'weight', 'halal', 'health', 'image_reflection', 
                  'image_clarity', 'product_type_confidence', 'weight_confidence',
                  'halal_confidence', 'health_confidence']
    parsed_data = df_from_strings
    # Normalizing column names for consistency across all DataFrames
    normalized_dfs = []
    tmp_valid_paths = []

    df = df_from_strings
    if df.shape[1] != 0:
        if df.shape[1] != 10:
            print("--\n")
            print(i)
            print(df.shape[1])
            if df.columns[-1]=="Note":
                df.drop(columns=['Note'], inplace=True)
                df.columns = col_names
                normalized_dfs.append(df)
                tmp_valid_paths.append(tmp_paths[i])
            elif df.shape[1] == 11:
                df = df.iloc[:,:-1]
                df.columns = col_names
        else:
            # Renaming columns to have consistent names across all DataFrames
            df.columns = col_names

    # Combining all DataFrames into a single DataFrame
    gpt_pred_df = df
    
    return gpt_pred_df, content

# Video stream loop
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

def reset_multi_frame():
    global sublabel_list_prdtype3, sublabel_list_weight3, sublabel_list_halal3, sublabel_list_health3
    global countdown_active3, countdown_start_time3, countdown_duration3, show_result3
    global countdown_active4, countdown_start_time4, countdown_duration4, show_result4
    global all_frames_logits_prdtype, all_frames_logits_weight, all_frames_logits_halal, all_frames_logits_health
    global sublabel_list_prdtype4, sublabel_list_weight4, sublabel_list_halal4, sublabel_list_health4

    # Lists to hold label predictions for each sublabel in multi-frame mode3
    sublabel_list_prdtype3 = []
    sublabel_list_weight3 = []
    sublabel_list_halal3 = []
    sublabel_list_health3 = []

    # Lists to hold label predictions for each sublabel in multi-frame mode4
    sublabel_list_prdtype4 = []
    sublabel_list_weight4 = []
    sublabel_list_halal4 = []
    sublabel_list_health4 = []

    # Lists to hold logits of all label types for all frames
    all_frames_logits_prdtype = []
    all_frames_logits_weight = []
    all_frames_logits_halal = []
    all_frames_logits_health = []

    # Initially, countdown is not active
    countdown_active3 = False
    countdown_start_time3 = 0
    countdown_duration3 = 3  # Countdown from 3 seconds
    show_result3 = False

    countdown_active4 = False
    countdown_start_time4 = 0
    countdown_duration4 = 3.1  # Countdown from 3 seconds+
    show_result4 = False

reset_multi_frame()

while True:
    frame = vs.read()
    orig = frame.copy()

    # Define the instructions text
    instructions = [
        "1: Real time simple",
        "2: Real time complex",
        "3: Multiple frames simple",
        "4: Multiple frames GPT",
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

        if CONFIGS['MULTIPLE_FRAME3']:
            # print("starting scanning...")
            countdown_trigger = True
            # cv2.imshow("Frame", orig)

            # Display countdown if triggered
            # Countdown display logic
            if countdown_active3:
                current_time = time.time()
                elapsed_time = current_time - countdown_start_time3
                countdown_time = countdown_duration3 - int(elapsed_time)

                # Display the countdown time if within the countdown period
                if 0 < countdown_time <= countdown_duration3:
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
                    
                    sublabel_list_prdtype3.append(top_preds_indices[0])
                    sublabel_list_weight3.append(top_preds_indices[1])
                    sublabel_list_halal3.append(top_preds_indices[2])
                    sublabel_list_health3.append(top_preds_indices[3])

                    # print(top_preds_indices)
                    # print(sublabel_list_1)

                # Stop the countdown after reaching 0
                if countdown_time < 0:
                    countdown_active3 = False
                    show_result3 = True
                    # print("count down complete")

            if show_result3:
                # print("show results")
                
                # print(sublabel_list_1)
                if first_time_process3:
                    # Extract top two predictions and their relative frequencies for each sublabel list
                    top_two_labels_1, top_two_probs_1 = get_top_two_predictions_and_frequencies(sublabel_list_prdtype3)
                    top_two_labels_2, top_two_probs_2 = get_top_two_predictions_and_frequencies(sublabel_list_weight3)
                    top_two_labels_3, top_two_probs_3 = get_top_two_predictions_and_frequencies(sublabel_list_halal3)
                    top_two_labels_4, top_two_probs_4 = get_top_two_predictions_and_frequencies(sublabel_list_health3)

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

                    first_time_process3 = False

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

                # print(label_scan)
                # print(prob_scan)

                # Draw each label text in a new line
                for label_text in label_texts:
                    cv2.putText(orig, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    text_y += line_offset  # Move to the next line

        if CONFIGS['MULTIPLE_FRAME4']:
            # print("starting scanning...")
            countdown_trigger4 = True

            # Display countdown if triggered
            # Countdown display logic
            if countdown_active4:
                current_time = time.time()
                elapsed_time = current_time - countdown_start_time4
                countdown_time = countdown_duration4 - elapsed_time

                # Display the countdown time if within the countdown period
                if 0 < countdown_time <= countdown_duration4:
                    if countdown_time <= 0.1:
                        text = "Querying ChatGPT..."
                        orig2 = orig.copy()
                    else:
                        text = f"{int(countdown_time)+1}"  # Display numbers 3 to 1 
                    
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
                    
                    sublabel_list_prdtype4.append(top_preds_indices[0])
                    sublabel_list_weight4.append(top_preds_indices[1])
                    sublabel_list_halal4.append(top_preds_indices[2])
                    sublabel_list_health4.append(top_preds_indices[3])

                    # save logits of all frames
                    all_frames_logits_prdtype.append(labelPreds_prdtype[0])
                    all_frames_logits_weight.append(labelPreds_weight[0])
                    all_frames_logits_halal.append(labelPreds_halal[0])
                    all_frames_logits_health.append(labelPreds_healthy[0])

                    

                # Stop the countdown after reaching 0
                if countdown_time < 0:
                    countdown_active4 = False
                    show_result4 = True
                    # print("count down complete")
                    # print(all_frames_logits_prdtype[:2])

                    # big model prediction on prdtype as input for bayes model
                    big_model_input = pd.DataFrame([t.tolist() for t in all_frames_logits_prdtype], columns=prdtype_colnames)
                    # print(big_model_input.shape)

                    # gpt prediction
                    if not gpt_triggered:
                        print("Calling GPT...")
                        gpt_pred_df, gpt_content = gpt_scoring(orig, testflag=False)
                        gpt_triggered = True  # Set the flag to True after triggering function
                        # print(gpt_pred_df)
                        print(gpt_content)


            if show_result4:
                # print("show results")

                if first_time_process4:
                    # print(len(all_frames_logits_prdtype))

                    # Extract top two predictions and their relative frequencies for each sublabel list
                    top_two_labels_1, top_two_probs_1 = get_top_two_predictions_and_frequencies(sublabel_list_prdtype4)
                    top_two_labels_2, top_two_probs_2 = get_top_two_predictions_and_frequencies(sublabel_list_weight4)
                    top_two_labels_3, top_two_probs_3 = get_top_two_predictions_and_frequencies(sublabel_list_halal4)
                    top_two_labels_4, top_two_probs_4 = get_top_two_predictions_and_frequencies(sublabel_list_health4)

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

                    # Generate all possible label combinations and their joint probabilities
                    all_possible_labels_scan, all_joint_probs_scan = generate_labels_and_probs(top_probs_combined, top_labels_combined, 
                                                                                    [le_prdtype, le_weight, le_halal, le_healthy])

                    # Filter out the valid labels and their joint probabilities
                    valid_labels_and_probs_scan = [(label, prob) for label, prob in zip(all_possible_labels_scan, all_joint_probs_scan) if label in all_categories]

                    # If there are no valid labels, set label to 'Nonfood'
                    if not valid_labels_and_probs_scan:
                        big_model_label_scan = "Nonfood"
                        big_model_prob_scan = "NA"
                    else:
                        # Otherwise, pick the valid label with the highest joint probability
                        big_model_label_scan, big_model_prob_scan = max(valid_labels_and_probs_scan, key=lambda x: x[1])
                        big_model_prob_scan *= 100
                        # print(big_model_prob_scan)

                    print("Big model prediction")
                    print(big_model_label_scan)

                    ########## Enter into Bayes model #######
                    # print(top_probs_combined)
                    # print(top_labels_combined)
                    top_pred_idx_big_model = [sub_list[0] for sub_list in top_labels_combined]
                    # print(top_pred_idx_big_model)

                    # top prediction for each label
                    top_pred_label_big_model = [encoder.inverse_transform([idx])[0] for idx, encoder in zip(top_pred_idx_big_model, label_encoders)]
                    # print(top_pred_label_big_model)

                    # Prepare predictions on product type and image quality from GPT
                    # print("GPT prediction")
                    # print(gpt_pred_df)
                    gpt_pred_label = gpt_pred_df['product_type'] + "_" + gpt_pred_df["weight"] + "_" + gpt_pred_df["halal"] + "_" + gpt_pred_df["health"]
                    gpt_pred_label = gpt_pred_label.values[0]
                    gpt_pred_confidence = "Product_" + gpt_pred_df['product_type_confidence'] + "-Weight_" + gpt_pred_df['weight_confidence'] + "-Halal_" + gpt_pred_df['halal_confidence'] + "-Health_" + gpt_pred_df['health_confidence']
                    gpt_pred_confidence = gpt_pred_confidence.values[0]

                    (frame_height, frame_width) = orig.shape[:2]
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 2)[0]
                    text_x = (frame_width - text_size[0]) // 2
                    text_y = (frame_height + text_size[1]) // 2
                    text = "Applying Bayesian model..."
                    cv2.putText(orig2, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                    cv2.imshow("Frame", orig2)
                    cv2.waitKey(1)

                    print("Applying Bayesian model...")
                    # Call R script to perform scoring
                    # Pass the JSON directly as an argument (ensure it's not too large)
                    # input1 = df_json
                    input1 = big_model_input.to_json(orient='records')
                    input2 = gpt_pred_df.to_json(orient='records')

                    # The command (ensure correct paths)
                    command = ['Rscript', '../bayesian_model/v2/bayes_model_real_time.R', input1, input2]

                    # Run the command and capture the output
                    result = subprocess.run(command, capture_output=True, text=True)

                    # Check if the command was executed successfully
                    if result.returncode == 0:
                        # Print stdout for debug messages and output
                        # print("STDOUT from R:\n", result.stdout)
                        # Parse the JSON output from the R script if needed
                        try:
                            output = json.loads(result.stdout)
                            print("bayes model output")
                            print(output)
                            bayes_pred_prdtype = output['pred'][0]
                            print("Output from R:", output)
                        except json.JSONDecodeError:
                            print("Failed to parse JSON output.")
                    else:
                        # Print stderr for errors
                        print("Error running R script:\n", result.stderr)
                        bayes_pred_prdtype = "NA"

                    # time.sleep(3)
                    # print(top_pred_label_big_model)
                    # print(top_pred_label_big_model[1])
                    # print(bayes_pred_prdtype)
                    bayes_pred_label = bayes_pred_prdtype + "_" + top_pred_label_big_model[1] + "_" + top_pred_label_big_model[2] + "_" + top_pred_label_big_model[3]

                    first_time_process4 = False

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
                    f"-------- Deep Learning Model -------",
                    f"Prediction: {big_model_label_scan}",
                    f"Confidence: {float(big_model_prob_scan):.2f}%",
                    f"-------- ChatGPT -------",
                    f"Prediction: {gpt_pred_label}",
                    f"Confidence: {gpt_pred_confidence}",
                    f"-------- Bayesian Combination Model -------",
                    f"Prediction: {bayes_pred_label}",
                ]

                # print(label_scan)
                # print(prob_scan)

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
        CONFIGS['MULTIPLE_FRAME3'] = False
        CONFIGS['MULTIPLE_FRAME4'] = False
        reset_multi_frame()
    elif key == ord("2"):
        CONFIGS['RECOGNITION_ENABLED'] = True
        CONFIGS['SINGLE_FRAME_TOP_PRED'] = False
        CONFIGS['SINGLE_FRAME_TOP_TWO_PRED'] = True
        CONFIGS['MULTIPLE_FRAME3'] = False
        CONFIGS['MULTIPLE_FRAME4'] = False
        reset_multi_frame()
    elif key == ord("3"):
        if not countdown_active3 and not show_result3:
            CONFIGS['RECOGNITION_ENABLED'] = True
            CONFIGS['SINGLE_FRAME_TOP_PRED'] = False
            CONFIGS['SINGLE_FRAME_TOP_TWO_PRED'] = False
            CONFIGS['MULTIPLE_FRAME3'] = True
            CONFIGS['MULTIPLE_FRAME4'] = False
            countdown_active3 = True
            countdown_start_time3 = time.time()  # Reset countdown start time
            show_result3 = False
            first_time_process3 = True
        else:
            reset_multi_frame()
    elif key == ord("4"):
        if not countdown_active4 and not show_result4:
            CONFIGS['RECOGNITION_ENABLED'] = True
            CONFIGS['SINGLE_FRAME_TOP_PRED'] = False
            CONFIGS['SINGLE_FRAME_TOP_TWO_PRED'] = False
            CONFIGS['MULTIPLE_FRAME3'] = False
            CONFIGS['MULTIPLE_FRAME4'] = True
            countdown_active4 = True
            countdown_start_time4 = time.time()  # Reset countdown start time
            show_result4 = False
            first_time_process4 = True
            gpt_triggered = False
        else:
            reset_multi_frame()
    elif key == ord("0"):
        CONFIGS['RECOGNITION_ENABLED'] = False
        reset_multi_frame()
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
