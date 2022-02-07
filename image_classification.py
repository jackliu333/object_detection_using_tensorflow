import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image
# from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model

# Pipeline configuration file
PATH_TO_PIPELINE_CONFIG = 'Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config'
configs = config_util.get_configs_from_pipeline_file(PATH_TO_PIPELINE_CONFIG)

# Load detection model
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
CHECKPOINT_PATH = 'Tensorflow/workspace/models/my_ssd_mobnet'
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-3')).expect_partial()

# List of the strings that is used to add correct label for each box
PATH_TO_LABELS = 'Tensorflow/workspace/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(model, image):
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
  
    # image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`
    # after expanding the dimension
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)

    # Run inference
    output_dict = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


def predict_class(img_path, min_score_thresh):
    img = cv2.imread(img_path)
    img = np.array(img)

    # Actual detection.
    detections = run_inference_for_single_image(detection_model, img)
    
    # min_score_thresh = 0.5
    
    # rst = {category_index[i+1]['name']:score for i, score in enumerate(detections['detection_scores']) if score >= min_score_thresh}
    rst = {category_index[detections['detection_classes'][i]+1]['name']:score for i, score in enumerate(detections['detection_scores']) \
           if score >= min_score_thresh}
    # rst = {category_index[detections['detection_classes'][i]]:score for i, score in enumerate(detections['detection_scores']) \
    #    if score >= min_score_thresh}
    # rst = {detections['detection_classes'][i]:score for i, score in enumerate(detections['detection_scores']) \
    #    if score >= min_score_thresh}
           
    # print(detections['detection_classes'])
           
    # save image
    vis_util.visualize_boxes_and_labels_on_image_array(
      img,
      detections['detection_boxes'],
      detections['detection_classes']+1,
      detections['detection_scores'],
      category_index,
      instance_masks=detections.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      min_score_thresh=min_score_thresh,
      line_thickness=8)

    # im = Image.fromarray(img)
    # 
    # im.save("tmp_output/img.jpg")
    cv2.imwrite("tmp_output/img.jpg", img)
    
    return rst

