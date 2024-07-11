import json
import copy
import os
import cv2
from dataset_utils import get_information_of_dataset
# from training_utils import write_proccess_to_log


def read_annotation(json_path):
    """
    Read and parse a JSON file containing annotation data.

    Parameters:
    json_path (str): The path to the JSON file containing annotation data.

    Returns:
    dict: A dictionary representing the annotation data.

    Raises:
    FileNotFoundError: If the specified JSON file does not exist.
    json.JSONDecodeError: If the JSON file contains invalid data.
    """
    with open(json_path, 'r') as f:
        annotation = json.load(f)
    return annotation


def convert_data_for_augmentation(json_path):
    """
    Convert annotation data into a format suitable for augmentation using Albumentations library.

    Parameters:
    json_path (str): The path to the JSON file containing annotation data.

    Returns:
    tuple: A tuple containing the original annotation data, bounding box data for Albumentations format, and keypoint data for Albumentations format.

    Raises:
    FileNotFoundError: If the specified JSON file does not exist.
    json.JSONDecodeError: If the JSON file contains invalid data.
    """
    annotation = read_annotation(json_path)
    copy_of_annotations = copy.deepcopy(annotation)
    # proccesed_image_file_name = copy_of_annotations['images'][0]['file_name']
    _, bbox_data, keypoints_data, categories, class_name = get_information_of_dataset(annotation)
    
    bbox_data_for_albumentations = [
        [int(bbox_data[0]), int(bbox_data[1]), int(bbox_data[0] + bbox_data[2]), int(bbox_data[1] + bbox_data[3]), class_name]
    ]

    keypoints_data_for_albumentations = []
    
    for i in range(0, len(keypoints_data), 3):
        x = keypoints_data[i]
        y = keypoints_data[i + 1]
        keypoints_data_for_albumentations.append((x, y))
    # write_proccess_to_log(bbox_data, keypoints_data, proccesed_image_file_name, "Old")
    return annotation, bbox_data_for_albumentations, keypoints_data_for_albumentations


def get_path_for_saving_augmented_image(json_file, path, augmented_type):
    """
    Generate the file path for saving the augmented image.

    Parameters:
    json_file (str): The original JSON file name.
    path (str): The directory path where the augmented image will be saved.
    augmented_type (str): The type of augmentation applied to the image.

    Returns:
    str: The file path for saving the augmented image.

    The function constructs the file name for the augmented image by combining the original JSON file name,
    the augmented type, and the file extension ".jpg". It then joins the directory path and the file name
    to generate the full file path for saving the augmented image.
    """
    image_file_name = os.path.splitext(json_file)[0] + "_" + augmented_type + "_augmented" ".jpg"
    saved_image_path = os.path.join(path, image_file_name)
    return saved_image_path


def get_path_for_saving_augmented_json(json_file, path, augmented_type):
    """
    Generate the file path for saving the augmented JSON data.

    Parameters:
    json_file (str): The original JSON file name.
    path (str): The directory path where the augmented JSON data will be saved.
    augmented_type (str): The type of augmentation applied to the JSON data.

    Returns:
    str: The file path for saving the augmented JSON data.

    The function constructs the file name for the augmented JSON data by combining the original JSON file name,
    the augmented type, and the file extension ".json". It then joins the directory path and the file name
    to generate the full file path for saving the augmented JSON data.
    """
    json_file_name = os.path.splitext(json_file)[0] + "_" + augmented_type + "_augmented" ".json"
    saved_json_path = os.path.join(path, json_file_name)
    return saved_json_path


def write_augmented_image_to_file(output_filename, augmented):
    """
    Write the augmented image to a file.

    Parameters:
    output_filename (str): The name of the file where the augmented image will be saved.
    augmented (dict): A dictionary containing the augmented image data. The dictionary should have a key 'image' which maps to the augmented image.

    Returns:
    None

    This function uses the OpenCV library to write the augmented image to a file. The image is first converted from BGR to RGB color space before saving.
    """
    cv2.imwrite(output_filename, cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB))
    
    
def create_new_augmented_dataset(annotation, augmented):
    """
    Create a new augmented dataset based on the original annotation data.

    Parameters:
    annotation (dict): The original annotation data.
    augmented (dict): The augmented data containing bounding boxes, keypoints, and other relevant information.

    Returns:
    dict: The new augmented annotation data.

    This function creates a deep copy of the original annotation data, modifies the keypoints and bounding box
    information based on the augmented data, and returns the new annotation data.
    """
    copy_of_annotations = copy.deepcopy(annotation)
    # proccesed_image_file_name = copy_of_annotations['images'][0]['file_name']

    modified_keypoint = copy_of_annotations['annotations'][0]['keypoints']
    for i in range(len(augmented['keypoints'])):
        modified_keypoint[i * 3] = int(augmented['keypoints'][i][0])
        modified_keypoint[i * 3 + 1] = int(augmented['keypoints'][i][1])
    copy_of_annotations['annotations'][0]['keypoints'] = modified_keypoint

    new_bbox = augmented['bboxes'][0][:4]
    # write_proccess_to_log(new_bbox, modified_keypoint, proccesed_image_file_name, "New", "Augmented Dataset created")
    copy_of_annotations['annotations'][0]['bbox'] = [int(coord) for coord in new_bbox]
    return copy_of_annotations


def write_augmented_data_to_json(augmented_json_output_filename, new_augmented_json_data):
    """
    Write the augmented JSON data to a file.

    Parameters:
    augmented_json_output_filename (str): The name of the file where the augmented JSON data will be saved.
    new_augmented_json_data (dict): The new augmented annotation data to be saved.

    Returns:
    None

    This function opens the specified file in write mode, converts the new augmented JSON data to a JSON string,
    and writes the JSON string to the file. The file is then closed.
    """
    with open(augmented_json_output_filename , 'w') as f:
        json.dump(new_augmented_json_data, f)
        
        
def get_array_for_augmented():
    """
    This function returns three empty lists. These lists are intended to be used as placeholders for
    image, bounding box, keypoint from the augmentation process.

    Returns:
    tuple: A tuple containing three empty lists. The first list is for bounding image, the second list is for bbox data, and the third list is for keypoint data.
    """
    return([],[],[])