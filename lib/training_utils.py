import os
import cv2
import matplotlib.pyplot as plt
import shutil
from PIL import Image as Image
from shared_utils import get_all_global_variable
import sys
sys.path.append('../utils')

(class_labels,
augmented_images, augmented_bboxes, augmented_keypoints,
json_training_files, json_testing_files,
log_file,
train_img_dir, train_annot_dir,
test_img_dir, test_annot_dir,
json_training_path, image_training_path, 
json_testing_path, image_testing_path
) = get_all_global_variable()

def delete_augmented_image_and_annotation(folder_path, substrings, mode, dataset_type):
    """
    Deletes augmented image and annotation files from a specified folder based on certain substrings.

    Parameters:
    folder_path (str): The path to the folder containing the image and annotation files.
    substrings (list): A list of substrings that the filenames should contain to be considered for deletion.
    mode (str): Indicates whether the files are 'train' or 'test'. Write it with 'train' or 'test'
    dataset_type (str): Indicates whether the files are 'annotation' or 'image'. Write it with 'annotation' or 'image'

    Returns:
    None

    Prints:
    A success message indicating that all the augmented image and annotation files have been deleted.
    """
    if(mode == "train" and dataset_type == "annotation"):
        folder_path = train_annot_dir
    elif(mode == "train" and dataset_type == "image"):
        folder_path = train_img_dir
    elif(mode == "test" and dataset_type == "annotation"):
        folder_path = test_annot_dir
    elif(mode == "test" and dataset_type == "image"):
        folder_path = test_img_dir
    else:
        raise ValueError("Mode should be either 'train' or 'test' and dataset_type should be either 'annotation or 'image'")
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if any(substring in filename for substring in substrings):
            if os.path.isfile(file_path):
                os.remove(file_path)
    print(f"Deleted all the augmented image and annotation successfully")
    
def move_augmented_image_and_annotation(mode, dataset_type, substrings):
    """
    Moves augmented image and annotation files from a specified folder based on certain substrings.

    Parameters:
    mode (str): Indicates whether the files are 'train' or 'test'. Write it with 'train' or 'test'
    dataset_type (str): Indicates whether the files are 'annotation' or 'image'. Write it with 'annotation' or 'image'
    substrings (list): A list of substrings that the filenames should contain to be considered for moving.

    Returns:
    None

    Prints:
    A success message indicating that all the augmented image and annotation files have been moved.
    """
    
    substrings_concat = '_'.join(substrings)  # Join substrings into a single string
    
    if (dataset_type == 'annotation' and mode == 'train'):
        destination_folder = os.path.join(train_annot_dir, substrings_concat)
        folder_path = train_annot_dir
    elif (dataset_type == 'image' and mode == 'train'):
        destination_folder = os.path.join(train_img_dir, substrings_concat)
        folder_path = train_img_dir
    elif (dataset_type == ' annotation' and mode == 'test'):
        destination_folder = os.path.join(test_annot_dir, substrings_concat)
        folder_path = test_annot_dir
    elif (dataset_type == 'image' and mode == 'test'):
        destination_folder = os.path.join(test_img_dir, substrings_concat)
        folder_path = test_img_dir
    else:
        raise ValueError("dataset_type should be either 'annotation' or 'image' or ")
        
    os.makedirs(destination_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if any(substring in filename for substring in substrings):
            if os.path.isfile(file_path):
                shutil.move(file_path, destination_folder)
    
    print(f"Moved all the augmented image and annotation files successfully")
                
def get_image_file_and_path_and_renderable_image_from_json(json_file, mode):
    """
    This function extracts the image file name, its path, and a renderable image from a given JSON file.

    Parameters:
    json_file (str): The name of the JSON file containing the image information.
    mode (str): Indicates whether the image information is for 'train' or 'test'. Write it with 'train' or 'test'

    Returns:
    tuple: A tuple containing three elements:
           - The name of the image file.
           - The path to the image file.
           - The image in RGB format for rendering.

    Note:
    - The image file name is extracted from the JSON file name by removing the extension and appending ".jpg".
    - The image path is constructed by joining the image_training_path with the image file name.
    - The image is read using cv2.imread and then converted to RGB format using cv2.cvtColor.
    """
    
    if(mode == "train"):
        destination_path = train_img_dir
    elif(mode == "test"):
        destination_path = test_img_dir
        
    image_file = os.path.splitext(json_file)[0] + ".jpg"
    image_path = os.path.join(destination_path, image_file)
    image = cv2.imread(image_path)
    renderable_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_file, image_path, renderable_image
        
def display_image_with_bboxes_and_keypoints_cv2(image, bboxes, keypoints):
    """
    Displays an image with bounding boxes and keypoints using OpenCV and Matplotlib.

    Parameters:
    image (numpy.ndarray): The input image in RGB format.
    bboxes (list): A list of bounding boxes. Each bounding box is represented as a tuple of (xmin, ymin, xmax, ymax, label).
    keypoints (list): A list of keypoints. Each keypoint is represented as a tuple of (x, y).

    Returns:
    None

    Displays the image with bounding boxes and keypoints using OpenCV and Matplotlib.
    """
    pil_image = Image.fromarray(image)
    fig, ax = plt.subplots(1)
    ax.imshow(pil_image)
    
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, label = bbox
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin, ymin - 10, label, color='g', fontsize=12, weight='bold')

    for keypoint in keypoints:
        x, y = keypoint
        ax.plot(x, y, 'ro')

    plt.axis('off')
    plt.show()
    
def delete_debug_log_file():
    """
    Deletes the debug log file if it exists.

    Parameters:
    None

    Returns:
    None

    Prints:
    A success message indicating that the debug log file has been deleted,
    or a message indicating that the debug log file does not exist.

    Note:
    The log file name is hardcoded as "debug_log.txt".
    """
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Deleted {log_file} successfully.")
    else:
        print(f"{log_file} does not exist.")
        
# delete_augmented_image_and_annotation(image_training_path, ["scale_down", "shift_scale_rotate", "resize", "shift_scale_rotate_negative"])
# move_augmented_image_and_annotation(image_testing_path, "image", ["scale_down", "shift_scale_rotate", "resize", "shift_scale_rotate_negative"])