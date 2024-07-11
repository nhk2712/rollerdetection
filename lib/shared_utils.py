from global_variable import get_json_files

def get_array_for_augmented():
    """
    This function returns three empty lists. These lists are intended to be used as placeholders for
    image, bounding box, keypoint from the augmentation process.

    Returns:
    tuple: A tuple containing three empty lists. The first list is for bounding image, the second list is for bbox data, and the third list is for keypoint data.
    """
    return([],[],[])

def get_all_global_variable():
    """
    This function retrieves all global variables required for the project.

    Returns:
    tuple: A tuple containing the following elements:
        - class_labels: A list of class labels.
        - augmented_images: A list of augmented images.
        - augmented_bboxes: A list of augmented bounding boxes.
        - augmented_keypoints: A list of augmented keypoints.
        - json_training_files: A list of filenames for training JSON files.
        - json_testing_files: A list of filenames for testing JSON files.
        - log_file: The path to the log file.
        - train_img_dir: The path to the training images directory.
        - train_annot_dir: The path to the training annotations directory.
        - test_img_dir: The path to the testing images directory.
        - test_annot_dir: The path to the testing annotations directory.
    """
    class_labels = get_class_labels()
    augmented_images, augmented_bboxes, augmented_keypoints = ([],[],[])
    log_file = '/home/citiai-cygnus/RollerDetection_HoangKhanh/debug_log.txt'
    train_img_dir = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset/train/images'
    train_annot_dir = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset/train/annotations'
    test_img_dir = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset/test/images'
    test_annot_dir = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset/test/annotations'
    json_training_files = get_json_files(train_annot_dir)
    json_testing_files = get_json_files(test_annot_dir)
    return ( 
            class_labels, 
            augmented_images, augmented_bboxes, augmented_keypoints,
            json_training_files, json_testing_files,
            log_file,
            train_img_dir, train_annot_dir,
            test_img_dir, test_annot_dir,
            train_annot_dir, train_img_dir, 
            test_annot_dir, test_img_dir
        )
    
def get_class_labels():
    """
    Returns a list of class labels for a dataset.

    Parameters:
    None

    Returns:
    list: A list of class labels. Each label corresponds to a specific class in the dataset.

    Raises:
    None

    Note:
    The number and names of the class labels should be consistent with the dataset.
    """
    return [
        "keypoint",
        "keypoint",
        "keypoint",
        "keypoint",
        "keypoint"
    ]