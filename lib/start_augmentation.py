from augmentation_utils import *
from augmentation_pipeline import *
from dataset_utils import *
from training_utils import *
from shared_utils import get_class_labels, get_all_global_variable

class_labels = get_class_labels()
transformations = ["horizontal_flip", "scale_down", "shift_scale_rotate", "shift_scale_rotate_negative", "resize"]
scale_down_transform= scale_down_pipeline()
resize_transform = resize_pipeline()
horizontal_flip_transform = horizontal_flip_pipeline()
shift_scale_rotate_transform = shift_scale_rotate_pipeline()
shift_scale_rotate_negative_transform = shift_scale_rotate_negative_pipeline()

(class_labels,
augmented_images, augmented_bboxes, augmented_keypoints,
json_training_files, json_testing_files,
log_file,
train_img_dir, train_annot_dir,
test_img_dir, test_annot_dir,
json_training_path, image_training_path, 
json_testing_path, image_testing_path
) = get_all_global_variable()

augmented_image_train_path = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset_augmented/train_all_augmented/images'
augmented_json_train_path = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset_augmented/train_all_augmented/annotations'
augmented_image_test_path = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset_augmented/test_all_augmented/images'
augmented_json_test_path = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset_augmented/test_all_augmented/annotations'

def perform_augmentation(image, bboxes, keypoints, augmentation_type):
    if augmentation_type == "horizontal_flip":
        augmented = horizontal_flip_transform(image=image, bboxes=bboxes, keypoints=keypoints, class_labels=class_labels)
    elif augmentation_type == "scale_down":
        augmented = scale_down_transform(image=image, bboxes=bboxes, keypoints=keypoints, class_labels=class_labels)
    elif augmentation_type == "shift_scale_rotate":
        augmented = shift_scale_rotate_transform(image=image, bboxes=bboxes, keypoints=keypoints, class_labels=class_labels)
    elif augmentation_type == "shift_scale_rotate_negative":
        augmented = shift_scale_rotate_negative_transform(image=image, bboxes=bboxes, keypoints=keypoints, class_labels=class_labels)
    elif augmentation_type == "resize":
        augmented = resize_transform(image=image, bboxes=bboxes, keypoints=keypoints, class_labels=class_labels)
    return augmented


def visualize_all_augmented_images():
    for img, bbox, kpts in zip(augmented_images, augmented_bboxes, augmented_keypoints):
        display_image_with_bboxes_and_keypoints_cv2(img, bbox, kpts)

def main():
    error_log = []
    for json_file in json_training_files:
        json_path = os.path.join(json_training_path, json_file)
        image_file, image_path, image = get_image_file_and_path_and_renderable_image_from_json(json_file, 'train')
        
        annotation, bbox_data_for_albumentations, keypoints_data_for_albumentations = convert_data_for_augmentation(json_path)

        for transformation in transformations:
            # print(image)
            print("Trying to augment {} of type {}".format(json_file, transformation))
            try:
                augmented = perform_augmentation(image, bbox_data_for_albumentations, keypoints_data_for_albumentations, transformation)
                new_augmented_json_data = create_new_augmented_dataset(annotation, augmented)
            
                augmented_json_output_filename = get_path_for_saving_augmented_json(json_file, augmented_json_train_path, transformation)
                write_augmented_data_to_json(augmented_json_output_filename, new_augmented_json_data)
                print("Written {}".format(augmented_json_output_filename))
            
                augmented_image_output_filename = get_path_for_saving_augmented_image(json_file, augmented_image_train_path, transformation)
                write_augmented_image_to_file(augmented_image_output_filename, augmented)
                print("Written {}".format(augmented_image_output_filename))
                
                augmented_images.append(augmented['image'])
                augmented_bboxes.append(augmented['bboxes'])
                augmented_keypoints.append(augmented['keypoints'])
            except:
                error_log.append("Failed to augment {} of type {}".format(json_file, transformation))
                continue
       
    print()     
    print("Done augmentation! Failures:")
    for error in error_log:
        print(error)
    
main()