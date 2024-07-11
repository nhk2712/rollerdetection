import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Directory paths
dest_annotation_dir = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset/horizontal_flip_train/annotations'
dest_image_dir = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset/horizontal_flip_train/images'
output_dir = '/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset/visualize_horizontal_flip_train'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate through files in dest_annotation_dir (assuming both annotation and image filenames match)
for json_file in os.listdir(dest_annotation_dir):
    if json_file.endswith('.json'):
        # Load JSON annotation file
        json_path = os.path.join(dest_annotation_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Find corresponding image filename with .jpg extension
        image_filename = json_file[:-5] + '.jpg'  # Replace .json with .jpg
        
        # Check if the corresponding image file exists
        image_path = os.path.join(dest_image_dir, image_filename)
        if os.path.exists(image_path):
            # Load corresponding image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Coco format: [xmin, ymin, width, height]
            # OpenCV format: [xmin, ymin, xmax, ymax]
            # Albumentation format (during augmentation): same as OpenCV 
            
            # Draw bounding box on the image
            for annotation in data['annotations']:
                if 'bbox' in annotation and 'keypoints' in annotation:
                    bbox = annotation['bbox']
                    keypoints = annotation['keypoints']
                    
                    # Draw bounding box
                    # For visualizing data directly
                    # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+int(bbox[2]), int(bbox[1])+int(bbox[3])), (255, 0, 0), 1)
                    # For visualizing data after augmentation
                    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
                    
                    # Draw keypoints
                    for i in range(0, len(keypoints), 3):
                        x = int(keypoints[i])
                        y = int(keypoints[i+1])
                        visibility = keypoints[i+2]
                        if visibility > 0:
                            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            
            # Save the modified image with annotations
            output_image_path = os.path.join(output_dir, image_filename)
            cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f'Saved annotated image: {output_image_path}')
        else:
            print(f'Corresponding image file not found for {json_file}')
