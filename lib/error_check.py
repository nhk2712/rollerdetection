import os
import json

json_checking_dir = "/home/citiai-cygnus/RollerDetection_HoangKhanh/roller_dataset_augmented/train_all_augmented/annotations"

files_lacking_area = []
areas = []

for json_file in os.listdir(json_checking_dir):
    if json_file.endswith('.json'):
         # Load JSON annotation file
        json_path = os.path.join(json_checking_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        for annotation in data['annotations']:
            bbox = annotation['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            if xmin>=xmax or ymin>=ymax:
                print(json_path)
     

