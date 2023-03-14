# import os
# import sys
# import yaml
# import pickle
# from tqdm import tqdm
# import cv2
# import fnmatch
# import math

# import splitfolders


# if len(sys.argv) != 3:
#     sys.stderr.write('Arguments error. Usage:\n')
#     sys.stderr.write(
#         '\tpython3 src/split.py data/prepared data/split\n'
#     )
#     sys.exit(1)





# def main():
#     params = yaml.safe_load(open('params.yaml'))
#     outputsplit = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
#     #makeBatches(outputsplit)
#     infer_batch = params['split']['val']
#     train_batch = params['split']['train']
#     input_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")
   
#     os.makedirs(outputsplit, exist_ok = True)
#     print("-------------------------------")
#     print("Splitting.....")
#     print("-------------------------------")
#     splitfolders.ratio(input_path, output=outputsplit, ratio=(train_batch, infer_batch), group_prefix=None, move=False)


# if __name__ == '__main__':
#     main()




import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/split_yolo.py data/prepared data/split\n'
    )
    sys.exit(1)

random.seed(108)

class_name_to_id_mapping = {"person-like": 0,
                           "person": 1}


def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict




# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict,annot_path,class_name_to_id_mapping):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join(annot_path, info_dict["filename"].replace("xml", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))




# random.seed(0)

# class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))


# Split the dataset into train-valid-test splits 


def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders


def main():


    
    class_name_to_id_mapping = {}
    
    
    params = yaml.safe_load(open('params.yaml'))
    class_ids = params['class_id']
    for id in class_ids.keys():
        class_name_to_id_mapping[id] = class_ids[id]
    print(class_name_to_id_mapping)
    input_path = sys.argv[1]
    annotations = [os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations', x) for x in os.listdir(os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations') )if x[-3:] == "xml"]
    annotations.sort()
    image_path = os.path.join(input_path,f"v{params['ingest']['dcount']}",'images')
    annot_path = os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations')


# Convert and save the annotations
    for ann in tqdm(annotations):
        info_dict = extract_info_from_xml(ann)
        convert_to_yolov5(info_dict,annot_path,class_name_to_id_mapping)
    #annotations = [os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations', x) for x in os.listdir(os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations')) if x[-3:] == "txt"]

    images = [os.path.join(input_path,f"v{params['ingest']['dcount']}",'images', x) for x in os.listdir(os.path.join(input_path,f"v{params['ingest']['dcount']}",'images'))]
    annotations = [os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations', x) for x in os.listdir(os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations')) if x[-3:] == "txt"]

    images.sort()
    annotations.sort()

    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.1, random_state = 1)
    


    split_train_image_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}",'images','train')
    split_train_annot_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}",'labels','train')
    split_val_image_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}",'images','val')
    split_val_annot_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}",'labels','val')
    os.makedirs(split_train_annot_path,exist_ok=True)
    os.makedirs(split_train_image_path,exist_ok=True)
    os.makedirs(split_val_image_path,exist_ok=True)
    os.makedirs(split_val_annot_path,exist_ok=True)
    move_files_to_folder(train_images, split_train_image_path)
    move_files_to_folder(val_images, split_val_image_path)
    move_files_to_folder(train_annotations, split_train_annot_path)
    move_files_to_folder(val_annotations, split_val_annot_path)





if __name__ == '__main__':

    main()