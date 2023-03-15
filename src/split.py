

import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from tqdm import tqdm
import yaml
import sys

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/split.py data/prepared data/split\n'
    )
    sys.exit(1)

random.seed(108)

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



def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders

def class_id_mapping():
    class_ids = params['class_id']
    name_to_id = {}
    for id in class_ids.keys():
        name_to_id[id] = class_ids[id]
    
    return name_to_id

def convert_and_save_annotations(class_name_to_id_mapping):
    input_path = sys.argv[1]
    annotations = [os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations', x) for x in os.listdir(os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations') )if x[-3:] == "xml"]
    annotations.sort()
    image_path = os.path.join(input_path,f"v{params['ingest']['dcount']}",'images')
    annot_path = os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations')


# Convert and save the annotations
    for ann in tqdm(annotations):
        info_dict = extract_info_from_xml(ann)
        convert_to_yolov5(info_dict,annot_path,class_name_to_id_mapping)


    return input_path


def split_and_save(t_img,t_annot,v_img,v_annot):
    split_train_image_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}",'images','train')
    split_train_annot_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}",'labels','train')
    split_val_image_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}",'images','val')
    split_val_annot_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}",'labels','val')
    os.makedirs(split_train_annot_path,exist_ok=True)
    os.makedirs(split_train_image_path,exist_ok=True)
    os.makedirs(split_val_image_path,exist_ok=True)
    os.makedirs(split_val_annot_path,exist_ok=True)
    move_files_to_folder(t_img, split_train_image_path)
    move_files_to_folder(v_img, split_val_image_path)
    move_files_to_folder(t_annot, split_train_annot_path)
    move_files_to_folder(v_annot, split_val_annot_path)

    return

def get_img_annots(input_path):
    images = [os.path.join(input_path,f"v{params['ingest']['dcount']}",'images', x) for x in os.listdir(os.path.join(input_path,f"v{params['ingest']['dcount']}",'images'))]
    annotations = [os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations', x) for x in os.listdir(os.path.join(input_path,f"v{params['ingest']['dcount']}",'annotations')) if x[-3:] == "txt"]
    return images,annotations


def yolov5Model():
    class_name_to_id_mapping = class_id_mapping()
    #Get input Path with conversion
    input_path = convert_and_save_annotations(class_name_to_id_mapping)
    #Get images and annotations path
    images, annotations = get_img_annots(input_path)
    images.sort()
    annotations.sort()

    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.1, random_state = 1)
    split_and_save(train_images,train_annotations,val_images,val_annotations)




def main():
    print("-------------------------------")
    print("Splitting.....")
    print("-------------------------------")

    if params['model'] == 'yolov5':
        yolov5Model()
    


if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))
    main()






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
