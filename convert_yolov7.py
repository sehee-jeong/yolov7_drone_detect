'''
* @name : convert_yolov7.py
* @date : 2023-02-20 오전 10:24
* @author : ssum
* @version : 1.0.0
* @modifyed :
'''
import os
import random
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil

random.seed(108)

class_name_to_id_mapping = {"AIRPLANE": 0,
                            "BIRD": 1,
                            "DRONE": 2,
                            "HELICOPTER": 3}

def extract_coord_from_txt(file_path):
    # Initialise the info dict
    info_dict = {}
    info_dict['bboxes'] = []
    info_dict['txt_filename'] = os.path.basename(file_path)
    info_dict['img_filepath'] = file_path.replace('labels', 'images').replace('txt', 'png')
    with open(file_path, 'r', encoding='utf-8') as f:
        img = Image.open(info_dict['img_filepath'])
        info_dict['image_size'] = img.size  # (width, height)

        bbox_dict = {}
        tg_list = list(map(lambda s: s.strip(), f.readlines()))
        for coord in tg_list:
            bbox_dict['class'] = info_dict['txt_filename'].split('_')[1]  # IR_AIRPLANE_0011_001
            coord_list = [float(c) for c in coord.split(',')]  # 150.171249389648,79.0561218261719,23.9634246826172,15.0070953369141
            # bbox가 1개 있는 경우
            factor = int(len(coord_list)/4)
            if factor==1:
                #  'xmin': 20, 'ymin': 109, 'xmax': 81, 'ymax': 237}
                bbox_dict['xmin'] = coord_list[0]
                bbox_dict['ymin'] = coord_list[1]
                bbox_dict['xmax'] = coord_list[2]
                bbox_dict['ymax'] = coord_list[3]
                info_dict['bboxes'].append(bbox_dict)
            # bbox가 2개 이상 있는 경우; matlab 에서 변환된 좌표 위치가 순서대로 정렬되어 있지 않음...
            elif factor>1:
                for i in range(0,factor):
                    bbox_dict = {}
                    bbox_dict['class'] = info_dict['txt_filename'].split('_')[1]
                    bbox_dict['xmin'] = coord_list[i]
                    bbox_dict['ymin'] = coord_list[i+factor]
                    bbox_dict['xmax'] = coord_list[i+factor*2]
                    bbox_dict['ymax'] = coord_list[i+factor*3]
                    info_dict['bboxes'].append(bbox_dict)

    # print(info_dict)
    return info_dict


# PASCAL VOC (yolov5); x_min, y_min, x_max, y_max
# yolov7; class, x_center, y_center, width, and heigh
#         => 영상사이즈에 맞춰 비율로 환산
#         => index 번호는 0번부터 시작
def convert_to_yolov5(info_dict, save_root):
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
        b_width = (b["xmax"] - b["xmin"])
        b_height = (b["ymax"] - b["ymin"])

        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # Write the bbox details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

    print("\n".join(print_buffer), file=open(f'{save_root}/{info_dict["txt_filename"]}', "w"))


def convert_to_yolov2(info_dict, save_root):
    print_buffer = []
    # [x,y,width,height]. The format specifies the upper-left corner location and size of the bounding box in the corresponding image.
    # x_min, y_min, width, height

    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

        x_min = b['xmin']
        y_min = b['ymin']
        x_max = b['xmax'] + x_min   # info dict 상에서 xmax 밸류에 width, ymax 밸류에 height가 있음
        y_max = b['ymax'] + y_min

        b_center_x = (x_min + x_max) / 2
        b_center_y = (y_min + y_max) / 2
        b_width = b['xmax']
        b_height = b['ymax']

        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # Write the bbox details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

    print("\n".join(print_buffer), file=open(f'{save_root}/{info_dict["txt_filename"]}', "w"))
    # 이미지도 같이 옮긴다
    shutil.copy(info_dict["img_filepath"], f'{save_root_img}/{os.path.basename(info_dict["img_filepath"])}')


def plot_bounding_box(image, annotation_list):
    class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

    annotations = np.array(annotation_list)
    w, h = image.size

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (transformed_annotations[:, 3] / 2)
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (transformed_annotations[:, 4] / 2)
    transformed_annotations[:, 3] = transformed_annotations[:, 1] + transformed_annotations[:, 3]
    transformed_annotations[:, 4] = transformed_annotations[:, 2] + transformed_annotations[:, 4]

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0, y0), (x1, y1)))

        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])

    plt.imshow(np.array(image))
    plt.show()

def plot_bbox_yolov7_format():
    random.seed(0)

    # Get any random annotation file
    annotation_file = random.choice(conv_annotations)
    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x] for x in annotation_list]

    # Get the corresponding image file
    image_file = annotation_file.replace("labels", "images").replace("txt", "png")
    assert os.path.exists(image_file)

    # Load the image
    image = Image.open(image_file)

    # Plot the Bounding Box
    plot_bounding_box(image, annotation_list)


if __name__ == '__main__':

    # Get the annotations
    dataset_type = ['V'] #['IR', 'V']
    for dstype in dataset_type:
        annotations = glob.glob(f'C:/Data/data_{dstype}/labels/*.txt')

        save_root=f'data_{dstype}/labels'
        save_root_img= save_root.replace('labels', 'images')
        os.makedirs(save_root, exist_ok=True)
        os.makedirs(save_root_img, exist_ok=True)

        # Convert and save the annotations
        for ann_f in tqdm(annotations):
            info_dict = extract_coord_from_txt(ann_f)
            convert_to_yolov2(info_dict, save_root)

        # check
        conv_annotations=''
        conv_annotations = glob.glob(f'{save_root}/*.txt')
        plot_bbox_yolov7_format()