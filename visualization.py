'''
* @name : visualization.py
* @date : 2023-03-07 오전 9:16
* @author : ssum
* @version : 1.0.0
* @modifyed :
'''
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import glob


class_name_to_id_mapping = {'Hubsan H107D+': 0,
                            'Phantom 4 Pro': 1,
                            'F450': 2}


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

    for idx, ann in enumerate(transformed_annotations):
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0, y0), (x1, y1)))

        plotted_image.text((x0, y0 - 10), f'{idx}_{obj_cls}_{class_id_to_name_mapping[(int(obj_cls))]}')

    plt.imshow(np.array(image))
    # plt.show()
    plt.savefig(f'{save_path}/{os.path.basename(image_file)}', dpi=300)
    # plt.imsave(f'{save_path}/{os.path.basename(image_file)}', image)


if __name__ == '__main__':
    for sensor in ['IR', 'V']:
        target_path = f'dataset/data_{sensor}/identification_target/labels'
        save_path = target_path.replace('/labels', '/visual')
        os.makedirs(save_path, exist_ok=True)
        for annotation_file in tqdm(glob.glob(f'{target_path}/*.txt')):

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
