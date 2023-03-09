'''
* @name : split_dataset.py
* @date : 2023-02-22 오후 1:44
* @author : ssum
* @version : 1.0.0
* @modifyed :
'''
# import splitfolders

# # Split with a ratio.
# # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
# splitfolders.ratio("data/origin/recognition", output="data/dataset/recognition",
#     seed=1337, ratio=(.8, .1, .1), group_prefix=2, move=False) # default values
# => 왜인지 split folders 모듈이 먹히지 않음...

import os
import glob
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import pandas as pd


def labeled_raw_data(root_path):
    labels=[]
    images = glob.glob(f'{root_path}/images/*.png')
    for img_file in tqdm(images):
        class_name = os.path.basename(img_file).split('_')[1]
        labels.append(class_name_to_id_mapping[class_name])
    return images, labels


def save_dataset(tp, save_path):
    print(f'start to save {tp[0]} set')
    save_img_dir = f'{save_path}/{tp[0]}/images'
    save_label_dir = f'{save_path}/{tp[0]}/labels'
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    for o_file in tqdm(tp[1]):
        shutil.copy(o_file, f'{save_img_dir}/{os.path.basename(o_file)}')
        shutil.copy(o_file.replace('images', 'labels').replace('.png', '.txt')
                    , f'{save_label_dir}/{os.path.basename(o_file).replace(".png", ".txt")}')
    print('done!')


def labeled_raw_data_from_xlsx(root_path):
    images = []
    labels = []

    for ann_f in tqdm(glob.glob(f'{root_path}/labels/*.txt')):
        with open(ann_f, 'r', encoding='utf-8') as f:
            # 0 0.469 0.346 0.075 0.059
            ori_txt = f.readline().strip()
            class_id = ori_txt.split(' ')[0]
            labels.append(class_id)

            # IR_AIRPLANE_0011_001
            filename = os.path.splitext(os.path.basename(ann_f))[0]
            images.append(f'{root_path}/images/{filename}.png')
    return images, labels


def random_sampling(images,labels):
    df=pd.DataFrame({'img':images,'label': labels})

    # drone identification
    target_df=df[df['label']=='2']
    print(f'target_df = {len(target_df)}')

    result_df = df.drop(target_df.index)
    target_df = target_df.drop_duplicates(['img'], keep = 'first')

    print(f'result_df={len(result_df)}, deleted dup;target_df={len(target_df)}')

    # ration는1/6
    target_df=target_df.sample(frac=0.2,random_state=512)
    print(f'sampling; target_df = {len(target_df)}')

    result_df = pd.concat([result_df, target_df], axis=0)
    result_df = result_df.sort_values('img')

    print(f'concatenated result_df={len(result_df)}')
    # res_images = result_df['img'].values.tolist()
    # res_labels = result_df['label'].values.tolist()

    target_imgs = result_df['img'].values.tolist()
    result_df=df.query("img in @target_imgs")
    print(result_df.head())
    print(f'latest result_df = {len(result_df)}')

    result_df = result_df.sort_values('img')

    res_images = result_df['img'].values.tolist()
    res_labels = result_df['label'].values.tolist()


    return res_images, res_labels



if __name__=='__main__':
    class_name_to_id_mapping = {"AIRPLANE": 0,
                                "BIRD": 1,
                                "DRONE": 2,
                                "HELICOPTER": 3}

    classes = { 'recognition': {"AIRPLANE": 0,
                                "BIRD": 1,
                                "DRONE": 2,
                                "HELICOPTER": 3},
                'distance': {'CLOSE': 0,
                            'MEDIUM': 1,
                            'DISTANT': 2},
               'identification': {'Hubsan H107D+': 0,
                                  'Phantom 4 Pro': 1,
                                  'F450': 2}}


    target_list = ['identification', 'cognition']

    # root_path = 'data/origin'
    # save_path = 'data/dataset/recognition'

    for target in target_list:
        root_path = f'dataset/{target}'
        save_path = f'dataset/{target}'

        images, labels = labeled_raw_data_from_xlsx(root_path)
        if target == 'identification':
            images, labels = random_sampling(images,labels)

        # 8:1:1
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=512, shuffle=True, stratify=labels)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=512, shuffle=True, stratify=y_test)

        print(f'train={len(X_train)}/{len(y_train)} : test={len(X_test)}/{len(y_test)} : val={len(X_val)}/{len(y_val)}')

        dataset = [('train', X_train), ('test', X_test), ('val', X_val)]
        for tp in dataset:
            save_dataset(tp, save_path)

