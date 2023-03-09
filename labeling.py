'''
* @name : labeling.py
* @date : 2023-02-20 오전 10:42
* @author : ssum
* @version : 1.0.0
* @modifyed :
'''

import os
import pandas as pd
import glob
from tqdm import tqdm
import pandas as pd


def create_labels_from_xlsx(root_path):
    df = pd.read_excel(f'{root_path}/Video_dataset_description.xlsx', sheet_name='Blad1')
    # SENSOR	CLASS	NUMBER	DISTANCE  BIN	INTER BIN NUMBER	DRONE TYPE / INTERNET SOURCE
    target_df = df[df['CLASS'] == 'DRONE']

    classes = {'Hubsan H107D+': 0,
                'Phantom 4 Pro': 1,
                'F450': 2}

    sensor_type = ['IR', 'V']
    dstype = ('identification', 'DRONE TYPE / INTERNET SOURCE')
    for sensor in sensor_type:
        annotations = glob.glob(f'{root_path}/data_{sensor}/labels/*DRONE*.txt')

        sensor_df = target_df[target_df['SENSOR'] == sensor]

        save_root = f'{root_path}/data_{sensor}/{dstype[0]}/labels'
        os.makedirs(save_root, exist_ok=True)

        # Convert and save the annotations
        for ann_f in tqdm(annotations):
            result_lines = []
            with open(ann_f, 'r', encoding='utf-8') as f:
                # 0 0.469 0.346 0.075 0.059
                # 여러 객체
                tg_list = list(map(lambda s: s.strip(), f.readlines()))
                i = 0
                for ori_txt in tg_list:
                    ori_list = ori_txt.split(' ')

                    # IR_AIRPLANE_0011_001.txt
                    filename = os.path.splitext(os.path.basename(ann_f))[0]
                    file_idx = int(filename.split('_')[2][:-1])
                    coord = ' '.join(ori_list[1:])
                    class_name = sensor_df[sensor_df['NUMBER'] == file_idx][dstype[1]].iloc[0]
                    class_idx = -1
                    try:
                        class_idx = classes[class_name]
                    except Exception as e:
                        trg_cls_list = class_name.split(', ')
                        if len(trg_cls_list) > 1:
                            excp_trg_dict = {'IR': [(7, 178), (157, 155)],
                                             'V': [(5, 253), (99, 124)]}
                            img_idx = int(filename.split('_')[-1])
                            '''
                            예외
                            IR 007 1~177; F450
                            IR 007 178~325 ; Hubsan H107D+

                            IR 157 1~154; Phantom 4 Pro
                            IR 157 155~308; F450

                            V 005 1~252; F450
                                253~ ; Hubsan H107D+ 
                            V 099 ~123
                                    124~
                            V 
                            '''
                            for tp in excp_trg_dict[sensor]:
                                if file_idx == tp[0]:
                                    if img_idx > tp[1]:
                                        class_idx = classes[trg_cls_list[0]]
                                    else:
                                        class_idx = classes[trg_cls_list[1]]
                        else:
                            # 당장은... 왼 phantom 오 f450 인 것같으니 이렇게...
                            # bbox 사이즈로 계산하기에도 무리가 있을 게, 영상을 돌려보니 무조건 팬텀이 작은건 아님
                            trg_cls_list = class_name.split(' and ')
                            class_idx = classes[trg_cls_list[i]]
                            i += 1
                    print(f'{class_idx} {coord}')
                    result_lines.append(f'{class_idx} {coord}')
                with open(f'{save_root}/{os.path.basename(ann_f)}', 'w', encoding='utf-8') as sf:
                    for result_line in result_lines:
                        sf.write(result_line + '\n')


def get_target_data(root_path):
    df = pd.read_excel(f'{root_path}/Video_dataset_description.xlsx', sheet_name='Blad1')
    # SENSOR	CLASS	NUMBER	DISTANCE  BIN	INTER BIN NUMBER	DRONE TYPE / INTERNET SOURCE
    target_df = df[df['CLASS'] == 'DRONE']

    classes = {'Hubsan H107D+': 0,
                'Phantom 4 Pro': 1,
                'F450': 2}

    sensor_type = ['IR', 'V']
    dstype = ('identification', 'DRONE TYPE / INTERNET SOURCE')
    for sensor in sensor_type:
        annotations = glob.glob(f'{root_path}/data_{sensor}/labels/*DRONE*.txt')

        sensor_df = target_df[target_df['SENSOR'] == sensor]

        save_root = f'{root_path}/data_{sensor}/{dstype[0]}_target/labels'
        os.makedirs(save_root, exist_ok=True)

        # Convert and save the annotations
        for ann_f in tqdm(annotations):
            result_lines = []
            with open(ann_f, 'r', encoding='utf-8') as f:
                # 0 0.469 0.346 0.075 0.059
                # 여러 객체
                tg_list = list(map(lambda s: s.strip(), f.readlines()))
                i = 0
                for ori_txt in tg_list:
                    ori_list = ori_txt.split(' ')

                    # IR_AIRPLANE_0011_001.txt
                    filename = os.path.splitext(os.path.basename(ann_f))[0]
                    file_idx = int(filename.split('_')[2][:-1])
                    coord = ' '.join(ori_list[1:])
                    class_name = sensor_df[sensor_df['NUMBER'] == file_idx][dstype[1]].iloc[0]
                    class_idx = -1
                    try:
                        class_idx = classes[class_name]
                    except Exception as e:
                        trg_cls_list = class_name.split(', ')
                        if len(trg_cls_list) > 1:
                            continue
                        else:
                            # 당장은... 왼 phantom 오 f450 인 것같으니 이렇게...
                            # bbox 사이즈로 계산하기에도 무리가 있을 게, 영상을 돌려보니 무조건 팬텀이 작은건 아님
                            trg_cls_list = class_name.split(' and ')
                            class_idx = classes[trg_cls_list[i]]
                            i += 1
                    print(f'{class_idx} {coord}')
                    result_lines.append(f'{class_idx} {coord}')
                with open(f'{save_root}/{os.path.basename(ann_f)}', 'w', encoding='utf-8') as sf:
                    for result_line in result_lines:
                        sf.write(result_line + '\n')



def create_detect_labels(root_path):
    for one_file in tqdm(glob.glob(f'{root_path}/*.txt')):
        with open(one_file, 'r', encoding='utf-8') as f:
            result_lines = []
            for line in f.readlines():
                spl_line = line.split(' ')
                spl_line[0] = '0'
                result_lines.append(' '.join(spl_line))
        # 썼던 파일 재사용
        with open(one_file, 'w', encoding='utf-8') as wf:
            wf.writelines(result_lines)


def count_bbox_obj(root_path, save_ok=True):
    result_dict = {}

    for one_file in tqdm(glob.glob(f'{root_path}/*.txt')):
        with open(one_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                spl_line = line.split(' ')

                if not spl_line[0] in result_dict.keys():
                    result_dict[spl_line[0]] = [1,1]
                else:
                    if idx == 0:
                        result_dict[spl_line[0]][0] += 1
                    result_dict[spl_line[0]][1] += 1

    df = pd.DataFrame.from_dict(result_dict, orient='index', columns=['images', 'bboxes'])
    df = df.sort_index()
    # df.to_excel(f'{root_path.replace("/labels", "")}/count_obj.xlsx')
    if save_ok:
        df.to_csv(f'{root_path.replace("/labels", "")}/count_obj.csv')

    return df


def count_trainset(root_path):
    result_df = pd.DataFrame(columns=['dataset', 'images', 'bboxes'])
    for d in ['train', 'test', 'val']:
        df = count_bbox_obj(f'{root_path}/{d}/labels', save_ok=False)
        df['dataset'] = d
        result_df = pd.concat([result_df, df], axis=0, ignore_index=True)

    result_df.to_csv(f'{root_path}/count_obj_trainset.csv')



if __name__ =='__main__':
    # root_path = 'dataset/data_IR/identification/labels'
    # root_path = 'data_IR/labels'
    # create_detect_labels(root_path)
    # count_bbox_obj(root_path)

    # for sensor in ['IR', 'V']:
    #     count_bbox_obj(root_path=f'dataset/data_{sensor}/identification/labels')

    root_path='dataset'
    create_labels_from_xlsx(root_path)

    # for target_task in ['identification', 'cognition']:
    #     root_path = f'dataset/{target_task}'
    #     count_trainset(root_path)