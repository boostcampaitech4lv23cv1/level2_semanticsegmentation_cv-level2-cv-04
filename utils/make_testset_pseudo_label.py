import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser(description='Hello world!')
    parser.add_argument('--infered_csv_path', default="/opt/ml/input/data/mIoU7866.csv", help="input absolute path")
    parser.add_argument('--train_json_path', default="/opt/ml/input/data/train_fold_3.json", help="input absolute path")
    parser.add_argument('--train_all_json_path', default="/opt/ml/input/data/train_all.json", help="input absolute path")
    parser.add_argument('--start_img_idx', default=0, type=int)
    parser.add_argument('--num_img', default=100, type=int, help="num of pseudo image")
    args = parser.parse_args()
    return args

# 다 코딩하고 나서야... 리스트 인덱스를 사용할껄이란 생각이 들었다...
def mask_to_name(mask:int)-> str:
    if mask == 1:
        name = "General trash"
    elif mask == 2:
        name = "Paper"
    elif mask == 3:
        name = "Paper pack"
    elif mask == 4:
        name = "Metal"
    elif mask == 5:
        name = "Glass"
    elif mask == 6:
        name = "Plastic"
    elif mask == 7:
        name = "Styrofoam"
    elif mask == 8:
        name = "Plastic bag"
    elif mask == 9:
        name = "Battery"
    elif mask == 10:
        name = "Clothing"
    else: # 0
        name = "Backgroud"
    return name

def main():
    args = parse_args()
    print("인퍼런스 csv파일 패스:", args.infered_csv_path)
    print("합칠 train_json_path:", args.train_json_path)
    # print("인퍼런스 csv파일에서 합쳐질 ratio(이미지 기준)", args.mixing_ratio)
    print()
    print("mix.json과 pseudo.json이 output으로 생성됩니다")

    sub_file = pd.read_csv(args.infered_csv_path)
    # num_of_row = int(len(sub_file)*float(args.mixing_ratio))
    sub_file = sub_file[args.start_img_idx:args.start_img_idx+args.num_img]
    print(f"{args.start_img_idx}~{args.start_img_idx+args.num_img-1} 인덱스 이미지만 쉐도 라벨을 만듭니다")
    annotation = args.train_json_path

    with open(annotation) as f: 
        train_annot = json.load(f)
    
    print("="*20, "CHECK", "="*20)
    if len(sub_file["PredictionString"][0].split(' ')) == 512*512:
        pass
    else:
        print("inference 결과는 512 x 512만 가능합니다, 더 이상 진행이 불가능합니다")
        return
    ## check
    
    id_list = [i["id"] for i in train_annot["images"]] # 이미지의 id만 따온다
    id_list = sorted(id_list) # 정렬
    # print(id_list[0]) # 시작 이미지 id
    # print(id_list[-1]) # 마지막 이미지 id
    # print(len(id_list))
    if len(id_list) == id_list[-1]+1:
        print("image의 아이디는 순차적으로 부여되어 있습니다")

    # 없는 annot_id를 확인하기 위해서 train_all.json을 가져옵니다.
    # 이 용도 외에는 사용하지 않습니다.
    with open(args.train_all_json_path) as f: 
        train_all_annot = json.load(f)

    # train_all의 annot입니다.
    annot_all_id_list = [d["id"] for d in train_all_annot["annotations"]]
    print("train_all의 가장 작은 annot_id", min(annot_all_id_list))
    print("train_all의 가장 큰 annot_id",max(annot_all_id_list))

    if max(annot_all_id_list)+1 != len(annot_all_id_list):
        print("총", len(annot_all_id_list), "개의 annot 존재하므로 train_all의 annot_id는 순차적으로 부여되어 있지 않습니다")
    print()

    # test set에 부여할 아이디를 넉넉히 담아놀 리스트
    remain_annot_ids = []

    for i in range(max(annot_all_id_list)+10000): # 넉넉히 잡기 위해 10000까지 범위를 잡음
        # 기존에 속한 아이디가 아니면
        if i not in annot_all_id_list:
            remain_annot_ids.append(i)

    classes_dict={
        'Backgroud':[[]] , # 0
        'General trash':[[]] , # 1 
        'Paper':[[]] , # 2
        'Paper pack': [[]], # 3 
        'Metal': [[]], # 4
        'Glass': [[]], # 5
        'Plastic':[[]], # 6
        'Styrofoam':[[]], # 7
        'Plastic bag':[[]], # 8
        'Battery':[[]], # 9
        'Clothing':[[]] # 10
    }

    cls_name_list = [
        'Backgroud', 
        'General trash', 
        'Paper', 
        'Paper pack', 
        'Metal', 
        'Glass', 
        'Plastic', 
        'Styrofoam', 
        'Plastic bag', 
        'Battery', 
        'Clothing']

    # cls_name_list.index()
    pseudo_annot = {} # 최종으로 output할 json(dict)
    pseudo_annot["info"] = train_annot["info"]
    pseudo_annot["licenses"]  = train_annot["licenses"]
    pseudo_annot["categories"] = train_annot["categories"]
    pseudo_annot["images"] = []
    pseudo_annot["annotations"] = []

    pseudo_img_id = id_list[-1] + 1
    # 이미지마다 iter 순환
    for image_file_name, preds in tqdm(zip(sub_file["image_id"], sub_file["PredictionString"])):
        pred = preds.split(' ')
        # 이미지 딕셔너리생성 및 추가
        dict_image = {}
        dict_image["license"] = 0
        dict_image["url"] = None
        dict_image["file_name"] = image_file_name
        dict_image["height"] = 512
        dict_image["width"] = 512
        dict_image["date_captured"] = None
        dict_image["id"] = pseudo_img_id
        pseudo_annot["images"].append(dict_image)

        '''동영스
        # [x0, y0, x1, y1, ..., xn, yn]
        pred = np.array(list(map(int, preds.split(" "))), dtype='uint8').reshape(512,512)
        unique, counts = np.unique(pred, return_counts=True)
        count_dict = dict(zip(unique, counts))
        print("unique",unique)
        
        # 2차원 배열로 변환
        pred = cv.resize(pred, dsize=(512,512))

        for cls in unique:
            if cls == 0:
                continue
            else:
                print(cls)
                y_points, x_points = np.where(pred==0)
                print("y_points", y_points)
                print("x_points", x_points)
        # print(pred)
        # print(pred.shape)
        # print()
        return
        '''

        for idx, mask in enumerate(pred):
            if mask != '0':
                cls_name = mask_to_name(int(mask))
                height, width = idx//512, idx%512 # divmod(idx, 512)
                classes_dict[cls_name][0].append(width)
                classes_dict[cls_name][0].append(height)

        # per image class dict를 돌면서
        for cls_name, coordinates in classes_dict.items():
            # mask가 있는 놈이면
            if cls_name != "Backgroud": 
                dict_annot = {}
                annot_id = remain_annot_ids.pop(0)
                dict_annot["id"] = annot_id
                dict_annot["image_id"] = pseudo_img_id
                dict_annot["category_id"] = cls_name_list.index(cls_name)
                dict_annot["segmentation"] = coordinates
                dict_annot["area"] = len(coordinates[0])
                dict_annot["bbox"] = [1,2,3,4]
                dict_annot["iscrowd"] = 0
                pseudo_annot["annotations"].append(dict_annot)
        pseudo_img_id += 1 # 모두 끝나고 이미지 아이디 1 증가

    train_img_ids = [i["id"] for i in train_annot["images"]]
    psuedo_img_ids = [i["id"] for i in pseudo_annot["images"]]
    train_annot_ids = [i["id"] for i in train_annot["annotations"]]
    psuedo_annot_ids = [i["id"] for i in pseudo_annot["annotations"]]
    full_img_ids = train_img_ids + psuedo_img_ids
    full_annot_ids = train_annot_ids + psuedo_annot_ids

    annot_ids = [d["id"] for d in pseudo_annot["annotations"]]
    print("original set의 이미지의 수:", len(train_img_ids))
    print("original set의 annot 개수:", len(train_annot_ids))
    print()
    print("추가된 이미지의 수:", pseudo_annot["images"][-1]["id"]-pseudo_annot["images"][0]["id"]+1)
    print("추가된 pseudo annot 개수:", len(annot_ids))
    print()
    print("합쳐진 이후 총 이미지 개수", len(full_img_ids))
    print("합쳐진 이후 총 annot 개수", len(full_annot_ids))
    print()
    # return 0

    with open("./pseudo.json", 'w') as f:
        json_str = json.dumps(pseudo_annot, indent=4)
        f.write(json_str)
    print("pseudo.json 이름으로 psudo label annotation을 생성했습니다")


if __name__ == '__main__':
    main()