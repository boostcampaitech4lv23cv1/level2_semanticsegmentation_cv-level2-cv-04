import streamlit as st
import os.path as osp
import argparse
import pandas as pd
import cv2 as cv
import numpy as np
import mmcv

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


parser = argparse.ArgumentParser(description='basic Argparse')
parser.add_argument('--submission_csv', type=str, help='Infered된 csv 파일의 경로 ex)~/output.csv', default='./output.csv')
parser.add_argument('--dataset_path', type=str, help='데이터셋 폴더 경로', default='/opt/ml/input/data')
args = parser.parse_args()

classes = ['Backgroud',
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

palette = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0],
           [64, 0, 128], [64, 0, 192], [192, 128, 64], [192, 192, 128],
           [64, 64, 128], [128, 0, 192]]

def imshow_semantic(img,
                    seg,
                    class_names = classes,
                    palette=palette,
                    win_name='',
                    show=False,
                    wait_time=0,
                    opacity=0.5):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        seg (Tensor): The semantic segmentation results to draw over
            `img`.
        class_names (list[str]): Names of each classes.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    if palette is None:
        palette = np.random.randint(0, 255, size=(len(class_names), 3))
    palette = np.array(palette)
    assert palette.shape[0] == len(class_names)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    # if out_file specified, do not show image in window

    if show:
        mmcv.imshow(img, win_name, wait_time)

    return img


def main():
    st.title("Visualize your submission file")
    
    df = pd.read_csv(args.submission_csv, index_col=False)
        
    #image index 설정
    image_index = int(st.sidebar.number_input('보고싶은 이미지의 인덱스:', value=0))
    st.sidebar.image('./resources/colormap.png')
    
    image = df.iloc[image_index, 0]
    image = cv.imread(osp.join(args.dataset_path, image))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    pred = df.iloc[image_index, 1]
    pred = np.array(list(map(int, pred.split(" "))), dtype='uint8').reshape(256,256)
    unique, counts = np.unique(pred, return_counts=True)
    count_dict = dict(zip(unique, counts))
    pred = cv.resize(pred, dsize=(512,512))
    
    pred_image = imshow_semantic(image, pred)
    

    col1, col2 = st.columns(2)
    col1.text('Original Image')
    col1.image(image)
    col2.text('Infered Result')
    col2.image(pred_image)
    
    for c in sorted(unique, key=lambda x : count_dict[x], reverse=True):
        st.text(f'{classes[c]} : {round((count_dict[c] / 65536) * 100, 2)} %')
    

main()