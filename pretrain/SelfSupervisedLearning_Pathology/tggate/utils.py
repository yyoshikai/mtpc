import random

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
try:
    from openslide import OpenSlide
except:
    print("openslide is not available")

def check_file(str_file):
    """check input file can be opened or can't"""
    try:
        temp = openslide.OpenSlide(str_file)
        print("OK")
    except:
        print("can't open")

def get_patch_mask(image, patch_size, threshold=None,):
    """
    get_patchesに加え, WSI内に複数ある切片を分けて認識するようにする。

    Parameters
    ----------
    iamge: openslide.OpenSlide
    patch_size: int
    threshold: float or None
        各patchが背景かどうかを判定する彩度のthreshold.
        Noneの場合, OTSU法で決定する。
    Returns
    -------
    mask: np.array(int)[wsi_height, wsi_width]
        各patchがどの切片に属するか(-1=背景)
    """
    level = image.get_best_level_for_downsample(patch_size)
    downsample = image.level_downsamples[level]
    ratio = patch_size / downsample
    whole = image.read_region(location=(0,0), level=level,
        size = image.level_dimensions[level]).convert('HSV')
    whole = whole.resize((int(whole.width / ratio), int(whole.height / ratio)))
    whole = np.array(whole, dtype=np.uint8)
    saturation = whole[:,:,1]

    if threshold is None:
        threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_OTSU)
    mask = saturation > threshold
    return mask

def make_patch(filein:str="", patch_size:int=256, patch_number:int=1000, seed:int=24771):
    """extract patch from WSI"""
    # set seed
    random.seed(seed)
    # load
    wsi = OpenSlide(filein)
    # get patch mask
    mask = get_patch_mask(image_file=filein, patch_size=patch_size)
    mask_shape=mask.shape
    # extract / append
    lst_number=np.array(range(len(mask.flatten())))[mask.flatten()]
    lst_number=random.sample(list(lst_number), patch_number)
    res = []
    ap = res.append
    for number in lst_number:
        v_h, v_w = divmod(number, mask_shape[1])
        patch_image=wsi.read_region(
            location=(int(v_w*patch_size), int(v_h*patch_size)),
            level=0,
            size=(patch_size, patch_size))
        ap(np.array(patch_image, np.uint8)[:,:,:3])
    return res, lst_number

def make_groupkfold(group_col, n_splits:int=5):
    temp_arr = np.zeros((len(group_col),1))
    kfold = GroupKFold(n_splits = n_splits).split(temp_arr, groups=group_col)
    kfold_arr = np.zeros((len(group_col),1), dtype=int)
    for n, (tr_ind, val_ind) in enumerate(kfold):
        kfold_arr[val_ind]=int(n)
    return kfold_arr

def sampling_patch_from_wsi(patch_number:int=200, all_number:int=2000, len_df:int=0, seed:int=None):
    if seed is not None:
        random.seed(seed)
    random_lst = list(range(all_number))
    return [random.sample(random_lst, patch_number) for i in range(len_df)]
