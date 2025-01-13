import os
from os.path import join, exists
import numpy as np
import nibabel as nb
import pickle
from tqdm import tqdm

data_root = r'F:\Data\runtime\TransMorph\inhouse_mini'


def pkload(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    in_dir = join(data_root, 'Train')
    out_dir = join(data_root, 'Unpacked')
    subs = sorted([i for i in os.listdir(in_dir) if i.endswith('.pkl')])

    for sub_pkl in tqdm(subs, desc='Registering'):
        sub = sub_pkl[:-4]

        sub_dir = join(out_dir, sub)
        if not exists(sub_dir):
            os.makedirs(sub_dir)

        img, seg = pkload(join(in_dir, sub_pkl))
        nb.Nifti1Image(img.astype(np.float32), np.eye(4)).to_filename(join(sub_dir, 'img.nii.gz'))
        nb.Nifti1Image(seg.astype(np.uint8), np.eye(4)).to_filename(join(sub_dir, 'seg.nii.gz'))
