import os
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join, exists

np.random.seed(12345)

dr = ''


def nifty_reg(fixed_file, moving_file, out_file, to_dir, fixed_arr=None, moving_arr=None, out_arr=None):
    if out_arr is None:
        out_arr = []
    if moving_arr is None:
        moving_arr = []
    if fixed_arr is None:
        fixed_arr = []
    os.system(f'reg_f3d -be 0.0002 --ssd -ref {fixed_file} -flo {moving_file} '
              f'-res {to_dir}/{out_file} -voff '
              f'-cpp {to_dir}/ref_template_flo_new_image_nrr_cpp.nii')
    for i in range(len(fixed_arr)):
        os.system(f'reg_resample -ref {fixed_arr[i]} -flo {moving_arr[i]} '
                  f'-res {to_dir}/{out_arr[i]} -cpp {to_dir}/ref_template_flo_new_image_nrr_cpp.nii -inter 0')
    os.remove(f'{to_dir}/ref_template_flo_new_image_nrr_cpp.nii')


def main(dataset):
    src_dir = f'/home/jerry/Desktop/test_niftyreg/{dataset}'
    dst_dir = f'/home/jerry/Desktop/test_niftyreg/results/{dataset}'
    atlas_img = f'{src_dir}/{dataset.split("_")[-1]}-template.nii.gz'
    atlas_seg = f'{src_dir}/{dataset.split("_")[-1]}-atlas.nii.gz'

    imagesTr = join(src_dir, 'imagesTr')
    labelsTr = join(src_dir, 'labelsTr')
    atlas2sub = join(dst_dir, 'atlas2sub')
    sub2atlas = join(dst_dir, 'sub2atlas')

    if not exists(atlas2sub):
        os.makedirs(atlas2sub)
    if not exists(sub2atlas):
        os.makedirs(sub2atlas)

    test_keys = np.load(join(src_dir, 'vals.npy'))

    for img_key in tqdm(test_keys, desc='Registering'):
        sub_img = join(imagesTr, f'{img_key}_0000.nii.gz')
        sub_seg = join(labelsTr, f'{img_key}.nii.gz')

        # subject to atlas
        if not exists(join(sub2atlas, f'{img_key}_seg.nii.gz')):
            nifty_reg(fixed_file=atlas_img, moving_file=sub_img, to_dir=sub2atlas, out_file=f'{img_key}_img.nii.gz',
                      fixed_arr=[atlas_seg], moving_arr=[sub_seg], out_arr=[f'{img_key}_seg.nii.gz'])

        # atlas to subject
        if not exists(join(atlas2sub, f'{img_key}_seg.nii.gz')):
            nifty_reg(fixed_file=sub_img, moving_file=atlas_img, to_dir=atlas2sub, out_file=f'{img_key}_img.nii.gz',
                      fixed_arr=[sub_seg], moving_arr=[atlas_seg], out_arr=[f'{img_key}_seg.nii.gz'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, default='Dataset004_OASIS-Sim',
                        choices=['Dataset001_OASIS', 'Dataset002_InHouse', 'Dataset004_OASIS-Sim', 'Dataset005_InHouse-Sim'])
    args = parser.parse_args()
    main(args.ds)
