import os
import ants
import numpy as np
import nibabel as nb
from tqdm import tqdm
from os.path import join


def syn_registration(fixed_img, moving_img, fixed_arr, moving_arr):
    _fixed_img = ants.from_numpy(fixed_img.astype(np.float32))
    _moving_img = ants.from_numpy(moving_img.astype(np.float32))
    _fixed_arr = [ants.from_numpy(i.astype(np.float32)) for i in fixed_arr]
    _moving_arr = [ants.from_numpy(i.astype(np.float32)) for i in moving_arr]

    reg12 = ants.registration(fixed=_fixed_img, moving=_moving_img, type_of_transform='SyN', syn_metric='meansquares')

    def_img = ants.apply_transforms(fixed=_fixed_img,
                                    moving=_moving_img,
                                    transformlist=reg12['fwdtransforms'],
                                    interpolator='nearestNeighbor')

    def_arr = []
    for fixed, moving in zip(_fixed_arr, _moving_arr):
        _def = ants.apply_transforms(fixed=fixed, moving=moving,
                                     transformlist=reg12['fwdtransforms'],
                                     interpolator='nearestNeighbor')
        def_arr.append(_def.numpy())

    return def_img.numpy().astype(np.float32), def_arr


def main():
    data_set = 'Dataset007_MCI-CSVD'
    base_dir = f'/remote-home/hejj/Data/runtime/SoR_folder/SoR_raw/{data_set}'
    dst_dir = f'/remote-home/hejj/Data/runtime/SoR_folder/SyN_results/{data_set}'

    T1_dir = join(base_dir, 'T1')
    QSM_dir = join(base_dir, 'QSM')
    ROI_dir = join(base_dir, 'ROIs')

    atlas2sub = join(dst_dir, 'atlas2sub')
    sub2atlas = join(dst_dir, 'sub2atlas')
    os.makedirs(atlas2sub, exist_ok=True)
    os.makedirs(sub2atlas, exist_ok=True)

    tpl_file = '/remote-home/hejj/Data/runtime/SoR_folder/SoR_raw/Dataset002_InHouse/InHouse-template.nii.gz'
    ats_file = '/remote-home/hejj/Data/runtime/SoR_folder/SoR_raw/Dataset002_InHouse/InHouse-atlas.nii.gz'

    fixed_nii = nb.load(tpl_file)
    affine = fixed_nii.affine
    atlas_img = fixed_nii.get_fdata()
    atlas_seg = nb.load(ats_file).get_fdata()

    img_keys = sorted([i[:-7] for i in os.listdir(ROI_dir)])
    for im_key in tqdm(img_keys, desc='registering T1-QSM'):
        t1_file = join(T1_dir, f'{im_key}_0000.nii.gz')
        QSM_file = join(QSM_dir, f'{im_key}.nii.gz')
        ROI_file = join(ROI_dir, f'{im_key}.nii.gz')

        sub_t1 = nb.load(t1_file).get_fdata()
        sub_qsm = nb.load(QSM_file).get_fdata()
        sub_seg = nb.load(ROI_file).get_fdata()

        # subject to atlas
        s2a_img, s2a_arr = syn_registration(atlas_img, sub_t1, fixed_arr=[atlas_img, atlas_seg], moving_arr=[sub_qsm, sub_seg])
        nb.Nifti1Image(s2a_img, affine).to_filename(join(sub2atlas, f'{im_key}_img.nii.gz'))
        nb.Nifti1Image(s2a_arr[0].astype(np.float32), affine).to_filename(join(sub2atlas, f'{im_key}_qsm.nii.gz'))
        nb.Nifti1Image(s2a_arr[1].astype(np.uint8), affine).to_filename(join(sub2atlas, f'{im_key}_seg.nii.gz'))

        # atlas to subject
        a2s_img, s2a_arr = syn_registration(sub_t1, atlas_img, fixed_arr=[sub_qsm, sub_seg], moving_arr=[atlas_img, atlas_seg])
        nb.Nifti1Image(a2s_img, affine).to_filename(join(atlas2sub, f'{im_key}_img.nii.gz'))
        nb.Nifti1Image(s2a_arr[1].astype(np.uint8), affine).to_filename(join(atlas2sub, f'{im_key}_seg.nii.gz'))


if __name__ == '__main__':
    main()
