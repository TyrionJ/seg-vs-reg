import os
import ants
import argparse
import numpy as np
import nibabel as nb
from tqdm import tqdm
from os.path import join


def syn_registration(fixed_img, fixed_seg, moving_img, moving_seg):
    _fixed_img = ants.from_numpy(fixed_img.astype(np.float32))
    _fixed_seg = ants.from_numpy(fixed_seg.astype(np.float32))

    _moving_img = ants.from_numpy(moving_img.astype(np.float32))
    _moving_seg = ants.from_numpy(moving_seg.astype(np.float32))

    reg12 = ants.registration(fixed=_fixed_img, moving=_moving_img, type_of_transform='SyN', syn_metric='meansquares')

    def_img = ants.apply_transforms(fixed=_fixed_img,
                                    moving=_moving_img,
                                    transformlist=reg12['fwdtransforms'],
                                    interpolator='nearestNeighbor')

    def_seg = ants.apply_transforms(fixed=_fixed_seg,
                                    moving=_moving_seg,
                                    transformlist=reg12['fwdtransforms'],
                                    interpolator='nearestNeighbor')

    return def_img.numpy().astype(np.float32), def_seg.numpy().astype(np.uint8)


def main(dataset):
    dr = '/remote-home/hejj/Data/runtime/SoR_folder'

    src_dir = f'{dr}/SoR_raw/{dataset}'
    dst_dir = f'{dr}/SyN_results/{dataset}'

    imagesTr = join(src_dir, 'imagesTr')
    labelsTr = join(src_dir, 'labelsTr')
    atlas2sub = join(dst_dir, 'atlas2sub')
    sub2atlas = join(dst_dir, 'sub2atlas')

    tpl_file = f'{dr}/SoR_raw/{dataset}/{dataset.split("_")[-1]}-template.nii.gz'
    ats_file = f'{dr}/SoR_raw/{dataset}/{dataset.split("_")[-1]}-atlas.nii.gz'

    os.makedirs(atlas2sub, exist_ok=True)
    os.makedirs(sub2atlas, exist_ok=True)

    fixed_nii = nb.load(tpl_file)
    affine = fixed_nii.affine
    atlas_img = fixed_nii.get_fdata()
    atlas_seg = nb.load(ats_file).get_fdata()

    img_keys = sorted([i[:-7] for i in os.listdir(labelsTr) if i.endswith('.nii.gz')])
    for im_key in tqdm(img_keys, desc=f'registering {dataset}'):
        subject_img = nb.load(os.path.join(imagesTr, f'{im_key}_0000.nii.gz')).get_fdata()
        subject_seg = nb.load(os.path.join(labelsTr, f'{im_key}.nii.gz')).get_fdata()

        s2a_img, s2a_seg = syn_registration(atlas_img, atlas_seg, subject_img, subject_seg)
        nb.Nifti1Image(s2a_img, affine).to_filename(join(sub2atlas, f'{im_key}_img.nii.gz'))
        nb.Nifti1Image(s2a_seg, affine).to_filename(join(sub2atlas, f'{im_key}_seg.nii.gz'))

        a2s_img, a2s_seg = syn_registration(subject_img, subject_seg, atlas_img, atlas_seg)
        nb.Nifti1Image(a2s_img, affine).to_filename(join(atlas2sub, f'{im_key}_img.nii.gz'))
        nb.Nifti1Image(a2s_seg, affine).to_filename(join(atlas2sub, f'{im_key}_seg.nii.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, default='Dataset005_InHouse-Sim',
                        choices=['Dataset001_OASIS', 'Dataset002_InHouse', 'Dataset004_OASIS-Sim', 'Dataset005_InHouse-Sim'])
    args = parser.parse_args()
    main(args.ds)
