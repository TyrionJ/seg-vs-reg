import os
import torch
import argparse
import warnings
import numpy as np
import nibabel as nb
from os.path import join
from batchgenerators.utilities.file_and_folder_operations import load_pickle

import utils
from common.clip_data import clip_data
from models.cycleMorph_model import CycleMorph
from models.cycleMorph_model import CONFIGS as CONFIGS

warnings.filterwarnings('ignore')


def create_clip(t1_file, roi_file, func_files, img_size):
    t1_data = nb.load(t1_file).get_fdata()
    roi_data = nb.load(roi_file).get_fdata()
    func_data = [nb.load(i).get_fdata() for i in func_files]

    msk_data = roi_data.copy()
    msk_data[msk_data > 0] = 1
    t1_data *= msk_data
    t1_data = (t1_data - t1_data.min()) / (t1_data.max() - t1_data.min())

    t1_data, roi_data, func_data = clip_data(t1_data, roi_data, func_data, img_size)
    return t1_data.astype(np.float16), roi_data.astype(np.int8), func_data


def main(device):
    model_file = '/remote-home/hejj/Data/runtime/SoR_folder/CycleMorph/Dataset002_InHouse/experiments/dsc0.501.pth'
    data_set = 'Dataset007_MCI-CSVD'
    atlas_file = '/remote-home/hejj/Data/runtime/SoR_folder/morph_data/Dataset002_InHouse/atlas.pkl'
    base_dir = rf'/remote-home/hejj/Data/runtime/SoR_folder/SoR_raw/{data_set}'

    T1_dir = join(base_dir, 'T1')
    QSM_dir = join(base_dir, 'QSM')
    ROI_dir = join(base_dir, 'ROIs')

    atlas_img, atlas_seg = load_pickle(atlas_file)

    opt = CONFIGS['Cycle-Morph-v0']
    opt.device = device
    opt.inputSize = atlas_img.shape
    model = CycleMorph()
    model.initialize(opt)
    model = model.netG_A
    best_model = torch.load(model_file)['state_dict_A']
    model.load_state_dict(best_model)
    model = model.to(device)
    reg_model = utils.register_model(opt.inputSize, 'bilinear').to(device)
    seg_model = utils.register_model(opt.inputSize, 'nearest').to(device)

    dst_dir = f'/remote-home/hejj/Data/runtime/SoR_folder/CycleMorph/results/{data_set}'
    atlas2sub = join(dst_dir, 'atlas2sub')
    sub2atlas = join(dst_dir, 'sub2atlas')

    os.makedirs(atlas2sub, exist_ok=True)
    os.makedirs(sub2atlas, exist_ok=True)

    img_keys = sorted([i[:-7] for i in os.listdir(ROI_dir)])
    num_classes = atlas_seg.max() + 1
    atlas_img = torch.from_numpy(atlas_img[None, None].astype(np.float32)).to(device)
    atlas_seg = torch.from_numpy(atlas_seg[None, None].astype(np.float32)).to(device)
    print(f'Class number = {num_classes}')

    print(f'\nThere are {len(img_keys)} cases to infer ({data_set}):')
    with torch.no_grad():
        model.eval()
        dsc_arr = []
        for idx, img_key in enumerate(img_keys):
            print(f'[{idx + 1}/{len(img_keys)}] {img_key}')
            t1_file = join(T1_dir, f'{img_key}_0000.nii.gz')
            QSM_file = join(QSM_dir, f'{img_key}.nii.gz')
            ROI_file = join(ROI_dir, f'{img_key}.nii.gz')

            sub_img, sub_seg, sub_funcs = create_clip(t1_file, ROI_file, [QSM_file], opt.inputSize)
            sub_qsm = sub_funcs[0].astype(np.float16)
            affine = nb.load(t1_file).affine

            sub_img = torch.from_numpy(sub_img[None, None].astype(np.float32)).to(device)
            sub_seg = torch.from_numpy(sub_seg[None, None].astype(np.float32)).to(device)
            sub_qsm = torch.from_numpy(sub_qsm[None, None].astype(np.float32)).to(device)

            # subject to atlas
            def_img, flow = model(torch.cat((sub_img, atlas_img), dim=1))
            def_qsm = reg_model([sub_qsm, flow])
            def_seg = seg_model([sub_seg, flow])

            dsc_trans = utils.dice_val(def_seg.long(), atlas_seg.long(), num_classes)
            dsc_arr.append(np.mean(dsc_trans))
            print(f'  sub2at dsc={np.mean(dsc_trans):.4f}')

            def_img = def_img[0, 0].cpu().numpy().astype(np.float32)
            def_qsm = def_qsm[0, 0].cpu().numpy().astype(np.float32)
            def_seg = def_seg[0, 0].cpu().numpy().astype(np.uint8)
            nb.Nifti1Image(def_img, affine).to_filename(join(sub2atlas, f'{img_key}_img.nii.gz'))
            nb.Nifti1Image(def_qsm, affine).to_filename(join(sub2atlas, f'{img_key}_qsm.nii.gz'))
            nb.Nifti1Image(def_seg, affine).to_filename(join(sub2atlas, f'{img_key}_seg.nii.gz'))

            # atlas to subject
            def_img, flow = model(torch.cat((atlas_img, sub_img), dim=1))
            def_seg = seg_model([atlas_seg, flow])

            dsc_trans = utils.dice_val(def_seg.long(), sub_seg.long(), num_classes)
            dsc_arr.append(np.mean(dsc_trans))
            print(f'  at2sub dsc={np.mean(dsc_trans):.4f}')

            def_img = def_img[0, 0].cpu().numpy().astype(np.float32)
            def_seg = def_seg[0, 0].cpu().numpy().astype(np.uint8)
            nb.Nifti1Image(def_img, affine).to_filename(join(atlas2sub, f'{img_key}_img.nii.gz'))
            nb.Nifti1Image(def_seg, affine).to_filename(join(atlas2sub, f'{img_key}_seg.nii.gz'))

            print(f'result saved\n')

    print(f'Infer done, mean dsc: {np.mean(dsc_arr):.04f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, default='0', help='device: 0, 1, 2, ...')
    args = parser.parse_args()

    d = torch.device('cpu') if args.d == 'cpu' else torch.device(f'cuda:{args.d}')
    main(d)
