import os
import torch
import argparse
import warnings
import numpy as np
import nibabel as nb
from os.path import join
from batchgenerators.utilities.file_and_folder_operations import load_pickle

from utils import losses
from utils.config import args
from common.clip_data import clip_data
from Models.TransMatch import TransMatch
from Models.STN import SpatialTransformer

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


def compute_label_dice(gt, pred, num_classes):
    gt = gt[0, 0].cpu().numpy()
    pred = pred[0, 0].cpu().numpy()
    cls_lst = list(range(1, num_classes))
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return dice_lst


def main(device):
    model_file = '/remote-home/hejj/Data/runtime/SoR_folder/TransMatch/results/Dataset002_InHouse/checkpoint/dsc-0.6034_epoch-188.pth.tar'
    data_set = 'Dataset007_MCI-CSVD'
    atlas_pkl = '/remote-home/hejj/Data/runtime/SoR_folder/morph_data/Dataset002_InHouse/atlas.pkl'
    base_dir = f'/remote-home/hejj/Data/runtime/SoR_folder/SoR_raw/{data_set}'

    T1_dir = join(base_dir, 'T1')
    QSM_dir = join(base_dir, 'QSM')
    ROI_dir = join(base_dir, 'ROIs')

    img_size = (160, 224, 160)

    checkpoint = torch.load(model_file)
    net = TransMatch(args).to(device)
    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(device)
    reg_model = SpatialTransformer(img_size, 'bilinear').to(device)
    seg_model = SpatialTransformer(img_size, 'nearest').to(device)
    net.eval()
    reg_model.eval()
    seg_model.eval()

    dst_dir = f'/remote-home/hejj/Data/runtime/SoR_folder/TransMatch/preds/{data_set}'
    atlas2sub = join(dst_dir, 'atlas2sub')
    sub2atlas = join(dst_dir, 'sub2atlas')

    os.makedirs(atlas2sub, exist_ok=True)
    os.makedirs(sub2atlas, exist_ok=True)

    img_keys = sorted([i[:-7] for i in os.listdir(ROI_dir)])
    atlas_img, atlas_seg = load_pickle(atlas_pkl)
    atlas_img = np.transpose(atlas_img, (2, 1, 0))
    atlas_seg = np.transpose(atlas_seg, (2, 1, 0))
    num_classes = np.max(atlas_seg) + 1
    atlas_img = torch.from_numpy(atlas_img[None, None].astype(np.float32)).to(device)
    atlas_seg = torch.from_numpy(atlas_seg[None, None].astype(np.float32)).to(device)
    print(f'Class number = {num_classes}')

    print(f'\nThere are {len(img_keys)} cases to infer ({data_set}):')
    with torch.no_grad():
        dsc_arr = []
        for idx, img_key in enumerate(img_keys):
            print(f'[{idx + 1}/{len(img_keys)}] {img_key}')
            t1_file = join(T1_dir, f'{img_key}_0000.nii.gz')
            QSM_file = join(QSM_dir, f'{img_key}.nii.gz')
            ROI_file = join(ROI_dir, f'{img_key}.nii.gz')

            sub_img, sub_seg, sub_funcs = create_clip(t1_file, ROI_file, [QSM_file], img_size)
            sub_qsm = sub_funcs[0].astype(np.float16)
            affine = nb.load(t1_file).affine
            sub_img = np.transpose(sub_img, (2, 1, 0))
            sub_seg = np.transpose(sub_seg, (2, 1, 0))
            sub_qsm = np.transpose(sub_qsm, (2, 1, 0))

            sub_img = torch.from_numpy(sub_img[None, None].astype(np.float32)).to(device)
            sub_seg = torch.from_numpy(sub_seg[None, None].astype(np.float32)).to(device)
            sub_qsm = torch.from_numpy(sub_qsm[None, None].astype(np.float32)).to(device)

            # subject to atlas
            flow = net(sub_img, atlas_img)
            def_img = reg_model(sub_img, flow)
            def_qsm = reg_model(sub_qsm, flow)
            def_seg = seg_model(sub_seg, flow)

            dsc_trans = compute_label_dice(def_seg.long(), atlas_seg.long(), num_classes)
            dsc_arr.append(np.mean(dsc_trans))
            print(f'  sub2at dsc={np.mean(dsc_trans):.4f}')

            def_img = def_img[0, 0].cpu().numpy().astype(np.float32).transpose((2, 1, 0))
            def_qsm = def_qsm[0, 0].cpu().numpy().astype(np.float32).transpose((2, 1, 0))
            def_seg = def_seg[0, 0].cpu().numpy().astype(np.uint8).transpose((2, 1, 0))
            nb.Nifti1Image(def_img, affine).to_filename(join(sub2atlas, f'{img_key}_img.nii.gz'))
            nb.Nifti1Image(def_qsm, affine).to_filename(join(sub2atlas, f'{img_key}_qsm.nii.gz'))
            nb.Nifti1Image(def_seg, affine).to_filename(join(sub2atlas, f'{img_key}_seg.nii.gz'))

            # atlas to subject
            flow = net(atlas_img, sub_img)
            def_img = reg_model(atlas_img, flow)
            def_seg = seg_model(atlas_seg, flow)

            dsc_trans = compute_label_dice(def_seg.long(), sub_seg.long(), num_classes)
            dsc_arr.append(np.mean(dsc_trans))
            print(f'  at2sub dsc={np.mean(dsc_trans):.4f}')

            def_img = def_img[0, 0].cpu().numpy().astype(np.float32).transpose((2, 1, 0))
            def_seg = def_seg[0, 0].cpu().numpy().astype(np.uint8).transpose((2, 1, 0))
            nb.Nifti1Image(def_img, affine).to_filename(join(atlas2sub, f'{img_key}_img.nii.gz'))
            nb.Nifti1Image(def_seg, affine).to_filename(join(atlas2sub, f'{img_key}_seg.nii.gz'))

            print(f'result saved\n')

    print(f'Infer done, mean dsc: {np.mean(dsc_arr):.04f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, default='2', help='device: 0, 1, 2, ...')
    args = parser.parse_args()

    d = torch.device('cpu') if args.d == 'cpu' else torch.device(f'cuda:{args.d}')
    main(d)
