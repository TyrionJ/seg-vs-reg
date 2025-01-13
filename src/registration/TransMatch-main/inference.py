import os
import glob
import shutil

import torch
import warnings
import numpy as np
import SimpleITK as sitk
from os.path import join
from natsort import natsorted

from utils import losses
from utils.config import args
from Models.TransMatch import TransMatch
from Models.STN import SpatialTransformer

warnings.filterwarnings('ignore')


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def compute_label_dice(gt, pred, num_classes):
    gt = gt[0, 0].cpu().numpy()
    pred = pred[0, 0].cpu().numpy()
    cls_lst = list(range(1, num_classes))
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return dice_lst


def make_itk(data, itk_sam):
    data_itk = sitk.GetImageFromArray(data)
    data_itk.SetSpacing(itk_sam.GetSpacing())
    data_itk.SetOrigin(itk_sam.GetOrigin())
    data_itk.SetDirection(itk_sam.GetDirection())
    return data_itk


def infer():
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    atlas2sub = join(args.save_dir, 'atlas2sub')
    sub2atlas = join(args.save_dir, 'sub2atlas')
    os.makedirs(atlas2sub, exist_ok=True)
    os.makedirs(sub2atlas, exist_ok=True)

    f_img = sitk.ReadImage(args.template_file)
    atlas_img = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = atlas_img.shape[2:]

    atlas_img = torch.from_numpy(atlas_img).to(device).float()
    atlas_seg = sitk.GetArrayFromImage(sitk.ReadImage(args.atlas_file))[np.newaxis, np.newaxis, ...]
    atlas_seg = torch.from_numpy(atlas_seg).to(device).float()
    num_classes = int(atlas_seg.max()) + 1

    shutil.copy(args.template_file, f'{args.save_dir}/template.nii.gz')
    shutil.copy(args.atlas_file, f'{args.save_dir}/atlas.nii.gz')

    check_dir = args.save_model_dir
    model_file = join(check_dir, natsorted([i for i in os.listdir(check_dir) if 'latest' not in i])[-1])
    checkpoint = torch.load(model_file)
    net = TransMatch(args).to(device)
    net.load_state_dict(checkpoint['state_dict'])

    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    net.eval()
    STN.eval()

    # Training loop.
    test_file_lst = sorted(glob.glob(os.path.join(args.test_dir, "*.nii.gz")))

    with torch.no_grad():
        net.eval()
        STN.eval()
        STN_label.eval()
        s2a_dice_arr, a2s_dice_arr = [], []
        for N, file in enumerate(test_file_lst):
            file_name = os.path.split(file)[1]
            img_key = '_'.join(file_name[:-7].split('_')[::-1])
            print(f'[{N+1}/{len(test_file_lst)}] infer {img_key}:')

            sub_itk = sitk.ReadImage(file)
            sub_img = sitk.GetArrayFromImage(sub_itk)[np.newaxis, np.newaxis, ...]
            sub_img = torch.from_numpy(sub_img).to(device).float()
            sub_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, file_name)))
            sub_seg = sub_seg[np.newaxis, np.newaxis, ...]
            sub_seg = torch.from_numpy(sub_seg).to(device).float()

            # atlas to subject
            flow = net(atlas_img, sub_img)
            def_img = STN(atlas_img, flow)
            def_seg = STN_label(atlas_seg, flow)
            a2s_dice = compute_label_dice(sub_seg, def_seg, num_classes)
            a2s_dice_arr.append(a2s_dice)
            print(f'  mean a2s_dice: {np.nanmean(a2s_dice):.4f}')

            def_img = def_img[0, 0].cpu().numpy().astype(np.float32)
            def_seg = def_seg[0, 0].cpu().numpy().astype(np.uint8)

            def_img_itk = make_itk(def_img, sub_itk)
            sitk.WriteImage(def_img_itk, join(atlas2sub, f'{img_key}_img.nii.gz'))
            def_seg_itk = make_itk(def_seg, sub_itk)
            sitk.WriteImage(def_seg_itk, join(atlas2sub, f'{img_key}_seg.nii.gz'))

            # subject to atlas
            flow = net(sub_img, atlas_img)
            def_img = STN(sub_img, flow)
            def_seg = STN_label(sub_seg, flow)
            s2a_dice = compute_label_dice(atlas_seg, def_seg, num_classes)
            s2a_dice_arr.append(s2a_dice)
            print(f'  mean s2a_dice: {np.nanmean(s2a_dice):.4f}')

            def_img = def_img[0, 0].cpu().numpy().astype(np.float32)
            def_seg = def_seg[0, 0].cpu().numpy().astype(np.uint8)

            def_img_itk = make_itk(def_img, sub_itk)
            sitk.WriteImage(def_img_itk, join(sub2atlas, f'{img_key}_img.nii.gz'))
            def_seg_itk = make_itk(def_seg, sub_itk)
            sitk.WriteImage(def_seg_itk, join(sub2atlas, f'{img_key}_seg.nii.gz'))

            print('  results saved\n')

    print(f'total mean s2a_dice: {np.nanmean(s2a_dice_arr):.4f}')
    print(f'total mean a2s_dice: {np.nanmean(a2s_dice_arr):.4f}')


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    infer()
