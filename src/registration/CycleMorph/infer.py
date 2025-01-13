import os
import glob
import torch
import argparse
import numpy as np
import nibabel as nb
from natsort import natsorted
from os.path import join, split
from torchvision import transforms
from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import load_pickle

import utils
from data import datasets, trans
from models.cycleMorph_model import CycleMorph
from models.cycleMorph_model import CONFIGS as CONFIGS


def main(device, dataset, dr):
    CM_dir = '/remote-home/hejj/Data/runtime/SoR_folder/CycleMorph'
    exp_dir = f'{CM_dir}/{dataset}/experiments'
    to_dir = f'{CM_dir}/results/{dataset}'

    atlas2sub = join(to_dir, 'atlas2sub')
    sub2atlas = join(to_dir, 'sub2atlas')

    atlas_file = f'{dr}/atlas.pkl'
    val_dir = f'{dr}/Val/'

    os.makedirs(atlas2sub, exist_ok=True)
    os.makedirs(sub2atlas, exist_ok=True)

    affine = np.load(join(dr, 'affine.npy'))
    template, atlas = load_pickle(atlas_file)
    nb.Nifti1Image(template.astype(np.float32), affine).to_filename(join(to_dir, 'template.nii.gz'))
    nb.Nifti1Image(atlas.astype(np.uint8), affine).to_filename(join(to_dir, 'atlas.nii.gz'))
    num_classes = atlas.max() + 1

    opt = CONFIGS['Cycle-Morph-v0']
    opt.device = device
    model = CycleMorph()
    model.initialize(opt)
    model = model.netG_A
    model_file = join(exp_dir, natsorted([i for i in os.listdir(exp_dir) if 'latest' not in i])[-1])
    best_model = torch.load(model_file)['state_dict_A']
    print('Best model: {}'.format(model_file))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(opt.inputSize, 'nearest')
    reg_model.cuda()

    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    test_files = natsorted(glob.glob(val_dir + '*.pkl'))
    test_set = datasets.ValidationDataset(test_files, atlas_file, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    with torch.no_grad():
        model.eval()
        s2a_dice_arr, a2s_dice_arr = [], []
        for idx, data in enumerate(test_loader):
            img_key = split(test_files[idx])[-1][:-4]
            print(f'[{idx + 1}/{len(test_files)}] Infer {img_key}')

            atlas_img, sub_img, atlas_seg, sub_seg = [t.cuda().float() for t in data]

            # subject to atlas
            def_img, flow = model(torch.cat((sub_img, atlas_img), dim=1))
            def_seg = reg_model([sub_seg, flow])
            s2a_dice = utils.dice_val(def_seg.long(), atlas_seg.long(), num_classes)
            s2a_dice_arr.append(s2a_dice)
            print(f'  mean s2a_dice: {np.nanmean(s2a_dice):.4f}')

            def_img = def_img[0, 0].cpu().numpy().astype(np.float32)
            def_seg = def_seg[0, 0].cpu().numpy().astype(np.uint8)
            nb.Nifti1Image(def_img, affine).to_filename(join(sub2atlas, f'{img_key}_img.nii.gz'))
            nb.Nifti1Image(def_seg, affine).to_filename(join(sub2atlas, f'{img_key}_seg.nii.gz'))

            # atlas to subject
            def_img, flow = model(torch.cat((atlas_img, sub_img), dim=1))
            def_seg = reg_model([atlas_seg, flow])
            a2s_dice = utils.dice_val(def_seg.long(), sub_seg.long(), num_classes)
            a2s_dice_arr.append(a2s_dice)
            print(f'  mean a2s_dice: {np.nanmean(a2s_dice):.4f}')

            def_img = def_img[0, 0].cpu().numpy().astype(np.float32)
            def_seg = def_seg[0, 0].cpu().numpy().astype(np.uint8)
            nb.Nifti1Image(def_img, affine).to_filename(join(atlas2sub, f'{img_key}_img.nii.gz'))
            nb.Nifti1Image(def_seg, affine).to_filename(join(atlas2sub, f'{img_key}_seg.nii.gz'))

            print('  results saved\n')

    print(f'total mean s2a_dice: {np.nanmean(s2a_dice_arr):.4f}')
    print(f'total mean a2s_dice: {np.nanmean(a2s_dice_arr):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument('-ds', type=str, default='Dataset005_InHouse-Sim',
                        choices=['Dataset001_OASIS', 'Dataset002_InHouse', 'Dataset004_OASIS-Sim',
                                 'Dataset005_InHouse-Sim'])
    args = parser.parse_args()

    '''
    GPU configuration
    '''
    GPU_iden = args.d
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    d = torch.device('cpu') if args.d == 'cpu' else torch.device(f'cuda:{args.d}')
    _dr = f'/remote-home/hejj/Data/runtime/SoR_folder/morph_data/{args.ds}'
    main(d, args.ds, _dr)

