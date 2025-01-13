import os
import pickle
import shutil
import argparse
import numpy as np
import nibabel as nb
from tqdm import tqdm
from os.path import join, exists
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import load_pickle

np.random.seed(12345)


def create_SymTrans(data_type):
    img_size = (96, 112, 96)

    print('create_SymTrans, img_size =', img_size)
    dataset = 'Dataset001_OASIS' if data_type == 'OASIS' else 'Dataset002_InHouse'
    dr = f'/remote-home/hejj/Data/runtime/SoR_folder/SoR_raw/{dataset}'
    to_dir = f'/remote-home/hejj/Data/runtime/SoR_folder/SymTrans/{dataset}'

    templ_data = nb.load(join(dr, f'{dataset.split("_")[-1]}-template.nii.gz')).get_fdata().astype(np.float16)
    atlas_data = nb.load(join(dr, f'{dataset.split("_")[-1]}-atlas.nii.gz')).get_fdata().astype(np.uint8)
    templ_data[atlas_data == 0] = 0

    images_dir, labels_dir = join(dr, 'imagesTr'), join(dr, 'labelsTr')
    data_keys = sorted([i[:-7] for i in os.listdir(labels_dir) if i.endswith('.nii.gz')])
    train_keys, test_keys = train_test_split(data_keys, test_size=0.3)

    tr_dir = join(to_dir, 'train_data')
    atlas_file = join(to_dir, 'atlas_file')
    atlases_label = join(to_dir, 'atlas_label')
    valsets_file = join(to_dir, 'valset_file')
    valsets_label = join(to_dir, 'valset_label')

    if not exists(tr_dir):
        os.makedirs(tr_dir)
        os.makedirs(atlas_file)
        os.makedirs(atlases_label)
        os.makedirs(valsets_file)
        os.makedirs(valsets_label)

    with tqdm(total=2, desc=' atlas ') as p:
        _templ = resize(templ_data, img_size, order=3).astype(np.float16)
        _atlas = resize(atlas_data, img_size, order=3).astype(np.uint8)

        np.save(join(atlas_file, 'atlas_0001.npy'), _templ)
        p.update()
        np.save(join(atlases_label, 'atlas_0001_label.npy'), _atlas)
        p.update()

    for data_key in tqdm(train_keys, desc=' training set'):
        data = nb.load(join(images_dir, f'{data_key}_0000.nii.gz')).get_fdata()
        labl = nb.load(join(labels_dir, f'{data_key}.nii.gz')).get_fdata()
        data[labl == 0] = 0
        data = resize(data, img_size, order=3).astype(np.float16)
        np.save(join(tr_dir, f'{data_key}.npy'), data)

    for data_key in tqdm(test_keys, desc=' validation set'):
        val = nb.load(join(images_dir, f'{data_key}_0000.nii.gz')).get_fdata()
        lbl = nb.load(join(labels_dir, f'{data_key}.nii.gz')).get_fdata()
        if len(set(lbl.flatten())) != 110:
            print(data_key)
        val[lbl == 0] = 0
        val = resize(val, img_size, order=3).astype(np.float16)
        lbl = resize(lbl, img_size, order=3).astype(np.uint8)
        np.save(join(valsets_file, f'{data_key}.npy'), val)
        np.save(join(valsets_label, f'{data_key}_label.npy'), lbl)


def create_Morph(dataset):
    img_size = [160, 224, 160]
    data_root = f'/remote-home/hejj/Data/runtime/SoR_folder/SoR_raw/{dataset}'
    to_dir = f'/remote-home/hejj/Data/runtime/SoR_folder/morph_data/{dataset}'
    dirs = ['Train', 'Val', 'Test', 'Check']
    print(img_size)

    def pkload(f_name):
        with open(f_name, 'rb') as f:
            return pickle.load(f)

    def write_pickle(data, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_center(msk_data):
        pos = np.where(msk_data > 0)
        x = (pos[0].max() + pos[0].min()) // 2
        y = (pos[1].max() + pos[1].min()) // 2
        z = (pos[2].max() + pos[2].min()) // 2

        return x, y, z

    def clip_data(t1_data, roi_data, msk_data):
        cx, cy, cz = get_center(msk_data)
        hx, hy, hz = [i // 2 for i in img_size]
        xx, yy, zz = [[max(i - j, 0), i + j] for i, j in zip([cx, cy, cz], [hx, hy, hz])]

        t1_data = t1_data[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]
        roi_data = roi_data[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]

        pad_width = [(k // 2, k - k // 2) for k in [i - j for i, j in zip(img_size, roi_data.shape)]]
        t1_data = np.pad(t1_data, pad_width, mode='constant')
        roi_data = np.pad(roi_data, pad_width, mode='constant')

        return t1_data, roi_data

    def create_pickle(t1_file, roi_file):
        t1_data = nb.load(t1_file).get_fdata()
        roi_data = nb.load(roi_file).get_fdata()

        msk_data = roi_data.copy()
        msk_data[msk_data > 0] = 1
        t1_data *= msk_data
        t1_data = (t1_data - t1_data.min()) / (t1_data.max() - t1_data.min())

        t1_data, roi_data = clip_data(t1_data, roi_data, msk_data)
        return [t1_data.astype(np.float16), roi_data.astype(np.int8)]

    def check_pickles():
        import matplotlib.pyplot as plt

        x, y, z = [i // 2 for i in img_size]

        def subplot(c, r, i):
            plt.subplot(c, r, i)
            plt.xticks([])
            plt.yticks([])

        for pkl in tqdm(os.listdir(join(to_dir, dirs[0])), desc=f'Checking {dirs[0]}'):
            img, seg = pkload(join(to_dir, dirs[0], pkl))

            fig = plt.figure(1, figsize=(15, 10))

            subplot(2, 3, 1)
            plt.imshow(img[:, :, z], cmap='gray')
            subplot(2, 3, 2)
            plt.imshow(img[:, y, :], cmap='gray')
            subplot(2, 3, 3)
            plt.imshow(img[z, :, :], cmap='gray')
            subplot(2, 3, 4)
            plt.imshow(seg[:, :, z], cmap='jet')
            subplot(2, 3, 5)
            plt.imshow(seg[:, y, :], cmap='jet')
            subplot(2, 3, 6)
            plt.imshow(seg[z, :, :], cmap='jet')

            plt.savefig(join(to_dir, dirs[3], f'{dirs[0]}_{pkl[8:-4]}.jpg'))
            plt.close(fig)

            assert np.all([i == j for i, j in zip(img.shape, img_size)]), pkl

    def create():
        if not exists(to_dir):
            os.makedirs(to_dir)
        for t in dirs:
            if exists(join(to_dir, t)):
                shutil.rmtree(join(to_dir, t))
            os.mkdir(join(to_dir, t))

        t1_file = join(data_root, f'{dataset.split("_")[-1]}-template.nii.gz')
        roi_file = join(data_root, f'{dataset.split("_")[-1]}-atlas.nii.gz')
        pkl_data = create_pickle(t1_file, roi_file)
        write_pickle(pkl_data, join(to_dir, 'atlas.pkl'))
        np.save(join(to_dir, 'affine'), nb.load(t1_file).affine)

        imgs = sorted(set([i[:-12] for i in os.listdir(join(data_root, 'imagesTr'))]))
        train_imgs, test_imgs = train_test_split(imgs, test_size=0.3)
        for i, img_keys in enumerate([train_imgs, test_imgs]):
            for img_key in tqdm(img_keys, desc=f'Creating {dirs[i]}'):
                t1_file = join(data_root, 'imagesTr', f'{img_key}_0000.nii.gz')
                roi_file = join(data_root, 'labelsTr', f'{img_key}.nii.gz')
                pkl_data = create_pickle(t1_file, roi_file)
                write_pickle(pkl_data, join(to_dir, dirs[i], f'{img_key}.pkl'))

    create()
    check_pickles()


def create_TransMatch_Fr_Morph(dataset):
    data_dir = f'/remote-home/hejj/Data/runtime/SoR_folder/morph_data/{dataset}'
    to_dir = f'/remote-home/hejj/Data/runtime/SoR_folder/TransMatch/data/{dataset}'
    train_dir = join(to_dir, 'train')
    test_dir = join(to_dir, 'test')
    label_dir = join(to_dir, 'label')
    template, atlas = load_pickle(join(data_dir, 'atlas.pkl'))

    train_keys = sorted([i[:-4] for i in os.listdir(join(data_dir, 'Train'))])
    valid_keys = sorted([i[:-4] for i in os.listdir(join(data_dir, 'Val'))])

    if not exists(train_dir):
        os.makedirs(train_dir)
    if not exists(test_dir):
        os.makedirs(test_dir)
    if not exists(label_dir):
        os.makedirs(label_dir)

    nb.Nifti1Image(template.astype(np.float32), np.eye(4)).to_filename(join(to_dir, 'Template.nii.gz'))
    nb.Nifti1Image(atlas, np.eye(4).astype(np.uint8)).to_filename(join(to_dir, 'Atlas.nii.gz'))

    for dat_key in tqdm(train_keys, desc='Creating train'):
        data, segm = load_pickle(join(data_dir, 'Train', f'{dat_key}.pkl'))
        data = data.astype(np.float32)
        segm = segm.astype(np.uint8)
        nb.Nifti1Image(data, np.eye(4)).to_filename(join(train_dir, f'{dat_key[-4:]}_{dat_key[:-5]}.nii.gz'))
        nb.Nifti1Image(segm, np.eye(4)).to_filename(join(label_dir, f'{dat_key[-4:]}_{dat_key[:-5]}.nii.gz'))

    for dat_key in tqdm(valid_keys, desc='Creating test'):
        data, segm = load_pickle(join(data_dir, 'Val', f'{dat_key}.pkl'))
        data = data.astype(np.float32)
        segm = segm.astype(np.uint8)
        nb.Nifti1Image(data, np.eye(4)).to_filename(join(test_dir, f'{dat_key[-4:]}_{dat_key[:-5]}.nii.gz'))
        nb.Nifti1Image(segm, np.eye(4)).to_filename(join(label_dir, f'{dat_key[-4:]}_{dat_key[:-5]}.nii.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Dataset005_InHouse-Sim',
                        choices=['Dataset001_OASIS', 'Dataset002_InHouse', 'Dataset004_OASIS-Sim', 'Dataset005_InHouse-Sim'])
    args = parser.parse_args()

    # create_SymTrans(args.d)
    # create_Morph(args.d)
    # create_TransMatch_Fr_Morph(args.d)
