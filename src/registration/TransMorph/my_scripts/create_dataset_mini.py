import os
import pickle
import numpy as np
from tqdm import tqdm
import nibabel as nib
from os.path import join, exists

from TransMorph.data.data_utils import pkload

img_size = [160, 224, 160]
data_root = r'F:\Data\runtime\00RawFolder\QSMT1-Mod'

to_dir = r'F:\Data\runtime\TransMorph\inhouse_mini'
dirs = ['Train', 'Val', 'Test', 'Check']

if not exists(to_dir):
    os.makedirs(to_dir)
    for t in dirs:
        os.mkdir(join(to_dir, t))


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
    t1_data = nib.load(t1_file).get_fdata()
    roi_data = nib.load(roi_file).get_fdata()

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

        fig = plt.figure(1, figsize=[15, 10])

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
    imgs = sorted(set([i[:-12] for i in os.listdir(join(data_root, 'imagesTr'))]))
    i = 1
    for img_key in tqdm(imgs, desc='Creating'):
        t1_file = join(data_root, 'imagesTr', f'{img_key}_0001.nii.gz')
        roi_file = join(data_root, 'labelsTr', f'{img_key}.nii.gz')
        pkl_data = create_pickle(t1_file, roi_file)
        write_pickle(pkl_data, join(to_dir, dirs[0], f'subject_{i:04d}.pkl'))
        i += 1


def main():
    create()
    # check_pickles()


if __name__ == '__main__':
    main()
