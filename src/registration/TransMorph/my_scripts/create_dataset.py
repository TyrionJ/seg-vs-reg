import os
import pickle
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import nibabel as nib
from os.path import join, isdir, exists

from tqdm import tqdm

from TransMorph.my_scripts.waiting_process import waiting_proc, mp
from TransMorph.data.data_utils import pkload

img_size = [192, 224, 192]
data_roots = [r'F:\Data\Researches\MRI\Raw\ASL&QSM',
              r'F:\Data\Researches\MRI\Raw\CSVD',
              r'F:\Data\Researches\MRI\Raw\NormalControl']

to_dir = r'F:\Data\runtime\TransMorph\inhouse_mini'
dirs = ['All', 'Train', 'Val', 'Test', 'Check']

if not exists(to_dir):
    os.makedirs(to_dir)
for t in dirs:
    if exists(join(to_dir, t)):
        shutil.rmtree(join(to_dir, t))
    os.mkdir(join(to_dir, t))


def create_atlas():
    print('Creating atlas')
    dr = r'E:\Data\Researches\Registration\MNI152NLin2009cAsym'
    t1_brain = join(dr, 'tpl-MNI152NLin2009cAsym_res-01_T1w_brain.nii.gz')
    t1_mask = join(dr, 'tpl-MNI152NLin2009cAsym_res-01_T1w_mask.nii.gz')
    pkl_data = create_pickle(t1_brain, t1_mask)
    write_pickle(pkl_data, join(to_dir, 'atlas.pkl'))


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
    xx, yy, zz = [[max(i-j, 0), i+j] for i, j in zip([cx, cy, cz], [hx, hy, hz])]

    t1_data = t1_data[xx[0]:xx[1], yy[0]:yy[1],  zz[0]:zz[1]]
    roi_data = roi_data[xx[0]:xx[1], yy[0]:yy[1],  zz[0]:zz[1]]
    pad_width = [(k//2, k-k//2) for k in [i-j for i, j in zip(img_size, roi_data.shape)]]

    t1_data = np.pad(t1_data, pad_width, mode='constant')
    roi_data = np.pad(roi_data, pad_width, mode='constant')

    return t1_data, roi_data


def create_pickle(t1_file, roi_file):
    t1_nii = nib.load(t1_file)
    roi_nii = nib.load(roi_file)
    t1_data = t1_nii.get_fdata()
    roi_data = roi_nii.get_fdata()

    roi_data[roi_data > 114] = 0
    roi_data[roi_data > 0] -= 5

    msk_data = np.zeros_like(roi_data)
    msk_data[roi_data > 0] = 1
    t1_data *= msk_data
    t1_data = (t1_data - t1_data.min()) / (t1_data.max() - t1_data.min())

    t1_data, roi_data = clip_data(t1_data, roi_data, msk_data)
    return [t1_data.astype(np.float16), roi_data.astype(np.int8)]


def check_pickles():
    import matplotlib.pyplot as plt

    x, y, z = [i//2 for i in img_size]

    def subplot(c, r, i):
        plt.subplot(c, r, i)
        plt.xticks([])
        plt.yticks([])

    for ts in dirs[1:4]:
        for pkl in tqdm(os.listdir(join(to_dir, ts)), desc=f'Checking {ts}'):
            img, seg = pkload(join(to_dir, ts, pkl))

            fig = plt.figure(1, figsize=[15, 10])

            subplot(2, 3, 1)
            plt.imshow(img[:, :, z], cmap='gray')
            subplot(2, 3, 2)
            plt.imshow(img[:, y, :], cmap='gray')
            subplot(2, 3, 3)
            plt.imshow(img[z, :, :], cmap='gray')
            subplot(2, 3, 4)
            plt.imshow(seg[:, :, z], cmap='gray')
            subplot(2, 3, 5)
            plt.imshow(seg[:, y, :], cmap='gray')
            subplot(2, 3, 6)
            plt.imshow(seg[z, :, :], cmap='gray')

            plt.savefig(join(to_dir, dirs[4], f'{ts}_{pkl[8:-4]}.jpg'))
            plt.close(fig)

            assert np.all([i == j for i, j in zip(img.shape, img_size)]), pkl


def create_one(t1_file, roi_file, data_index):
    pkl_data = create_pickle(t1_file, roi_file)
    write_pickle(pkl_data, join(to_dir, dirs[0], f'subject_{data_index:04d}.pkl'))


def create1(data_index=0):
    rs = []
    with mp.get_context('spawn').Pool(18) as p:
        for dr in data_roots:
            data_dir = dr.replace(r'\Raw', '')
            for typ in os.listdir(dr):
                if not isdir(join(dr, typ)):
                    continue
                for sub in os.listdir(join(dr, typ)):
                    sub_dir = join(dr, typ, sub)
                    if not isdir(sub_dir):
                        continue
                    roi_file = join(data_dir, typ, sub, 'T1', 'T1_mask.nii.gz')
                    if exists(roi_file):
                        data_index += 1
                        t1_file = join(data_dir, typ, sub, 'T1', 'T1.nii.gz')
                        rs.append(p.starmap_async(create_one, ((t1_file, roi_file, data_index),)))
        waiting_proc(rs, p, 'Creating 1')
    return data_index


def create2(data_index):
    dr = r'F:\Data\Researches\MRI\UAI_Pull'
    acc = [i[:-4] for i in os.listdir(r'F:\Data\Researches\MRI\Raw\UAI_Pull\accept')]

    rs = []
    with mp.get_context('spawn').Pool(18) as p:
        for sub in sorted(os.listdir(dr)):
            if sub in acc:
                data_index += 1
                t1_file = join(dr, sub, 'T1', 'T1.nii.gz')
                roi_file = join(dr, sub, 'T1', 'T1_mask.nii.gz')
                rs.append(p.starmap_async(create_one, ((t1_file, roi_file, data_index),)))
        waiting_proc(rs, p, 'Creating 2')
    return data_index


def split_data():
    all_dir = join(to_dir, dirs[0])

    pkls = os.listdir(all_dir)
    train_set, val_test_set = train_test_split(pkls, test_size=0.3, shuffle=True)
    test_set, val_set = train_test_split(val_test_set, test_size=0.333, shuffle=True)

    for tr in train_set:
        shutil.move(join(all_dir, tr), join(to_dir, dirs[1], tr))
    for tr in val_set:
        shutil.move(join(all_dir, tr), join(to_dir, dirs[2], tr))
    for tr in test_set:
        shutil.move(join(all_dir, tr), join(to_dir, dirs[3], tr))


def main():
    # create_atlas()
    total = create1()
    print(f'Dataset size = {total}')
    split_data()
    check_pickles()


if __name__ == '__main__':
    main()
