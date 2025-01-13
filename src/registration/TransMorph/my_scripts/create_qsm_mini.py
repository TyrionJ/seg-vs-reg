import os
import shutil
from os.path import exists, join
import nibabel as nb
import numpy as np

from batchgenerators.utilities.file_and_folder_operations import load_json
from tqdm import tqdm

from TransMorph.my_scripts.create_dataset_mini import clip_data, write_pickle

ref_imgs = ['001', '005', '011', '018', '023', '031', '039', '044', '050', '060']
data_info = load_json(r'F:\Data\runtime\00RawFolder\QSMT1-Mod\data_info.json')
data_root = r'F:\Data\runtime\00RawFolder\QSMT1-Mod'


if __name__ == '__main__':
    to_dir = r'F:\Data\runtime\TransMorph\inhouse_mini\QSM'
    if not exists(to_dir):
        os.makedirs(to_dir)

    for im_key in tqdm(ref_imgs):
        sub_dir = data_info[f'QT_{im_key}'][0]
        qsm_data = nb.load(join(sub_dir, 'QSM', 'iLSQR_T1.nii.gz')).get_fdata()
        roi_file = join(data_root, 'labelsTr', f'QT_{im_key}.nii.gz')
        roi_data = nb.load(roi_file).get_fdata()

        msk_data = roi_data.copy()
        msk_data[msk_data > 0] = 1
        qsm_data, _ = clip_data(qsm_data, roi_data, msk_data)
        write_pickle([qsm_data.astype(np.float16)], join(to_dir, f'subject_{int(im_key):04d}.pkl'))
