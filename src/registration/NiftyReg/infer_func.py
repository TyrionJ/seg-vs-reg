import os
from tqdm import tqdm
from os.path import join, exists


def nifty_reg(fixed_file, moving_file, out_file, to_dir, fixed_arr=None, moving_arr=None, out_arr=None):
    if out_arr is None:
        out_arr = []
    if moving_arr is None:
        moving_arr = []
    if fixed_arr is None:
        fixed_arr = []
    os.system(f'reg_f3d -be 0.0002 --ssd -ref {fixed_file} -flo {moving_file} '
              f'-res {to_dir}/{out_file} -voff '
              f'-cpp {to_dir}/ref_template_flo_new_image_nrr_cpp.nii')
    for i in range(len(fixed_arr)):
        os.system(f'reg_resample -ref {fixed_arr[i]} -flo {moving_arr[i]} '
                  f'-res {to_dir}/{out_arr[i]} -cpp {to_dir}/ref_template_flo_new_image_nrr_cpp.nii -inter 0')
    os.remove(f'{to_dir}/ref_template_flo_new_image_nrr_cpp.nii')


def main():
    base_dir = '/home/jerry/Desktop/test_niftyreg/Dataset007_MCI-CSVD'
    tpl_file = '/home/jerry/Desktop/test_niftyreg/Dataset002_InHouse/InHouse-template.nii.gz'
    ats_file = '/home/jerry/Desktop/test_niftyreg/Dataset002_InHouse/InHouse-atlas.nii.gz'
    dst_dir = f'/home/jerry/Desktop/test_niftyreg/results/Dataset007_MCI-CSVD'

    T1_dir = join(base_dir, 'T1')
    QSM_dir = join(base_dir, 'QSM')
    ROI_dir = join(base_dir, 'ROIs')

    atlas2sub = join(dst_dir, 'atlas2sub')
    sub2atlas = join(dst_dir, 'sub2atlas')
    os.makedirs(atlas2sub, exist_ok=True)
    os.makedirs(sub2atlas, exist_ok=True)

    img_keys = sorted([i[:-7] for i in os.listdir(ROI_dir)])
    for img_key in tqdm(img_keys, desc='Registering'):

        t1_file = join(T1_dir, f'{img_key}_0000.nii.gz')
        QSM_file = join(QSM_dir, f'{img_key}.nii.gz')
        ROI_file = join(ROI_dir, f'{img_key}.nii.gz')

        # subject to atlas
        if not exists(join(sub2atlas, f'{img_key}_seg.nii.gz')):
            nifty_reg(fixed_file=tpl_file, moving_file=t1_file, to_dir=sub2atlas, out_file=f'{img_key}_img.nii.gz',
                      fixed_arr=[tpl_file, ats_file], moving_arr=[QSM_file, ROI_file],
                      out_arr=[f'{img_key}_qsm.nii.gz', f'{img_key}_seg.nii.gz'])

        # atlas to subject
        if not exists(join(atlas2sub, f'{img_key}_seg.nii.gz')):
            nifty_reg(fixed_file=t1_file, moving_file=tpl_file, to_dir=atlas2sub, out_file=f'{img_key}_img.nii.gz',
                      fixed_arr=[ROI_file], moving_arr=[ats_file], out_arr=[f'{img_key}_seg.nii.gz'])


if __name__ == '__main__':
    main()
