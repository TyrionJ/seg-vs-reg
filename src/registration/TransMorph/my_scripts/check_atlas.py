import nibabel as nib
from TransMorph.data.data_utils import pkload


if __name__ == '__main__':
    aff = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    # dr = r'F:\Data\Researches\MRI\Registration\IXI_data'
    # img, seg = pkload(fr'{dr}\atlas.pkl')
    # img = np.flip(img.swapaxes(1, 2), 2)
    # seg = np.flip(seg.swapaxes(1, 2), 2)
    # nib.Nifti1Image(img, aff).to_filename(fr'{dr}\atlas_img.nii.gz')
    # nib.Nifti1Image(seg, aff).to_filename(fr'{dr}\atlas_seg.nii.gz')
    # print(img.shape)

    dr = r'F:\Data\runtime\TransMorph\inhouse_smini'
    img, seg = pkload(fr'{dr}\atlas.pkl')
    nib.Nifti1Image(img.astype(float), aff).to_filename(fr'{dr}\atlas_img.nii.gz')
    nib.Nifti1Image(seg.astype(float), aff).to_filename(fr'{dr}\atlas_seg.nii.gz')

    print(img.shape)
