# class DownloadProgressBar(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)

import numpy as np
import nibabel as nib
from nilearn import image, datasets

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def direct_file_download_open_neuro(file_list, file_types, dataset_id, dataset_version, save_dirs):
    # https://openneuro.org/crn/datasets/ds004226/snapshots/1.0.0/files/sub-001:sub-001_scans.tsv
    for i, n in enumerate(file_list):
        subject = n.split('_')[0]
        download_link = 'https://openneuro.org/crn/datasets/{}/snapshots/{}/files/{}:{}:{}'.format(dataset_id, dataset_version, subject, file_types[i],n)
        print('Attempting download from ', download_link)
        download_url(download_link, op.join(save_dirs[i], n))
        print('Ok')


def mkdir_no_exist(path):
    """
    Function to create a directory if it does not exist already.

    Parameters
    ----------
    path: string
        Path to create
    """
    import os
    import os.path as op
    if not op.isdir(path):
        os.makedirs(path)

def make_mask_from_aal(mask_value, mask_name):
    # Load the AAL atlas
    aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
    atlas_img = image.load_img(aal_atlas.maps)
    atlas_data = atlas_img.get_fdata()  # Extract atlas data as numpy array

    # Create a binary mask for the specified region
    mask_data = atlas_data == mask_value

    # Create a new Nifti image with the mask
    mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), atlas_img.affine, atlas_img.header)

    # Save the mask
    if ".nii" not in mask_name:
        mask_name += ".nii"
    nib.save(mask_img, mask_name)