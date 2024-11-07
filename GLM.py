import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.image import resample_to_img, mean_img
import ants
import os
import os.path as op
import nibabel as nib

def reg_motion(motion_outliers, n_scans):
    new_regs = []
    new_reg_names = []
    for idx, out_frame in enumerate(motion_outliers):
        column_values = np.zeros(n_scans)
        column_values[out_frame] = 1
        new_regs.append(column_values)
        new_reg_names.append(f"motion_outlier_{idx}")
    return np.vstack(new_regs), new_reg_names
    
def condition_vector(position: int, n_regressors: int) -> np.array:
        vec = np.zeros((1, n_regressors))
        vec[0, position] = 1
        return vec
    
def get_conditions_of_interest(design: pd.DataFrame, keys_to_keep: list) -> dict:
    """
    Creates a dictionary of conditions of interest based on specified keys to keep.

    Parameters:
    - design (pd.DataFrame): Design matrix with columns representing regressors.
    - keys_to_keep (list): List of condition names to keep.

    Returns:
    - dict: A dictionary where keys are condition names and values are condition vectors.
    """
    
    n_regressors = design.shape[1]
    conditions = {col: condition_vector(idx, n_regressors) for idx, col in enumerate(design.columns)}
    conditions_of_interest = {key: conditions[key] for key in keys_to_keep if key in conditions}
    
    return conditions_of_interest

def atlas_fMRI_MNI(fmri_img, atlas_path, fMRI_MNI_path, preproc_path):
    aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
    atlas_img = image.load_img(aal_atlas.maps)
    atlas_img.to_filename(atlas_path)
    
    fmri_resampled = resample_to_img(fmri_img, atlas_img, interpolation='continuous')
    mean_fmri=mean_img(fmri_resampled)
    fmri_resampled_path = op.join(preproc_path,"sub-control01/func/sub-control01_task-music_run-all_bold_moco_resampled.nii")
    mean_fmri.to_filename(fmri_resampled_path)
    
    moving_image = ants.image_read(fmri_resampled_path)
    fixed_image = ants.image_read("data/derivatives/atlas_template.nii")
    
    transformation = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform = 'SyN')
    warpedImage = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=transformation['fwdtransforms'])  
    ants.image_write(warpedImage, fMRI_MNI_path)

    return atlas_img, nib.load(fMRI_MNI_path)

def make_combined_mask_from_aal(atlas_img, mask_values, combined_mask_name):
    
    atlas_data = atlas_img.get_fdata()  
    combined_mask_data = np.zeros_like(atlas_data, dtype=np.uint8)

    # Combine all regions into ta mask
    for mask_value in mask_values:
        combined_mask_data += (atlas_data == mask_value).astype(np.uint8)

    combined_mask_img = nib.Nifti1Image(combined_mask_data, atlas_img.affine, atlas_img.header)

    nib.save(combined_mask_img, combined_mask_name)
    print(f"Saved combined mask as {combined_mask_name}")