import os.path as op
import os
import ants
from utils import mkdir_no_exist
from fsl.wrappers import fast, bet, mcflirt
from fsl.wrappers.misc import fslroi
from fsl.wrappers import flirt

def get_skull_stripped_anatomical(bids_root, preproc_path, subject, robust=False):
    """
    Function to perform skull-stripping (removing the skull around the brain).
    This is a simple wrapper around the brain extraction tool (BET) in FSL's suite
    It assumes data to be in the BIDS format (which we will cover in the following weeks).
    The method also saves the brain mask which was used to extract the brain.

    The brain extraction is conducted only on the T1w of the participant.

    Parameters
    ----------
    bids_root: string
        The root of the BIDS directory
    preproc_root: string
        The root of the preprocessed data, where the result of the brain extraction will be saved.
    subject_id: string
        Subject ID, the subject on which brain extraction should be conducted.
    robust: bool
        Whether to conduct robust center estimation with BET or not. Default is False.
    """
    # For now all you need to do is that we remove the bones and flesh from the MRI to get the brain!
    anatomical_path = op.join(bids_root, subject, 'anat' , '{}_T1w.nii.gz'.format(subject))
    betted_brain_path = op.join(preproc_path, subject, 'anat')
    mkdir_no_exist(betted_brain_path)
    betted_brain_path = op.join(preproc_path, subject, 'anat', '{}_T1w'.format(subject))
    os.system('bet {} {} -m {}'.format(anatomical_path, betted_brain_path, '-R' if robust else ''))
    print("Done with BET.")
    resulting_mask_path = op.join(preproc_path, subject, 'anat', '{}_T1w_mask.nii.gz'.format(subject))
    return resulting_mask_path

def apply_fsl_mask(dataset_path, mask_path, preproc_path, subject, subfolder='anat'):
    anatomical_path = op.join(dataset_path, subject, subfolder , '{}_T1w.nii.gz'.format(subject))
    betted_brain_path = op.join(preproc_path, subject, subfolder, '{}_T1w.nii.gz'.format(subject))
    os.system('fslmaths {} -mas {} {}'.format(anatomical_path, mask_path, betted_brain_path))
    return betted_brain_path

def apply_fast(preproc_path, subject, n_classes=3, subfolder='anat'):
    bet_path = op.join(preproc_path, subject,subfolder, '{}_T1w'.format(subject))
    segmentation_path = op.join(preproc_path, subject, subfolder, '{}_T1w_fast'.format(subject))
    fast(imgs=[bet_path], out=segmentation_path, n_classes=n_classes)
    return bet_path

def apply_ants(preproc_path, subject, reference, type_of_transform = 'SyN', subfolder='anat' ):
    target_path = op.join(preproc_path, '{}'.format(subject), 'anat', '{}_T1w.nii.gz'.format(subject))
    moving_image = ants.image_read(target_path)
    fixed_image = ants.image_read(reference + '.nii.gz')

    # Compute the transformation (non linear) to put align the moving image to the fixed image
    transformation = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform = type_of_transform)

    # After the transformation has been computed, apply it
    warpedImage = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=transformation['fwdtransforms'])

    # Save the image to disk
    resultAnts = op.join(preproc_path, subject, subfolder, '{}_T1w_mni_{}.nii.gz'.format(subject, type_of_transform))
    ants.image_write(warpedImage, resultAnts)
    return resultAnts

def apply_mcflirt(bids_root, preproc_root, subject, task, run, subfolder='func'):
    path_original_data = os.path.join(bids_root, subject, subfolder, '{}_task-{}_run-{}_bold'.format(subject, task, run))
    path_moco_data = os.path.join(preproc_root, subject, subfolder)
    mkdir_no_exist(path_moco_data)
    path_moco_data = op.join(path_moco_data, '{}_task-{}_run-{}_bold_moco'.format(subject, task, run))
    mcflirt(infile=path_original_data,o=path_moco_data, plots=True, report=True, dof=6, mats=True)
    return path_moco_data
    
