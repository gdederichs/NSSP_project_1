import os.path as op
import os
import ants
from utils import mkdir_no_exist
from fsl.wrappers import fast, bet, mcflirt
from fsl.wrappers.misc import fslroi
from fsl.wrappers import flirt
import nibabel as nib
import subprocess

def get_skull_stripped_anatomical(bids_root, preproc_path, subject, robust=False):
    """
    Function to perform skull-stripping (removing the skull around the brain).
    This is a simple wrapper around the brain extraction tool (BET) in FSL's suite
    It assumes data to be in the BIDS format
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
    bet_path = op.join(preproc_path, subject, subfolder, '{}_T1w'.format(subject))
    segmentation_path = op.join(preproc_path, subject, subfolder, '{}_T1w_fast'.format(subject))
    fast(imgs=[bet_path], out=segmentation_path, n_classes=n_classes)
    print('Done with fast')
    return segmentation_path

def apply_ants(preproc_path, subject, reference, type_of_transform = 'SyN', subfolder='anat'):
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
    
    #Determine the middle volume to use as a reference for the motion correction algorithm
    reference_moco = extract_middle_vol(path_original_data)
    
    mcflirt(infile=path_original_data, o=path_moco_data, plots=True, report=True, dof=6, mats=True, reffile=reference_moco) 
    return path_moco_data, reference_moco
    



def extract_middle_vol(path_4d_series):
    output_path = path_4d_series.replace('ds000171', 'derivatives/preprocessed_data') + 'middle-vol'
    img = nib.load(path_4d_series+'.nii.gz')
    _, _, _, nbr_volumes = img.shape
    fslroi(path_4d_series, output_path, str(nbr_volumes //2), str(1))
    return output_path
    
def apply_epi_reg(bids_root, preproc_root, moco_path, subject, task, run, subfolder='func'):
    epi_reg_path = op.join(preproc_root, subject, subfolder, '{}_task-{}_run-{}_bold_anat-space_epi'.format(subject, task, run))
    
    anatomical_path = op.join(bids_root, subject, 'anat', '{}_T1w.nii.gz').format(subject)
    betted_brain_path = op.join(preproc_root, subject, 'anat', '{}_T1w.nii.gz'.format(subject))
    white_matter = op.join(preproc_root, 'sub-control01', 'anat', 'sub-control01_T1w_fast_pve_2')
    reference_epi = extract_middle_vol(moco_path) # apply on moco data
    
    subprocess.run(['epi_reg','--epi={}'.format(reference_epi), 
                    '--t1={}'.format(anatomical_path), # original t1w mri scan
                    '--t1brain={}'.format(betted_brain_path), # brain without skull
                    '--out={}'.format(epi_reg_path), # output file
                    '--wmseg={}'.format(white_matter), #white matter segmentation
                    ])
    print("Done with EPI to anatomical registration")
    return epi_reg_path, reference_epi
















#delete these later
def combine_all_transforms(reference_volume, warp_save_name, is_linear, epi_2_moco=None, epi_2_anat_warp=None, anat_2_standard_warp=None):
    """
    Combines transformation BEFORE motion correction all the way to standard space transformation
    The various transformation steps are optional. As such, the final warp to compute is based on 
    which transforms are provided.

    Parameters
    ----------
    reference_volume: str
        Reference volume. The end volume after all transformations have been applied, relevant for final resolution and field of view (resampling).
    warp_save_name: str
        Under which name to save the total warp
    is_linear: bool
        Whether the transformation is linear or non linear.
    epi_2_moco: str
        Transformation of the EPI volume to motion-correct it (located in the .mat/ folder of the EPI)
    epi_2_anat_warp: str
        Transformation of the EPI volume to the anatomical space, typically obtained by epi_reg. Assumed to include fieldmap correction and thus be non-linear.
    anat_2_standard_warp: str
        Transformation of the anatomical volume to standard space, such as the MNI152 space. Might be linear or non linear, which affects is_linear value accordingly.
    """
    from fsl.wrappers import convertwarp
    args_base = {'premat': epi_2_moco, 'warp1': epi_2_anat_warp}
    if is_linear:
        args_base['postmat'] = anat_2_standard_warp
    else:
        args_base['warp2'] = anat_2_standard_warp
    args_filtered = {k: v for k, v in args_base.items() if v is not None}

    convertwarp(warp_save_name, reference_volume, **args_filtered)
    print("Done with warp conversion")

def apply_transform(reference_volume, target_volume, output_name, transform):
    """
    Applies a warp field to a target volume and resamples to the space of the reference volume.

    Parameters
    ----------
    reference_volume: str
        The reference volume for the final interpolation, resampling and POV setting
    target_volume: str
        The target volume to which the warp should be applied
    output_name: str
        The filename under which to save the new transformed image
    transform: str
        The filename of the warp (assumed to be a .nii.gz file)
    """
    from fsl.wrappers import applywarp
    applywarp(target_volume,reference_volume, output_name, w=transform, rel=False)


