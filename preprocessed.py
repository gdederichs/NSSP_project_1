import os.path as op
import os
import ants
from utils import mkdir_no_exist
from fsl.wrappers import fast, bet, mcflirt
from fsl.wrappers.misc import fslroi
from fsl.wrappers import flirt
import nibabel as nib
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import progressbar
import gc

def get_skull_stripped_anatomical(bids_root, preproc_path, subject, robust=False):
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
    print('Mask applied')
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

    # Compute the (non linear) transformation to align the moving image to the fixed image and apply it
    transformation = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform = type_of_transform)
    warpedImage = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=transformation['fwdtransforms'])
    
    resultAnts = op.join(preproc_path, subject, subfolder, '{}_T1w_mni_{}.nii.gz'.format(subject, type_of_transform))
    ants.image_write(warpedImage, resultAnts)
    return resultAnts

"""_________________Functional Imaging preprocessing_________________"""

def minmax_std(data, eps = 1e-20, verbose = False):
    min = np.min(data)
    max = np.max(data)
    if verbose:
        print(f'Min : {min}      Max : {max}')
    normalized_data = (data - min) / (max - min + eps)
    return normalized_data
    
def zscore_std(data, eps = 1e-20, verbose = False):
    mean = np.mean(data)
    std = np.std(data)
    std_adj = np.where(std > eps, std, eps)
    if verbose:
        print(f'Mean : {pd.DataFrame(mean)}      std : {pd.DataFrame(std_adj)}')
    return (data - mean) / std_adj

def concatenate_mri_runs(bids_root, preproc_root, subject, task, fct='minmax', verbose = False, subfolder = 'func'):
    standardized_runs = []
    
    # Load and standardize each run
    for i in range(3):
        run = os.path.join(bids_root, subject, 'func', '{}_task-{}_run-{}_bold.nii.gz'.format(subject, task, i+1))
        img = nib.load(run)
        data = img.get_fdata()
        if verbose:
            print(f'Shape of the series of volumes of run {i}:', data.shape)
        if fct == 'minmax':
            standardized_data = minmax_std(data)
        elif fct == 'zscore':
            standardized_data = zscore_std(data)
        else:
            raise Exception("Error: This standardization technique is not implemented, please run again with a valid technique!")
        standardized_runs.append(standardized_data) 
    
    # Concatenate data along the time axis and save corresponding image
    concatenated_data = np.concatenate(standardized_runs, axis=-1)
    concatenated_img = nib.Nifti1Image(concatenated_data, img.affine)
    
    output_path = op.join(preproc_root, subject, subfolder)
    mkdir_no_exist(output_path)
    output_path = op.join(output_path, '{}_task-{}_run-{}_bold.nii.gz'.format(subject, task, 'all'))

    nib.save(concatenated_img, output_path)
    print(f"Concatenation complete. \nOutput saved to {output_path}")
    return output_path


def apply_mcflirt(preproc_root, subject, task, run, subfolder='func', ref = 'mean'):
    path_original_data = os.path.join(preproc_root, subject, subfolder, '{}_task-{}_run-{}_bold'.format(subject, task, run))
    path_moco_data = op.join(preproc_root, subject, subfolder, '{}_task-{}_run-{}_bold_moco'.format(subject, task, run))
    
    #Determine the middle volume to use as a reference
    if ref == 'mean':
        meanvol_bool = True
        reference_moco = 'mean but no saved file' 
    elif ref == 'middle':
        meanvol_bool = False
        reference_moco = extract_middle_vol(path_original_data)
        
    mcflirt(infile=path_original_data, o=path_moco_data, plots=True, report=True, dof=6, mats=True, reffile = reference_moco if not meanvol_bool else None, meanvol = meanvol_bool) 
    return path_moco_data, reference_moco


def extract_middle_vol(path_4d_series):
    output_path = path_4d_series.replace('ds000171', 'derivatives/preprocessed_data') + 'middle-vol'
    img = nib.load(path_4d_series+'.nii.gz')
    _, _, _, nbr_volumes = img.shape
    fslroi(path_4d_series, output_path, str(nbr_volumes //2), str(1))
    return output_path

def apply_slice_timer(preproc_root, subject, task, run, input_file, slice_timing, tr, dim, subfolder='func'):
    slice_order = np.argsort(slice_timing) + 1

    # Write to a file the corresponding sorted timings :)
    timing_path = op.join(preproc_root, subject, subfolder, '{}_task-{}_run-{}_slice-timings.txt'.format(subject, task, run)) # same for all runs 
    file = open(timing_path, mode='w')
    for t in slice_order:
        file.write(str(t) + '\n')
    file.close()
    output_path = op.join(preproc_root, subject, 'func', 'sub-control01_task-music_run-all_bold_slice-corr')
    subprocess.run(['slicetimer', '-i', input_file, '-o', output_path, '-r', str(tr), '-d', str(dim), '--ocustom={}'.format(timing_path)])
    return output_path
    

"""___________________________Coregistration__________________________________"""
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


def combine_splits(preproc_path, splits_epi_path, subject):
    import progressbar
    import gc
    split_vols_epi = sorted(glob.glob(op.join(splits_epi_path, '*_bold_epi*')))
    
    first_vol = nib.load(split_vols_epi[0])
    v_shape = first_vol.get_fdata().shape
    mkdir_no_exist(op.join('/data', 'splits_epi'))
    filename = op.join('/data', 'splits_epi', 'sub-control01_task-music_run-all_bold_epi_concat.dat') # data file on Neurodesk needs to be onon desktop
    large_array = np.memmap(filename, dtype=np.float64, mode='w+', shape=(v_shape[0],v_shape[1],v_shape[2], len(split_vols_epi)))
    nbr_batches= 10
    batch_size = len(split_vols_epi)//nbr_batches
    
    A = np.zeros((v_shape[0],v_shape[1],v_shape[2], batch_size))
    
    with progressbar.ProgressBar(max_value=len(split_vols_epi)) as bar:
        for batch_i in range(nbr_batches):
            print('Starting for batch {}/{}'.format(batch_i+1, nbr_batches))
            start_batch = batch_size * batch_i
            end_batch = min(batch_size * (batch_i+1),len(split_vols_epi))
            max_len = end_batch - start_batch + 1
            for i in range(start_batch, end_batch):
                vol = nib.load(split_vols_epi[i])
                A[:,:,:,i-start_batch] = vol.get_fdata()
                bar.update(i)
            large_array[:,:,:, start_batch:end_batch] = A[:,:,:,:max_len]
            gc.collect()
    
    large_array = np.memmap(filename, dtype=np.float64, mode='r', shape=(v_shape[0],v_shape[1],v_shape[2], len(split_vols_epi)))
    
    # Step 2: Modify the header to indicate that we have 4D data, and specify its TR.
    header = first_vol.header.copy()  # Copy the header of the first volume (to get right resolution, affine, Q-form etc)
    header['dim'][0] = 4  # Specifies that this is a 4D dataset
    header['dim'][1:5] = large_array.shape  # Update dimensions (x, y, z, t)
    header['pixdim'][4] = 3 # Set the TR in the 4th dimension.
    print("Done with header")
    
    # Step 3: Create the Nifti1 image and save it to disk
    epi_all = op.join('/data', 'splits_epi', 'sub-control01_task-music_run-all_bold_epi_concat.nii.gz')
    img = nib.Nifti1Image(large_array, first_vol.affine, first_vol.header)
    print("Done creating the image")
    img.to_filename(epi_all)
    os.system('rm -rf {}'.format(op.join('/data', 'splits_epi', '*split*'))) # remove the splits from disk
    print("Done writing it to disk")
    return epi_all


"""___________________________PLOTTING__________________________________"""
def viz_fsleyes(fsleyesDisplay, to_load:list, viz:bool=True):
    if viz:
        try:
            fsleyesDisplay.resetOverlays()
            for i, path in enumerate(to_load):
                fsleyesDisplay.load(path)
                colors = ['Red','Green','Blue']
                if 'pve' in path: 
                    fsleyesDisplay.displayCtx.getOpts(fsleyesDisplay.overlayList[i]).cmap = colors[i-1]
        except:
            print('Error: fsleyesDisplay has most likely not been initialized...')
        

def plot_bold_data(path, timepoints:list=[30,150,250]):
    data = nib.load(path).get_fdata()
    nb = len(timepoints)
    fig, axs = plt.subplots(1, nb, sharey = True, figsize = (20,6))
    
    for i, tp in enumerate(timepoints):
        volume = data[..., tp]
        # Choose a slice e.g. the middle one
        z_slice = volume.shape[2] // 2 
        slice_data = volume[:, :, z_slice]
        
        # Plot the slice
        axs[i%nb].imshow(slice_data.T, cmap="gray", origin="lower")
        
        axs[i%nb].set_title(f"Time Point {tp}, z-slice {z_slice} (run {int(tp/100)+1})")
        axs[i%nb].set_xlabel("X-axis")
        axs[i%nb].set_ylabel("Y-axis")
    fig.suptitle('BOLD Signal')
    plt.show()


def plot_mean_voxel_intensity(all, len_run, nb_runs = 3):
    data = nib.load(all).get_fdata()
    for i in range(nb_runs):
        plt.plot(data[:,:,:,i*105:(i+1)*105].mean(axis=(0,1,2)), label =f'run {i+1}')
    plt.xlabel('Time (volume)')
    plt.ylabel('Mean voxel intensity')
    plt.title('Mean voxel intensity per run after standardization')
    plt.legend(loc = 4)
    plt.show()

