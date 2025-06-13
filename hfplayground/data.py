from nilearn import datasets, image
from nilearn.maskers import NiftiMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
from pathlib import Path
from tqdm import tqdm
import os
from nilearn._utils import check_niimg
import numpy as np

def preprocess_development_dataset(sourcedata_dir, processed_dir):
    """Download and preprocess the nilearn development dataset.

    post fmriprep processing details
    Denoising: Simple strategy with 6 motion parameters.
    Scaling: None.
    Mask: generic MNI152 whole brain mask.
    """
    development_dataset = datasets.fetch_development_fmri(data_dir=sourcedata_dir)

    mni_mask = datasets.fetch_icbm152_2009()['mask']
    mni_mask = downsample_for_tutorial(mni_mask, '/tmp/')
    # quick preprocessing
    # TODO: standardization is done at the time window creation stage
    # However, when refactoring, I will change the code to handle
    # standardisation at time series creation stage
    # (so we can take outputs from giga connectome)
    # If standardize of time series per voxel was not performed,
    # The BrainLM workflow will produce very different outputs based on the
    # scaling options
    # Not smoothing here as the data is heavily downsampled
    masker = NiftiMasker(mask_img=mni_mask, smoothing_fwhm=None, standardization=True)
    for func in tqdm(development_dataset['func'], desc="Denoising data..."):
        conf, sm = load_confounds_strategy(img_files=func, denoise_strategy='simple', motion='basic', global_signal='basic')
        ts = masker.fit_transform(func, confounds=conf, sample_mask=sm)
        nii = masker.inverse_transform(ts)
        del ts
        nii.to_filename(processed_dir)
        nii_name = func.split('/')[-1].replace('preproc', 'preprocSimpleFwhm8mm')
        nii.to_filename(f"{processed_dir}/{nii_name}")
        del nii
    return None

def downsample_for_tutorial(nii_file, output_dir):
    """Downsample atlas to match developmental dataset.
    Modified from https://osf.io/wjtyq
    """
    fname = os.path.basename(nii_file)
    aff_orig = np.array([ -96., -132.,  -78.,    1.])  # from developmental dataset
    target_affine = np.column_stack([np.eye(4, 3) * 4, aff_orig])
    downsample_data = image.resample_img(
        nii_file,
        target_affine=target_affine,
        target_shape=(50, 59, 50),
        interpolation='nearest',
        force_resample=True,
        copy_header=True
    )
    downsample_data.to_filename(Path(output_dir) / f'downsample_{fname}')
    return Path(output_dir) / f'downsample_{fname}'
