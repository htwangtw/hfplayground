from nilearn import datasets, image
from nilearn.maskers import NiftiMasker,NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
from datasets import Dataset
from importlib.resources import files
import pandas as pd
from sklearn.preprocessing import RobustScaler
import nibabel as nib

seg_name = 'A424+2mm'
ATLAS_FILE = 'data/development_fmri/downsample_A424+2mm.nii.gz'
denoise_strategy_name = 'simple+gsr'
denoise_strategy = {
    'denoise_strategy': 'simple',
    'motion': 'basic',
    'global_signal': 'basic',
}

def preprocess_development_dataset(sourcedata_dir, processed_dir, arrow_dir=None):
    """Download and preprocess the nilearn development dataset to arrow dataset.

    This is an extremely lazy version as for code implementation.

    post fmriprep processing details
    Denoising: Simple strategy with 6 motion parameters.
    Scaling: None.
    Mask: generic MNI152 whole brain mask.
    """
    development_dataset = datasets.fetch_development_fmri(data_dir=sourcedata_dir)
    # mni_gm_mask = nib.save(datasets.load_mni152_gm_mask(), Path('/tmp/mni_mask.nii.gz'))
    # mni_gm_mask = downsample_for_tutorial(Path('/tmp/mni_mask.nii.gz'), '/tmp/')
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
    masker = NiftiMasker(mask_img=mni_mask, smoothing_fwhm=8, standardize=True)
    complete_labels = (np.arange(424)+1).tolist()
    for func in tqdm(development_dataset['func'], desc="Denoising data..."):
        nii_name = func.split('/')[-1].replace('preproc', denoise_strategy_name)
        if not Path(f"{processed_dir}/{nii_name}").exists():
            conf, sm = load_confounds_strategy(img_files=func, **denoise_strategy)
            ts = masker.fit_transform(func, confounds=conf, sample_mask=sm)
            nii = masker.inverse_transform(ts)
            nii.to_filename(f"{processed_dir}/{nii_name}")
            del ts
            del nii

    atlas_masker = NiftiLabelsMasker(labels_img=files('hfplayground') / ATLAS_FILE, mask_img=mni_mask).fit()  # no scaling here
    Path(f"{processed_dir}_gigaconnectome_a424").mkdir(exist_ok=True, parents=True)
    for func in tqdm(development_dataset['func'], desc="Extract time series..."):
        nii_name = func.split('/')[-1].replace('preproc', denoise_strategy_name)
        matches = nii_name.split('_space-')[0]
        ts_filename = f"{matches}_seg-{seg_name}_desc-{denoise_strategy_name}_timeseries.tsv"
        if Path(f"{processed_dir}_gigaconnectome_a424/{ts_filename}").exists():
            continue
        seg_ts = atlas_masker.fit_transform(f"{processed_dir}/{nii_name}")
        seg_ts = pd.DataFrame(seg_ts, columns=[int(l) for l in atlas_masker.labels_])
        # they filled missing values with 0, so we do the same..... this is bad
        seg_ts = seg_ts.reindex(columns=complete_labels, fill_value=0)
        seg_ts.to_csv(Path(f"{processed_dir}_gigaconnectome_a424") / ts_filename, sep='\t', index=False)
        del seg_ts

    if not arrow_dir:
        print("No arrow_dir provided, skipping arrow conversion.")
        return None

    convert_data = ['Age', 'Gender', 'AgeGroup', 'Child_Adult']
    phenotype = pd.read_csv("data/external/development_fmri/development_fmri/participants.tsv", index_col='participant_id', sep='\t')
    timeseries_files = list(Path(f"{processed_dir}_gigaconnectome_a424").glob('*seg-*_timeseries.tsv'))
    dataset_dict = {
        "robustscaler_timeseries": [],
        "raw_timeseries": [],
        "filename":[],
        "participant_id":[]
    }
    for col in convert_data:
        dataset_dict[col] = []

    for file_path in tqdm(timeseries_files, desc="convert to arrow"):
        seg_ts = pd.read_csv(file_path, sep='\t').values.astype(np.float32)
        # apply robust scaling to the time series
        scaler = RobustScaler()
        seg_ts_robustscaler = scaler.fit_transform(seg_ts)
        participant_id = file_path.stem.split('_')[0]
        dataset_dict["raw_timeseries"].append(seg_ts)
        dataset_dict["robustscaler_timeseries"].append(seg_ts_robustscaler)
        dataset_dict["filename"].append(str(file_path.name))
        dataset_dict["participant_id"].append(participant_id)
        for col in convert_data:
            dataset_dict[col].append(phenotype.loc[participant_id, col])
    arrow_train_dataset = Dataset.from_dict(dataset_dict)
    arrow_train_dataset.save_to_disk(
        dataset_path=Path(arrow_dir) / "fmri_development.arrow"
    )

    # --- Save Brain Region Coordinates Into Another Arrow Dataset ---#
    coords_dat = np.loadtxt(files('hfplayground') / "data/brainlm/atlases/A424_Coordinates.dat").astype(np.float32)
    coords_pd = pd.DataFrame(coords_dat, columns=["Index", "X", "Y", "Z"])
    coords_dataset = Dataset.from_pandas(coords_pd)
    coords_dataset.save_to_disk(
        dataset_path=Path(arrow_dir) / "brainregion_coordinates.arrow")
    print("Done.")


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
