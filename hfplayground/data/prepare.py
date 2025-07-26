from nilearn import datasets, image
from nilearn.maskers import MultiNiftiMasker, MultiNiftiLabelsMasker
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
ATLAS_FILE = f'resource/development_fmri/downsample_{seg_name}.nii.gz'
denoise_strategy_name = 'simple+gsr'
denoise_strategy = {
    'denoise_strategy': 'simple',
    'motion': 'basic',
    'global_signal': 'basic',
}
ts_min_length = 160

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

    # Check if the processed directory exists
    raw_to_preproc = []
    niis_preproc_path = []
    niis_to_extract = []
    ts_file_paths = []
    for path_raw in development_dataset['func']:
        nii_name = path_raw.split('/')[-1].replace('preproc', denoise_strategy_name)
        nii_path = Path(f"{processed_dir}/{nii_name}")
        matches = nii_name.split('_space-')[0]
        ts_filename = f"{matches}_seg-{seg_name}_desc-{denoise_strategy_name}_timeseries.tsv"
        ts_path = Path(f"{processed_dir}_gigaconnectome_a424/{ts_filename}")
        if not nii_path.exists():
            raw_to_preproc.append(path_raw)
            niis_preproc_path.append(nii_path)
        if not ts_path.exists():
            niis_to_extract.append(nii_path)
            ts_file_paths.append(ts_path)

    if len(raw_to_preproc)>0:  # giga connectome preprocessing and brainlm workflow shares denoising
        # I did not do signal normalisation here. It will throw brainlm results off.
        masker = MultiNiftiMasker(mask_img=mni_mask, smoothing_fwhm=8, verbose=2)
        conf, sm = load_confounds_strategy(img_files=raw_to_preproc, **denoise_strategy)
        fmri_data = masker.fit_transform(raw_to_preproc, confounds=conf, sample_mask=sm)

        for reproc_path, fd in tqdm(zip(niis_preproc_path, fmri_data), desc="Save denoising data..."):
            nii = masker.inverse_transform(fd)
            nii.to_filename(reproc_path)

    if len(niis_to_extract) > 0:
        Path(f"{processed_dir}_gigaconnectome_a424").mkdir(exist_ok=True, parents=True)
        complete_labels = (np.arange(424)+1).tolist()

        atlas_masker = MultiNiftiLabelsMasker(
            labels_img=files('hfplayground') / ATLAS_FILE,
            labels=complete_labels,
            mask_img=mni_mask, verbose=3
        ).fit()  # no scaling here
        parcellated_timeseries = atlas_masker.transform(niis_to_extract)

        for ts_path, seg_ts in tqdm(zip(ts_file_paths, parcellated_timeseries), desc="save time series..."):
            seg_ts = pd.DataFrame(seg_ts, columns=[int(l) for l in atlas_masker.labels_])
            seg_ts = seg_ts.reindex(columns=complete_labels)
            seg_ts.fillna("n/a", inplace=True)
            seg_ts.to_csv(ts_path, sep='\t', index=False)

    if not arrow_dir:
        print("No arrow_dir provided, skipping arrow conversion.")
        return None

    convert_data = ['Age', 'Gender', 'AgeGroup', 'Child_Adult']
    phenotype = pd.read_csv("data/external/development_fmri/development_fmri/participants.tsv", index_col='participant_id', sep='\t')
    timeseries_files = list(Path(f"{processed_dir}_gigaconnectome_a424").glob('*seg-*_timeseries.tsv'))
    timeseries_files.sort()
    dataset_dict = {
        "robustscaler_timeseries": [],
        "raw_timeseries": [],
        "filename":[],
        "participant_id":[]
    }
    for col in convert_data:
        dataset_dict[col] = []

    for file_path in tqdm(timeseries_files, desc="convert to arrow"):
        seg_ts = pd.read_csv(file_path, sep='\t', header=0).values.astype(np.float32)
        # apply robust scaling to the time series
        scaler = RobustScaler()
        seg_ts_robustscaler = scaler.fit_transform(seg_ts)
        # they filled missing values with 0, so we do the same..... this is bad
        seg_ts_robustscaler = np.nan_to_num(seg_ts_robustscaler, nan=0.0, posinf=0.0, neginf=0.0)
        seg_ts_z = (seg_ts - np.mean(seg_ts, axis=0)) / np.std(seg_ts, axis=0)
        participant_id = file_path.stem.split('_')[0]
        dataset_dict["raw_timeseries"].append(seg_ts)
        dataset_dict["robustscaler_timeseries"].append(seg_ts_robustscaler)
        dataset_dict["zscore_timeseries"].append(seg_ts_z)
        dataset_dict["filename"].append(str(file_path.name))
        dataset_dict["participant_id"].append(participant_id)
        for col in convert_data:
            dataset_dict[col].append(phenotype.loc[participant_id, col])
    arrow_train_dataset = Dataset.from_dict(dataset_dict)
    arrow_train_dataset.save_to_disk(
        dataset_path=Path(arrow_dir)
    )
    print("Done.")


def brain_region_coord_to_arrow():
    """Save Brain Region Coordinates Into Another Arrow Dataset"""
    coords_dat = np.loadtxt(files('hfplayground') / "resource/brainlm/atlases/A424_Coordinates.dat").astype(np.float32)
    coords_pd = pd.DataFrame(coords_dat, columns=["Index", "X", "Y", "Z"])
    coords_dataset = Dataset.from_pandas(coords_pd)
    coords_dataset.save_to_disk(
        dataset_path=files('hfplayground') / "resource/brainlm/atlases/brainregion_coordinates.arrow")


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
