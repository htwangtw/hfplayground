import invoke
from nilearn.datasets import fetch_development_fmri
from hfplayground.data import preprocess_development_dataset, downsample_for_tutorial
from hfplayground.brainlm_data import convert_fMRIvols_to_A424, convert_to_arrow_datasets

@invoke.task
def models(c):
    c.run("HF_HUB_ENABLE_HF_TRANSFER=1")
    c.run("huggingface-cli download vandijklab/brainlm --local-dir ./models/")

@invoke.task
def atlas(c):
    c.run("mkdir -p ./hfplayground/data/development_fmri")
    downsample_for_tutorial("hfplayground/data/brainlm/atlases/A424+4mm.nii.gz", "hfplayground/data/development_fmri")
    downsample_for_tutorial("hfplayground/data/brainlm/atlases/A424+2mm.nii.gz", "hfplayground/data/development_fmri")

@invoke.task
def data(c):
    print("Downloading the nilearn development fMRI dataset...")
    fetch_development_fmri(data_dir="data/external")
    print("Preprocessing the development dataset...")
    c.run("mkdir -p ./data/interim/development_fmri")
    preprocess_development_dataset("data/external", "data/interim/development_fmri")

@invoke.task
def brainlm_workflow_timeseries(c):
    c.run("mkdir -p ./data/interim/development_fmri_brainlm_a424")
    convert_fMRIvols_to_A424("./data/interim/development_fmri/", "./data/interim/development_fmri_brainlm_a424")
    c.run("mkdir -p ./data/processed/development_fmri_brainlm_a424")
    convert_to_arrow_datasets("./data/interim/development_fmri_brainlm_a424", "data/processed/development_fmri_brainlm_a424", ts_min_length=160, compute_Stats=True)

@invoke.task
def gigaconnectome_workflow_timeseries(c):
    c.run("mkdir -p ./data/interim/development_fmri_gigaconnectome_a424")
    preprocess_development_dataset(
        "data/external",
        "data/interim/development_fmri",
        "data/processed/development_fmri_gigaconnectome_a424"
    )
