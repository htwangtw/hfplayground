import invoke
from nilearn.datasets import fetch_development_fmri
from hfplayground.data.prepare import denoise_development_dataset, gigaconnectome_development_dataset, downsample_for_tutorial, brain_region_coord_to_arrow
from hfplayground.data.brainlm import convert_fMRIvols_to_A424, convert_to_arrow_datasets

@invoke.task
def models(c):
    c.run("HF_HUB_ENABLE_HF_TRANSFER=1")
    c.run("huggingface-cli download vandijklab/brainlm --local-dir ./models/brainlm")

@invoke.task
def atlas(c):
    c.run("mkdir -p ./hfplayground/resource/development_fmri")
    downsample_for_tutorial("hfplayground/resource/brainlm/atlases/A424+4mm.nii.gz", "hfplayground/data/development_fmri")
    downsample_for_tutorial("hfplayground/resource/brainlm/atlases/A424+2mm.nii.gz", "hfplayground/data/development_fmri")
    brain_region_coord_to_arrow()

@invoke.task
def data(c):
    print("Downloading the nilearn development fMRI dataset...")
    fetch_development_fmri(data_dir="data/external")
    print("Preprocessing the development dataset...")
    c.run("mkdir -p ./data/interim/development_fmri")


@invoke.task
def brainlm_workflow_timeseries(c):
    denoise_development_dataset(
        "data/external",
        "data/interim/development_fmri.brainlm",
        grand_mean_scale=False
    )
    c.run("mkdir -p ./data/interim/development_fmri.brainlm.a424")
    convert_fMRIvols_to_A424(
        "./data/interim/development_fmri.brainlm",
        "./data/interim/development_fmri.brainlm.a424"
    )
    c.run("mkdir -p ./data/processed/development_fmri.brainlm.arrow")
    convert_to_arrow_datasets(
        "./data/interim/development_fmri.brainlm.a424",
        "./data/processed/development_fmri.brainlm.arrow",
        ts_min_length=160, compute_Stats=True
    )

@invoke.task
def gigaconnectome_workflow_timeseries(c):
    denoise_development_dataset(
        "data/external",
        "data/interim/development_fmri.gigaconnectome",
        grand_mean_scale=True
    )
    gigaconnectome_development_dataset(
        "data/external",
        "data/interim/development_fmri.gigaconnectome",
        "data/processed/development_fmri.gigaconnectome.arrow"
    )
