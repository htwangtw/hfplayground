.PHONY: env activate_env download_model download_data

env: 
	uv venv
	uv sync

download_models: 
	HF_HUB_ENABLE_HF_TRANSFER=1
	huggingface-cli download vandijklab/brainlm --local-dir ./models/

download_data: 
	mkdir -p ./data/interim/development_fmri
	python -c 'from hfplayground.data import preprocess_developmental_dataset; preprocess_developmental_dataset("data/external", "data/interim/development_fmri")'

downsample_atlas: ./data/interim/development_fmri/*.nii.gz
	mkdir -p ./hfplayground/data/development_fmri
	python -c 'from hfplayground.data import downsample_for_tutorial; downsample_for_tutorial("hfplayground/data/brainlm/atlases/A424+4mm.nii.gz", "hfplayground/data/development_fmri"); downsample_for_tutorial("hfplayground/data/brainlm/atlases/A424+2mm.nii.gz", "hfplayground/data/development_fmri")'