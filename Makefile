.PHONY: env activate_env download_model download_data clean

env: 
	uv venv
	uv sync

download_models: 
	HF_HUB_ENABLE_HF_TRANSFER=1
	huggingface-cli download vandijklab/brainlm --local-dir ./models/

download_data: 
	mkdir -p ./data/interim/development_fmri
	python -c 'from hfplayground.data import preprocess_development_dataset; preprocess_development_dataset("data/external", "data/interim/development_fmri")'

downsample_atlas: ./data/interim/development_fmri/*.nii.gz
	mkdir -p ./hfplayground/data/development_fmri
	python -c 'from hfplayground.data import downsample_for_tutorial; downsample_for_tutorial("hfplayground/data/brainlm/atlases/A424+4mm.nii.gz", "hfplayground/data/development_fmri"); downsample_for_tutorial("hfplayground/data/brainlm/atlases/A424+2mm.nii.gz", "hfplayground/data/development_fmri")'

timeseries: ./data/interim/development_fmri/*.nii.gz
	mkdir -p ./data/interim/brainlm_a424
	python -c 'from hfplayground.brainlm_data import convert_fMRIvols_to_A424; convert_fMRIvols_to_A424("./data/interim/development_fmri/", "./data/interim/brainlm_a424")'

clean:
	rm -rf data/interim/*