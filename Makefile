install_uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

env:
	uv sync

download_model:
	HF_HUB_ENABLE_HF_TRANSFER=1
	uv run huggingface-cli download vandijklab/brainlm --local-dir ./models/
