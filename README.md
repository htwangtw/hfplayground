# hfplayground

For learning and testing various hugging face stuff through BrainLM.

Here I am using the nilearn test dataset to walk through BrainLM fine tuning and downstream tutorial.

## Use this project

Please use [`uv`](https://docs.astral.sh/uv/) to install this project.

```
git clone git@github.com:htwangtw/hfplayground.git
```

The brainLM submodule is for record keeping.
However, if you wish pull it, run:

```
git submodule update --init --recursive
```

## Download model

```
HF_HUB_ENABLE_HF_TRANSFER=1
uv run huggingface-cli download vandijklab/brainlm --local-dir ./models/
```

## Other unrelated notes

### `uv`

[Quick tutorial with pip->uv table](https://www.datacamp.com/tutorial/python-uv)