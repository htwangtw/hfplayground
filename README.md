# hfplayground

For learning and testing various hugging face stuff through BrainLM.

Here I am using the nilearn test dataset to walk through BrainLM fine tuning and downstream tutorial.

The aim is to use the code as-is as much as possible.

Later we might try to improve the process....

## Run this project

Please use [`uv`](https://docs.astral.sh/uv/) to install this project for the smoothest experience.

```
git clone git@github.com:htwangtw/hfplayground.git
```

The brainLM submodule is for record keeping.
However, if you wish pull it, run:

```
git submodule update --init --recursive
```

## Create virtual environment

With `uv`
```
uv venv
uv sync
```

You can either activate the environment with `source .venv/bin/activate` and use this environment in the conventional python way,
or prepend any command you want to run with `uv run` to activate the environment.


## Download models and data

With `uv`
```
uv run invoke prepare.models
uv run invoke prepare.data
```

Or with your virtual environment
```
source .venv/bin/activate
invoke prepare.models
invoke prepare.atlas
invoke prepare.data
invoke prepare.timeseries
```
Check out `uv run invoke --list` for the commands and documentations.

## Other unrelated notes

### `uv`

[Quick tutorial with pip->uv table](https://www.datacamp.com/tutorial/python-uv)
