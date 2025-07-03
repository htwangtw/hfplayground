"""https://github.com/SalvoCalcagno/quantformer2024/blob/98286c88d79cf562966545b5509e93e611ae049b/src/trainers/trainer_brainlm.py#L55
https://github.com/Shef-AIRE/FMM_TC/blob/main/FMM_TC-tutorial.ipynb 
https://github.com/wenhui0206/MeTSK/blob/main/meta_learning.py#L8
BrainLM/continue_train_same_wandb.py
"""
from hfplayground.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
from hfplayground.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
from transformers import ViTMAEConfig
from datasets import load_from_disk, DatasetDict
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
from hfplayground.models.brainlm import preprocess_images
from hfplayground.utils.brainlm_trainer import BrainLMTrainer
from hfplayground.utils.metrics import MetricsCalculator

preprocessing = "development_fmri_gigaconnectome_a424"
# preprocessing = "development_fmri_brainlm_a424"
timeseries_length = 160

image_column_name_kw = {
    "development_fmri_gigaconnectome_a424": "robustscaler_timeseries",
    "development_fmri_brainlm_a424": "All_Patient_All_Voxel_Normalized_Recording"  # this works
}

preprocess_images_kargs = {
    "image_column_name": image_column_name_kw[preprocessing],
    "timeseries_length": timeseries_length, # this is for developmental dataset, full length
    "axis_index": "Y",
    "max_val_to_scale": None  # max_val_to_scale = 5.6430855  # this is weird.
}

model_params = "111M"  # Choose between 650M and 111M
model_arguments = {  # BrainLM/train.py::ModelArguments
    "mask_ratio": 0.75,  # The ratio of the number of masked tokens per brain region.
    "timepoint_patching_size": 20,  #Length of moving window of timepoints from each brain regions signal for 1 sample.
    "num_timepoints_per_voxel": timeseries_length,  # Number of timepoints for each brain region given in 1 sample input to model.  Must be divisible by timepoint_patching_size.
    "hidden_dropout_prob": 0.0,  # Dropout probability for layer activations in CellLM
    "attention_probs_dropout_prob": 0.0,  # Dropout probability for attention coefficients in CellLM.
    "output_attentions": True,
}
inputs_path = f"data/processed/{preprocessing}/fmri_development.arrow"
outputs_path = f"outputs/{preprocessing}_{model_params}"

fmri_ds = load_from_disk(inputs_path).class_encode_column("Child_Adult")
# train_ds.set_transform(transform_func)

# 90% train, 10% test + validation
train_testvalid = fmri_ds.train_test_split(train_size=0.8, stratify_by_column='Child_Adult')
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(train_size=0.5, stratify_by_column='Child_Adult')
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
replace_vitmae_attn_with_flash_attn()

def transform_func(batch):
    return preprocess_images(batch, **preprocess_images_kargs)

config = ViTMAEConfig.from_pretrained("vandijklab/brainlm", subfolder=f"vitmae_{model_params}")
config.update(model_arguments)
model = ViTMAEForPreTraining.from_pretrained(
        "vandijklab/brainlm",
        config=config,
        subfolder=f"vitmae_{model_params}",
    ).to(device)

model = model.half()  # half precision data type

train_test_valid_dataset.set_transform(transform_func)

metrics_calculator = MetricsCalculator()

# Initialize our trainer
trainer = BrainLMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_test_valid_dataset["train"],
    eval_dataset=train_test_valid_dataset["valid"],
    data_collator=collate_fn,
    compute_metrics=metrics_calculator
)
