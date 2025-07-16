"""https://github.com/SalvoCalcagno/quantformer2024/blob/98286c88d79cf562966545b5509e93e611ae049b/src/trainers/trainer_brainlm.py#L55
https://github.com/Shef-AIRE/FMM_TC/blob/main/FMM_TC-tutorial.ipynb
https://github.com/wenhui0206/MeTSK/blob/main/meta_learning.py#L8
BrainLM/continue_train_same_wandb.py
"""
from hfplayground.models.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
from hfplayground.models.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
from transformers import ViTMAEConfig, Trainer, TrainingArguments

from datasets import load_from_disk, DatasetDict
import torch
from hfplayground.models.brainlm_mae.utils import timeseires_to_images, collate_fn
from hfplayground.models.brainlm_mae.metrics import MetricsCalculator


preprocessing = "development_fmri_gigaconnectome_a424"
# preprocessing = "development_fmri_brainlm_a424"
timeseries_length = 160

image_column_name_kw = {
    "development_fmri_gigaconnectome_a424": "robustscaler_timeseries",
    "development_fmri_brainlm_a424": "All_Patient_All_Voxel_Normalized_Recording"  # this works
}

timeseires_to_images_kargs = {
    "image_column_name": image_column_name_kw[preprocessing],
    "timeseries_length": timeseries_length, # this is for developmental dataset, full length
    "axis_index": "Y",
    "max_val_to_scale": None  # max_val_to_scale = 5.6430855  # this is weird.
}
model_params = "650M"
# model_params = "111M"  # Choose between 650M and 111M
model_arguments = {  # BrainLM/train.py::ModelArguments
    "mask_ratio": 0.75,  # The ratio of the number of masked tokens per brain region.
    "timepoint_patching_size": 20,  #Length of moving window of timepoints from each brain regions signal for 1 sample.
    "num_timepoints_per_voxel": timeseries_length,  # Number of timepoints for each brain region given in 1 sample input to model.  Must be divisible by timepoint_patching_size.
    "hidden_dropout_prob": 0.0,  # Dropout probability for layer activations in CellLM
    "attention_probs_dropout_prob": 0.0,  # Dropout probability for attention coefficients in CellLM.
    "output_attentions": True,
}
inputs_path = f"data/processed/{preprocessing}/fmri_development.arrow"
outputs_path = f"outputs/{preprocessing}_{model_params}/finetuning"

fmri_ds = load_from_disk(inputs_path).class_encode_column("Child_Adult")
def transform_func(batch):
    return timeseires_to_images(batch, **timeseires_to_images_kargs)
# 80% train, 20% test
train_test = fmri_ds.train_test_split(train_size=0.8, stratify_by_column='Child_Adult')
# gather everyone if you want to have a single DatasetDict
train_test_dataset = DatasetDict({
    'train': train_test['train'],
    'test': train_test['test']})

train_test_dataset.set_transform(transform_func)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
replace_vitmae_attn_with_flash_attn()


config = ViTMAEConfig.from_pretrained("vandijklab/brainlm", subfolder=f"vitmae_{model_params}")
config.update(model_arguments)
config.train_mode = "auto_encode"
model = ViTMAEForPreTraining.from_pretrained(
        "vandijklab/brainlm",
        config=config,
        subfolder=f"vitmae_{model_params}",
    ).to(device)


metrics_calculator = MetricsCalculator()

training_args = TrainingArguments(
    output_dir=outputs_path,
    remove_unused_columns=False,
    include_for_metrics=['inputs']
)
# Initialize our trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_dataset["train"],
    eval_dataset=train_test_dataset["test"],
    data_collator=collate_fn,
    compute_metrics=metrics_calculator
)

train_result = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

# Evaluation
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
