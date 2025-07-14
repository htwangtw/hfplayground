"""https://github.com/SalvoCalcagno/quantformer2024/blob/98286c88d79cf562966545b5509e93e611ae049b/src/trainers/trainer_brainlm.py#L55
https://github.com/Shef-AIRE/FMM_TC/blob/main/FMM_TC-tutorial.ipynb
https://github.com/wenhui0206/MeTSK/blob/main/meta_learning.py#L8
BrainLM/continue_train_same_wandb.py
"""
from dataclasses import dataclass, field

from hfplayground.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
from hfplayground.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
from transformers import ViTMAEConfig, Trainer, TrainingArguments

from datasets import load_from_disk, DatasetDict
import torch
import torch.nn.functional as F
from hfplayground.models.utils import preprocess_images
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
def transform_func(batch):
    return preprocess_images(batch, **preprocess_images_kargs)
# 80% train, 20% test + validation
train_testvalid = fmri_ds.train_test_split(train_size=0.8, stratify_by_column='Child_Adult')
# Split the 20% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(train_size=0.5, stratify_by_column='Child_Adult')
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

train_test_valid_dataset.set_transform(transform_func)

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


from typing import Dict

import torch
import numpy as np
from sklearn.metrics import r2_score
from transformers.trainer_utils import EvalPrediction
from scipy.stats import pearsonr


class MetricsCalculator:
    """
    Class for metric calculation. An object of this class will be passed to the Huggingface Trainer
    class as a callable to calculate all metrics for training BrainLM models.

    Receives an EvalPrediction object from Trainer.
    Call:
        eval_pred_obj:              EvalPrediction object containing predictions, label_ids, and model inputs

    Returns:
        metrics_dict: dictionary containing metrics
            - mse:                  mean square error between masked expression values and model predictions
            - mae:                  mean absolute error between masked expression values and model predictions
            - cell_r2_avg:          average R2 across cells in minibatch, calculated on masked expression values
            - cell_r2_list:         list of R2 of individual cells in minibatch
            - gene_r2_avg:          average R2 across genes in minibatch, calculated on masked expression values
            - gene_r2_list:         list of R2 values of individual genes in minibatch
            - r2gene_idx_list:      indices of genes which were considered in gene_r2_avg calculation
            - cross_entropy_loss:   cross entropy loss between masked expression and model prediction
    """

    def __init__(self) -> None:
        self.cross_entropy_criterion = torch.nn.CrossEntropyLoss()
        self.current_epoch = 0  # Updated in log() function of CellLM Trainer

    def __call__(self, eval_pred_obj: EvalPrediction) -> Dict:
        # Current method for getting input expression vectors in compute_metrics function:
        #  Set training argument 'include_inputs_for_metrics' to True, and pass input expression vectors
        #  as input_ids through model forward() function. Trainer class passed the input_ids tensor to
        #  this function.
        print(eval_pred_obj.predictions)
        (pred_logits, encoder_latents), mask = eval_pred_obj.predictions
        # pred is numpy array, shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        # encoder_latents is numpy array shape [batch_size, num_masked_tokens + 1 CLS token, hidden_size]
        # mask is binary mask, shape [batch_size, num_voxels, num_tokens]


        # Get input expression vectors and sampled gene indices
        signal_vectors_padded = eval_pred_obj.inputs
        signal_vectors = signal_vectors_padded[: pred_logits.shape[0], :]
        signal_vectors = np.reshape(signal_vectors, pred_logits.shape)

        # Calculate MSE and MAE
        mse = self.calculate_mse(pred_logits, signal_vectors, mask)
        mae = self.calculate_mae(pred_logits, signal_vectors, mask)

        # Calculate R2
        mask = mask.astype(bool)
        unadjusted_r2 = self.calculate_r_squared_masked(
            pred_logits, signal_vectors, mask
        )

        p = self.calculate_pearson_masked(pred_logits, signal_vectors, mask)

        # # --- Plot figures for evaluation to weights & biases ---#
        # plot_cls_token_2d_umap(
        #     cls_tokens, age_labels, epoch=self.current_epoch, dataset_split="val"
        # )
        # plot_scatterplot(
        #     pred_logits,
        #     signal_vectors,
        #     mask,
        #     epoch=self.current_epoch,
        #     dataset_split="val",
        # )
        # plot_model_output_histogram(
        #     pred_logits,
        #     signal_vectors,
        #     mask,
        #     epoch=self.current_epoch,
        #     dataset_split="val",
        # )
        # plot_masked_pred_trends_one_sample(
        #     pred_logits=pred_logits,
        #     signal_vectors=signal_vectors,
        #     mask=mask,
        #     sample_idx=0,
        #     node_idxs=[0, 100, 200],
        #     dataset_split="val",
        #     epoch=self.current_epoch,
        # )
        # plot_masked_pred_trends_one_sample(
        #     pred_logits=pred_logits,
        #     signal_vectors=signal_vectors,
        #     mask=mask,
        #     sample_idx=1,
        #     node_idxs=[0, 100, 200],
        #     dataset_split="val",
        #     epoch=self.current_epoch,
        # )

        # # --- Return metrics dictionary ---#
        metrics_dict = {
            "mse": mse,
            "mae": mae,
            "r2": unadjusted_r2,
            "pearson r": p,
        }
        print(mse)
        return metrics_dict

    @staticmethod
    def calculate_mse(pred_values, signal_values, mask):
        """
        Helper function to calculate Mean Square Error (MSE) on predicted masked gene expression values.

        Args:
            pred_values:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            signal_values:  numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            mask:           binary mask of shape [batch_size, num_voxels, num_tokens]

        Returns:
            loss:           mean square error loss on only masked timepoints
        """
        mask = np.expand_dims(mask, axis=-1).repeat(pred_values.shape[-1], axis=-1)
        loss = (((pred_values - signal_values) ** 2) * mask).sum() / mask.sum()
        return loss.item()

    @staticmethod
    def calculate_mae(pred_values, signal_values, mask):
        """
        Helper function to calculate Mean Absolute Error (MAE) on predicted masked gene expression values.

        Args:
            pred_values:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            signal_values:  numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            mask:           binary mask of shape [batch_size, num_voxels, num_tokens]

        Returns:
            loss:           mean square error loss on only masked timepoints
        """
        mask = np.expand_dims(mask, axis=-1).repeat(pred_values.shape[-1], axis=-1)
        loss = abs((pred_values - signal_values) * mask).sum() / mask.sum()
        return loss.item()

    @staticmethod
    def calculate_r_squared_masked(pred_values, signal_values, mask):
        """
        Helper function to calculate R-squared between predicted pixel values and actual
        masked pixel values over all masked gene expression values from all cells and genes.

        Args:
            pred_values:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            signal_values:  numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            mask:           binary mask of shape [batch_size, num_voxels, num_tokens]
        """
        gt_list = []
        pred_vals_list = []
        for sample_idx in range(signal_values.shape[0]):
            for voxel_idx in range(signal_values.shape[1]):
                gt_list += list(
                    signal_values[sample_idx, voxel_idx][
                        mask[sample_idx, voxel_idx]
                    ].flatten()
                )
                pred_vals_list += list(
                    pred_values[sample_idx, voxel_idx][
                        mask[sample_idx, voxel_idx]
                    ].flatten()
                )

        r_squared = r2_score(y_true=gt_list, y_pred=pred_vals_list)
        if r_squared < 0.0:
            r_squared = 0.0
        return r_squared


    @staticmethod
    def calculate_pearson_masked(pred_values, signal_values, mask):
        """
        Helper function to calculate Pearson correlation between predicted pixel values and actual
        masked pixel values over all masked fMRI values from all voxels.

        Args:
            pred_values:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            signal_values:  numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            mask:           binary mask of shape [batch_size, num_voxels, num_tokens]
        """
        gt_list = []
        pred_vals_list = []
        for sample_idx in range(signal_values.shape[0]):
            for voxel_idx in range(signal_values.shape[1]):
                gt_list += list(
                    signal_values[sample_idx, voxel_idx][
                        mask[sample_idx, voxel_idx]
                    ].flatten()
                )
                pred_vals_list += list(
                    pred_values[sample_idx, voxel_idx][
                        mask[sample_idx, voxel_idx]
                    ].flatten()
                )

        pearson = pearsonr(x=gt_list, y=pred_vals_list)
        p = pearson.statistic
        if p < 0.0:
            p = 0.0
        return p

metrics_calculator = MetricsCalculator()

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    if "labels" not in examples:
        labels = torch.tensor([1 for _ in range(len(pixel_values))])
    labels = torch.stack([torch.tensor(example["labels"]) for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": pixel_values,
        "labels": labels
    }

training_args = TrainingArguments(
    output_dir="outputs/test",
    remove_unused_columns=False
)
# Initialize our trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_valid_dataset["train"],
    eval_dataset=train_test_valid_dataset["valid"],
    data_collator=collate_fn,
    # compute_metrics=metrics_calculator
)

train_result = trainer.train()
eval_metrics = trainer.evaluate()
