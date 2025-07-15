from dataclasses import dataclass, field
from typing import Optional
from random import randint
from transformers import TrainingArguments


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_dataset_path: str = field(
        metadata={
            "help": "Path to saved train arrow dataset of cell x gene expression matrix."
        }
    )
    val_dataset_path: str = field(
        metadata={
            "help": "Path to saved val arrow dataset of cell x gene expression matrix."
        }
    )
    coords_dataset_path: str = field(
        metadata={"help": "Path to saved arrow dataset of brain region coordinates."}
    )
    recording_col_name: str = field(
        default="Voxelwise_RobustScaler_Normalized_Recording",
        metadata={"help": "Column in dataset which contains recording for each patient. Choose from:"
                          "All_Patient_All_Voxel_Normalized_Recording, "
                          "Per_Patient_All_Voxel_Normalized_Recording, "
                          "Per_Patient_Per_Voxel_Normalized_Recording, "
                          "Per_Voxel_All_Patient_Normalized_Recording, "
                          "Subtract_Mean_Normalized_Recording, "
                          "or Subtract_Mean_Divide_Global_STD_Normalized_Recording"
                  }
    )
    variable_of_interest_col_name: str = field(
        default="Age.At.MHQ",
        metadata={
            "help": "Column in dataset containing desired label for each patient. Choose from:"
            "Order, eid, Gender, Age.At.MHQ, PHQ9.Severity, Depressed.At.Baseline"
            "Neuroticism, Self.Harm.Ever, Not.Worth.Living, PCL.Score, GAD7.Severity"
        },
    )
    num_timepoints_per_voxel: int = field(
        default=490,
        metadata={
            "help": "Number of timepoints for each voxel given in 1 sample input to model. "
            "Must be divisible by timepoint_patching_size."
        },
    )
    timepoint_patching_size: int = field(
        default=49,
        metadata={
            "help": "Length of moving window of timepoints from each brain "
            "regions signal for 1 sample."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        self.data_files = None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name_or_path"
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    hidden_size: int = field(default=128, metadata={"help": "Encoder hidden size."})
    num_hidden_layers: int = field(default=2, metadata={"help": "Encoder num layers."})
    num_attention_heads: int = field(
        default=2, metadata={"help": "Number of attention heads in encoder."}
    )
    intermediate_size: int = field(
        default=512, metadata={"help": "Intermediate size in MLP in encoder layers."}
    )
    decoder_hidden_size: int = field(
        default=128, metadata={"help": "Decoder hidden size."}
    )
    decoder_num_hidden_layers: int = field(
        default=2, metadata={"help": "Decoder num layers."}
    )
    decoder_num_attention_heads: int = field(
        default=2, metadata={"help": "Number of attention heads in the decoder."}
    )
    decoder_intermediate_size: int = field(
        default=512, metadata={"help": "Intermediate size in MLP in decoder layers."}
    )
    hidden_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "Dropout probability for layer activations in CellLM."},
    )
    attention_probs_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for attention coefficients in CellLM."},
    )
    mask_ratio: float = field(
        default=0.2,
        metadata={"help": "The ratio of the number of masked tokens per voxel."},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Don't remove unused columns."}
    )
    do_train: int = field(default=True, metadata={"help": "Whether to do training."})
    do_eval: int = field(default=True, metadata={"help": "Whether to do eval."})
    base_learning_rate: float = field(
        default=1e-3,
        metadata={
            "help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."
        },
    )
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={"help": "What learning rate scheduler to use."},
    )
    weight_decay: float = field(
        default=0.05,
        metadata={
            "help": "Weight decay (L2 regularization coefficient) for optimizer."
        },
    )
    num_train_epochs: int = field(
        default=100, metadata={"help": "Number of epochs to train for."}
    )
    warmup_ratio: float = field(
        default=0.05, metadata={"help": "Warmup ratio for learning rate scheduler."}
    )
    per_device_train_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size for each device used during training."},
    )
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size for each device used during evaluation."},
    )
    logging_strategy: str = field(
        default="steps",
        metadata={
            "help": "How often to log training metrics. If choose 'steps', specify logging_steps."
        },
    )
    logging_steps: int = field(
        default=10,
        metadata={
            "help": "If logging_strategy is 'steps', log training metrics every X iterations."
        },
    )
    evaluation_strategy: str = field(
        default="steps", metadata={"help": "How often to log eval results."}
    )
    eval_steps: int = field(
        default=10,
        metadata={
            "help": "If evaluation_strategy is 'steps', calculate validation metrics every X iterations."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "How often to save results and models."}
    )
    save_steps: int = field(
        default=10,
        metadata={
            "help": "If save_strategy is 'steps', save model checkpoint every X iterations."
        },
    )
    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "At the end, load the best model."}
    )
    save_total_limit: int = field(
        default=50, metadata={"help": "Maximum number of models to save."}
    )
    seed: int = field(default=1234, metadata={"help": "Random seed."})
    wandb_logging: bool = field(
        default=False,
        metadata={
            "help": "Whether to log metrics to weights & biases during training."
        },
    )
    include_inputs_for_metrics: bool = field(
        default=True,
        metadata={
            "help": "Trainer will include model inputs in call to metrics calculation function. Depends on 'input_ids' being one of the input parameters to model, comes from tokenizer used? Currently incompatible with single-cell dataloader, leave as False."
        },
    )
    loss_fn: str = field(
        default="mse",
        metadata={"help": "Loss function for CellLM to use for pretraining."},
    )
    use_tanh_decoder: bool = field(
        default=False,
        metadata={
            "help": "If we want to use TanH as the nonlinearity for the output layer."
        },
    )

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.stack([example["label"] for example in examples])
    # labels = torch.tensor([1 for _ in range(len(pixel_values))])

    return {
        "pixel_values": pixel_values,
        "input_ids": pixel_values,
        "labels": labels
    }