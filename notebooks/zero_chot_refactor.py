from hfplayground.brainlm_mae.modeling_brainlm import BrainLMForPretraining
from hfplayground.brainlm_mae.configuration_brainlm import BrainLMConfig
from transformers import ViTImageProcessor, ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEForPreTraining
from datasets import load_from_disk, concatenate_datasets
import numpy as np
import torch
from tqdm import tqdm
from random import randint, seed
import torch.nn.functional as F
from pathlib import Path

COORD_PATH = "data/processed/development_fmri_brainlm_a424/brainregion_coordinates.arrow"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
do_r2 = True
do_inference = True
aggregation_mode = "cls" # 'cls', 'mean', or 'max'
model_type="pad"
variable_of_interest_col_name = "Index"
image_column_name = "Voxelwise_RobustScaler_Normalized_Recording"  # In the paper they used robust scaling, wich has a sklearn implementation.
timeseries_length = 160  # this is for developmental dataset
num_parcels = 424

# max_val_to_scale = 5.6430855  # this is weird.
max_val_to_scale = None

model_params = "111M"  # Choose between 650M and 111M
model_arguments = {  # BrainLM/train.py::ModelArguments
    "mask_ratio": 0.75,  # The ratio of the number of masked tokens per brain region.
    "timepoint_patching_size": 20,  #Length of moving window of timepoints from each brain regions signal for 1 sample.
    "num_timepoints_per_voxel": timeseries_length,  # Number of timepoints for each brain region given in 1 sample input to model.  Must be divisible by timepoint_patching_size.
    "hidden_dropout_prob": 0.0,  # Dropout probability for layer activations in CellLM
    "attention_probs_dropout_prob": 0.0,  # Dropout probability for attention coefficients in CellLM.
    "output_attentions": True,
}

outputs_path = f"outputs/development_fmri_brainlm_a424_{model_params}"
Path(outputs_path).mkdir(parents=True, exist_ok=True)

def preprocess_images(examples, image_column_name=image_column_name,
                      length=timeseries_length,
                      max_val_to_scale=max_val_to_scale):
    """Preprocess a batch of images by applying transforms."""
    coords_ds = load_from_disk(COORD_PATH)

    parcel_coords_list = coords_ds["Y"]  # sort by y
    reorder_idxs_by_coord = sorted(range(len(parcel_coords_list)), key=lambda k: parcel_coords_list[k])
    reorder_idxs_by_coord = np.array(reorder_idxs_by_coord)

    fmri_images_list = []
    for idx in range(len(examples[image_column_name])):  # iterate over subject
        signal_window = torch.tensor(examples[image_column_name][idx], dtype=torch.float32) # [timepoints, num_parcel]

        # Choose random starting index, take window of moving_window_len points for each region
        start_idx = randint(0, signal_window.shape[0] - length)
        end_idx = start_idx + length
        signal_window = signal_window[start_idx: end_idx, :]
        signal_window = torch.movedim(signal_window, 0, 1)  # --> [num_parcel, moving_window_len]

        # reorder voxels according to y-coordinate
        signal_window = signal_window[reorder_idxs_by_coord, :]
        # signal_window = signal_window / max_val_to_scale

        # Repeat tensor for 3 channels (R,G,B)
        signal_window = signal_window.unsqueeze(0).repeat(3, 1, 1)

        fmri_images_list.append(signal_window)


    examples["pixel_values"] = fmri_images_list  # No transformation or resizing; model will do padding
    return examples

def get_attention_cls_token(attn_probs):
    attn_probs_heads = attn_probs[-1].squeeze(0)  # last attention layer, first head
    attn_probs_avg = attn_probs_heads.mean(dim=0, keepdim=True)
    cls_attn = attn_probs_avg[:, 0, :].cpu().numpy()
    return cls_attn

# config = BrainLMConfig.from_pretrained("models/vitmae_111M")
# config.update(model_arguments)
config = ViTMAEConfig.from_pretrained("vandijklab/brainlm", subfolder=f"vitmae_{model_params}")
config.update(model_arguments)
model = ViTMAEForPreTraining.from_pretrained(
        "vandijklab/brainlm",
        config=config,
        subfolder=f"vitmae_{model_params}",
    ).to(device)

model = model.half()  # half precision data type
model.eval()
print(model.dtype)
print(model.config.mask_ratio)
print(model.vit.embeddings.config.mask_ratio)

# multiple train modes (auto-encoder, causal attention, predict last, etc)
model.config.train_mode = "auto_encode"

train_ds = load_from_disk("data/processed/development_fmri_brainlm_a424/fmri_development.arrow")
image_processor = ViTImageProcessor(
    size={
        "height": model.config.image_size[0],
        "width": model.config.image_size[1]
    }
)
train_ds.set_transform(preprocess_images)

list_cls_tokens = []
list_attn_cls_tokens = []
all_embeddings = []
# all_index = []
with torch.no_grad():
    # for recording in tqdm(train_ds, desc="Getting CLS tokens"):
    recording = train_ds[0]  # For testing, use only the first recording
    # pixel_values = recording["pixel_values"].unsqueeze(0).half().to(device)
    pixel_values = recording["pixel_values"].half().to(device)
    if model_type == "pad":
        # pixel_values is [batch, channels=3, number of parcels, number of time points]. Pad to [batch, channels=3, 434, 434]
        height_pad_total = model.config.image_size[0] - pixel_values.shape[2]
        height_pad_total_half = height_pad_total // 2
        width_pad_total = model.config.image_size[1] - pixel_values.shape[3]
        width_pad_total_half = width_pad_total // 2
        pixel_values = F.pad(pixel_values, (width_pad_total_half, width_pad_total_half, height_pad_total_half, height_pad_total_half), "constant", -1)

    encoder_output = model.vit(
        pixel_values=pixel_values,
        output_hidden_states=True
    )

    cls_token = encoder_output.last_hidden_state[:,0,:].detach().cpu().numpy()  # torch.Size([1, 256])? (I got 1, 241)
    embedding = encoder_output.last_hidden_state[:,1:,:].detach().cpu().numpy()

    attn_probs_heads = encoder_output.attentions[-1].squeeze(0)  # last attention layer, first head
    attn_probs_avg = attn_probs_heads.mean(dim=0, keepdim=True)
    attn_cls_token = attn_probs_avg[:, 0, :].cpu().numpy()

    list_attn_cls_tokens.append(attn_cls_token)
    list_cls_tokens.append(cls_token)
    if aggregation_mode != "cls":
        all_embeddings.append(embedding)
    # all_index.append(recording["labels"].detach().numpy())

# save results
preds_name = Path(outputs_path) / f"{model_params}_cls_token_brainlm.npy"
print("Saving inference results to: ", preds_name)
print(list_attn_cls_tokens[0].shape)
np.save(preds_name, list_attn_cls_tokens)

if aggregation_mode == "cls":
    print("cls aggregation")
    all_embeds = np.concatenate(list_cls_tokens, axis=0)
elif aggregation_mode == "mean":
    print("mean pool aggregation")
    all_mean_embeddings = [e.mean(axis=1) for e in all_embeddings]
    all_embeds = np.concatenate(all_mean_embeddings, axis=0)
elif aggregation_mode == "max":
    print("max pool aggregation")
    all_sum_embeddings = [e.max(axis=1) for e in all_embeddings]
    all_embeds = np.concatenate(all_sum_embeddings, axis=0)

print("Shape of all cls embeddings: ")
print(all_embeds.shape)
np.save(Path(outputs_path) / f"{aggregation_mode}_all_{timeseries_length}recordinglength_brainlm.npy", all_embeds)

# Save raw, padded recordings as well
all_recordings = []
for idx, batch in enumerate(tqdm(train_ds)):
    signal = batch["pixel_values"]  #(1, 3, 434, 434)
    recording = signal.flatten(start_dim = 1)
    recording = np.array(recording, dtype=np.float32)
    all_recordings.append(recording)
all_recordings = np.vstack(all_recordings)
print("Shape of all padded recordings: ")
print(all_recordings.shape)
np.save(Path(outputs_path) / f"all_recordings_{timeseries_length}length_brainlm.npy", all_recordings)
