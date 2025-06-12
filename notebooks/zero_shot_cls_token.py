from hfplayground.brainlm_mae.modeling_brainlm import BrainLMForPretraining
from transformers import ViTImageProcessor, ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEForPreTraining
from datasets import load_from_disk, concatenate_datasets
import numpy as np
import torch
from tqdm import tqdm
from random import randint, seed
import torch.nn.functional as F

# from hfplayground.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn  #no module named 'flash_attn'

# relevant notebook: BrainLM/brainlm_tutorial.ipynb, BrainLM/toolkit/BrainLM_Tutorial.ipynb

# # loading pretrain
# model = BrainLMForPretraining.from_pretrained("models/old_13M/")
# model = BrainLMForPretraining.from_pretrained("BrainLM/pretrained_models/2023-06-06-22_15_00-checkpoint-1400/")
# model = ViTMAEForPreTraining.from_pretrained("models/vitmae_111M")
# model = ViTMAEForPreTraining.from_pretrained("models/vitmae_650M")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# zero-shot inference -save cls token
# replace_vitmae_attn_with_flash_attn()
params = "650M" #Choose between 650M and 111M
config = ViTMAEConfig.from_pretrained("vandijklab/brainlm", subfolder=f"vitmae_{params}")
config.update({  # BrainLM/train.py::ModelArguments
    "mask_ratio": 0.75,  # The ratio of the number of masked tokens per voxel.
    "timepoint_patching_size": 20,  #Length of moving window of timepoints from each brain regions signal for 1 sample.
    "num_timepoints_per_voxel": 160,  # Number of timepoints for each voxel given in 1 sample input to model.  Must be divisible by timepoint_patching_size.
    "hidden_dropout_prob": 0.0,  # Dropout probability for layer activations in CellLM
    "attention_probs_dropout_prob": 0.0,  # Dropout probability for attention coefficients in CellLM.
    "output_attentions": True,
})

model = ViTMAEForPreTraining.from_pretrained(
        "vandijklab/brainlm",
        config=config,
        subfolder=f"vitmae_{params}",
    ).to(device)

model = model.half()  # half precision data type
model.eval()
print(model.dtype)
print(model.config.mask_ratio)
print(model.vit.embeddings.config.mask_ratio)

do_r2 = True
do_inference = True
aggregation_mode = "cls" # 'cls', 'mean', or 'max'

variable_of_interest_col_name = "Index"
image_column_name = "All_Patient_All_Voxel_Normalized_Recording"
length = 160  # this is for developmental dataset
num_voxels = 424  # this should be atlas....

# multiple train modes (auto-encoder, causal attention, predict last, etc)
model.config.train_mode = "auto_encode"

coords_ds = load_from_disk("data/processed/brainlm_a424/Brain_Region_Coordinates")
train_ds = load_from_disk("data/processed/brainlm_a424/train")

# split = "train"
# used_ds = dataset_split[split]
# print(used_ds)

image_processor = ViTImageProcessor(size={"height": model.config.image_size[0], "width": model.config.image_size[1]})
if "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
else:
    size = (image_processor.size["height"], image_processor.size["width"])
voxel_x_coords_list = coords_ds["Y"]
reorder_idxs_by_x_coord = sorted(range(len(voxel_x_coords_list)), key=lambda k: voxel_x_coords_list[k])
reorder_idxs_by_x_coord = np.array(reorder_idxs_by_x_coord)
max_val_to_scale = 5.6430855

def preprocess_images(examples):
    """Preprocess a batch of images by applying transforms."""
    fmri_images_list = []
    for idx in range(len(examples[image_column_name])):
        signal_window = torch.tensor(examples[image_column_name][idx], dtype=torch.float32).t()

        # Choose random starting index, take window of moving_window_len points for each region
        start_idx = randint(0, signal_window.shape[0] - length)
        end_idx = start_idx + length
        signal_window = signal_window[start_idx: end_idx, :]
        signal_window = torch.movedim(signal_window, 0, 1)  # --> [num_voxels, moving_window_len]

        # reorder voxels according to x-coordinate
        signal_window = signal_window[reorder_idxs_by_x_coord, :]
        signal_window = signal_window / max_val_to_scale

        # Repeat tensor for 3 channels (R,G,B)
        signal_window = signal_window.unsqueeze(0).repeat(3, 1, 1)

        fmri_images_list.append(signal_window)


    examples["pixel_values"] = fmri_images_list  # No transformation or resizing; model will do padding
    return examples


def get_attention_cls_token(attn_probs):
    attn_probs_heads = attn_probs[31].squeeze(0)
    attn_probs_avg = attn_probs_heads.mean(dim=0, keepdim=True)
    cls_attn = attn_probs_avg[:, 0, :].cpu().numpy()
    return cls_attn

train_ds.set_transform(preprocess_images)

model_type="pad"
list_cls_tokens = []
list_attn_cls_tokens = []
all_embeddings = []
all_index = []
with torch.no_grad():
    for recording in tqdm(train_ds, desc="Getting CLS tokens"):

        pixel_values = recording["pixel_values"].unsqueeze(0).half().to(device)
        if model_type == "pad":
            # pixel_values is [batch, channels=3, 424 (number of parcels), 200 (number of time points)]. Pad to [batch, channels=3, 432, 432]
            height_pad_total = model.config.image_size[0] - pixel_values.shape[2]
            height_pad_total_half = height_pad_total // 2
            width_pad_total = model.config.image_size[1] - pixel_values.shape[3]
            width_pad_total_half = width_pad_total // 2
            pixel_values = F.pad(pixel_values, (width_pad_total_half, width_pad_total_half, height_pad_total_half, height_pad_total_half), "constant", -1)

        encoder_output = model.vit(
            pixel_values=pixel_values,
            output_hidden_states=True
        )

        cls_token = encoder_output.last_hidden_state[:,0,:]  # torch.Size([1, 256])
        embedding = encoder_output.last_hidden_state[:,1:,:]
        all_embeddings.append(embedding.detach().cpu().numpy())
        list_cls_tokens.append(cls_token.detach().cpu().numpy())
        all_index.append(recording["labels"].detach().numpy())
        attn_cls_token = get_attention_cls_token(encoder_output.attentions)
        list_attn_cls_tokens.append(attn_cls_token)
print(all_embeddings[0].shape)

# save results
preds_name = f"outputs/{params}_cls_token.npy"
print("Saving inference results to: ", preds_name)
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


print(all_embeds.shape)
all_index = np.concatenate(all_index, axis=0)

# check if dataloader messed up order in any way
if not np.all(all_index[:-1] <= all_index[1:]):
    # reorder everything
    print("reordering")
    all_embeds = all_embeds[all_index, :]

np.save(f"outputs/{aggregation_mode}_all_{length}recordinglength.npy", all_embeds)


# extract patch tokens with data
if aggregation_mode != "cls":
    print(length)
    hor_img_start_idx = (model.config.image_size[1] - length) // 2
    hor_token_start = hor_img_start_idx // model.config.patch_size
    hor_token_end = hor_token_start + np.ceil(length / model.config.patch_size).astype(int)

    vert_img_start_idx = (model.config.image_size[0] - num_voxels) // 2
    vert_token_start = vert_img_start_idx // model.config.patch_size
    vert_token_end = vert_token_start + np.ceil(num_voxels / model.config.patch_size).astype(int)
    print(hor_token_start, hor_token_end, vert_token_start, vert_token_end)

    for i, e in enumerate(all_embeddings):
        e = e.reshape(e.shape[0], int(np.sqrt(e.shape[1])), int(np.sqrt(e.shape[1])), -1)
        e = e[:, vert_token_start:vert_token_end, hor_token_start:hor_token_end]
        all_embeddings[i] = e.reshape(e.shape[0], -1, e.shape[-1])

    print(all_embeddings[0].shape)

    if aggregation_mode == "mean":
        print("mean pool aggregation")
        all_mean_embeddings = [e.mean(axis=1) for e in all_embeddings]
        all_embeds = np.concatenate(all_mean_embeddings, axis=0)
    if aggregation_mode == "max":
        print("max pool aggregation")
        all_sum_embeddings = [e.max(axis=1) for e in all_embeddings]
        all_embeds = np.concatenate(all_sum_embeddings, axis=0)

    np.save(f"outputs/{aggregation_mode}_only_data_{length}recordinglength.npy", all_embeds)

# Save raw recordings as well
all_recordings = []
for idx, batch in enumerate(tqdm(train_ds)):
    signal = batch["pixel_values"]
    recording = signal.flatten(start_dim = 1)
    recording = np.array(recording, dtype=np.float32)
    all_recordings.append(recording)
all_recordings = np.vstack(all_recordings)
all_recordings.shape

np.save(f"outputs/all_recordings_{length}length.npy", all_recordings)

# BrainLM/toolkit/BrainLM_Tutorial.ipynb plotting with PCA etc