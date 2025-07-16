# cleaned up for using published weights for direct transfer with CLS token
from transformers import ViTMAEConfig
from datasets import load_from_disk, Dataset
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path

from hfplayground.utils import timeseires_to_images, get_attention_cls_token
from hfplayground.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
from hfplayground.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn

preprocessing = "development_fmri_gigaconnectome_a424"
# preprocessing = "development_fmri_brainlm_a424"
# model_params = "111M"  # Choose between 650M and 111M
model_params = "650M"
timeseries_length = 160

image_column_name_kw = {
    "development_fmri_gigaconnectome_a424": "robustscaler_timeseries",
    "development_fmri_brainlm_a424": "All_Patient_All_Voxel_Normalized_Recording"  # this works
    # In the paper brainlm claimed to use the robust scalar but that option produces NaN
}

timeseires_to_images_kargs = {
    "image_column_name": image_column_name_kw[preprocessing],
    "timeseries_length": timeseries_length, # this is for developmental dataset, full length
    "axis_index": "Y",
    "max_val_to_scale": None  # max_val_to_scale = 5.6430855  # this is weird.
}

model_arguments = {  # BrainLM/train.py::ModelArguments
    "mask_ratio": 0.75,  # The ratio of the number of masked tokens per brain region.
    "timepoint_patching_size": 20,  #Length of moving window of timepoints from each brain regions signal for 1 sample.
    "num_timepoints_per_voxel": timeseries_length,  # Number of timepoints for each brain region given in 1 sample input to model.  Must be divisible by timepoint_patching_size.
    "hidden_dropout_prob": 0.0,  # Dropout probability for layer activations in CellLM
    "attention_probs_dropout_prob": 0.0,  # Dropout probability for attention coefficients in CellLM.
    "output_attentions": True,
}


def padding_timeseries_For_vitmae(pixel_values, image_size):
    # pixel_values is [batch, channels=3, number of parcels, number of time points]. Pad to [batch, channels=3, 434, 434]
    height_pad_total = image_size[0] - pixel_values.shape[2]
    height_pad_total_half = height_pad_total // 2
    width_pad_total = image_size[1] - pixel_values.shape[3]
    width_pad_total_half = width_pad_total // 2
    return F.pad(pixel_values, (width_pad_total_half, width_pad_total_half, height_pad_total_half, height_pad_total_half), "constant", -1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replace_vitmae_attn_with_flash_attn()

    def transform_func(batch):
        return timeseires_to_images(batch, **timeseires_to_images_kargs)
    outputs_path = f"outputs/{preprocessing}_{model_params}"

    # load model
    config = ViTMAEConfig.from_pretrained("vandijklab/brainlm", subfolder=f"vitmae_{model_params}")
    config.update(model_arguments)
    model = ViTMAEForPreTraining.from_pretrained(
            "vandijklab/brainlm",
            config=config,
            subfolder=f"vitmae_{model_params}",
        ).to(device)

    model = model.half()  # half precision data type
    model.eval()
    # multiple train modes (auto-encoder, causal attention, predict last, etc)
    model.config.train_mode = "auto_encode"

    train_ds = load_from_disk(f"data/processed/{preprocessing}/fmri_development.arrow")
    train_ds.set_transform(transform_func)

    list_subject_id = []
    list_sex = []
    list_ageclass = []
    list_cls_tokens = []
    list_attn_cls_tokens = []
    all_embeddings = []
    # all_index = []
    with torch.no_grad():
        for recording in tqdm(train_ds, desc="Getting CLS tokens"):
            pixel_values = recording["pixel_values"].unsqueeze(0).half().to(device)
            pixel_values = padding_timeseries_For_vitmae(pixel_values, model.config.image_size)

            encoder_output = model.vit(
                pixel_values=pixel_values,
                output_hidden_states=True
            )

            cls_token = encoder_output.last_hidden_state[:,0,:].detach().cpu().numpy()  # torch.Size([1, 256])? (I got 1, 241)
            embedding = encoder_output.last_hidden_state[:,1:,:].detach().cpu().numpy()

            attn_cls_token = get_attention_cls_token(encoder_output.attentions)
            list_subject_id.append(recording['participant_id'])
            list_sex.append(recording['Gender'])
            list_ageclass.append(recording['Child_Adult'])
            list_attn_cls_tokens.append(attn_cls_token)
            list_cls_tokens.append(cls_token)
            all_embeddings.append(embedding)

    # pooling or 
    cls_embeds = np.concatenate(list_cls_tokens, axis=0)
    all_mean_embeddings = [e.mean(axis=1) for e in all_embeddings]
    all_mean_embeddings = np.concatenate(all_mean_embeddings, axis=0)
    all_maxpool_embeddings = [e.max(axis=1) for e in all_embeddings]
    all_maxpool_embeddings = np.concatenate(all_maxpool_embeddings, axis=0)
    
    # save all padded recording
    all_recordings = []
    for idx, batch in enumerate(tqdm(train_ds)):
        signal = batch["pixel_values"]  # (1, 3, num_parcel, timeseries_length)
        recording = signal.flatten(start_dim=1)
        recording = np.array(recording, dtype=np.float32)
        all_recordings.append(recording)

    results = {
        'participant_id': list_subject_id,
        'Gender': list_sex,
        'Child_Adult': list_ageclass,
        'cls_token': list_attn_cls_tokens,
        'cls_embedding': cls_embeds,
        'mean_embedding': all_mean_embeddings,
        'max_embedding': all_maxpool_embeddings,
        'padded_recording': all_recordings
    }
    arrow_results = Dataset.from_dict(results)
    arrow_results.save_to_disk(Path(outputs_path) / "direct_transfer.arrow")
