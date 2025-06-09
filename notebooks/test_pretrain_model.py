from hfplayground.brainlm_mae.modeling_brainlm import BrainLMForPretraining, ViTMAEForPreTraining

model = BrainLMForPretraining.from_pretrained("models/old_13M/")
model = BrainLMForPretraining.from_pretrained("BrainLM/pretrained_models/2023-06-06-22_15_00-checkpoint-1400/")
model = ViTMAEForPreTraining.from_pretrained("models/vitmae_111M")
model = ViTMAEForPreTraining.from_pretrained("models/vitmae_650M")
