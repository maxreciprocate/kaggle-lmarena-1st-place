import os
import torch
from transformers import AutoTokenizer
from packing.models.modeling_qwen2 import Qwen2ForSequenceClassification

model_paths = [
    "ckpts/qwen14b_useall_nopv_clean31",
    "ckpts/qwen14b_useall_nopvhs_clean31",
]

output_path = f"ckpts/qwen14b_merge_useall_nopvhs_nopv"

def average_models(model_paths, output_path):
    if not model_paths:
        raise ValueError("At least one model must be provided")

    if len(model_paths) == 1:
        print("Only one model provided, no need to average")
        return model_paths[0]

    models = [Qwen2ForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.float16) for model_path in model_paths]

    model_type = type(models[0])
    if not all(isinstance(model, model_type) for model in models):
        raise TypeError("All models must have the same architecture")

    state_dicts = [model.state_dict() for model in models]
    averaged_state_dict = {}

    for key in state_dicts[0].keys():
        averaged_state_dict[key] = torch.stack([sd[key] for sd in state_dicts]).mean(dim=0)

    models[0].load_state_dict(averaged_state_dict)

    # Save the averaged model
    h = sum(hash(model_path) for model_path in model_paths)

    print(f"Saving averaged model to {output_path}")
    models[0].save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    tokenizer.save_pretrained(output_path)

average_models(model_paths, output_path)
