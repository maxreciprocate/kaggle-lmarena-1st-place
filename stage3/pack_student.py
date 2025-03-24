import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from inference.packing.models.modeling_qwen2_pack import Qwen2ForSequenceClassification as Qwen2ForSequenceClassificationPack

from argparse import ArgumentParser
argsp = ArgumentParser()
argsp.add_argument("-i")
argsp.add_argument("-o")
args = argsp.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.i)
model = AutoModelForSequenceClassification.from_pretrained(
    args.i,
    torch_dtype=torch.float16,
)

state_dict = model.state_dict()

for idx, layer in enumerate(model.model.layers):
    state_dict[f"model.layers.{idx}.mlp.gate_up_proj.weight"] = torch.cat(
        [
            state_dict.pop(f"model.layers.{idx}.mlp.gate_proj.weight"),
            state_dict.pop(f"model.layers.{idx}.mlp.up_proj.weight"),
        ],
        dim=0,
    )

packed_model = Qwen2ForSequenceClassificationPack(AutoConfig.from_pretrained(args.i))
packed_model.half()
packed_model.load_state_dict(state_dict=state_dict)

print(packed_model)
print(next(packed_model.parameters()).dtype)

tokenizer.save_pretrained(args.o)
packed_model.save_pretrained(args.o)
