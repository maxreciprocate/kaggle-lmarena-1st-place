# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_utils import PreTrainedModel
from transformers import Qwen2PreTrainedModel, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.activations import ACT2FN

from .ops.rms_norm import rms_norm
from .ops.silu_and_mul import silu_and_mul_fwd
from .ops.fused_rotary_emb import fused_rotary_emb
from .ops.flash_attention_nopad import context_attention_fwd

class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # return down_proj
        gate_up = self.gate_up_proj(x)
        with torch.cuda.device(x.device):
            gate_out = silu_and_mul_fwd(gate_up)
        return self.down_proj(gate_out)

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        with torch.cuda.device(x.device):
            return rms_norm(x.contiguous(), self.weight, self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, seq_info, inv_freq):
        q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(q_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)

        with torch.cuda.device(hidden_states.device):
            query_states, key_states = fused_rotary_emb(
                query_states[None],
                key_states[None],
                seq_info["position_ids"],
                inv_freq=inv_freq,
                scaling_factor=1.0,
                out_q=query_states[None],
                out_k=key_states[None],
            )

        query_states = query_states[0]
        key_states = key_states[0]

        cu_seqlens = seq_info["cu_seqlens"]
        max_seq_len = seq_info["max_seq_len"]
        with torch.cuda.device(hidden_states.device):
            context_attention_fwd(
                query_states,
                key_states,
                value_states,
                query_states,  # write to query_states
                cu_seqlens[:-1],
                cu_seqlens[1:] - cu_seqlens[:-1],
                max_seq_len,
            )
        attn_output = query_states.reshape(q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #if config.sliding_window and config._attn_implementation != "flash_attention_2":
        #    logger.warning_once(
        #        f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
        #        "unexpected results may be encountered."
        #    )
        self.sliding_window = False

    def forward(self, hidden_states, seq_info, inv_freq):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states, seq_info, inv_freq)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        #self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, seq_info, inv_freq):
        inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        hidden_states = inputs_embeds

        for decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, seq_info, inv_freq)

        hidden_states = self.norm(hidden_states)

        return hidden_states

class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, 2, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(self, input_ids, seq_info, inv_freq):
        assert input_ids.size(0) == 1
        hidden_states = self.model(input_ids.squeeze(0), seq_info, inv_freq)

        last_token_inds = seq_info["cu_seqlens"][1:] - 1
        hidden_states = hidden_states[last_token_inds]

        logits = self.score(hidden_states)
        logits = logits.float()
        return logits

    def forward_part1(self, input_ids, seq_info, inv_freq, show=1):
        input_ids = input_ids.squeeze(0)
        model = self.model
        inputs_embeds = model.embed_tokens(input_ids)
        # embed positions
        hidden_states = inputs_embeds

        n = len(model.layers)
        for decoder_layer in model.layers[: n // 2]:
            for _ in range(show):
                hidden_states = decoder_layer(hidden_states, seq_info, inv_freq)

        return hidden_states

    def forward_part2(self, hidden_states, seq_info, inv_freq, show=1):
        model = self.model
        for decoder_layer in model.layers[len(model.layers) // 2 :]:
            for _ in range(show):
                hidden_states = decoder_layer(hidden_states, seq_info, inv_freq)

        hidden_states = model.norm(hidden_states)

        last_token_inds = seq_info["cu_seqlens"][1:] - 1
        hidden_states = hidden_states[last_token_inds]

        logits = self.score(hidden_states)
        logits = logits.float()
        return logits

