from typing import List

import json

from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file


class MistralConfig:
    def __init__(
        self,
        _name_or_path: str = "echarlaix/tiny-random-mistral",
        _attn_implementation: str = "sdpa",
        architectures: List[str] = [
            "MistralForCausalLM"
        ],
        attention_dropout: float = 0.0,
        attention_probs_dropout_prob: float = 0.1,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        hidden_size: int = 32,
        initializer_range: float = 0.02,
        intermediate_size: int = 37,
        is_decoder: bool = True,
        max_position_embeddings: int = 512,
        model_type: str = "mistral",
        num_attention_heads: int = 4,
        num_hidden_layers: int = 2,
        num_key_value_heads: int = 2,
        pad_token_id: int = 0,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 10000.0,
        sliding_window: int = 4096,
        tie_word_embeddings: bool = False,
        torch_dtype: str = "float32",
        transformers_version: str = "4.39.1",
        type_vocab_size: int = 16,
        use_cache: bool = True,
        vocab_size: int = 32000,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> None:
        self._name_or_path = _name_or_path
        self._attn_implementation = _attn_implementation
        self.architectures = architectures
        self.attention_dropout = attention_dropout
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.is_decoder = is_decoder
        self.max_position_embeddings = max_position_embeddings
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.pad_token_id = pad_token_id
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.type_vocab_size = type_vocab_size
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        
    @staticmethod
    def from_pretrained_model_or_path(pretrained_model_name_or_path: str) -> "MistralConfig":
        resolved_archive_file = cached_file(
            path_or_repo_id=pretrained_model_name_or_path,
            filename=CONFIG_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        
        config_content = json.load(open(resolved_archive_file))
        return MistralConfig(**config_content)        