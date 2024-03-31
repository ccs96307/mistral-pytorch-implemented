from typing import OrderedDict, Optional, List, Tuple, Union

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import WEIGHTS_NAME
from transformers.utils.hub import cached_file

from config import MistralConfig
from cache_utils import Cache, DynamicCache
from modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


def _my_prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
):
    """
    Prepares the correct `attention_mask` argument to be used by `torch.nn.functional.scaled_dot_product_attention`.
    """
    batch_size, query_length = input_shape
    key_value_length = query_length + past_key_values_length

    if attention_mask is None:
        # Creating a causal mask for all positions
        mask = torch.full((query_length, key_value_length), torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device)
        mask_cond = torch.arange(key_value_length, device=inputs_embeds.device)
        mask[:, :query_length] = (mask_cond[None, :] < (mask_cond[:query_length] + 1)[:, None]).float()
        expanded_4d_mask = mask[None, None, :, :].expand(batch_size, 1, query_length, key_value_length)
    elif len(attention_mask.shape) == 4:
        # If a 4D attention mask is already provided, use it directly
        expanded_4d_mask = attention_mask
    else:
        # Expanding a 2D mask to 4D
        expanded_mask = attention_mask[:, None, None, :]
        expanded_4d_mask = expanded_mask.expand(batch_size, 1, query_length, key_value_length).to(dtype=inputs_embeds.dtype)

        # Apply causal mask to the expanded mask
        causal_mask = torch.triu(torch.ones((query_length, key_value_length), device=inputs_embeds.device, dtype=torch.bool), diagonal=1)
        padding_mask = attention_mask == 0
        padding_mask = padding_mask.view(batch_size, 1, 1, key_value_length)
        expanded_4d_mask = expanded_4d_mask.masked_fill(~padding_mask, 0.)
        expanded_4d_mask = expanded_4d_mask.masked_fill(padding_mask, torch.finfo(inputs_embeds.dtype).min)
        expanded_4d_mask = expanded_4d_mask.masked_fill(causal_mask, torch.finfo(inputs_embeds.dtype).min)

    return expanded_4d_mask


def repeat_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_kv_groups: int,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_size = key_states.size()
    key_states = key_states[:, :, None, :, :].expand(
        batch_size,
        num_heads,
        num_kv_groups,
        seq_len,
        head_size,
    )

    value_states = value_states[:, :, None, :, :].expand(
        batch_size,
        num_heads,
        num_kv_groups,
        seq_len,
        head_size,
    )

    return (
        key_states.reshape(batch_size, num_heads * num_kv_groups, seq_len, head_size),
        value_states.reshape(batch_size, num_heads * num_kv_groups, seq_len, head_size),
    )


class MistralRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6) -> None:
        """
        The RMSNorm is implemented according `modeling_mistral.py`.
        It is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(original_input_dtype)


class MistralRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_size: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        super().__init__()
        self.theta = 1 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
        self.theta = torch.cat([self.theta, self.theta], dim=-1).to(device)
        self.position_ids = torch.arange(0, max_position_embeddings).to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        unsqueeze_dim: int = 1,
    ) -> torch.Tensor:
        position_maxtrix = torch.outer(self.position_ids, self.theta)
        cos = torch.cos(position_maxtrix)
        sin = torch.sin(position_maxtrix)

        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)

        x1 = hidden_states[..., :hidden_states.shape[-1] // 2]
        x2 = hidden_states[..., hidden_states.shape[-1] // 2 :]
        _x = torch.cat([-x2, x1], dim=-1)

        out = hidden_states * cos + _x * sin

        return out
    

class MistralAttention(torch.nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None) -> None:
        super().__init__()

        # Init
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_q_heads // self.num_kv_heads
        self.head_size = config.hidden_size // self.num_q_heads
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.layer_idx = layer_idx

        # QKVO Layer
        self.q_proj = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
        )
        self.k_proj = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_kv_heads * self.head_size,
            bias=False,
        )
        self.v_proj = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_kv_heads * self.head_size,
            bias=False,
        )
        self.o_proj = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
        )

        # RoPE
        self.rotary_emb = MistralRotaryEmbedding(
            head_size=self.head_size,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Init
        batch_size, seq_len, hidden_size = hidden_states.size()

        # QKV
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(batch_size, seq_len, self.num_q_heads, self.head_size).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)

        # KV Cache
        kv_seq_len = key_states.size(2)
        if past_key_value is not None and self.layer_idx is not None:
            kv_seq_len += past_key_value.get_usable_length(
                new_seq_length=kv_seq_len,
                layer_idx=self.layer_idx,
            )

        query_states = self.rotary_emb(
            hidden_states=query_states,
            position_ids=position_ids,
        )
        key_states = self.rotary_emb(
            hidden_states=key_states,
            position_ids=position_ids,
        )

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states=key_states,
                value_states=value_states,
                layer_idx=self.layer_idx,
            )
        
        # Repeat kv heads
        key_states, value_states = repeat_kv(
            key_states=key_states,
            value_states=value_states,
            num_kv_groups=self.num_kv_groups,
        )

        # Attention weights (Q * K^T)
        attention_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_size)

        # Attention mask
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask

        # Upcast attention to fp32
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_weights = torch.nn.functional.dropout(attention_weights, p=self.attention_dropout, training=self.training)

        # Attention output (A = Q * K^T, A * V)
        attention_output = torch.matmul(attention_weights, value_states).reshape(batch_size, seq_len, self.hidden_size)
        attention_output = self.o_proj(attention_output)

        if not output_attentions:
            attention_weights = None

        return attention_output, attention_weights, past_key_value


class MistralSdpaAttention(MistralAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        
        batch_size, seq_len, hidden_size = hidden_states.size()

        # QKV
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(batch_size, seq_len, self.num_q_heads, self.head_size).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)

        # KV Cache
        query_states = self.rotary_emb(
            hidden_states=query_states,
            position_ids=position_ids,
        )
        key_states = self.rotary_emb(
            hidden_states=key_states,
            position_ids=position_ids,
        )

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states=key_states,
                value_states=value_states,
                layer_idx=self.layer_idx,
            )
        
        # Repeat kv heads
        key_states, value_states = repeat_kv(
            key_states=key_states,
            value_states=value_states,
            num_kv_groups=self.num_kv_groups,
        )

        # Contiguous
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # SDPA
        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and seq_len > 1,
        )

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, hidden_size)
        attention_output = self.o_proj(attention_output)

        return attention_output, None, past_key_value
    

class MistralMLP(torch.nn.Module):
    def __init__(self, config: MistralConfig) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=False,
        )
        self.up_proj = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=False,
        )
        self.down_proj = torch.nn.Linear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=False,
        )
        self.act_fn = torch.nn.functional.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_output = self.up_proj(x)
        gate_output = self.gate_proj(x)
        intermediate_output = self.act_fn(gate_output) * up_output
        down_output = self.down_proj(intermediate_output)
        return down_output
    

class MistralDecoderLayer(torch.nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = MistralSdpaAttention(config=config, layer_idx=layer_idx)
        self.mlp = MistralMLP(config=config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attention_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Redisual connection
        hidden_states = hidden_states + residual

        # Fully connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attention_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MistralModel(torch.nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self._attn_implementation = config._attn_implementation

        self.embed_tokens = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = torch.nn.ModuleList([MistralDecoderLayer(config=config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = MistralRMSNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Config
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Init
        batch_size, seq_length = input_ids.shape
        past_key_values_length = 0

        # If use cache
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values=past_key_values)

            past_key_values_length = past_key_values.get_usable_length(seq_length)

        # Position ids
        if position_ids is None:
            position_ids = torch.arange(
                start=past_key_values_length,
                end=past_key_values_length + seq_length,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # Input embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Attention mask (output_attentions is not supported when using SDPA)
        if self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _my_prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            raise NotImplementedError(
                "_my_prepare_4d_causal_attention_mask() if not implemented for now.",
            )
        
        # Feed-Forward
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            # No matter how many data returned, the first one is the `hidden_states`
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    

class MyMistralForCausalLM(torch.nn.Module):
    def __init__(self, config: MistralConfig) -> None:
        super().__init__()
        # Settings
        self.config = config
        self.use_cache = config.use_cache
        self.eos_token_id = config.eos_token_id

        self.model = MistralModel(config=config)
        self.lm_head = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Settings
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)

        # decoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,   
        )

        hidden_states = outputs.last_hidden_state
        past_key_values = outputs.past_key_values
        logits = self.lm_head(hidden_states).float()

        # Loss
        loss = None
        if labels is not None:
            criterion = torch.nn.CrossEntropyLoss()

            # Shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Make sure they are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss = criterion(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attention_mask,
        )
    
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str) -> "MyMistralForCausalLM":
        """Load pretrained weights from HuggingFace into model.
        
        Args:
            pretrained_model_name_or_path: One of
                * "echarlaix/tiny-random-mistral"
                * "mistralai/Mistral-7B-Instruct-v0.2"
                ...

        Returns:
            model: MyMistralModelForCausalLM model with weights loaded
        """

        def load_state_dict_hf(path_or_repo_id: str) -> OrderedDict:
            resolved_archive_file = cached_file(
                path_or_repo_id=path_or_repo_id,
                filename=WEIGHTS_NAME,
            )
            return torch.load(resolved_archive_file, weights_only=True)

        # Load config
        config = MistralConfig.from_pretrained_model_or_path(pretrained_model_name_or_path=pretrained_model_name_or_path)

        # Load weights
        state_dict = load_state_dict_hf(pretrained_model_name_or_path)

        # Load model
        model = MyMistralForCausalLM(config=config)
        model.load_state_dict(state_dict=state_dict)

        return model
    

    @torch.no_grad
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 20,
        top_p: float = 0.9,
        top_k: int = 10,
        no_repeat_ngram_size: int = 2,
        early_stopping: bool = False,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # Prepare input
        batch_size = input_ids.shape[0]
        past_key_values = None
        generation_mode = "greedy"
        finished = torch.zeros(batch_size, dtype=torch.bool)
        all_sequences = input_ids
        use_cache = use_cache if use_cache is not None else self.use_cache

        # Position ids
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        # Greedy search
        if generation_mode == "greedy":
            for idx in range(max_length):
                outputs = self(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=self.use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                past_key_values = outputs.past_key_values if use_cache else None
                lm_logits = outputs.logits[:, -1, :]

                # Next token
                next_token = torch.argmax(lm_logits, dim=-1, keepdim=True)

                # Determine finished
                just_finished = next_token.squeeze(-1) == self.eos_token_id
                finished = finished | just_finished

                # Update input_ids
                next_token = torch.where(
                    condition=finished.unsqueeze(-1),
                    input=torch.full_like(next_token, self.eos_token_id),
                    other=next_token,
                )
                all_sequences = torch.cat([all_sequences, next_token], dim=1)

                if use_cache:
                    input_ids = next_token
                else:
                    input_ids = all_sequences

                # Update position_ids
                new_position_ids = position_ids[:, -1:] + 1
                new_position_ids = torch.where(
                    condition=finished.unsqueeze(-1),
                    input=torch.ones_like(new_position_ids),
                    other=new_position_ids,
                )

                if use_cache:
                    position_ids = new_position_ids
                else:
                    position_ids = torch.cat([position_ids, new_position_ids], dim=1)

                # Update attention_mask
                new_attention_mask_column = torch.ones((batch_size, 1), device=input_ids.device, dtype=torch.long)
                attention_mask = torch.cat([attention_mask, new_attention_mask_column], dim=1)

                if finished.all():
                    break
            
            return all_sequences


if __name__ == "__main__":
    # Settings
    pretrained_model_name_or_path = "echarlaix/tiny-random-mistral"
    config = MistralConfig.from_pretrained_model_or_path(pretrained_model_name_or_path=pretrained_model_name_or_path)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.eos_token_id = config.eos_token_id
    tokenizer.pad_token_id = config.pad_token_id

    # Model
    official_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path).eval()
    custom_model = MyMistralForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path).eval()

    # Testing
    texts = [
        "Today is a nice day.",
        "I want to go to play, do you want to join us?",
        "???",
    ]
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    print("custom_model == official_model:", torch.allclose(custom_model(**inputs).logits, official_model(**inputs).logits, atol=1e-12))
