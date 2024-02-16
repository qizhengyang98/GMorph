from pathlib import Path
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
# from transformers.models.bert.modeling_bert import BertLayer, BertPooler
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.activations import ACT2FN


class Squeeze_before_Linear(nn.Module):
    def forward(self, x):
        return x[:, 0, :] if len(x.shape)==3 else x
    
class Tuple2Tensor(nn.Module):
    def forward(self, x):
        return x[0] if isinstance(x, Tuple) else x
    
class BertLayer_Tuple2Tensor(nn.Module):
    def __init__(self, bert_config):
        super(BertLayer_Tuple2Tensor, self).__init__()
        self.bertLayer = BertLayer(bert_config)
        
    def forward(self, x):
        out = self.bertLayer(x)
        return out[0]

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length : int=0
    ):
        assert input_ids is not None
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, 0 : seq_length]

        token_type_ids = torch.zeros(input_shape, dtype=torch.long).cuda()

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds #+ token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        # new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # print(new_x_shape)
        # new_x_shape = torch.Size([1, 128, 16, 64]) if self.num_attention_heads==16 else torch.Size([1, 128, 12, 64])
        x = x.view(1,128,16,64) if self.num_attention_heads==16 else x.view(1,128,12,64)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions:bool=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # print(new_context_layer_shape)
        # context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = context_layer.view(1,128,1024) if self.num_attention_heads==16 else context_layer.view(1,128,768)

        outputs = (context_layer,)

        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions:bool=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions:bool=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class TransformHW(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformHW, self).__init__()
        self.input_size = input_size
        if len(output_size) == 4:
            self.output_size = (output_size[2], output_size[3])
        else:
            self.output_size = (output_size[2],)
    
    def forward(self, x):
        if len(self.output_size) > 1:
            return F.interpolate(x, size=self.output_size, mode='bilinear')
        else:
            return F.interpolate(x, size=self.output_size, mode='linear')

class glue_origin(nn.Module):
    def __init__(self):
        super(glue_origin, self).__init__()
        bert_config_1 = AutoConfig.from_pretrained(Path("../transformer_model/cola/checkpoint"))
        bert_config_2 = AutoConfig.from_pretrained(Path("../transformer_model/sst2/checkpoint"))
        
        self.task1 = nn.Sequential(
            BertEmbeddings(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1), 
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertPooler(bert_config_1),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(1024, 2, bias=True),
        )
        self.task2 = nn.Sequential(
            BertEmbeddings(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2), 
            BertLayer_Tuple2Tensor(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2),
            BertLayer_Tuple2Tensor(bert_config_2),
            BertPooler(bert_config_2),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(768, 2, bias=True),
        )
    
    def forward(self, x):
        out1 = self.task1(x)
        out2 = self.task2(x)
        return out1, out2
    
class glue_SA_t002(nn.Module):
    def __init__(self):
        super(glue_SA_t002, self).__init__()
        bert_config_1 = AutoConfig.from_pretrained(Path("../transformer_model/cola/checkpoint"))
        bert_config_2 = AutoConfig.from_pretrained(Path("../transformer_model/sst2/checkpoint"))
        
        self.shared = nn.Sequential(
            BertEmbeddings(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1), 
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
            BertLayer_Tuple2Tensor(bert_config_1),
        )
        
        self.task1 = nn.Sequential(
            BertPooler(bert_config_1),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(1024, 2, bias=True),
        )
        self.task2 = nn.Sequential(
            TransformHW((1,128,1024), (1,128,768)),
            BertPooler(bert_config_2),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(768, 2, bias=True),
        )
    
    def forward(self, x):
        out = self.shared(x)
        out1 = self.task1(out)
        out2 = self.task2(out)
        return out1, out2