import numpy as np
from typing import Optional, Tuple, Union
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import copy
import warnings
from transformers import T5Config, AutoTokenizer
from transformers.models.t5.modeling_t5 import *
from transformers.modeling_outputs  import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from torch.nn import CrossEntropyLoss
from torch.nn import init

from utils import *


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head: int, d_model: int, d_kv: int):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_q = d_kv
        self.d_kv = d_kv
        self.head = head

        self.fc_q = nn.Linear(d_model, head * d_kv)
        self.fc_k = nn.Linear(d_model, head * d_kv)
        self.fc_v = nn.Linear(d_model, head * d_kv)

    def forward(self, queries, keys, values, group_prob, attention_mask):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.head, self.d_q).permute(0, 2, 1, 3)   # (b_s, h, nq, d_q)
        k = self.fc_k(keys).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 3, 1)     # (b_s, h, nk, d_kv)
        v = self.fc_v(values).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 1, 3)   # (b_s, h, nk, d_kv)

        att = torch.matmul(q, k) / np.sqrt(self.d_kv)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # att.masked_fill(attention_mask == 0, -1e4)
            att.masked_fill(attention_mask == 0, -1e4)

        att = torch.softmax(att, dim=-1)
        att = att * group_prob
        output = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b_s, -1, self.d_model)

        return output
    

class GroupAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.8):
        super(GroupAttention, self).__init__()
        self.h = head
        self.d_k = d_model // head
        self.linear_key = nn.Linear(self.d_k, self.d_k)
        self.linear_query = nn.Linear(self.d_k, self.d_k)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, eos_mask, prior):
        bs, seq_len = context.size()[:2]

        context = self.norm(context).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)

        a = torch.diag(torch.ones(seq_len - 1), 1).long().to(context.device)
        b = torch.diag(torch.ones(seq_len), 0).long().to(context.device)
        c = torch.diag(torch.ones(seq_len - 1), -1).long().to(context.device)

        mask = torch.logical_and(eos_mask, (a+c))
        
        key = self.linear_key(context)
        query = self.linear_query(context)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k
        
        scores = scores.masked_fill(mask == 0, -1e4)
        neibor_attn = F.softmax(scores, dim = -1)
        neibor_attn = torch.sqrt(neibor_attn*neibor_attn.transpose(-2,-1) + 1e-4)
        neibor_attn = prior + (1. - prior)*neibor_attn

        tri_matrix = torch.triu(torch.ones(seq_len, seq_len), diagonal = 0).float().to(context.device)
        tri_matrix = tri_matrix.unsqueeze(0).unsqueeze(0)
        t = torch.log(neibor_attn + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-4)
        
        return g_attn, neibor_attn

# class GroupAttention(nn.Module):
#     """
#     GroupAttention như paper ViWordFormer, đã sửa mask cho đúng shape.
#     """
#     def __init__(self, head, d_model, dropout=0.8):
#         super(GroupAttention, self).__init__()
#         self.h = head
#         self.d_k = d_model // head
#         self.linear_key = nn.Linear(self.d_k, self.d_k)
#         self.linear_query = nn.Linear(self.d_k, self.d_k)
#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, context, eos_mask, prior):
#         """
#         context:  [B, L, d_model]
#         eos_mask: [B, L] (1: token, 0: pad)
#         prior:    0. (layer 1) hoặc [B, h, L, L] (các layer sau)
#         """
#         bs, seq_len = context.size()[:2]

#         # [B, h, L, d_k]
#         context = self.norm(context).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)

#         a = torch.diag(torch.ones(seq_len - 1, device=context.device), 1)
#         b = torch.diag(torch.ones(seq_len, device=context.device), 0)
#         c = torch.diag(torch.ones(seq_len - 1, device=context.device), -1)

#         # neighbor mask (trên & dưới đường chéo): [L,L] -> [1,1,L,L]
#         neighbor = (a + c).bool().unsqueeze(0).unsqueeze(0)  # [1,1,L,L]

#         if eos_mask is not None:
#             token = eos_mask.bool()                              # [B,L]
#             pair_mask = token.unsqueeze(2) & token.unsqueeze(1)  # [B,L,L]
#             pair_mask = pair_mask.unsqueeze(1)                   # [B,1,L,L]
#             mask = neighbor & pair_mask                          # [B,1,L,L]
#         else:
#             mask = neighbor.expand(bs, -1, -1, -1)               # [B,1,L,L]

#         key = self.linear_key(context)      # [B,h,L,d_k]
#         query = self.linear_query(context)  # [B,h,L,d_k]

#         scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k  # [B,h,L,L]
#         scores = scores.masked_fill(~mask, -1e4)

#         neibor_attn = F.softmax(scores, dim=-1)
#         neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(-2, -1) + 1e-4)

#         if isinstance(prior, torch.Tensor):
#             neibor_attn = prior + (1. - prior) * neibor_attn

#         tri_matrix = torch.triu(torch.ones(seq_len, seq_len, device=context.device), diagonal=0).float()
#         tri_matrix = tri_matrix.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]

#         t = torch.log(neibor_attn + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
#         g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
#         g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-4)

#         return g_attn, neibor_attn

    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class ViWordFormerLayer(nn.Module):
    def __init__(self, head, d_model, d_kv, d_ff, dropout=0.1):
        super(ViWordFormerLayer, self).__init__()
        self.group_attn = GroupAttention(head, d_model)
        self.self_attn = ScaledDotProductAttention(head, d_model, d_kv)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.size = d_model
        
    def forward(self, x, mask, group_prob):
        group_prob, break_prob = self.group_attn(x, mask, group_prob)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
        
        return self.sublayer[1](x, self.feed_forward), group_prob, break_prob
    
# ------------------------------------------------------------------
# 3. ViWordFormerEncoder: encode 1 sequence (doc / query / option)
# ------------------------------------------------------------------
class ViWordFormerEncoder(nn.Module):
    def __init__(self, head, d_model, d_kv, d_ff, word_embed):
        super().__init__()
        self.word_embed = word_embed
        self.layers = clones(ViWordFormerLayer(head, d_model, d_kv, d_ff), 3)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs, mask):
        """
        input:    [B, L]
        mask: [B, L] (1: token, 0: pad)
        """
        break_probs = []
        x = self.word_embed(inputs)
        group_prob = 0.
        for layer in self.layers:
            x, group_prob, break_prob = layer(x, mask,group_prob)
            break_probs.append(break_prob)

        x = self.norm(x)
        break_probs = torch.stack(break_probs, dim=1)

        return x, break_probs
        
        
class ViWordFormerViT5Encoder(T5Stack):
    def __init__(self, config: T5Config, embed_tokens=None,
                 max_doc_len: int=0,
                 max_query_len: int=0,
                 max_option_len: int=0):
        super(T5Stack, self).__init__(config)
        
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.viwordformer_encoder = ViWordFormerEncoder(config.num_heads, config.d_model, config.d_kv, config.d_ff, self.embed_tokens)

        self.max_doc_len = max_doc_len
        self.max_query_len = max_query_len
        self.max_option_len = max_option_len
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        doc_ids=None,
        doc_attn_mask=None,
        ques_ids=None,
        ques_attn_mask=None,
        opt_ids=None,
        opt_attn_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        assert doc_ids is not None and ques_ids is not None and opt_ids is not None

        batch_size = doc_ids.size(0)
        
        doc_embs, _ = self.viwordformer_encoder(doc_ids, doc_attn_mask.unsqueeze(1).unsqueeze(1))
        ques_embs, _ = self.viwordformer_encoder(ques_ids, ques_attn_mask.unsqueeze(1).unsqueeze(1))
        opt_embs, _ = self.viwordformer_encoder(opt_ids, opt_attn_mask.unsqueeze(1).unsqueeze(1))
        
        inputs_embeds = torch.cat([doc_embs, ques_embs, opt_embs], dim=1)
        attention_mask = torch.cat([doc_attn_mask, ques_attn_mask, opt_attn_mask], dim=1)
        
        seq_len = inputs_embeds.size(1)

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )
        
        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)
        
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, (batch_size, seq_len))
        
        # If a 2D or 3D attention mask is provided for the cross-attention  
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None
        
        hidden_states = self.dropout(inputs_embeds)
        
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]
            
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
                    
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )            
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

# --------------------------- OCN Modules ---------------------------
class AttentivePooling(nn.Module):

    def __init__(self, input_size):
        super(AttentivePooling, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, input, mask):
        bsz, length, size = input.size()
        score = self.fc(input.contiguous().view(-1, size)).view(bsz, length)
        score = score.masked_fill((~mask), -float('inf'))
        prob = F.softmax(score, dim=-1)
        attn = torch.bmm(prob.unsqueeze(1), input)
        return attn


class TriLinear(nn.Module):

    def __init__(self, input_size):
        super(TriLinear, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(1, input_size))
        self.w2 = nn.Parameter(torch.FloatTensor(1, input_size))
        self.w3 = nn.Parameter(torch.FloatTensor(1, input_size))

        self.init_param()

    def forward(self, query, key):
        ndim = query.dim()
        q_logit = F.linear(query, self.w1)
        k_logit = F.linear(key, self.w2)

        shape = [1] * (ndim - 1) + [-1]
        dot_k = self.w3.view(shape) * key
        dot_logit = torch.matmul(query, torch.transpose(dot_k, -1, -2))

        logit = q_logit + torch.transpose(k_logit, -1, -2) + dot_logit
        return logit

    def init_param(self):
        init.normal_(self.w1, 0., 0.02)
        init.normal_(self.w2, 0., 0.02)
        init.normal_(self.w3, 0., 0.02)


class OCNAttention(nn.Module):

    def __init__(self, sim):
        super(OCNAttention, self).__init__()
        self.sim = sim

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        ndim = query.dim()
        logit = self.sim(query, key)
        if query_mask is not None and key_mask is not None:
            mask = query_mask.unsqueeze(ndim - 1) * key_mask.unsqueeze(ndim - 2)
            logit = logit.masked_fill((~mask), -float('inf'))

        attn_weight = F.softmax(logit, dim=-1)
        if query_mask is not None and key_mask is not None:
            attn_weight = attn_weight.masked_fill((~mask), 0.)

        attn = torch.matmul(attn_weight, value)

        kq_weight = F.softmax(logit, dim=1)
        if query_mask is not None and key_mask is not None:
            kq_weight = kq_weight.masked_fill((~mask), 0.)

        co_weight = torch.matmul(attn_weight, torch.transpose(kq_weight, -1, -2))
        co_attn = torch.matmul(co_weight, query)

        return (attn, attn_weight), (co_attn, co_weight)

class ViWordFormerOCNModel(T5PreTrainedModel):
    def __init__(
        self,
        config: T5Config,
        num_labels: int,
        max_doc_len: int,
        max_query_len: int,
        max_option_len: int,
        explanation_loss_weight: float = 1.0,  # trọng số loss giải thích
    ):
        super().__init__(config)
        self.num_labels = num_labels
        self.max_doc_len = max_doc_len
        self.max_query_len = max_query_len
        self.max_option_len = max_option_len

        self.hidden_size = config.d_model
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # ---- Encoder: ViWordFormer + T5 ----
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = ViWordFormerViT5Encoder(
            encoder_config,
            embed_tokens=self.shared,
            max_doc_len=max_doc_len,
            max_query_len=max_query_len,
            max_option_len=max_option_len,
        )

        # ---- Decoder T5 để sinh explanation ----
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ---- OCN modules (giống MCRCModel) ----
        self.attn_sim = TriLinear(self.hidden_size)
        self.attention = OCNAttention(sim=self.attn_sim)
        self.attn_fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)

        self.opt_attn_sim = TriLinear(self.hidden_size)
        self.opt_attention = OCNAttention(sim=self.opt_attn_sim)
        self.comp_fc = nn.Linear(self.hidden_size * 7, self.hidden_size, bias=True)

        self.query_attentive_pooling = AttentivePooling(input_size=self.hidden_size)
        self.gate_fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)

        self.opt_selfattn_sim = TriLinear(self.hidden_size)
        self.opt_self_attention = OCNAttention(sim=self.opt_selfattn_sim)
        self.opt_selfattn_fc = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=True)

        self.score_fc = nn.Linear(self.hidden_size, 1)

        self.explanation_loss_weight = explanation_loss_weight

        self.post_init()

    # ====== Embedding & getter ======
    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    # ====== Helper: Encode + OCN + fusion (re-use cho forward & generation) ======
    def _encode_and_fuse(
        self,
        doc_ids, doc_mask,
        query_ids, query_mask,
        opt_ids, opt_mask,
    ):
        """
        Input:
          doc_ids:   [B, N, max_doc_len]
          doc_mask:  [B, N, max_doc_len]
          query_ids: [B, N, max_query_len]
          query_mask:[B, N, max_query_len]
          opt_ids:   [B, N, max_option_len]
          opt_mask:  [B, N, max_option_len]
        Return:
          logits: [B, N] (score mỗi option)
          fusion: [B*N, H] (fusion vector từng option, dùng cho explanation)
        """
        B = doc_ids.size(0)
        N = self.num_labels

        # 1) Flatten
        doc_ids_flat   = doc_ids.view(B * N, self.max_doc_len)
        doc_mask_flat  = doc_mask.view(B * N, self.max_doc_len)

        query_ids_flat  = query_ids.view(B * N, self.max_query_len)
        query_mask_flat = query_mask.view(B * N, self.max_query_len)

        opt_ids_flat   = opt_ids.view(B * N, self.max_option_len)
        opt_mask_flat  = opt_mask.view(B * N, self.max_option_len)

        # 2) Encode qua ViWordFormerViT5Encoder
        encoder_outputs = self.encoder(
            doc_ids=doc_ids_flat,
            doc_attn_mask=doc_mask_flat,
            ques_ids=query_ids_flat,
            ques_attn_mask=query_mask_flat,
            opt_ids=opt_ids_flat,
            opt_attn_mask=opt_mask_flat,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        enc = encoder_outputs.last_hidden_state  # [B*N, L_total, H]
        _, L_total, hidden = enc.size()
        assert hidden == self.hidden_size
        assert L_total == self.max_doc_len + self.max_query_len + self.max_option_len

        # 3) Tách doc / query / option
        doc_enc   = enc[:, :self.max_doc_len, :]                         # [B*N, max_doc_len, H]
        query_enc = enc[:, self.max_doc_len:self.max_doc_len + self.max_query_len, :]
        opt_enc   = enc[:, self.max_doc_len + self.max_query_len:, :]    # [B*N, max_option_len, H]

        doc_mask_bool   = doc_mask_flat > 0
        query_mask_bool = query_mask_flat > 0
        opt_mask_bool   = opt_mask_flat > 0

        # 4) Query attentive pooling
        query_attn = self.query_attentive_pooling(query_enc, query_mask_bool)   # [B*N, 1, H]

        # 5) reshape option cho OCN
        opt_total_len = opt_enc.size(1)
        opt_mask_comp = opt_mask_bool.view(B, N, opt_total_len)
        opt_enc_comp  = opt_enc.view(B, N, opt_total_len, self.hidden_size)

        # 6) Option Comparison (OCN)
        correlation_list = []
        for i in range(self.num_labels):
            cur_opt  = opt_enc_comp[:, i, :, :]   # [B, L_opt, H]
            cur_mask = opt_mask_comp[:, i, :]     # [B, L_opt]

            comp_info = []
            for j in range(self.num_labels):
                if j == i:
                    continue
                tmp_opt  = opt_enc_comp[:, j, :, :]
                tmp_mask = opt_mask_comp[:, j, :]

                (attn, _), _ = self.opt_attention(cur_opt, tmp_opt, tmp_opt, cur_mask, tmp_mask)
                comp_info.append(cur_opt * attn)
                comp_info.append(cur_opt - attn)

            correlation = torch.tanh(self.comp_fc(torch.cat([cur_opt] + comp_info, dim=-1)))
            correlation_list.append(correlation)

        correlation_list = [c.unsqueeze(1) for c in correlation_list]   # list[B,1,L,H]
        opt_correlation = torch.cat(correlation_list, dim=1)           # [B,N,L,H]

        # 7) Gate với query context
        opt_mask_bool_flat = opt_mask_comp.view(B * N, opt_total_len)
        opt_enc_flat = opt_enc_comp.view(B * N, opt_total_len, self.hidden_size)
        opt_correlation_flat = opt_correlation.view(B * N, opt_total_len, self.hidden_size)

        gate = torch.sigmoid(
            self.gate_fc(torch.cat((opt_enc_flat, opt_correlation_flat, query_attn.expand_as(opt_enc_flat)), -1))
        )
        option = opt_enc_flat * gate + opt_correlation_flat * (1.0 - gate)   # [B*N, L_opt, H]

        # 8) Co-attention với doc_enc
        (attn_ctx, _), (coattn, _) = self.attention(option, doc_enc, doc_enc, opt_mask_bool_flat, doc_mask_bool)

        fusion = self.attn_fc(torch.cat((option, attn_ctx, coattn), -1))
        fusion = F.relu(fusion)

        # 9) Self-attention
        (attn_self, _), _ = self.opt_self_attention(fusion, fusion, fusion, opt_mask_bool_flat, opt_mask_bool_flat)
        fusion = self.opt_selfattn_fc(torch.cat((fusion, attn_self, fusion * attn_self, fusion - attn_self), -1))
        fusion = F.relu(fusion)
        
        fusion_seq = fusion                             # [B*N, L_opt, H]

        # 10) Max-pooling time
        fusion = fusion.masked_fill(
            (~opt_mask_bool_flat).unsqueeze(-1).expand(B * N, opt_total_len, self.hidden_size),
            -float('inf'),
        )
        fusion, _ = fusion.max(dim=1)       # [B*N, H]
        fusion_pooled = fusion

        # 11) Scoring
        logits = self.score_fc(fusion).view(B, N)   # [B,N]

        return logits, fusion_seq, fusion_pooled, encoder_outputs

    # ====== Forward ======
    def forward(
        self,
        doc_ids,        # [B, num_labels, max_doc_len]
        doc_mask,       # [B, num_labels, max_doc_len]
        query_ids,      # [B, num_labels, max_query_len]
        query_mask,     # [B, num_labels, max_query_len]
        opt_ids,        # [B, num_labels, max_option_len]
        opt_mask,       # [B, num_labels, max_option_len]
        labels=None,                    # [B] - index đáp án
        explanation_labels=None,        # [B, L_y] - token ids giải thích (pad->-100)
        decoder_input_ids=None,
        decoder_attention_mask=None,
        return_dict: bool = True,
        **kwargs
    ):

        B = doc_ids.size(0)

        # 1) Encode + OCN
        logits, fusion_seq, fusion_pooled, encoder_outputs = self._encode_and_fuse(
            doc_ids, doc_mask,
            query_ids, query_mask,
            opt_ids, opt_mask,
        )

        # ----- Answer loss -----
        answer_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            answer_loss = loss_fct(logits, labels)

        # ----- Explanation loss -----
        explanation_loss = None
        lm_logits = None

        if explanation_labels is not None:
            # nếu mọi token đều -100 (không có data), bỏ qua để tránh NaN
            if (explanation_labels != -100).any():
                if labels is not None:
                    chosen_idx = labels            # [B]
                else:
                    chosen_idx = logits.argmax(dim=-1)  # [B]
                    
                B = doc_ids.size(0)
                N = self.num_labels
                L_opt = opt_ids.size(2)
                H = self.hidden_size

                # fusion_seq: [B*N, L_opt, H] -> [B, N, L_opt, H]
                fusion_seq_4d = fusion_seq.view(B, N, L_opt, H)
                opt_mask_bool = (opt_mask > 0)          # [B, N, L_opt]
                
                idx = torch.arange(B, device=fusion_seq.device)
                
                # sequence representation của option được chọn
                chosen_seq = fusion_seq_4d[idx, chosen_idx]    # [B, L_opt, H]
                chosen_mask = opt_mask_bool[idx, chosen_idx]   # [B, L_opt]

                encoder_hidden_for_expl = chosen_seq           # [B, L_opt, H]
                encoder_attention_mask_expl = chosen_mask.long()  # [B, L_opt]

                if decoder_input_ids is None:
                    decoder_input_ids = self._shift_right(explanation_labels)

                if decoder_attention_mask is None and decoder_input_ids is not None:
                    decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()

                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=encoder_hidden_for_expl,
                    encoder_attention_mask=encoder_attention_mask_expl,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                sequence_output = decoder_outputs.last_hidden_state   # [B,L,H]
                if self.config.tie_word_embeddings:
                    sequence_output = sequence_output * (self.model_dim ** -0.5)
                lm_logits = self.lm_head(sequence_output)             # [B,L,V]

                expl_loss_fct = CrossEntropyLoss(ignore_index=-100)
                explanation_loss = expl_loss_fct(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    explanation_labels.view(-1),
                )

        # ----- Total loss -----
        total_loss = None
        if (answer_loss is not None) and (explanation_loss is not None):
            total_loss = answer_loss + self.explanation_loss_weight * explanation_loss
        elif answer_loss is not None:
            total_loss = answer_loss
        elif explanation_loss is not None:
            total_loss = explanation_loss

        if not return_dict:
            output = (logits,)
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=logits,  # classification logits
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # ====== Generation: sinh explanation (greedy) ======
    @torch.no_grad()
    def generate_explanation(
        self,
        doc_ids,
        doc_mask,
        query_ids,
        query_mask,
        opt_ids,
        opt_mask,
        chosen_idx,  
        max_length,
        tokenizer,
        device,
    ):
        self.eval()
        B = doc_ids.size(0)
        N = self.num_labels
        L_opt = opt_ids.size(2)
        H = self.hidden_size

        # Lấy logits, fusion từ helper
        logits, fusion_seq, fusion_pooled, encoder_outputs = self._encode_and_fuse(
            doc_ids, doc_mask,
            query_ids, query_mask,
            opt_ids, opt_mask,
        )

        # Chuẩn bị context cho decoder giống forward
        fusion_seq_4d = fusion_seq.view(B, N, L_opt, H)   # [B,N,L_opt,H]
        opt_mask_bool = (opt_mask > 0)                    # [B,N,L_opt]

        idx = torch.arange(B, device=device)
        chosen_seq = fusion_seq_4d[idx, chosen_idx]       # [B,L_opt,H]
        chosen_mask = opt_mask_bool[idx, chosen_idx]      # [B,L_opt]

        encoder_hidden_for_expl = chosen_seq              # [B,L_opt,H]
        encoder_attention_mask_expl = chosen_mask.long()  # [B,L_opt]
        
        # decoder start token (dùng pad hoặc decoder_start_token_id nếu có)
        start_id = (
            self.config.decoder_start_token_id
            if self.config.decoder_start_token_id is not None
            else tokenizer.pad_token_id
        )
        generated = torch.full(
            (B, 1),
            start_id,
            dtype=torch.long,
            device=device,
        )

        for _ in range(max_length - 1):
            decoder_outputs = self.decoder(
                input_ids=generated,
                encoder_hidden_states=encoder_hidden_for_expl,
                encoder_attention_mask=encoder_attention_mask_expl,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            seq_out = decoder_outputs.last_hidden_state      # [B,L,H]
            if self.config.tie_word_embeddings:
                seq_out = seq_out * (self.model_dim ** -0.5)
            lm_logits = self.lm_head(seq_out[:, -1:, :])     # [B,1,V]
            next_token = lm_logits.argmax(-1)                # [B,1]
            generated = torch.cat([generated, next_token], dim=1)

        explanations = []
        for i in range(B):
            text = tokenizer.decode(generated[i], skip_special_tokens=True)
            explanations.append(text.strip())
        return explanations, logits