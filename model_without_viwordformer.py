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
from transformers.models.t5.modeling_t5 import (
    T5LayerNorm,
    T5Stack,
    T5Block,
    T5PreTrainedModel
)
from transformers.modeling_outputs  import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from torch.nn import CrossEntropyLoss
from torch.nn import init

from utils import *

        
class ViT5Encoder(T5Stack):
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
        
        input_ids = torch.cat([doc_ids, ques_ids, opt_ids], dim=1)
        attention_mask = torch.cat([doc_attn_mask, ques_attn_mask, opt_attn_mask], dim=1)
        
        inputs_embeds = self.embed_tokens(input_ids)
        
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

class viT5OCNModel(T5PreTrainedModel):
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

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = ViT5Encoder(
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
        
        enc_seq = torch.cat([doc_enc, query_enc, fusion_seq], dim=1)

        return logits, enc_seq, fusion_pooled, encoder_outputs

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
                fusion_seq_4d = fusion_seq.view(B, N, -1, H)
                query_mask_bool = (query_mask > 0)          # [B, N, L_opt]
                doc_mask_bool = (doc_mask > 0)          # [B, N, L_opt]
                opt_mask_bool = (opt_mask > 0)
                
                idx = torch.arange(B, device=fusion_seq.device)
                
                # sequence representation của option được chọn
                chosen_seq = fusion_seq_4d[idx, chosen_idx]    # [B, L_opt, H]
                chosen_query_mask = query_mask_bool[idx, chosen_idx]   # [B, L_opt]
                chosen_doc_mask = doc_mask_bool[idx, chosen_idx]   # [B, L_opt]
                chosen_opt_mask = opt_mask_bool[idx, chosen_idx]   # [B, L_opt]
                chosen_mask = torch.cat([chosen_doc_mask, chosen_query_mask, chosen_opt_mask], dim=-1)

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
        # fusion_seq: [B*N, L_opt, H] -> [B, N, L_opt, H]
        fusion_seq_4d = fusion_seq.view(B, N, -1, H)
        query_mask_bool = (query_mask > 0)          # [B, N, L_opt]
        doc_mask_bool = (doc_mask > 0)          # [B, N, L_opt]
        opt_mask_bool = (opt_mask > 0)

        idx = torch.arange(B, device=fusion_seq.device)        
        # sequence representation của option được chọn
        chosen_seq = fusion_seq_4d[idx, chosen_idx]    # [B, L_opt, H]
        chosen_query_mask = query_mask_bool[idx, chosen_idx]   # [B, L_opt]
        chosen_doc_mask = doc_mask_bool[idx, chosen_idx]   # [B, L_opt]
        chosen_opt_mask = opt_mask_bool[idx, chosen_idx]   # [B, L_opt]
        chosen_mask = torch.cat([chosen_doc_mask, chosen_query_mask, chosen_opt_mask], dim=-1)

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