import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.MultiheadAttention):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, input_size, output_size, num_heads, dropout=0, causal=False, bias=True, add_bias_kv=False, add_zero_attn=False, batch_first=True, groups=None, weight_norm=None):
        super(MultiHeadAttention, self).__init__(input_size, num_heads, dropout=dropout,
                                                 bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        assert(input_size % num_heads == 0)
        assert(input_size == output_size)
        self.causal = causal
        self.batch_first = batch_first

    def set_mask_q(self, masked_tq):
        self.mask_q = masked_tq

    def set_mask_k(self, masked_tk):
        # applies a mask of b x tk length
        self.mask_k = masked_tk

    def forward(self, query, key, value, need_weights=False, static_kv=False):
        key_padding_mask = attn_mask = None
        time_dim = 1 if self.batch_first else 0
        t_q = query.size(time_dim)
        t_k = key.size(time_dim)
        with torch.no_grad():
            if self.causal and t_q > 1:
                attn_mask = torch.full((t_q, t_k), float('-inf'),
                                       device=query.device, dtype=query.dtype).triu_(1)
            key_padding_mask = self.mask_k

        if self.batch_first:
            qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
            kv_same = key.data_ptr() == value.data_ptr()
            key = key.transpose(0, 1)
            if kv_same:
                value = key
            else:
                value = value.transpose(0, 1)
            if qkv_same:
                query = key
            else:
                query = query.transpose(0, 1)
        elif key_padding_mask is not None:
            key_padding_mask.t()

        attn_output, attn_output_weights = super(
            MultiHeadAttention, self).forward(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=need_weights)
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_output_weights