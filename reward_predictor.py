import math
import numpy as np

import torch
import torch.nn as nn


class TransformerEncoderCustom(nn.TransformerEncoderLayer):
    '''Modified to return *attention weights* in addition to output'''

    def forward(self, src, src_mask=None, src_key_padding_mask=None, output_attention=False):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attention = self.self_attn(src, src, src, attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask, need_weights=output_attention)
        if attention is not None:
            attention = attention.detach().clone()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear1(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attention


class PositionalEncoding(nn.Module):
    '''From: https://github.com/pytorch/examples/blob/master/word_language_model/model.py'''

    def __init__(self, d_model, device, max_len=250):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(
            0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).type(
            torch.cuda.FloatTensor) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class RewardPredictor(nn.Module):
    '''The self-attentional reward prediction model from https://arxiv.org/abs/1907.08027'''

    def __init__(
        self,
        observation_dims,
        action_size,
        device,
        dim_feedforward=128,
        n_filters=32,
        pad_val=10,
        max_len=250,
        verbose=False,
    ):
        super(RewardPredictor, self).__init__()

        self.verbose = verbose
        self.device = device
        self.pad_val = pad_val
        self.encoder = nn.Sequential(  # TODO: add dropout?
            nn.Conv2d(observation_dims[0], n_filters, 3, padding=1),
            nn.ReLU()
        )
        encoded_size = np.prod(observation_dims[1:]) * n_filters

        self.linear1 = nn.Sequential(  # TODO: add dropout?
            nn.Linear(encoded_size + action_size, dim_feedforward),
            nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(
            dim_feedforward, self.device, max_len)
        self.self_attention = TransformerEncoderCustom(
            dim_feedforward, nhead=1, dim_feedforward=dim_feedforward, dropout=0.2)
        # 3 reward classes (-1,0,1)
        self.fc_out = nn.Linear(dim_feedforward, 3)

    def forward(self, observations, actions, output_attention=False):
        seq_length, N, _, _, _ = observations.shape
        seq_length, N, action_size = actions.shape

        observations = observations.refine_names('S', 'N', 'C', 'H', 'W')
        observations_flat = observations.align_to('N', 'S', 'C', 'H', 'W')
        observations_flat = observations_flat.flatten(['N', 'S'], 'B').rename(
            None)  # drop names because Conv2D doesn't support them yet

        x = self.encoder(observations_flat)
        x = x.refine_names('B', 'C', 'H', 'W')
        x = x.unflatten('B', [('N', N), ('S', seq_length)])
        x = x.flatten(['C', 'H', 'W'], 'E')
        x = x.align_to('S', 'N', 'E').rename(None)
        x = torch.cat((x, actions), dim=2)
        x = self.linear1(x)
        x = self.pos_encoder(x)

        src_mask = self.generate_square_subsequent_mask(sz=seq_length)
        padding_mask = self.make_padding_mask(observations.rename(None))

        out, attention = self.self_attention(
            x,
            src_mask=src_mask,
            src_key_padding_mask=padding_mask,
            output_attention=output_attention
        )
        out = out.refine_names('S', 'N', 'E').align_to(
            'N', 'S', 'E').rename(None)
        out = self.fc_out(out)

        if output_attention:
            return (out, attention)
        else:
            return out

    # from pytorch Transformer source
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def make_padding_mask(self, batch_of_sequences):
        # batch_of_sequences (S, N, C, H, W)
        # return (N, S)
        padding_mask = batch_of_sequences.transpose(
            0, 1)[:, :, 0, 0, 0] == self.pad_val
        return padding_mask.to(self.device)
