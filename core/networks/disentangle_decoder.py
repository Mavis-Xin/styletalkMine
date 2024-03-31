import torch
from torch import nn

from .transformer import (
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from core.networks.dynamic_fc_decoder import DynamicFCDecoderLayer, DynamicFCDecoder
from core.utils import _reset_parameters


def get_decoder_network(
    network_type,
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    activation,
    normalize_before,
    num_decoder_layers,
    return_intermediate_dec,
    dynamic_K,
    dynamic_ratio,
):
    decoder = None
    if network_type == "TransformerDecoder":
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        norm = nn.LayerNorm(d_model)
        decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            norm,
            return_intermediate_dec,
        )
    elif network_type == "DynamicFCDecoder":
        d_style = d_model
        decoder_layer = DynamicFCDecoderLayer(
            d_model,
            nhead,
            d_style,
            dynamic_K,
            dynamic_ratio,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        norm = nn.LayerNorm(d_model)
        decoder = DynamicFCDecoder(decoder_layer, num_decoder_layers, norm, return_intermediate_dec)
    else:
        raise ValueError(f"Invalid network_type {network_type}")

    return decoder


class DisentangleDecoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pos_embed_len=80,
        upper_face3d_indices=tuple(list(range(19)) + list(range(46, 51))),
        lower_face3d_indices=tuple(range(19, 46)),
        network_type="None",
        dynamic_K=None,
        dynamic_ratio=None,
        **_,
    ) -> None:
        super().__init__()

        self.upper_face3d_indices = upper_face3d_indices
        self.lower_face3d_indices = lower_face3d_indices

        # upper_decoder_layer = TransformerDecoderLayer(
        #     d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        # )
        # upper_decoder_norm = nn.LayerNorm(d_model)
        # self.upper_decoder = TransformerDecoder(
        #     upper_decoder_layer,
        #     num_decoder_layers,
        #     upper_decoder_norm,
        #     return_intermediate=return_intermediate_dec,
        # )
        self.upper_decoder = get_decoder_network(
            network_type,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            num_decoder_layers,
            return_intermediate_dec,
            dynamic_K,
            dynamic_ratio,
        )
        _reset_parameters(self.upper_decoder)

        # lower_decoder_layer = TransformerDecoderLayer(
        #     d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        # )
        # lower_decoder_norm = nn.LayerNorm(d_model)
        # self.lower_decoder = TransformerDecoder(
        #     lower_decoder_layer,
        #     num_decoder_layers,
        #     lower_decoder_norm,
        #     return_intermediate=return_intermediate_dec,
        # )
        self.lower_decoder = get_decoder_network(
            network_type,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            num_decoder_layers,
            return_intermediate_dec,
            dynamic_K,
            dynamic_ratio,
        )
        _reset_parameters(self.lower_decoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        tail_hidden_dim = d_model // 2
        self.upper_tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, len(upper_face3d_indices)),
        )
        self.lower_tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, len(lower_face3d_indices)),
        )

    def forward(self, content, style_code):
        """

        Args:
            content (_type_): (B, num_frames, window, C_dmodel)
            style_code (_type_): (B, C_dmodel)

        Returns:
            face3d: (B, L_clip, C_3dmm)
        """
        print('in decoder')
        B, N, W, C = content.shape
        print(B, N, W, C ) # 1 780 11 256
        print('style 0', style_code.size()) # torch.Size([1, 256])
        style = style_code.reshape(B, 1, 1, C).expand(B, N, W, C) 
        print('style 1', style.size()) # torch.Size([1, 780, 11, 256])
        style = style.permute(2, 0, 1, 3).reshape(W, B * N, C)
        print('style 2', style.size()) # orch.Size([11, 780, 256])
        # (W, B*N, C)

        print('content 0', content.size()) # torch.Size([1, 780, 11, 256])
        content = content.permute(2, 0, 1, 3).reshape(W, B * N, C)
        print('content 1', content.size()) # torch.Size([11, 780, 256])

        # (W, B*N, C)
        tgt = torch.zeros_like(style)
        print('tgt', tgt.size()) # torch.Size([11, 780, 256])
        pos_embed = self.pos_embed(W)
        print('pos_embed 1', pos_embed.size()) # torch.Size([1, 11, 256])
        pos_embed = pos_embed.permute(1, 0, 2)
        print('pos_embed 2', pos_embed.size()) # torch.Size([11, 1, 256])

        upper_face3d_feat = self.upper_decoder(tgt, content, pos=pos_embed, query_pos=style)[0]
        print('upper_face3d_feat 0', upper_face3d_feat.size()) # torch.Size([11, 780, 256])
        # (W, B*N, C)
        upper_face3d_feat = upper_face3d_feat.permute(1, 0, 2).reshape(B, N, W, C)[:, :, W // 2, :]
        print('upper_face3d_feat 1', upper_face3d_feat.size()) # torch.Size([1, 780, 256])

        # (B, N, C)
        upper_face3d = self.upper_tail_fc(upper_face3d_feat)
        print('upper_face3d', upper_face3d.size()) # torch.Size([1, 780, 51])
        # (B, N, C_exp)

        lower_face3d_feat = self.lower_decoder(tgt, content, pos=pos_embed, query_pos=style)[0]
        print('lower_face3d_feat 1', lower_face3d_feat.size()) # torch.Size([11, 780, 256])
        lower_face3d_feat = lower_face3d_feat.permute(1, 0, 2).reshape(B, N, W, C)[:, :, W // 2, :]
        print('lower_face3d_feat 2', lower_face3d_feat.size()) # torch.Size([1, 780, 256])
        lower_face3d = self.lower_tail_fc(lower_face3d_feat)
        print('lower_face3d',lower_face3d.size())  # torch.Size([1, 780, 13])
        C_exp = len(self.upper_face3d_indices) + len(self.lower_face3d_indices)
        print('C_e', C_exp) # 64
        face3d = torch.zeros(B, N, C_exp).to(upper_face3d)

        face3d[:, :, self.upper_face3d_indices] = upper_face3d
        face3d[:, :, self.lower_face3d_indices] = lower_face3d
        print('face3d', face3d.size()) # torch.Size([1, 780, 64])
        return face3d
