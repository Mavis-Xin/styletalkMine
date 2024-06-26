import torch
from torch import nn

from .transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from core.utils import _reset_parameters
from core.networks.self_attention_pooling import SelfAttentionPooling


class ContentEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pos_embed_len=80,
        ph_embed_dim=128,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        _reset_parameters(self.encoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        self.ph_embedding = nn.Embedding(41, ph_embed_dim)
        self.increase_embed_dim = nn.Linear(ph_embed_dim, d_model)

    def forward(self, x):
        """

        Args:
            x (_type_): (B, num_frames, window)

        Returns:
            content: (B, num_frames, window, C_dmodel)
        """
        x_embedding = self.ph_embedding(x)
        x_embedding = self.increase_embed_dim(x_embedding)
        # (B, N, W, C)
        B, N, W, C = x_embedding.shape
        x_embedding = x_embedding.reshape(B * N, W, C)
        x_embedding = x_embedding.permute(1, 0, 2)
        # (W, B*N, C)

        pos = self.pos_embed(W)
        pos = pos.permute(1, 0, 2)
        # (W, 1, C)

        content = self.encoder(x_embedding, pos=pos)
        # (W, B*N, C)
        content = content.permute(1, 0, 2).reshape(B, N, W, C)
        # (B, N, W, C)

        return content


class StyleEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pos_embed_len=80,
        input_dim=128,
        aggregate_method="average",
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        _reset_parameters(self.encoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        self.increase_embed_dim = nn.Linear(input_dim, d_model)

        self.aggregate_method = None
        if aggregate_method == "self_attention_pooling":
            self.aggregate_method = SelfAttentionPooling(d_model)
        elif aggregate_method == "average":
            pass
        else:
            raise ValueError(f"Invalid aggregate method {aggregate_method}")

    def forward(self, x, pad_mask=None):
        """

        Args:
            x (_type_): (B, num_frames(L), C_exp)
            pad_mask: (B, num_frames)

        Returns:
            style_code: (B, C_model)
        """
        print('x1', x.size()) # torch.Size([1, 256, 64])
        x = self.increase_embed_dim(x)
        # (B, L, C)
        print('x2', x.size()) # torch.Size([1, 256, 256])
        x = x.permute(1, 0, 2)
        print('x3', x.size()) # torch.Size([256, 1, 256])
        # (L, B, C)

        pos = self.pos_embed(x.shape[0]) 
        print('pos 1', pos.size()) # torch.Size([1, 256, 256])
        pos = pos.permute(1, 0, 2)
        print('pos 2', pos.size()) # torch.Size([256, 1, 256])
        # (L, 1, C)

        print('pad_mask', pad_mask.size()) # torch.Size([1, 256])
        style = self.encoder(x, pos=pos, src_key_padding_mask=pad_mask)
        print('style', style.size()) # torch.Size([256, 1, 256])
        # (L, B, C)

        print('aggregate_method', self.aggregate_method) # not None
        if self.aggregate_method is not None:
            permute_style = style.permute(1, 0, 2)
            # (B, L, C)
            style_code = self.aggregate_method(permute_style, pad_mask)
            print('ss', style_code.size()) # torch.Size([1, 256])
            return style_code

        print('pad_mask', pad_mask)
        if pad_mask is None:
            style = style.permute(1, 2, 0)
            # (B, C, L)
            style_code = style.mean(2)
            # (B, C)
        else:
            permute_style = style.permute(1, 0, 2)
            # (B, L, C)
            permute_style[pad_mask] = 0
            sum_style_code = permute_style.sum(dim=1)
            # (B, C)
            valid_token_num = (~pad_mask).sum(dim=1).unsqueeze(-1)
            # (B, 1)
            style_code = sum_style_code / valid_token_num
            # (B, C)

        print('style_code', style_code.size())
        return style_code


class Decoder(nn.Module):
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
        output_dim=64,
        **_,
    ) -> None:
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        _reset_parameters(self.decoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        tail_hidden_dim = d_model // 2
        self.tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, output_dim),
        )

    def forward(self, content, style_code):
        """

        Args:
            content (_type_): (B, num_frames, window, C_dmodel)
            style_code (_type_): (B, C_dmodel)

        Returns:
            face3d: (B, num_frames, C_3dmm)
        """
        B, N, W, C = content.shape
        style = style_code.reshape(B, 1, 1, C).expand(B, N, W, C)
        style = style.permute(2, 0, 1, 3).reshape(W, B * N, C)
        # (W, B*N, C)

        content = content.permute(2, 0, 1, 3).reshape(W, B * N, C)
        # (W, B*N, C)
        tgt = torch.zeros_like(style)
        pos_embed = self.pos_embed(W)
        pos_embed = pos_embed.permute(1, 0, 2)
        face3d_feat = self.decoder(tgt, content, pos=pos_embed, query_pos=style)[0]
        # (W, B*N, C)
        face3d_feat = face3d_feat.permute(1, 0, 2).reshape(B, N, W, C)[:, :, W // 2, :]
        print('face3d_feat 1', face3d_feat.size())
        # (B, N, C)
        face3d = self.tail_fc(face3d_feat)
        print('face3d', face3d.size())
        # (B, N, C_exp)
        return face3d
