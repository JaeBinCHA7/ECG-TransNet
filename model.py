import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ------------------------------------------------------------
# ASPP (Atrous Spatial Pyramid Pooling) Blocks
# ------------------------------------------------------------
class AsppBlock(nn.Module):
    """
    1D ASPP with 4 parallel dilated 1D convolutions.
    Input:  [B, C_in, T]
    Output: [B, 4*C_out, T]
    """
    def __init__(self, in_channel: int = 4, out_channel: int = 4, kernel: int = 3):
        super().__init__()

        self.atrous_block1 = nn.Conv1d(in_channel, out_channel, kernel, padding=1, dilation=1)
        self.atrous_block6 = nn.Conv1d(in_channel, out_channel, kernel, padding=2, dilation=2)
        self.atrous_block12 = nn.Conv1d(in_channel, out_channel, kernel, padding=3, dilation=3)
        self.atrous_block18 = nn.Conv1d(in_channel, out_channel, kernel, padding=4, dilation=4)

        self.bn1 = nn.BatchNorm1d(out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.bn3 = nn.BatchNorm1d(out_channel)
        self.bn4 = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        # Four parallel dilated convs → concat on channel dim
        b1 = self.relu(self.bn1(self.atrous_block1(x)))
        b2 = self.relu(self.bn2(self.atrous_block6(x)))
        b3 = self.relu(self.bn3(self.atrous_block12(x)))
        b4 = self.relu(self.bn4(self.atrous_block18(x)))
        x = torch.cat([b1, b2, b3, b4], dim=1)
        return x


class ASPPConvBlock(nn.Module):
    """
    ASPP → Conv+BN+ReLU → AvgPool1d(4) → Conv+BN+ReLU
    Keeps original channel schedule: in → (4*out) → (4*out).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, padding: int = 1):
        super().__init__()

        self.aspp_block = AsppBlock(in_channels, out_channels)

        self.conv_block1 = nn.Conv1d(out_channels * 4, out_channels * 4, kernel_size=kernel, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU(inplace=False)

        self.pooling = nn.AvgPool1d(4)  # original choice maintained

        self.conv_block2 = nn.Conv1d(out_channels * 4, out_channels * 4, kernel_size=kernel, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels * 4)

    def forward(self, x: Tensor) -> Tensor:
        x = self.aspp_block(x)
        x = self.relu(self.bn1(self.conv_block1(x)))
        x = self.pooling(x)
        out = self.relu(self.bn2(self.conv_block2(x)))
        return out


# ------------------------------------------------------------
# Lead-wise Attention
# ------------------------------------------------------------
class LeadAttention(nn.Module):
    """
    Inter-lead attention per time step.
    Input:  x [B, L, C, T]
    Output: out [B, L, C, T], attn [B, T, L, L]
    """
    def __init__(self, in_channels, hidden_dim = 64, num_leads = 12):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_leads = num_leads

        self.query = nn.Linear(in_channels, hidden_dim)
        self.key = nn.Linear(in_channels, hidden_dim)
        self.value = nn.Linear(in_channels, in_channels)

        # LayerNorm over C (post-residual)
        self.norm_c = nn.LayerNorm(in_channels)
        self.proj = nn.Linear(in_channels, in_channels)

    def forward(self, x: Tensor):
        # x: [B, L, C, T]
        B, L, C, T = x.shape
        x_res = x

        # Reorder to [B, T, L, C] for lead-wise attention at each time step
        z = x.permute(0, 3, 1, 2)

        # Q, K, V
        q = self.query(z)  # [B, T, L, hd]
        k = self.key(z)    # [B, T, L, hd]
        v = self.value(z)  # [B, T, L, C]

        # Attention logits [B, T, L, L]
        logits = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1))
        logits = logits - logits.max(dim=-1, keepdim=True).values  # numerical stability
        attn = F.softmax(logits, dim=-1)

        out = torch.matmul(attn, v)  # [B, T, L, C]
        out = self.proj(out)         # [B, T, L, C]

        # Back to [B, L, C, T] and post-residual LayerNorm over C
        out = out.permute(0, 2, 3, 1)    # [B, L, C, T]
        out = out + x_res                # residual
        y = out.permute(0, 1, 3, 2)      # [B, L, T, C]
        y = self.norm_c(y)
        out = y.permute(0, 1, 3, 2)      # [B, L, C, T]

        return out, attn


# ------------------------------------------------------------
# Positional Encoding (sinusoidal)
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Standard sinusoidal PE. Expects [B, T, D]."""
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, D]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ------------------------------------------------------------
# Transformer-ish Blocks (Head-Independent Attention + Depthwise FF)
# ------------------------------------------------------------
class FeedForwardBlock(nn.Sequential):
    """
    Depthwise (grouped) MLP with 1x1 Conv1d:
      [B, D, T] → [B, D*exp, T] → GN → ReLU → Dropout → [B, D, T]
    """
    def __init__(self, emb_size: int, head: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Conv1d(emb_size, emb_size * expansion, kernel_size=1, stride=1, groups=head),
            nn.GroupNorm(num_groups=head, num_channels=emb_size * expansion),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Conv1d(emb_size * expansion, emb_size, kernel_size=1, stride=1, groups=head),
        )


class HiAttention(nn.Module):
    """
    Multi-head attention on [B, T, D], implemented with Linear projections
    and grouped 1x1 Conv projection + GroupNorm.
    Returns sequence output, attention map, and pre-projection per-head values.
    """
    def __init__(self, emb_size: int, num_heads: int):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.projection = nn.Conv1d(emb_size, emb_size, kernel_size=1, stride=1, groups=num_heads)
        self.gn = nn.GroupNorm(num_groups=num_heads, num_channels=emb_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, mask: Tensor = None):
        # x: [B, T, D]
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)

        scaling = self.head_dim ** 0.5

        # Attention: [B, H, T, T]
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        att = F.softmax(energy / scaling, dim=-1)

        # Weighted sum: -> [B, H, T, D_head]
        out = torch.einsum('bhal, bhlv -> bhav', att, v)
        feat_o = out  # keep for caller, as in original

        # Merge heads to [B, D, T], grouped 1x1 conv, GN+ReLU, residual; back to [B, T, D]
        residual = rearrange(out, "b h n d -> b (h d) n")
        out = self.projection(residual)
        out = self.relu(self.gn(out)) + residual
        out = out.permute(0, 2, 1)  # [B, T, D]

        return out, att, feat_o


class HiTransformer(nn.Module):
    """
    One layer of Hi-Transformer:
      - HiAttention on [B,T,D]
      - GroupNorm on [B,D,T]
      - Depthwise FF (grouped) with residual
      - Returns sequence [B,T,D], attention map, and pooled per-head features [B,H,D_head]
    """
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.head = h
        self.self_attn = HiAttention(d_model, h)
        self.feed_forward = FeedForwardBlock(d_model, head=h, expansion=d_ff, drop_p=dropout)

        self.gn_1 = nn.GroupNorm(num_groups=h, num_channels=d_model)
        self.gn_2 = nn.GroupNorm(num_groups=h, num_channels=d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor, mask: Tensor = None):
        # x: [B, T, D]
        x, feat_a, feat_o = self.self_attn(x, mask)  # (B, T, D)

        x = x.permute(0, 2, 1)            # (B, D, T) for GroupNorm/Conv1d
        attn_out = self.gn_1(x)

        ff_out = self.feed_forward(attn_out)   # (B, D, T)
        x = attn_out + self.dropout_2(ff_out)  # residual

        x = self.gn_2(x)
        x = x.permute(0, 2, 1)            # back to (B, T, D)

        # Pooled per-head features: [B,T,D] → [(B*H),D_head,T] → pool → [B,H,D_head]
        feat_pool = rearrange(x, 'b n (h d) -> (b h) d n', h=self.head)
        feat_pool = self.pool(feat_pool).squeeze(2)
        feat_pool = rearrange(feat_pool, '(b h) d -> b h d', h=self.head)

        return x, feat_a, feat_pool


class TransNetBlock(nn.Module):
    """
    Stack of HiTransformer layers with sinusoidal PE.
    Input:  [B, C, T]  (will be permuted to [B, T, C] internally)
    Output: seq [B, H, D_head, T], attn (from last layer), pooled [B, H, D_head]
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            HiTransformer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Tensor = None):
        # x: [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)
        x = self.positional_encoding(x)

        feat_a = None
        feat_o = None
        for layer in self.layers:
            x, feat_a, feat_o = layer(x, mask)

        # [B, T, C] -> [B, H, D_head, T]
        x = rearrange(x, "b n (h d) -> b h d n", h=self.num_heads)
        return x, feat_a, feat_o


# ------------------------------------------------------------
# Decision / Classification Head
# ------------------------------------------------------------
class DecisionBlock(nn.Module):
    """
    Decision head with per-head avg/max pooling + FC and proxy-sim fusion.
    Inputs:
      - x    : [B, H, D, T]  (sequence features per head)
      - feat : [B, H, D]     (pooled per-head features, L2-normalized inside)
      - proxy: [H, C, D]     (per-head class anchors)
    Returns:
      - logits: [B, C]
      - feat_h: [B, H, 2D]  (concat of avg/max pooled features)
    """
    def __init__(self, emb_dim, head, classes):
        super().__init__()
        self.head = head
        self.classes = classes

        # Pooling per head
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)

        # Base classifier FC: [B, 2D] -> [B, C]
        self.fc = nn.Linear(emb_dim * 2, classes)

        # Learnable scale (tau) for cosine logits and fusion weight alpha
        self.tau = nn.Parameter(torch.tensor(16.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: Tensor, feat: Tensor, proxy: Tensor):
        B, H, D, T = x.shape
        assert H == self.head, f"head mismatch: x has {H}, self.head={self.head}"

        # ----- (1) Per-head pooling (avg & max) -----
        x_h = rearrange(x, 'b h d n -> (b h) d n')     # [(B*H), D, T]
        x_avg = self.pool_avg(x_h).squeeze(2)          # [(B*H), D]
        x_max = self.pool_max(x_h).squeeze(2)          # [(B*H), D]
        feat_h = torch.cat([x_avg, x_max], dim=1)      # [(B*H), 2D]
        feat_h = rearrange(feat_h, '(b h) d2 -> b h d2', h=H)  # [B, H, 2D]

        # Collapse heads for FC as in original design (in_dim = emb_dim*2)
        x_flat = rearrange(feat_h, 'b h d2 -> b (h d2)')  # [B, 2D*H]
        # NOTE: The original implementation uses emb_dim*2 as FC input dimension.
        # Here the upstream defines emb_dim as per-head D, and the flatten keeps behavior
        # consistent with the original code path.

        # ----- (2) Cosine similarity with proxies -----
        f_sim = F.normalize(feat, dim=-1)                   # [B, H, D]
        P = proxy.detach()
        P = F.normalize(P, dim=-1)

        # Per-head cosine sim: [B,H,C] -> reduce heads
        sim_h = torch.einsum('bhd,hcd->bhc', f_sim, P)      # [B, H, C]
        sim = sim_h.mean(dim=1)                             # [B, C]

        # Temperature scaling
        tau_pos = F.softplus(self.tau) + 1e-6
        sim_logits = tau_pos * sim                          # [B, C]

        # ----- (3) Fusion with base logits -----
        base_logits = self.fc(x_flat)                       # [B, C]
        logits = base_logits + self.alpha * sim_logits      # [B, C]

        return logits, feat_h


# ------------------------------------------------------------
# Full Model
# ------------------------------------------------------------
class ECGTransNet(nn.Module):
    """
    End-to-end ECG model (Proxy-Sim Integrated):
      1) Frontend: Conv1d → 3× ASPPConvBlock
      2) LeadAttention on [B,L,C,T]
      3) Merge leads: [B,C,(L*T)] → TransNetBlock (Hi-Transformer stack)
      4) DecisionBlock with proxy-sim fusion
    Forward returns (logits, pooled_features, proxies) per original code.
    """
    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.d_model
        self.nhead = opt.head
        self.d_ff = opt.d_ff
        self.nOUT = opt.classes
        self.n_layer = opt.num_layers
        self.lead = opt.lead
        self.drop_out = opt.drop_out

        # Front-end
        self.conv_start = nn.Conv1d(1, 8, kernel_size=3, padding=1, stride=2, dilation=1)
        self.bn_start = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=False)

        # Channel expansion via ASPP blocks: 8 → 32 → 128 → 512 (= d_model)
        self.block = nn.Sequential(
            ASPPConvBlock(8, 8),        # -> 32 ch
            ASPPConvBlock(32, 32),      # -> 128 ch
            ASPPConvBlock(128, 128)     # -> 512 ch
        )

        self.lead_attention = LeadAttention(self.d_model)
        self.mltblock = TransNetBlock(self.d_model, self.nhead, self.d_ff, self.n_layer, dropout=self.drop_out)
        self.db = DecisionBlock(emb_dim=self.d_model, head=self.nhead, classes=self.nOUT)

        # Per-head class proxies
        self.proxies = nn.Parameter(torch.randn(self.nhead, self.nOUT, self.d_model // self.nhead))
        nn.init.kaiming_normal_(self.proxies)

    def forward(self, x: Tensor):
        """
        Forward keeps the original reshape behavior:
        - x reshaped to (-1, 5000) and unsqueezed to [?, 1, 5000]
        - batch is later reinterpreted with 'lead' at rearrange step
        Returns:
          logits [B,C], pooled features [B,H,D_head], proxies [H,C,D_head]
        """
        # --- Original brittle reshape maintained for exact behavior ---
        x = x.reshape(-1, 5000).unsqueeze(1)                 # [?, 1, 5000]

        x = self.relu(self.bn_start(self.conv_start(x)))     # [?, 8, ~2500]
        x = self.block(x)                                    # [?, 512, T']

        # [?, C, T] -> [B, L, C, T]
        x = rearrange(x, '(b l) c t -> b l c t', l=self.lead)

        # Lead-wise attention
        x, lead_atten = self.lead_attention(x)               # [B, L, 512, T']

        # Merge leads into temporal axis: [B, L, C, T] -> [B, C, L*T]
        x = rearrange(x, 'b l c t -> b c (l t)')

        # Transformer stack
        x, feat_a, feat_o = self.mltblock(x)                 # x: [B,H,D_head,T''], feat_o: [B,H,D_head]

        # Decision head (+ proxy fusion); proxies detached in call as in original
        out, feat_h = self.db(x, feat_o, self.proxies.detach())

        return out, feat_o, self.proxies
