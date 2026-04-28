import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F


class HeadwiseProxyBCELoss(nn.Module):
    """
    per-head proxies + no gating (uniform over heads) + BCE
    - features: [B, H, D]
    - logits  : [B, C]
    - target  : [B, C] (multi-hot)
    - proxies : [H, C, D]

    L = bce_weight * BCE + proxy_weight * ProxyLoss (+ align_weight * AlignLoss)

    ProxyLoss (basic / non-focal):
      sim[b,h,c] = <f_{b,h}, p_{h,c}>   # (cosine if use_cosine=True)
      pos_term_head = exp(-alpha * (sim - delta))
      neg_term_head = exp(+alpha * (sim + delta))
      pos/neg_term = mean over heads (uniform)
      per-sample aggregation: log(1 + sum_c term[b,c])

    AlignLoss (always cosine):
      L_align = mean_{h,c} [ 1 - cos(p_{h,c}, mu_c) ],  mu_c = mean_h (normalize(p_{h,c}))
    """

    def __init__(
        self,
        opt,
        bce_weight: float = 1.0,
        proxy_weight: float = 0.2,
        alpha: Optional[float] = None,
        delta: float = 0.0,
        use_cosine: bool = True,
        eps: float = 1e-8,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        align_weight: float = 0.05,         # >0면 cosine alignment 활성
        align_detach_centroid: bool = True  # 중심 stop-grad로 고정할지
    ):
        super().__init__()
        self.C = int(opt.classes)
        self.D = opt.d_model // opt.head
        self.H = opt.head

        self.bce_weight = bce_weight
        self.proxy_weight = getattr(opt, "proxy_weight", proxy_weight)

        self.alpha = getattr(opt, "proxy_a", 1.0 if alpha is None else alpha)
        self.delta = getattr(opt, "proxy_d", 1.0 if delta is None else delta)
        self.use_cosine = use_cosine
        self.eps = eps
        self.reduction = reduction

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

        # alignment (always cosine mode)
        self.align_weight = getattr(opt, "align_weight", align_weight)
        self.align_detach_centroid = getattr(opt, "align_detach_centroid", align_detach_centroid)

    @torch.no_grad()
    def _check_shapes(self, features, logits, target, proxies):
        B, H, D = features.shape
        assert H == self.H and D == self.D, f"features must be [B,{self.H},{self.D}]"
        assert logits.shape == (B, self.C), f"logits must be [B,{self.C}]"
        assert target.shape == (B, self.C), f"target must be [B,{self.C}]"
        assert proxies.shape == (self.H, self.C, self.D), f"proxies must be [{self.H},{self.C},{self.D}]"

    def _align_loss(self, proxies: torch.Tensor) -> torch.Tensor:
        """
        Cosine alignment regularizer.
        proxies: [H, C, D]
        returns scalar align loss.
        """
        P = F.normalize(proxies, dim=-1)       # [H,C,D] unit
        mu = P.mean(dim=0, keepdim=False)      # [C,D]
        mu = F.normalize(mu, dim=-1)           # [C,D]

        if self.align_detach_centroid:
            mu = mu.detach()                   # stop-grad for centroid

        # cos(h,c) = <p_{h,c}, mu_c>
        cos_hc = (P * mu.unsqueeze(0)).sum(dim=-1)  # [H,C] in [-1,1]
        loss_align = (1.0 - cos_hc).mean()          # maximize cos -> minimize (1 - cos)
        return loss_align

    def forward(
        self,
        features: torch.Tensor,  # [B,H,D]
        logits: torch.Tensor,    # [B,C]
        target: torch.Tensor,    # [B,C]
        proxies: torch.Tensor,   # [H,C,D]
    ):
        self._check_shapes(features, logits, target, proxies)
        device = features.device

        # -------- BCE part --------
        loss_bce = self.bce(logits, target.float())

        # -------- Proxy part (uniform over heads) --------
        x = features
        P = proxies
        if self.use_cosine:
            x = F.normalize(x, dim=-1)  # [B,H,D]
            P = F.normalize(P, dim=-1)  # [H,C,D]

        # sim[b,h,c] = <x[b,h,:], P[h,c,:]>
        sim = torch.einsum("bhd,hcd->bhc", x, P)  # [B,H,C]

        pos_mask = target.bool()                  # [B,C]
        neg_mask = ~pos_mask                      # [B,C]
        pos_mask_bhc = pos_mask[:, None, :].float()
        neg_mask_bhc = neg_mask[:, None, :].float()

        # Positive term
        pos_term_head = torch.exp(-self.alpha * (sim - self.delta)) * pos_mask_bhc  # [B,H,C]
        pos_term = pos_term_head.mean(dim=1)                                        # [B,C]
        pos_exists = pos_mask.any(dim=1)                                            # [B]
        if pos_exists.any():
            pos_sum = pos_term.sum(dim=1)                                           # [B]
            pos_loss_i = F.softplus(torch.log(pos_sum + self.eps))[pos_exists]
            loss_pos = pos_loss_i.mean()
        else:
            loss_pos = torch.zeros([], device=device)

        # Negative term
        neg_term_head = torch.exp(+self.alpha * (sim + self.delta)) * neg_mask_bhc  # [B,H,C]
        neg_term = neg_term_head.mean(dim=1)                                        # [B,C]
        neg_exists = neg_mask.any(dim=1)
        if neg_exists.any():
            neg_sum = neg_term.sum(dim=1)
            neg_loss_i = F.softplus(torch.log(neg_sum + self.eps))[neg_exists]
            loss_neg = neg_loss_i.mean()
        else:
            loss_neg = torch.zeros([], device=device)

        loss_proxy = loss_pos + loss_neg

        # -------- Cosine alignment regularizer --------
        if self.align_weight > 0.0:
            loss_align = self._align_loss(proxies)  # use raw proxies; normalized inside
        else:
            loss_align = torch.zeros([], device=device)

        # -------- Joint --------
        loss = (
            self.bce_weight * loss_bce
            + self.proxy_weight * loss_proxy
            + self.align_weight * loss_align
        )

        return loss, {
            "loss_total": loss.detach(),
            "loss_bce": loss_bce.detach(),
            "loss_proxy": loss_proxy.detach(),
            "loss_proxy_pos": loss_pos.detach(),
            "loss_proxy_neg": loss_neg.detach(),
            "loss_align": loss_align.detach(),
        }
