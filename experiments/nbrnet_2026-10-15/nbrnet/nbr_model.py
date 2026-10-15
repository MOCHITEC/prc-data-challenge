"""NbrNet (PREREG.md §3): a query MLP that attends over the neighbouring departures and arrivals.

`ctx=False` is the control twin `noctx`: the same parameters and code path with every token masked, which makes
the attention output, both pools, the token counts and the coverage scalar exactly zero."""

from __future__ import annotations

import math

import torch
from torch import nn


class TokenMLP(nn.Module):
    def __init__(self, n_in: int, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class MaskedCrossAttention(nn.Module):
    """Single-query multi-head attention with a key padding mask. A query with no valid key gets exactly 0."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, q, kv, mask):
        # q [B,d], kv [B,K,d], mask [B,K] bool (True = valid)
        B, K, d = kv.shape
        Q = self.q(q).view(B, self.h, self.dh)
        Kx = self.k(kv).view(B, K, self.h, self.dh)
        V = self.v(kv).view(B, K, self.h, self.dh)
        s = torch.einsum("bhd,bkhd->bhk", Q, Kx) / math.sqrt(self.dh)
        s = s.masked_fill(~mask[:, None, :], -1e9)
        w = torch.softmax(s, dim=-1)
        any_valid = mask.any(dim=1).to(q.dtype)[:, None]
        out = torch.einsum("bhk,bkhd->bhd", w, V).reshape(B, d)
        return self.o(out) * any_valid


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.att = MaskedCrossAttention(d, heads)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, q, kv, mask):
        q = self.ln1(q + self.att(q, kv, mask))
        return self.ln2(q + self.ff(q))


class NbrNet(nn.Module):
    def __init__(self, n_query: int, n_dep: int, n_arr: int, d: int = 64, heads: int = 4, rounds: int = 2, k_dep: int = 64, k_arr: int = 48):
        super().__init__()
        self.d, self.k_dep, self.k_arr = d, k_dep, k_arr
        self.query = nn.Sequential(nn.Linear(n_query, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, d))
        self.tok_dep, self.tok_arr = TokenMLP(n_dep, d), TokenMLP(n_arr, d)
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        nn.init.normal_(self.type_emb, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, 2 * d) for _ in range(rounds)])
        self.head = nn.Sequential(nn.Linear(4 * d + 3, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, xq, dep=None, mask_d=None, arr=None, mask_a=None, cov=None):
        """xq [B,n_query]; dep [B,Kd,n_dep], mask_d [B,Kd]; arr [B,Ka,n_arr], mask_a [B,Ka]; cov [B].
        Any of the token arguments None = the noctx twin (all tokens masked)."""
        q0 = self.query(xq)
        B = q0.shape[0]
        if dep is None:
            zero = q0.new_zeros(B, self.d)
            q = q0
            for blk in self.blocks:  # attention output is 0 with no valid key: q = LN(q), then the FF sublayer
                q = blk.ln1(q)
                q = blk.ln2(q + blk.ff(q))
            scal = q0.new_zeros(B, 3)
            return self.head(torch.cat([q0, q, zero, zero, scal], 1)).squeeze(1)
        ed = self.tok_dep(dep) + self.type_emb[0]
        ea = self.tok_arr(arr) + self.type_emb[1]
        kv = torch.cat([ed, ea], 1)
        mask = torch.cat([mask_d, mask_a], 1)
        q = q0
        for blk in self.blocks:
            q = blk(q, kv, mask)
        md, ma = mask_d.to(ed.dtype)[..., None], mask_a.to(ea.dtype)[..., None]
        nd, na = md.sum(1), ma.sum(1)
        pd = (ed * md).sum(1) / nd.clamp(min=1.0)
        pa = (ea * ma).sum(1) / na.clamp(min=1.0)
        scal = torch.cat([nd / self.k_dep, na / self.k_arr, cov[:, None]], 1)
        return self.head(torch.cat([q0, q, pd, pa, scal], 1)).squeeze(1)
