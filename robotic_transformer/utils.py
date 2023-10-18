# -*- coding: utf-8 -*-
from typing import TypeVar

import torch
from einops import pack, unpack

T = TypeVar("T")
U = TypeVar("U")


def exists(val: T) -> bool:
    return val is not None


def default(val: T, d: U) -> T | U:
    return val if exists(val) else d


def cast_tuple(val: T, length: int = 1) -> tuple:
    return val if isinstance(val, tuple) else ((val,) * length)


def pack_one(x: T, pattern: str) -> tuple:
    return pack([x], pattern)


def unpack_one(x: T, ps: list, pattern: str):
    return unpack(x, ps, pattern)[0]


def posemb_sincos_1d(seq, dim, temperature=10000, device=None, dtype=torch.float32):
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)
