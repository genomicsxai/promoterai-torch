"""
DNA one-hot encoding.

The lookup-table + single-write-per-base loop strategy here is adapted from
tangermeme's `_fast_one_hot_encode` (writing only the one hot channel into a
pre-zeroed array, rather than gathering a full one-hot row per base from a
table, avoids redundant writes and is measurably faster under numba):

    tangermeme (https://github.com/jmschrei/tangermeme)
    MIT License
    Copyright (c) 2024-2026 Jacob Schreiber

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

This implementation differs from tangermeme's in one deliberate way: any byte
outside the alphabet (not just 'N') silently encodes as an all-zero row rather
than raising, and it returns plain numpy rather than a torch.Tensor.
"""

from __future__ import annotations

import numba
import numpy as np

# Byte -> channel lookup: A/C/G/T map to 0-3; every other byte (e.g. 'N' or an
# IUPAC ambiguity code) maps to -1 and is skipped, leaving its row all-zero.
_BASE_TO_CHANNEL = np.full(256, -1, dtype=np.int8)
for _channel, _base in enumerate(b"ACGT"):
    _BASE_TO_CHANNEL[_base] = _channel


@numba.njit("void(float32[:, :], uint8[:], int8[:])", cache=True)
def _fill_onehot(out: np.ndarray, seq_bytes: np.ndarray, base_to_channel: np.ndarray) -> None:
    """Set out[i, base_to_channel[seq_bytes[i]]] = 1 for every recognized base."""
    for i in range(len(seq_bytes)):
        channel = base_to_channel[seq_bytes[i]]
        if channel >= 0:
            out[i, channel] = 1.0


def onehot_encode(seq: str) -> np.ndarray:
    """One-hot encode a DNA string. Unknown bases → all-zero row. Returns (L, 4) float32."""
    # bytearray (not bytes) so frombuffer's view is writable, matching the njit signature.
    seq_bytes = np.frombuffer(bytearray(seq.upper(), "ascii", errors="replace"), dtype=np.uint8)
    out = np.zeros((len(seq_bytes), 4), dtype=np.float32)
    _fill_onehot(out, seq_bytes, _BASE_TO_CHANNEL)
    return out
