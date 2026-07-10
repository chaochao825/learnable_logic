#!/usr/bin/env python3
"""Canonical ANF/XAG estimates for LUT truth tables."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class XagTruthResult:
    b: int
    truth_bits: str
    truth_hash: str
    anf_nonzero_terms: int
    anf_product_terms: int
    anf_constant_term: int
    xag_and_count: int
    xag_xor_count: int
    xag_not_count: int
    xag_level_estimate: int
    xag_product_depth: int
    xag_xor_depth: int
    xag_analysis_seconds: float
    xag_backend: str = "anf_xag"
    xag_status: str = "ok"


def infer_b(truth_bits: str) -> int:
    length = len(truth_bits)
    if length <= 0 or (length & (length - 1)):
        raise ValueError(f"truth table length must be power of two, got {length}")
    return int(math.log2(length))


def anf_coefficients(truth_bits: str) -> list[int]:
    coeffs = [1 if bit == "1" else 0 for bit in truth_bits]
    b = infer_b(truth_bits)
    for bit in range(b):
        step = 1 << bit
        for mask in range(1 << b):
            if mask & step:
                coeffs[mask] ^= coeffs[mask ^ step]
    return coeffs


def mask_degree(mask: int) -> int:
    return mask.bit_count()


@lru_cache(maxsize=None)
def monomial_plan(mask: int) -> tuple[frozenset[int], int]:
    degree = mask_degree(mask)
    if degree <= 1:
        return frozenset(), 0
    best_closure: frozenset[int] | None = None
    best_depth: int | None = None
    subset = (mask - 1) & mask
    while subset:
        other = mask ^ subset
        if other and subset <= other:
            left_closure, left_depth = monomial_plan(subset)
            right_closure, right_depth = monomial_plan(other)
            closure = left_closure | right_closure | frozenset({mask})
            depth = 1 + max(left_depth, right_depth)
            score = (len(closure), depth)
            if best_closure is None or score < (len(best_closure), best_depth if best_depth is not None else math.inf):
                best_closure = closure
                best_depth = depth
        subset = (subset - 1) & mask
    if best_closure is None or best_depth is None:
        raise ValueError(f"failed to plan mask={mask}")
    return best_closure, best_depth


def xag_from_truth_bits(truth_bits: str) -> XagTruthResult:
    started = time.perf_counter()
    b = infer_b(truth_bits)
    coeffs = anf_coefficients(truth_bits)
    nonzero_terms = [mask for mask, coeff in enumerate(coeffs) if coeff]
    product_terms = [mask for mask in nonzero_terms if mask_degree(mask) >= 2]
    product_closure: set[int] = set()
    product_depth = 0
    for mask in product_terms:
        closure, depth = monomial_plan(mask)
        product_closure.update(closure)
        product_depth = max(product_depth, depth)

    constant_term = 1 if coeffs[0] else 0
    nonconst_terms = [mask for mask in nonzero_terms if mask != 0]
    xag_xor_count = max(len(nonconst_terms) - 1, 0)
    xor_depth = 0 if len(nonconst_terms) <= 1 else math.ceil(math.log2(len(nonconst_terms)))
    if constant_term and nonconst_terms:
        xag_not_count = 1
        level = product_depth + xor_depth + 1
    else:
        xag_not_count = 0
        level = product_depth + xor_depth
    if not nonconst_terms and constant_term:
        level = 0

    truth_hash = hashlib.sha256(f"{b}:{truth_bits}".encode("ascii")).hexdigest()
    return XagTruthResult(
        b=b,
        truth_bits=truth_bits,
        truth_hash=truth_hash,
        anf_nonzero_terms=len(nonzero_terms),
        anf_product_terms=len(product_terms),
        anf_constant_term=constant_term,
        xag_and_count=len(product_closure),
        xag_xor_count=xag_xor_count,
        xag_not_count=xag_not_count,
        xag_level_estimate=int(level),
        xag_product_depth=int(product_depth),
        xag_xor_depth=int(xor_depth),
        xag_analysis_seconds=time.perf_counter() - started,
    )
