"""
LSB-first Spiking Mitchell multiplier verification.

This script checks whether a mantissa-bit streaming implementation can reproduce
the same result as the current parallel Mitchell C-2-style FP32 approximation.

The experiment is intentionally small and dependency-light:
  - FP32 values are decomposed with NumPy bit views.
  - Exponents/signs are handled digitally.
  - Mantissa fractions are added with an LSB-first bit-serial full-adder.
  - The serial result is compared against a parallel integer Mitchell model.

Run:
  python verify_lsb_first_spiking_mitchell.py
  python verify_lsb_first_spiking_mitchell.py --num_samples 100000 --mantissa_bits 16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


MITCHELL_C2_LUT = np.array(
    [
        [0.015625, 0.046875, 0.078125, 0.109375],
        [0.046875, 0.140625, 0.234375, 0.328125],
        [0.078125, 0.234375, 0.390625, 0.546875],
        [0.109375, 0.328125, 0.546875, 0.765625],
    ],
    dtype=np.float64,
)


@dataclass
class FP32Parts:
    sign: np.ndarray
    exponent_field: np.ndarray
    exponent: np.ndarray
    fraction: np.ndarray
    normal: np.ndarray


def fp32_to_bits(x: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    return x32.view(np.uint32)


def decompose_fp32(x: np.ndarray) -> FP32Parts:
    bits = fp32_to_bits(x)
    sign = (bits >> 31).astype(np.uint32)
    exponent_field = ((bits >> 23) & 0xFF).astype(np.int32)
    fraction = (bits & 0x7FFFFF).astype(np.uint32)
    normal = exponent_field > 0
    exponent = exponent_field - 127
    return FP32Parts(sign, exponent_field, exponent, fraction, normal)


def quantize_fraction(frac_q23: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """Keep the top mantissa_bits of the IEEE754 fractional mantissa."""
    if mantissa_bits == 23:
        return frac_q23.astype(np.uint64)
    shift = 23 - mantissa_bits
    return (frac_q23 >> shift).astype(np.uint64)


def correction_lut_q(mantissa_a_q: np.ndarray, mantissa_b_q: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """Return Mitchell C-2 correction in Q(mantissa_bits)."""
    if mantissa_bits < 2:
        raise ValueError("mantissa_bits must be >= 2 because LUT indexing uses the top 2 bits.")

    idx_shift = mantissa_bits - 2
    idx_a = ((mantissa_a_q >> idx_shift) & 0x3).astype(np.int64)
    idx_b = ((mantissa_b_q >> idx_shift) & 0x3).astype(np.int64)
    scale = 1 << mantissa_bits
    return np.rint(MITCHELL_C2_LUT[idx_a, idx_b] * scale).astype(np.uint64)


def lsb_first_add3(a_q: np.ndarray, b_q: np.ndarray, c_q: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """
    Add three Qm unsigned integers using an LSB-first bit-serial full-adder.

    In hardware terms, each cycle consumes one bit from a, b, and the correction
    constant, then propagates carry toward more significant bits.
    """
    a_q = a_q.astype(np.uint64)
    b_q = b_q.astype(np.uint64)
    c_q = c_q.astype(np.uint64)
    out = np.zeros_like(a_q, dtype=np.uint64)
    carry = np.zeros_like(a_q, dtype=np.uint64)

    # The sum can exceed Qm by up to two carry bits, so flush carry after m bits.
    for bit in range(mantissa_bits):
        bit_sum = ((a_q >> bit) & 1) + ((b_q >> bit) & 1) + ((c_q >> bit) & 1) + carry
        out |= (bit_sum & 1) << bit
        carry = bit_sum >> 1

    bit = mantissa_bits
    while np.any(carry):
        out |= (carry & 1) << bit
        carry >>= 1
        bit += 1

    return out


def normalize_mitchell_sum(q_sum: np.ndarray, exponent_sum: np.ndarray, mantissa_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert q_sum = frac_a + frac_b + correction into normalized mantissa fraction.

    This mirrors the repository's Mitchell implementation:
      M_sum = 1 + q_sum / 2^m
      if M_sum >= 2: M_out = M_sum / 2 and exponent += 1
    """
    threshold = np.uint64(1 << mantissa_bits)
    overflow = q_sum >= threshold
    frac_out = np.where(overflow, (q_sum - threshold) >> 1, q_sum)
    exponent_out = exponent_sum + overflow.astype(np.int32)
    return frac_out.astype(np.uint64), exponent_out.astype(np.int32)


def assemble_value(sign: np.ndarray, exponent: np.ndarray, frac_q: np.ndarray, mantissa_bits: int) -> np.ndarray:
    mantissa = 1.0 + frac_q.astype(np.float64) / float(1 << mantissa_bits)
    value = np.ldexp(mantissa, exponent.astype(np.int32))
    return np.where(sign == 0, value, -value).astype(np.float64)


def parallel_mitchell_product(a: np.ndarray, b: np.ndarray, mantissa_bits: int) -> np.ndarray:
    pa = decompose_fp32(a)
    pb = decompose_fp32(b)
    valid = pa.normal & pb.normal

    aq = quantize_fraction(pa.fraction, mantissa_bits)
    bq = quantize_fraction(pb.fraction, mantissa_bits)
    cq = correction_lut_q(aq, bq, mantissa_bits)
    q_sum = aq + bq + cq

    frac_out, exponent_out = normalize_mitchell_sum(q_sum, pa.exponent + pb.exponent, mantissa_bits)
    sign_out = pa.sign ^ pb.sign
    out = assemble_value(sign_out, exponent_out, frac_out, mantissa_bits)
    return np.where(valid, out, 0.0)


def serial_lsb_first_mitchell_product(a: np.ndarray, b: np.ndarray, mantissa_bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pa = decompose_fp32(a)
    pb = decompose_fp32(b)
    valid = pa.normal & pb.normal

    aq = quantize_fraction(pa.fraction, mantissa_bits)
    bq = quantize_fraction(pb.fraction, mantissa_bits)
    cq = correction_lut_q(aq, bq, mantissa_bits)
    q_sum_serial = lsb_first_add3(aq, bq, cq, mantissa_bits)

    frac_out, exponent_out = normalize_mitchell_sum(q_sum_serial, pa.exponent + pb.exponent, mantissa_bits)
    sign_out = pa.sign ^ pb.sign
    out = assemble_value(sign_out, exponent_out, frac_out, mantissa_bits)
    out = np.where(valid, out, 0.0)
    return out, q_sum_serial, aq + bq + cq


def summarize_error(name: str, approx: np.ndarray, reference: np.ndarray) -> None:
    abs_err = np.abs(approx - reference)
    rel_err = abs_err / (np.abs(reference) + 1e-12)
    print(f"\n[{name}]")
    print(f"  MAE       : {abs_err.mean():.8e}")
    print(f"  RMSE      : {np.sqrt(np.mean(abs_err ** 2)):.8e}")
    print(f"  MaxAE     : {abs_err.max():.8e}")
    print(f"  Mean RelE : {rel_err.mean():.8e}")
    print(f"  P99 RelE  : {np.quantile(rel_err, 0.99):.8e}")


def print_exact_product_comparison(approx: np.ndarray, exact: np.ndarray, a: np.ndarray, b: np.ndarray, top_k: int = 8) -> None:
    abs_err = np.abs(approx - exact)
    rel_err = abs_err / (np.abs(exact) + 1e-12)
    nonzero = np.abs(exact) > 1e-12

    print("\n[Exact FP32 Multiplication Difference]")
    print("  Reference product is computed as float32(a) * float32(b), then compared in float64.")
    print(f"  Mean absolute difference       : {abs_err.mean():.8e}")
    print(f"  Median absolute difference     : {np.median(abs_err):.8e}")
    print(f"  95th percentile absolute diff  : {np.quantile(abs_err, 0.95):.8e}")
    print(f"  99th percentile absolute diff  : {np.quantile(abs_err, 0.99):.8e}")
    print(f"  Maximum absolute difference    : {abs_err.max():.8e}")
    if np.any(nonzero):
        print(f"  Mean relative difference       : {rel_err[nonzero].mean():.8e}")
        print(f"  Median relative difference     : {np.median(rel_err[nonzero]):.8e}")
        print(f"  95th percentile relative diff  : {np.quantile(rel_err[nonzero], 0.95):.8e}")
        print(f"  99th percentile relative diff  : {np.quantile(rel_err[nonzero], 0.99):.8e}")
        print(f"  Maximum relative difference    : {rel_err[nonzero].max():.8e}")

    print(f"\n  Top-{top_k} worst absolute-error cases:")
    worst_abs = np.argsort(abs_err)[-top_k:][::-1]
    for idx in worst_abs:
        print(
            f"    a={a[idx]: .8g}, b={b[idx]: .8g}, exact={exact[idx]: .8g}, "
            f"approx={approx[idx]: .8g}, abs_err={abs_err[idx]:.8e}, rel_err={rel_err[idx]:.8e}"
        )

    if np.any(nonzero):
        print(f"\n  Top-{top_k} worst relative-error cases:")
        valid_indices = np.where(nonzero)[0]
        worst_rel_local = np.argsort(rel_err[valid_indices])[-top_k:][::-1]
        worst_rel = valid_indices[worst_rel_local]
        for idx in worst_rel:
            print(
                f"    a={a[idx]: .8g}, b={b[idx]: .8g}, exact={exact[idx]: .8g}, "
                f"approx={approx[idx]: .8g}, abs_err={abs_err[idx]:.8e}, rel_err={rel_err[idx]:.8e}"
            )


def make_test_values(num_samples: int, seed: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if mode == "weights":
        # Typical neural-network style values: activations and weights with moderate scale.
        a = rng.normal(loc=0.0, scale=1.0, size=num_samples).astype(np.float32)
        b = rng.normal(loc=0.0, scale=0.15, size=num_samples).astype(np.float32)
    elif mode == "uniform":
        a = rng.uniform(-4.0, 4.0, size=num_samples).astype(np.float32)
        b = rng.uniform(-4.0, 4.0, size=num_samples).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Include deterministic edge-ish cases.
    fixed_a = np.array([0.0, 1.0, -1.0, 1.5, -0.75, 0.03125, 1e-4, -3.14], dtype=np.float32)
    fixed_b = np.array([9.87, 1.0, 2.0, 2.3, 4.0, -0.5, 0.7, -2.71], dtype=np.float32)
    a[: len(fixed_a)] = fixed_a
    b[: len(fixed_b)] = fixed_b
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LSB-first bit-serial Spiking Mitchell multiplication.")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--mantissa_bits", type=int, default=23, choices=range(2, 24), metavar="[2-23]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["weights", "uniform"], default="weights")
    args = parser.parse_args()

    a, b = make_test_values(args.num_samples, args.seed, args.mode)
    exact = (a.astype(np.float64) * b.astype(np.float64)).astype(np.float64)

    parallel = parallel_mitchell_product(a, b, args.mantissa_bits)
    serial, q_serial, q_parallel = serial_lsb_first_mitchell_product(a, b, args.mantissa_bits)

    q_mismatch = np.count_nonzero(q_serial != q_parallel)
    out_mismatch = np.count_nonzero(serial != parallel)

    print("=" * 80)
    print("LSB-first Spiking Mitchell FP32 Multiplication Verification")
    print("=" * 80)
    print(f"Samples       : {args.num_samples}")
    print(f"Input mode    : {args.mode}")
    print(f"Mantissa bits : {args.mantissa_bits}")
    print(f"Q-sum mismatch(serial vs parallel) : {q_mismatch}")
    print(f"Output mismatch(serial vs parallel): {out_mismatch}")

    summarize_error("Serial LSB-first Mitchell vs exact FP32 product", serial, exact)
    summarize_error("Parallel Mitchell vs exact FP32 product", parallel, exact)
    summarize_error("Serial LSB-first Mitchell vs parallel Mitchell", serial, parallel)
    print_exact_product_comparison(serial, exact, a, b)

    print("\n[Examples]")
    for i in range(min(8, args.num_samples)):
        print(
            f"  a={a[i]: .8g}, b={b[i]: .8g}, "
            f"exact={exact[i]: .8g}, serial={serial[i]: .8g}, parallel={parallel[i]: .8g}"
        )

    if q_mismatch == 0 and out_mismatch == 0:
        print("\n[PASS] LSB-first bit-serial mantissa addition exactly reproduces the parallel Mitchell model.")
    else:
        print("\n[FAIL] Serial and parallel Mitchell models differ. Inspect carry/normalization logic.")


if __name__ == "__main__":
    main()
