# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSLX implementations of binary32 operations using round-to-nearest, ties-to-even."""

FP32_DSLX = """\
fn __allo_fp32_nan(x: u32) -> bool {
  (x & u32:0x7fffffff) > u32:0x7f800000
}

fn __allo_fp32_exponent(x: u32) -> u32 {
  let exp = ((x >> u32:23) as u8) as u32;
  if exp == u32:0 { u32:1 } else { exp }
}

fn __allo_fp32_significand(x: u32) -> u24 {
  let hidden = (x & u32:0x7f800000) != u32:0;
  ((hidden as u24) << u32:23) | ((x as u23) as u24)
}

// Preserve every discarded nonzero bit in the sticky bit.
fn __allo_fp32_shr_jam<N: u32>(x: bits[N], amount: u32) -> bits[N] {
  let shifted = x >> amount;
  shifted | (((shifted << amount) != x) as bits[N])
}

// The effective exponent is at least one, including for subnormal numbers.
// The leading bit of sig is followed by 23 fraction bits and three rounding bits.
fn __allo_fp32_round(sign: u1, exponent: u32, sig: u27) -> u32 {
  let significand = sig >> u32:3;
  let residue = sig as u3;
  let increment = residue > u3:4 || (residue == u3:4 && (significand as u1));
  let rounded = (significand as u25) + (increment as u25);
  let carry = (rounded >> u32:24) as u1;
  let normalized = if carry { rounded >> u32:1 } else { rounded };
  let exponent = exponent + (carry as u32);
  let leading = (normalized >> u32:23) as u1;
  let magnitude = if exponent >= u32:255 {
    u32:0x7f800000
  } else {
    ((if leading { exponent } else { u32:0 }) << u32:23) |
    ((normalized as u23) as u32)
  };
  ((sign as u32) << u32:31) | magnitude
}

fn __allo_fp32_add(a: u32, b: u32) -> u32 {
  let a_mag = a as u31;
  let b_mag = b as u31;
  let same_sign = ((a ^ b) >> u32:31) == u32:0;
  let (large, small) = if a_mag >= b_mag { (a, b) } else { (b, a) };
  let sign = (large >> u32:31) as u1;
  let exponent = __allo_fp32_exponent(large);
  let distance = exponent - __allo_fp32_exponent(small);
  let large_sig = (__allo_fp32_significand(large) as u28) << u32:3;
  let small_sig = __allo_fp32_shr_jam(
      (__allo_fp32_significand(small) as u28) << u32:3, distance);
  let sig = if same_sign { large_sig + small_sig } else { large_sig - small_sig };
  let carry = (sig >> u32:27) as u1;
  let available = exponent - u32:1;
  let leading_zeros = clz(sig as u27) as u32;
  let shift = if leading_zeros < available { leading_zeros } else { available };
  let norm_sig = if carry {
    __allo_fp32_shr_jam(sig, u32:1) as u27
  } else {
    (sig as u27) << shift
  };
  let norm_exp = if carry { exponent + u32:1 } else { exponent - shift };

  if __allo_fp32_nan(a) || __allo_fp32_nan(b) ||
     (!same_sign && a_mag == u31:0x7f800000 && b_mag == u31:0x7f800000) {
    u32:0x7fc00000
  } else if large as u31 == u31:0x7f800000 {
    large
  } else if sig == u28:0 {
    (a & b) & u32:0x80000000
  } else {
    __allo_fp32_round(sign, norm_exp, norm_sig)
  }
}

fn __allo_fp32_mul(a: u32, b: u32) -> u32 {
  let sign = ((a ^ b) >> u32:31) as u1;
  let a_mag = a as u31;
  let b_mag = b as u31;
  let infinity = a_mag == u31:0x7f800000 || b_mag == u31:0x7f800000;
  let zero = a_mag == u31:0 || b_mag == u31:0;
  let product = (__allo_fp32_significand(a) as u48) * (__allo_fp32_significand(b) as u48);
  let high_bit = u32:47 - (clz(product) as u32);
  // The product has 46 fraction bits. Switch the exponent bias from 254 to 127
  // while normalizing at high_bit: 254 + 46 - 127 = 173.
  let exp = ((__allo_fp32_exponent(a) + __allo_fp32_exponent(b) + high_bit) as s32) - s32:173;
  let sig = if high_bit > u32:26 {
    __allo_fp32_shr_jam(product, high_bit - u32:26) as u27
  } else {
    (product << (u32:26 - high_bit)) as u27
  };
  let subnormal = exp < s32:1;
  let norm_sig = if subnormal { __allo_fp32_shr_jam(sig, (s32:1 - exp) as u32) } else { sig };
  let norm_exp = if subnormal { u32:1 } else { exp as u32 };

  if __allo_fp32_nan(a) || __allo_fp32_nan(b) || (infinity && zero) {
    u32:0x7fc00000
  } else if infinity {
    ((sign as u32) << u32:31) | u32:0x7f800000
  } else if zero {
    (sign as u32) << u32:31
  } else {
    __allo_fp32_round(sign, norm_exp, norm_sig)
  }
}

fn __allo_fp32_equal(a: u32, b: u32) -> bool {
  a == b || ((a | b) as u31 == u31:0)
}

fn __allo_fp32_less(a: u32, b: u32) -> bool {
  let a_sign = (a >> u32:31) as u1;
  let b_sign = (b >> u32:31) as u1;
  if a_sign != b_sign { a_sign && !__allo_fp32_equal(a, b) } else {
    if a_sign { a > b } else { a < b }
  }
}
"""
