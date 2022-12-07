// Copyright Trapdoor-Tech
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>

#ifndef __EXTE_T_HPP__
#define __EXTE_T_HPP__

#ifndef __CUDA_ARCH__
# undef  __host__
# define __host__
# undef  __device__
# define __device__
# undef  __noinline__
# define __noinline__
#endif

template<class field_t> class exte_t {
public:
  field_t X, Y, T, Z;

public:
#ifdef __NVCC__
  class affine_inf_t { friend exte_t;
    field_t X, Y, T;

    inline __device__ bool is_inf() const
    {   return (bool)(X.is_zero() & (Y-field_t::one()).is_zero());   }
  };
#else
  class affine_inf_t { friend exte_t;
    field_t X, Y, T;

    inline __device__ bool is_inf() const
    {   return (bool)(X.is_zero() & (Y==field_t::one()));   }
  };
#endif

  class affine_t { friend exte_t;
    field_t X, Y, T;

    inline __device__ bool is_inf() const
    {   return (bool)(X.is_zero() & (Y==field_t::one()));   }

  public:
    inline affine_t& operator=(const exte_t& a)
    {
      Y = 1 / a.Z;
      X = Y * a.X;
      Y = Y * a.Y;
      T = X * Y;
      return *this;
    }
    inline affine_t(const exte_t& a)  { *this = a; }
  };

  inline operator affine_t() const      { return affine_t(*this); }

  template<class affine_t>
  inline __device__ exte_t& operator=(const affine_t& a)
  {
    X = a.X;
    Y = a.Y;
    T = a.T;
    // this works as long as |a| was confirmed to be non-infinity
    Z = field_t::one();
    return *this;
  }

  inline __device__ bool is_inf() const { return (bool)(X.is_zero() & (Y-Z).is_zero()); }
  inline __device__ void inf()          { X.zero(); Y = field_t::one(); T.zero(); Z = field_t::one(); }

  /*
   * http://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-4
   * http://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#doubling-dbl-2008-hwcd
   * Addition costs 8M, Double costs 2M+4M+4S
   */
  __device__ void add(const exte_t& p2)
  {
    if (p2.is_inf()) {
      return;
    }
#ifdef __CUDA_ARCH__
    exte_t p31 = *this;
#else
    exte_t& p31 = *this;
#endif
      field_t A, B, C, D, E, F, G, H, p31T;
      field_t y1_minus_x1, y2_add_x2, y1_add_x1, y2_minus_x2;

      y1_minus_x1 = p31.Y - p31.X;
      y2_minus_x2 = p2.Y - p2.X;
      y1_add_x1 = p31.Y + p31.X;
      y2_add_x2 = p2.Y + p2.X;

      A = y1_minus_x1 * y2_add_x2;
      B = y1_add_x1 * y2_minus_x2;

      F = B - A;

      if (F.is_zero()) {		// use doubling here
	A = p31.X^2;
	B = p31.Y^2;
	C = p31.Z^2;
	C <<= 1;
	D.zero();
        D -= A;

	E = y1_add_x1^2;
	E -= A;
	E -= B;
	G = D + B;
	F = G - C;
	H = D - B;
      } else {
	C = p31.Z * p2.T;
	C <<= 1;
	D = p31.T * p2.Z;
	D <<= 1;

	E = D + C;
	// F = B - A;
	G = B + A;
	H = D - C;

      }

      p31.X = E * F;
      p31.Y = G * H;
      p31.T = E * H;
      p31.Z = F * G;

#ifdef __CUDA_ARCH__
    *this = p31;
#endif
  }

  /*
   * http://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-madd-2008-hwcd-4
   * http://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#doubling-dbl-2008-hwcd
   * Mixed Addition costs 7M, Double costs 2M+4M+4S
   */
  template<class affine_t>
  __device__ void add(const affine_t& p2, bool subtract = false)
  {
#ifdef __CUDA_ARCH__
    exte_t p31 = *this;
#else
    exte_t& p31 = *this;
#endif
      field_t A, B, C, D, E, F, G, H, p31T;
      field_t y1_minus_x1, y2_add_x2, y1_add_x1, y2_minus_x2;

      y1_minus_x1 = p31.Y - p31.X;
      y2_minus_x2 = p2.Y - p2.X;
      y1_add_x1 = p31.Y + p31.X;
      y2_add_x2 = p2.Y + p2.X;

      A = y1_minus_x1 * y2_add_x2;
      B = y1_add_x1 * y2_minus_x2;

      F = B - A;

      if (F.is_zero()) {		// use doubling here
	A = p31.X^2;
	B = p31.Y^2;
	C = p31.Z^2;
	C <<= 1;
	D.zero();
        D -= A;

	E = y1_add_x1^2;
	E -= A;
	E -= B;
	G = D + B;
	F = G - C;
	H = D - B;
      } else {
	C = p31.Z * p2.T;
	C <<= 1;
	D = p31.T;
	D <<= 1;

	E = D + C;
	// F = B - A;
	G = B + A;
	H = D - C;

      }
      p31.X = E * F;
      p31.Y = G * H;
      p31.T = E * H;
      p31.Z = F * G;

#ifdef __CUDA_ARCH__
    *this = p31;
#endif
  }

  // for CPU
  static void dbl(exte_t& p3, const exte_t& p1)
  {
    if (p3.is_inf()) {
      return;
    }

    field_t A, B, C, D;

    A = p1.X^2;
    B = p1.Y^2;
    C = p1.Z^2;
    C <<= 1;
    D = -A;

    field_t x1_add_y1, E, F, G, H;

    x1_add_y1 = p1.X + p1.Y;
    E = x1_add_y1^2;
    E -= A;
    E -= B;
    G = D + B;
    F = G - C;
    H = D - B;

    p3.X = E * F;
    p3.Y = G * H;
    p3.T = E * H;
    p3.Z = F * G;
  }
  inline void dbl() { dbl(*this, *this); }

};
#endif
