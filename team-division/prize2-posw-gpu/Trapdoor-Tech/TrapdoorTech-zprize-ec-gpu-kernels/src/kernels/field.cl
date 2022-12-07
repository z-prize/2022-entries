// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#ifdef SN_CUDA
# define dfinline __device__ __forceinline__
# ifdef __GNUC__
#  define asm __asm__ __volatile__
# else
#  define asm asm volatile
# endif

dfinline FIELD& FIELD_add_sn(FIELD& a, const FIELD& b);
dfinline FIELD& FIELD_sub_sn(FIELD& a, const FIELD& b);
dfinline void FIELD_double_sn(FIELD& b);
#endif


#define FIELD_BITS (FIELD_LIMBS * FIELD_LIMB_BITS)
#if FIELD_LIMB_BITS == 32
  #define FIELD_mac_with_carry mac_with_carry_32
  #define FIELD_add_with_carry add_with_carry_32
#elif FIELD_LIMB_BITS == 64
  #define FIELD_mac_with_carry mac_with_carry_64
  #define FIELD_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool FIELD_gte(FIELD a, FIELD b) {
  for(char i = FIELD_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool FIELD_eq(FIELD a, FIELD b) {
  for(uchar i = 0; i < FIELD_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(OPENCL_NVIDIA) || defined(CUDA) || defined (SN_CUDA)
  #define FIELD_add_ FIELD_add_nvidia
  #define FIELD_sub_ FIELD_sub_nvidia
#else
  DEVICE FIELD FIELD_add_(FIELD a, FIELD b) {
    bool carry = 0;
    for(uchar i = 0; i < FIELD_LIMBS; i++) {
      FIELD_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  FIELD FIELD_sub_(FIELD a, FIELD b) {
    bool borrow = 0;
    for(uchar i = 0; i < FIELD_LIMBS; i++) {
      FIELD_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE FIELD FIELD_sub(FIELD a, FIELD b) {
#ifdef SN_CUDA
  FIELD res = FIELD_sub_sn(a, b);
#else
  FIELD res = FIELD_sub_(a, b);
  if(!FIELD_gte(a, b)) res = FIELD_add_(res, FIELD_P);
#endif
  return res;
}

// Modular addition
DEVICE FIELD FIELD_add(FIELD a, FIELD b) {
#ifdef SN_CUDA
  FIELD res = FIELD_add_sn(a, b);
#else
  FIELD res = FIELD_add_(a, b);
  if(FIELD_gte(res, FIELD_P)) res = FIELD_sub_(res, FIELD_P);
#endif
  return res;
}


#ifdef CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void FIELD_reduce(uint32_t accLow[FIELD_LIMBS], uint32_t np0, uint32_t fq[FIELD_LIMBS]) {
  // accLow is an IN and OUT vector
  // count must be even
  const uint32_t count = FIELD_LIMBS;
  uint32_t accHigh[FIELD_LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void FIELD_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = FIELD_LIMBS;
  const uint32_t yLimbs  = FIELD_LIMBS;
  const uint32_t xyLimbs = FIELD_LIMBS * 2;
  uint32_t temp[FIELD_LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE FIELD FIELD_mul_nvidia(FIELD a, FIELD b) {
  // Perform full multiply
  limb ab[2 * FIELD_LIMBS];
  FIELD_mult_v1(a.val, b.val, ab);

  uint32_t io[FIELD_LIMBS];
  #pragma unroll
  for(int i=0;i<FIELD_LIMBS;i++) {
    io[i]=ab[i];
  }
  FIELD_reduce(io, FIELD_INV, FIELD_P.val);

  // Add io to the upper words of ab
  ab[FIELD_LIMBS] = add_cc(ab[FIELD_LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < FIELD_LIMBS - 1; j++) {
    ab[j + FIELD_LIMBS] = addc_cc(ab[j + FIELD_LIMBS], io[j]);
  }
  ab[2 * FIELD_LIMBS - 1] = addc(ab[2 * FIELD_LIMBS - 1], io[FIELD_LIMBS - 1]);

  FIELD r;
  #pragma unroll
  for (int i = 0; i < FIELD_LIMBS; i++) {
    r.val[i] = ab[i + FIELD_LIMBS];
  }

  if (FIELD_gte(r, FIELD_P)) {
    r = FIELD_sub_(r, FIELD_P);
  }

  return r;
}

#endif

#ifdef SN_CUDA
// sn_dbl
dfinline void FIELD_double_sn(FIELD& a)
{
  size_t i;
  size_t n = FIELD_LIMBS;
  uint32_t tmp[FIELD_LIMBS + 1];
  asm("{ .reg.pred %top;");

  asm("add.cc.u32 %0, %0, %0;" : "+r"(a.val[0]));
  for (i = 1; i < n; i++)
      asm("addc.cc.u32 %0, %0, %0;" : "+r"(a.val[i]));
  if (253 % 32 == 0)
      asm("addc.u32 %0, 0, 0;" : "=r"(tmp[n]));

  asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(a.val[0]), "r"(FIELD_P.val[0]));
  for (i = 1; i < n; i++)
      asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(a.val[i]), "r"(FIELD_P.val[i]));
  if (253 % 32 == 0)
      asm("subc.u32 %0, %0, 0; setp.eq.u32 %top, %0, 0;" : "+r"(tmp[n]));
  else
      asm("subc.u32 %0, 0, 0; setp.eq.u32 %top, %0, 0;" : "=r"(tmp[n]));

  for (i = 0; i < n; i++)
      asm("@%top mov.b32 %0, %1;" : "+r"(a.val[i]) : "r"(tmp[i]));

  asm("}");
}

// sn_sub
dfinline FIELD& FIELD_sub_sn(FIELD& a, const FIELD& b)
{
    size_t i;
    size_t n = FIELD_LIMBS;
    uint32_t tmp[FIELD_LIMBS], borrow;

    asm("sub.cc.u32 %0, %0, %1;" : "+r"(a.val[0]) : "r"(b.val[0]));
    for (i = 1; i < n; i++)
        asm("subc.cc.u32 %0, %0, %1;" : "+r"(a.val[i]) : "r"(b.val[i]));
    asm("subc.u32 %0, 0, 0;" : "=r"(borrow));

    asm("add.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(a.val[0]), "r"(FIELD_P.val[0]));
    for (i = 1; i < n-1; i++)
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(a.val[i]), "r"(FIELD_P.val[i]));
    asm("addc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(a.val[i]), "r"(FIELD_P.val[i]));

    asm("{ .reg.pred %top; setp.ne.u32 %top, %0, 0;" :: "r"(borrow));
    for (i = 0; i < n; i++)
        asm("@%top mov.b32 %0, %1;" : "+r"(a.val[i]) : "r"(tmp[i]));
    asm("}");

    return a;
}

// sn_add
dfinline FIELD& FIELD_add_sn(FIELD& a, const FIELD& b)
{
    size_t i;
    size_t n = FIELD_LIMBS;
    uint32_t tmp[FIELD_LIMBS + 1];
    asm("{ .reg.pred %top;");

    asm("add.cc.u32 %0, %0, %1;" : "+r"(a.val[0]) : "r"(b.val[0]));
    for (i = 1; i < n; i++)
        asm("addc.cc.u32 %0, %0, %1;" : "+r"(a.val[i]) : "r"(b.val[i]));
    if (253 % 32 == 0)
        asm("addc.u32 %0, 0, 0;" : "=r"(tmp[n]));

    asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(a.val[0]), "r"(FIELD_P.val[0]));
    for (i = 1; i < n; i++)
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(a.val[i]), "r"(FIELD_P.val[i]));
    if (253 % 32 == 0)
        asm("subc.u32 %0, %0, 0; setp.eq.u32 %top, %0, 0;" : "+r"(tmp[n]));
    else
        asm("subc.u32 %0, 0, 0; setp.eq.u32 %top, %0, 0;" : "=r"(tmp[n]));

    for (i = 0; i < n; i++)
        asm("@%top mov.b32 %0, %1;" : "+r"(a.val[i]) : "r"(tmp[i]));

    asm("}");
    return a;
}


// sn_mul
static dfinline void FIELD_mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = FIELD_LIMBS)
{
    for (size_t j = 0; j < n; j += 2)
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(acc[j]), "=r"(acc[j+1])
            : "r"(a[j]), "r"(bi));
}

static dfinline void FIELD_cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = FIELD_LIMBS)
{
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
        : "+r"(acc[0]), "+r"(acc[1])
        : "r"(a[0]), "r"(bi));
    for (size_t j = 2; j < n; j += 2)
        asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(acc[j]), "+r"(acc[j+1])
            : "r"(a[j]), "r"(bi));
}

typedef struct {
    uint32_t even[2 * FIELD_LIMBS];
} FIELD_wide_t;

static dfinline void FIELD_mad_row(uint32_t* odd, uint32_t* even,
                           const uint32_t* a, uint32_t bi, size_t n = FIELD_LIMBS)
{
    FIELD_cmad_n(odd, a+1, bi, n-2);
    asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
        : "=r"(odd[n-2]), "=r"(odd[n-1])
        : "r"(a[n-1]), "r"(bi));

    FIELD_cmad_n(even, a, bi, n);
    asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
}

dfinline void FIELD_wide_mul(const uint32_t *a, const uint32_t *b, FIELD_wide_t& res, size_t n = FIELD_LIMBS)     //// |a|*|b|
{
    size_t i = 0;
    uint32_t odd[2* FIELD_LIMBS -2];

    FIELD_mul_n(res.even, a,  b[0]);
    FIELD_mul_n(odd, a+1, b[0]);
    ++i; FIELD_mad_row(&res.even[i+1], &odd[i-1], a, b[i]);

    #pragma unroll
    while (i < n - 2) {
        ++i; FIELD_mad_row(&odd[i],    &res.even[i],  a, b[i]);
        ++i; FIELD_mad_row(&res.even[i+1], &odd[i-1], a, b[i]);
    }

    // merge |even| and |odd|
    asm("add.cc.u32 %0, %0, %1;" : "+r"(res.even[1]) : "r"(odd[0]));
    for (i = 1; i < 2 * n - 2; i++)
        asm("addc.cc.u32 %0, %0, %1;" : "+r"(res.even[i+1]) : "r"(odd[i]));
    asm("addc.u32 %0, %0, 0;" : "+r"(res.even[i+1]));
}

static dfinline void FIELD_qad_row(uint32_t* odd, uint32_t* even,
                           const uint32_t* a, uint32_t bi, size_t n)
{
    FIELD_cmad_n(odd, a, bi, n-2);
    asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
        : "=r"(odd[n-2]), "=r"(odd[n-1])
        : "r"(a[n-2]), "r"(bi));

    FIELD_cmad_n(even, a+1, bi, n-2);
    asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
}

dfinline void FIELD_wide_sqr(const uint32_t *a, FIELD_wide_t& res, size_t n = FIELD_LIMBS)      //// |a|**2
{
    size_t i = 0, j;
    uint32_t odd[2 * FIELD_LIMBS -2];

    // perform |a[i]|*|a[j]| for all j>i
    FIELD_mul_n(res.even+2, a+2, a[0], n-2);
    FIELD_mul_n(odd,    a+1, a[0], n);

    #pragma unroll
    while (i < n-4) {
        ++i; FIELD_mad_row(&res.even[2*i+2], &odd[2*i], &a[i+1], a[i], n-i-1);
        ++i; FIELD_qad_row(&odd[2*i], &res.even[2*i+2], &a[i+1], a[i], n-i);
    }

    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(res.even[2*n-4]), "=r"(res.even[2*n-3])
        : "r"(a[n-1]), "r"(a[n-3]));
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
        : "+r"(odd[2*n-6]), "+r"(odd[2*n-5])
        : "r"(a[n-2]), "r"(a[n-3]));
    asm("addc.u32 %0, %0, 0;" : "+r"(res.even[2*n-3]));

    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(odd[2*n-4]), "=r"(odd[2*n-3])
        : "r"(a[n-1]), "r"(a[n-2]));

    // merge |even[2:]| and |odd[1:]|
    asm("add.cc.u32 %0, %0, %1;" : "+r"(res.even[2]) : "r"(odd[1]));
    for (j = 2; j < 2*n-3; j++)
        asm("addc.cc.u32 %0, %0, %1;" : "+r"(res.even[j+1]) : "r"(odd[j]));
    asm("addc.u32 %0, %1, 0;" : "+r"(res.even[j+1]) : "r"(odd[j]));

    // double |even|
    res.even[0] = 0;
    asm("add.cc.u32 %0, %1, %1;" : "=r"(res.even[1]) : "r"(odd[0]));
    for (j = 2; j < 2*n-1; j++)
        asm("addc.cc.u32 %0, %0, %0;" : "+r"(res.even[j]));
    asm("addc.u32 %0, 0, 0;" : "=r"(res.even[j]));

    // accumulate "diagonal" |a[i]|*|a[i]| product
    i = 0;
    asm("mad.lo.cc.u32 %0, %2, %2, %0; madc.hi.cc.u32 %1, %2, %2, %1;"
        : "+r"(res.even[2*i]), "+r"(res.even[2*i+1])
        : "r"(a[i]));
    for (++i; i < n; i++)
        asm("madc.lo.cc.u32 %0, %2, %2, %0; madc.hi.cc.u32 %1, %2, %2, %1;"
            : "+r"(res.even[2*i]), "+r"(res.even[2*i+1])
            : "r"(a[i]));
}

static dfinline void FIELD_madc_n_rshift(uint32_t* odd, const uint32_t *a, uint32_t bi, size_t n = FIELD_LIMBS)
{
    for (size_t j = 0; j < n-2; j += 2)
        asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, %5;"
            : "=r"(odd[j]), "=r"(odd[j+1])
            : "r"(a[j]), "r"(bi), "r"(odd[j+2]), "r"(odd[j+3]));
    asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
        : "=r"(odd[n-2]), "=r"(odd[n-1])
        : "r"(a[n-2]), "r"(bi));
}

static dfinline void FIELD_mad_n_redc(uint32_t *even, uint32_t* odd,
                              const uint32_t *a, uint32_t bi, bool first=false, size_t n = FIELD_LIMBS)
{
    if (first) {
        FIELD_mul_n(odd, a+1, bi);
        FIELD_mul_n(even, a,  bi);
    } else {
        asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
        FIELD_madc_n_rshift(odd, a+1, bi);
        FIELD_cmad_n(even, a, bi);
        asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
    }

    uint32_t mi = even[0] * FIELD_INV;

    FIELD_cmad_n(odd, FIELD_P.val + 1, mi);
    FIELD_cmad_n(even, FIELD_P.val,  mi);
    asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
}

static dfinline void FIELD_mul_by_1_row(uint32_t* even, uint32_t* odd, bool first=false, size_t n = FIELD_LIMBS)
{
    uint32_t mi;

    if (first) {
        mi = even[0] * FIELD_INV;
        FIELD_mul_n(odd, FIELD_P.val+1, mi);
        FIELD_cmad_n(even, FIELD_P.val,  mi);
        asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
    } else {
        asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
# if 1      // do we trust the compiler to *not* touch the carry flag here?
        mi = even[0] * FIELD_INV;
# else
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(mi) : "r"(even[0]), "r"(M0));
# endif
        FIELD_madc_n_rshift(odd, FIELD_P.val+1, mi);
        FIELD_cmad_n(even, FIELD_P.val, mi);
        asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
    }
}

dfinline void FIELD_mul_by_1(uint32_t *even, size_t n = FIELD_LIMBS)
{
    uint32_t odd[FIELD_LIMBS];
    size_t i;

    #pragma unroll
    for (i = 0; i < n; i += 2) {
        FIELD_mul_by_1_row(&even[0], &odd[0], i==0);
        FIELD_mul_by_1_row(&odd[0], &even[0]);
    }

    asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
    for (i = 1; i < n-1; i++)
        asm("addc.cc.u32 %0, %0, %1;" : "+r"(even[i]) : "r"(odd[i+1]));
    asm("addc.u32 %0, %0, 0;" : "+r"(even[i]));
}

dfinline  FIELD FIELD_mul_sn(const FIELD& a, const FIELD& b)
{
    if (FIELD_eq(a, b) && 0) {
        union { FIELD_wide_t w; FIELD s[2]; } ret;
        FIELD_wide_sqr(a.val, ret.w, FIELD_LIMBS);
        FIELD_mul_by_1(ret.s[0].val);
        return ret.s[0] = FIELD_add(ret.s[0], ret.s[1]);
    } else if (253 % 32 == 0) {
        union { FIELD_wide_t w; FIELD s[2]; } ret;
        FIELD_wide_mul(a.val, b.val, ret.w);
        FIELD_mul_by_1(ret.s[0].val);
        return ret.s[0] = FIELD_add(ret.s[0], ret.s[1]);
    } else {
        int n = FIELD_LIMBS;
        FIELD even;
        uint32_t odd[FIELD_LIMBS+1];
        size_t i;
        asm("{ .reg.pred %top;");

        #pragma unroll
        for (i = 0; i < n; i += 2) {
            FIELD_mad_n_redc(&even.val[0], &odd[0], a.val, b.val[i], i==0);
            FIELD_mad_n_redc(&odd[0], &even.val[0], a.val, b.val[i+1]);
        }

        // merge |even| and |odd|
        asm("add.cc.u32 %0, %0, %1;" : "+r"(even.val[0]) : "r"(odd[1]));
        for (i = 1; i < n-1; i++)
            asm("addc.cc.u32 %0, %0, %1;" : "+r"(even.val[i]) : "r"(odd[i+1]));
        asm("addc.u32 %0, %0, 0;" : "+r"(even.val[i]));

        // final subtraction
        asm("sub.cc.u32 %0, %1, %2;" : "=r"(odd[0]) : "r"(even.val[0]), "r"(FIELD_P.val[0]));
        for (i = 1; i < n; i++)
            asm("subc.cc.u32 %0, %1, %2;" : "=r"(odd[i]) : "r"(even.val[i]), "r"(FIELD_P.val[i]));
        asm("subc.u32 %0, 0, 0; setp.eq.u32 %top, %0, 0;" : "=r"(odd[i]));

        for (i = 0; i < n; i++)
            asm("@%top mov.b32 %0, %1;" : "+r"(even.val[i]) : "r"(odd[i]));

        asm("}");
        return even;
    }
}
#endif

// Modular multiplication
DEVICE FIELD FIELD_mul_default(FIELD a, FIELD b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  FIELD_limb t[FIELD_LIMBS + 2] = {0};
  for(uchar i = 0; i < FIELD_LIMBS; i++) {
    FIELD_limb carry = 0;
    for(uchar j = 0; j < FIELD_LIMBS; j++)
      t[j] = FIELD_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[FIELD_LIMBS] = FIELD_add_with_carry(t[FIELD_LIMBS], &carry);
    t[FIELD_LIMBS + 1] = carry;

    carry = 0;
    FIELD_limb m = FIELD_INV * t[0];
    FIELD_mac_with_carry(m, FIELD_P.val[0], t[0], &carry);
    for(uchar j = 1; j < FIELD_LIMBS; j++)
      t[j - 1] = FIELD_mac_with_carry(m, FIELD_P.val[j], t[j], &carry);

    t[FIELD_LIMBS - 1] = FIELD_add_with_carry(t[FIELD_LIMBS], &carry);
    t[FIELD_LIMBS] = t[FIELD_LIMBS + 1] + carry;
  }

  FIELD result;
  for(uchar i = 0; i < FIELD_LIMBS; i++) result.val[i] = t[i];

  if(FIELD_gte(result, FIELD_P)) result = FIELD_sub_(result, FIELD_P);

  return result;
}

#if defined(CUDA)
DEVICE FIELD FIELD_mul(FIELD a, FIELD b) {
  return FIELD_mul_nvidia(a, b);
}
#elif defined(SN_CUDA)
DEVICE FIELD FIELD_mul(FIELD a, FIELD b) {
  return FIELD_mul_sn(a, b);
}
#else
DEVICE FIELD FIELD_mul(FIELD a, FIELD b) {
  return FIELD_mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE FIELD FIELD_sqr(FIELD a) {
#ifdef SN_CUDA
  union { FIELD_wide_t w; FIELD s[2]; } ret;
  FIELD_wide_sqr(a.val, ret.w, FIELD_LIMBS);
  FIELD_mul_by_1(ret.s[0].val);
  return ret.s[0] = FIELD_add(ret.s[0], ret.s[1]);
#else
  return FIELD_mul(a, a);
#endif
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of FIELD_add(a, a)
DEVICE FIELD FIELD_double(FIELD a) {
#ifdef SN_CUDA
  FIELD_double_sn(a);
#else
  for(uchar i = FIELD_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (FIELD_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(FIELD_gte(a, FIELD_P)) a = FIELD_sub_(a, FIELD_P);
#endif
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE FIELD FIELD_pow(FIELD base, uint exponent) {
  FIELD res = FIELD_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = FIELD_mul(res, base);
    exponent = exponent >> 1;
    base = FIELD_sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE FIELD FIELD_pow_lookup(GLOBAL FIELD *bases, uint exponent) {
  FIELD res = FIELD_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = FIELD_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE FIELD FIELD_mont(FIELD a) {
  return FIELD_mul(a, FIELD_R2);
}

DEVICE FIELD FIELD_unmont(FIELD a) {
  FIELD one = FIELD_ZERO;
  one.val[0] = 1;
  return FIELD_mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool FIELD_get_bit(FIELD l, uint i) {
  return (l.val[FIELD_LIMBS - 1 - i / FIELD_LIMB_BITS] >> (FIELD_LIMB_BITS - 1 - (i % FIELD_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint FIELD_get_bits(FIELD l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= FIELD_get_bit(l, skip + i);
  }
  return ret;
}
