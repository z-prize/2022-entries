/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

__device__ __constant__ uint32_t BLS12377_MainField_NP0=0xFFFFFFFF;
__device__ __constant__ uint32_t zeroConstant=0;

namespace BLS12377 {

class MainField {
  public:
  static const uint32_t limbs=12;
  static const uint32_t bytes=limbs*4;
};

class G1Montgomery {
  public:
  static const uint32_t limbs=BLS12377::MainField::limbs;
  static const uint32_t bytes=limbs*4;
  
  typedef BLS12377::MainField MainField;
  
  typedef uint32_t Value[limbs];
  typedef uint64_t EvenOdd[limbs];

  __device__ __forceinline__ static uint32_t qTerm(uint32_t lowWord) {
    // Since np0 is 0xFFFFFFFF, this is just -lowWord
    return zeroConstant-lowWord;
  }
  
  __device__ __forceinline__ static void loadShared(Value& field, uint32_t offset) {
    uint4 u4a, u4b, u4c;
    
    u4a=load_shared_u4(offset);
    u4b=load_shared_u4(offset+16);
    u4c=load_shared_u4(offset+32);
   
    field[0]=u4a.x;
    field[1]=u4a.y;
    field[2]=u4a.z;
    field[3]=u4a.w;
    field[4]=u4b.x; 
    field[5]=u4b.y;
    field[6]=u4b.z;
    field[7]=u4b.w;
    field[8]=u4c.x;
    field[9]=u4c.y;
    field[10]=u4c.z;
    field[11]=u4c.w;
  }
  
  __device__ __forceinline__ static void load(Value& field, void* ptr) {
    uint4 u4a, u4b, u4c;
    
    u4a=*(uint4*)byteOffset(ptr, 0);
    u4b=*(uint4*)byteOffset(ptr, 16);
    u4c=*(uint4*)byteOffset(ptr, 32);

    field[0]=u4a.x;
    field[1]=u4a.y;
    field[2]=u4a.z;
    field[3]=u4a.w;
    field[4]=u4b.x; 
    field[5]=u4b.y;
    field[6]=u4b.z;
    field[7]=u4b.w;
    field[8]=u4c.x;
    field[9]=u4c.y;
    field[10]=u4c.z;
    field[11]=u4c.w;
  }
  
  __device__ __forceinline__ static void loadUnaligned(Value& field, void* ptr) {
    #pragma unroll
    for(int32_t i=0;i<limbs;i++) 
      field[i]=*(uint32_t*)byteOffset(ptr, i*4);
  }
  
  __device__ __forceinline__ static void store(void* ptr, const Value& field) {
    uint4 u4a, u4b, u4c;
    
    u4a.x=field[0];
    u4a.y=field[1];
    u4a.z=field[2];
    u4a.w=field[3];
    u4b.x=field[4];
    u4b.y=field[5];
    u4b.z=field[6];
    u4b.w=field[7];
    u4c.x=field[8];
    u4c.y=field[9];
    u4c.z=field[10];
    u4c.w=field[11];
    
    *(uint4*)byteOffset(ptr, 0)=u4a;
    *(uint4*)byteOffset(ptr, 16)=u4b;
    *(uint4*)byteOffset(ptr, 32)=u4c;
  }
  
  __device__ __forceinline__ static bool isEqual(const Value& a, const Value& b) {
    uint32_t compare=a[0] ^ b[0];
    
    #pragma unroll
    for(int32_t i=1;i<limbs;i++) 
      compare=compare | (a[i] ^ b[i]);
    return compare==0;
  }
  
  __device__ __forceinline__ static void setConstant(Value& field, uint32_t index) {
    uint4 u4a, u4b, u4c;
    
    u4a=load_shared_u4(512*0 + 16*index);
    u4b=load_shared_u4(512*1 + 16*index);
    u4c=load_shared_u4(512*2 + 16*index);
    
    field[0]=u4a.x;
    field[1]=u4a.y;
    field[2]=u4a.z;
    field[3]=u4a.w;
    field[4]=u4b.x; 
    field[5]=u4b.y;
    field[6]=u4b.z;
    field[7]=u4b.w;
    field[8]=u4c.x;
    field[9]=u4c.y;
    field[10]=u4c.z;
    field[11]=u4c.w;
  }
    
  __device__ __forceinline__ static void setZero(Value& field) {
    setConstant(field, 0);
  }
  
  __device__ __forceinline__ static void setN(Value& field) {
    setConstant(field, 1);    
  }
  
  __device__ __forceinline__ static void setOne(Value& field) {
    setConstant(field, 16);
  }
  
  __device__ __forceinline__ static void setRSquared(Value& field) {
    setConstant(field, 17);
  }
  
  __device__ __forceinline__ static void set(Value& field, const Value& a) {
    #pragma unroll
    for(int32_t i=0;i<12;i++)
      field[i]=a[i];
  }
  
  __device__ __forceinline__ static void setPermuteLow(Value& field, const Value& a) {
    #pragma unroll
    for(int32_t i=0;i<12;i++)
      field[i]=prmt(a[i], 0, 0x3210); 
  }
  
  __device__ __forceinline__ static void setPermuteHigh(Value& field, const Value& a) {
    #pragma unroll
    for(int32_t i=0;i<12;i++)
      field[i]=prmt(0, a[i], 0x7654); 
  }

  __device__ __forceinline__ static void swap(Value& a, Value& b) {
    #pragma unroll
    for(int32_t i=0;i<12;i++) {
      uint32_t swap=a[i];
      a[i]=b[i];
      b[i]=swap;
    }
  }
    
  __device__ __forceinline__ static void add(Value& field, const Value& a, const Value& b) {
    mp_add<limbs>(field, a, b);
  }
  
  __device__ __forceinline__ static bool sub(Value& field, const Value& a, const Value& b) {
    return mp_sub_carry<limbs>(field, a, b);
  }
  
  __device__ __forceinline__ static void mul(Value& r, const Value& a, const Value& b, const Value& n) {
    bool    carry;
    EvenOdd evenOdd;

    #if defined(PHONY)
      // useful for inspecting the SASS
      carry=false;
      #pragma unroll
      for(int i=0;i<12;i++) {
        evenOdd[i]=mulwide(a[i], 0xCAFEBABE);
        evenOdd[i]=madwide(b[i], 0xBAADF00D, evenOdd[i]);
      }
    #else
      carry=mp_mul_red_cl<G1Montgomery, limbs>(evenOdd, a, b, n);  
    #endif  
    mp_merge_cl<limbs>(r, evenOdd, carry);
  }

  __device__ __forceinline__ static void sqr(Value& r, Value& temp, const Value& a, const Value& n) {
    bool    carry;
    EvenOdd evenOdd;

    #if defined(PHONY)
      // useful for inspecting the SASS
      carry=false;
      #pragma unroll
      for(int i=0;i<12;i++) 
        evenOdd[i]=mulwide(a[i], 0xCAFEBABE);
    #else
      carry=mp_sqr_red_cl<G1Montgomery, limbs>(evenOdd, temp, a, n);
    #endif
    mp_merge_cl<limbs>(r, evenOdd, carry);
  }
  
  __device__ __forceinline__ static void mul(EvenOdd& evenOdd, bool& carry, const Value& a, const Value& b, const Value& n) {
    #if defined(PHONY)
      // useful for inspecting the SASS
      carry=false;
      #pragma unroll
      for(int i=0;i<12;i++) {
        evenOdd[i]=mulwide(a[i], 0xCAFEBABE);
        evenOdd[i]=madwide(b[i], 0xBAADF00D, evenOdd[i]);
      }
    #else
      carry=mp_mul_red_cl<G1Montgomery, limbs>(evenOdd, a, b, n);
    #endif
  }
  
  __device__ __forceinline__ static bool sqr(EvenOdd& evenOdd, Value& temp, bool& carry, const Value& a, const Value& n) {
    #if defined(PHONY)
      // useful for inspecting the SASS
      carry=false;
      #pragma unroll
      for(int i=0;i<12;i++) 
        evenOdd[i]=mulwide(a[i], 0xCAFEBABE);
    #else
      carry=mp_sqr_red_cl<G1Montgomery, limbs>(evenOdd, temp, a, n);
    #endif
  }

  __device__ __forceinline__ static void merge(Value& r, const EvenOdd& evenOdd, bool carry) {
    mp_merge_cl<limbs>(r, evenOdd, carry);
  }
  
  __device__ __forceinline__ static void addN(Value& field, const Value& a) {
    Value localN;

    localN[0]  = 0x00000001;
    localN[1]  = 0x8508C000;
    localN[2]  = 0x30000000;
    localN[3]  = 0x170B5D44;
    localN[4]  = 0xBA094800;
    localN[5]  = 0x1EF3622F;
    localN[6]  = 0x00F5138F;
    localN[7]  = 0x1A22D9F3;
    localN[8]  = 0x6CA1493B;
    localN[9]  = 0xC63B05C0;
    localN[10] = 0x17C510EA;
    localN[11] = 0x01AE3A46;
    add(field, a, localN);
  }

  __device__ __forceinline__ static void add2N(Value& field, const Value& a) {
    Value local2N;
    
    local2N[0]  = 0x00000002;
    local2N[1]  = 0x0A118000;
    local2N[2]  = 0x60000001;
    local2N[3]  = 0x2E16BA88;
    local2N[4]  = 0x74129000;
    local2N[5]  = 0x3DE6C45F;
    local2N[6]  = 0x01EA271E;
    local2N[7]  = 0x3445B3E6;
    local2N[8]  = 0xD9429276;
    local2N[9]  = 0x8C760B80;
    local2N[10] = 0x2F8A21D5;
    local2N[11] = 0x035C748C; 
    add(field, a, local2N);
  }
  
  __device__ __forceinline__ static void add3N(Value& field, const Value& a) {
    Value local3N;
        
    local3N[0]  = 0x00000003;
    local3N[1]  = 0x8F1A4000;
    local3N[2]  = 0x90000001;
    local3N[3]  = 0x452217CC;
    local3N[4]  = 0x2E1BD800;
    local3N[5]  = 0x5CDA268F;
    local3N[6]  = 0x02DF3AAD;
    local3N[7]  = 0x4E688DD9;
    local3N[8]  = 0x45E3DBB1;
    local3N[9]  = 0x52B11141;
    local3N[10] = 0x474F32C0;
    local3N[11] = 0x050AAED2;
    add(field, a, local3N);
  }
    
  __device__ __forceinline__ static void add4N(Value& field, const Value& a) {
    Value local4N;

    local4N[0]  = 0x00000004;
    local4N[1]  = 0x14230000;
    local4N[2]  = 0xC0000002;
    local4N[3]  = 0x5C2D7510;
    local4N[4]  = 0xE8252000;
    local4N[5]  = 0x7BCD88BE;
    local4N[6]  = 0x03D44E3C;
    local4N[7]  = 0x688B67CC;
    local4N[8]  = 0xB28524EC;
    local4N[9]  = 0x18EC1701;
    local4N[10] = 0x5F1443AB;
    local4N[11] = 0x06B8E918;
    add(field, a, local4N);
  }

  __device__ __forceinline__ static void add5N(Value& field, const Value& a) {
    Value local5N;

    local5N[0]  = 0x00000005;
    local5N[1]  = 0x992BC000;
    local5N[2]  = 0xF0000002;
    local5N[3]  = 0x7338D254;
    local5N[4]  = 0xA22E6800;
    local5N[5]  = 0x9AC0EAEE;
    local5N[6]  = 0x04C961CB;
    local5N[7]  = 0x82AE41BF;
    local5N[8]  = 0x1F266E27;
    local5N[9]  = 0xDF271CC2;
    local5N[10] = 0x76D95495;
    local5N[11] = 0x0867235E;
    add(field, a, local5N);
  }
  
  __device__ __forceinline__ static void add6N(Value& field, const Value& a) {
    Value local6N;

    local6N[0]  = 0x00000006;
    local6N[1]  = 0x1E348000;
    local6N[2]  = 0x20000003;
    local6N[3]  = 0x8A442F99;
    local6N[4]  = 0x5C37B000;
    local6N[5]  = 0xB9B44D1E;
    local6N[6]  = 0x05BE755A;
    local6N[7]  = 0x9CD11BB2;
    local6N[8]  = 0x8BC7B762;
    local6N[9]  = 0xA5622282;
    local6N[10] = 0x8E9E6580;
    local6N[11] = 0x0A155DA4;
    add(field, a, local6N);
  }
  
  __device__ __forceinline__ static void negateAddN(Value& field, const Value& a) {
    Value localN;

    localN[0]  = 0x00000001;
    localN[1]  = 0x8508C000;
    localN[2]  = 0x30000000;
    localN[3]  = 0x170B5D44;
    localN[4]  = 0xBA094800;
    localN[5]  = 0x1EF3622F;
    localN[6]  = 0x00F5138F;
    localN[7]  = 0x1A22D9F3;
    localN[8]  = 0x6CA1493B;
    localN[9]  = 0xC63B05C0;
    localN[10] = 0x17C510EA;
    localN[11] = 0x01AE3A46;
    sub(field, localN, a);
  }

  __device__ __forceinline__ static void negateAdd4N(Value& field, const Value& a) {
    Value local4N;

    local4N[0]  = 0x00000004;
    local4N[1]  = 0x14230000;
    local4N[2]  = 0xC0000002;
    local4N[3]  = 0x5C2D7510;
    local4N[4]  = 0xE8252000;
    local4N[5]  = 0x7BCD88BE;
    local4N[6]  = 0x03D44E3C;
    local4N[7]  = 0x688B67CC;
    local4N[8]  = 0xB28524EC;
    local4N[9]  = 0x18EC1701;
    local4N[10] = 0x5F1443AB;
    local4N[11] = 0x06B8E918;
    sub(field, local4N, a);
  }

  __device__ __forceinline__ static bool isZero(const Value& field) {
    uint32_t x, mult;
    Value    loaded;
    
    // note, this routine only works for field <= 8N
    
    x=__funnelshift_l(field[10], field[11], 4);
    mult=__umulhi(10, x);
    if((mult*0x1AE3A461u+3u-x)<=3u) {
      // if we pass this check, which should be very rare, we check all the words
      setConstant(loaded, mult);
      return mp_comp_eq<limbs>(loaded, field);
    }
    return false;
  }
  
  __device__ __forceinline__ static void reduce(Value& r, const Value& field) {
    uint32_t mult, x;
    Value    local;
    
    x=__funnelshift_l(field[10], field[11], 4);
    mult=__umulhi(10, x);
    setConstant(local, mult);
    if(!mp_sub_carry<limbs>(r, field, local))
      addN(r, r);
  }
  
  __device__ __forceinline__ static void warpShuffle(Value& r, const Value& field, uint32_t sourceLane) {
    #pragma unroll
    for(int32_t i=0;i<limbs;i++)
      r[i]=__shfl_sync(0xFFFFFFFF, field[i], sourceLane);
  }
  
  __device__ __forceinline__ static void toInternal(Value& r, const Value& external) {
    Value localRSquared;
    Value localN;
    
    setRSquared(localRSquared);
    setN(localN);
    mul(r, external, localRSquared, localN);
    if(mp_comp_ge<limbs>(r, localN))
      mp_sub<limbs>(r, r, localN);
  }
  
  __device__ __forceinline__ static void fromInternal(Value& r, const Value& internal) {
    Value one;
    Value localN;
    
    mp_zero<limbs>(one);
    one[0]=1;
    setN(localN);
    mul(r, internal, one, localN);
    if(mp_comp_ge<limbs>(r, localN))    // this compare/subtract might be unnecessary, but it's cheap, better safe than sorry
      mp_sub<limbs>(r, r, localN);
  }
};

}  /* namespace BLS12377 */

template<class Field>
class PointXY {
  typedef typename Field::Value FieldValue;
  
  public:
  FieldValue x;
  FieldValue y;
  
  __device__ __forceinline__ PointXY() {
  }

  __device__ __forceinline__ PointXY(const FieldValue& xValue, const FieldValue& yValue) {
    Field::set(x, xValue);
    Field::set(y, yValue);
  }
  
  __device__ __forceinline__ void loadShared(uint32_t offset) {
    Field::loadShared(x, offset + Field::bytes*0);
    Field::loadShared(y, offset + Field::bytes*1);
  }
  
  __device__ __forceinline__ void load(void* ptr) {
    Field::load(x, byteOffset(ptr, Field::bytes*0));
    Field::load(y, byteOffset(ptr, Field::bytes*1));
  }

  __device__ __forceinline__ void loadUnaligned(void* ptr) {
    Field::loadUnaligned(x, byteOffset(ptr, Field::bytes*0));
    Field::loadUnaligned(y, byteOffset(ptr, Field::bytes*1));
  }
  
  __device__ __forceinline__ void store(void* ptr) {
    Field::store(byteOffset(ptr, Field::bytes*0), x);
    Field::store(byteOffset(ptr, Field::bytes*1), y);
  }
  
  __device__ __forceinline__ void negate() {
    Field::negateAddN(y, y);
  }
  
  __device__ __forceinline__ void reduce() {
    Field::reduce(x, x);
    Field::reduce(y, y);
  }
  
  __device__ __forceinline__ void warpShuffle(uint32_t sourceLane) {
    Field::warpShuffle(x, x, sourceLane);
    Field::warpShuffle(y, y, sourceLane);
  }
  
  __device__ __forceinline__ void toInternal() {
    Field::toInternal(x, x);
    Field::toInternal(y, y);
  }
  
  __device__ __forceinline__ void fromInternal() {
    Field::fromInternal(x, x);
    Field::fromInternal(y, y);
  }
};

template<class Field>
class PointXYZZ {
  typedef typename Field::Value FieldValue;
  
  public:
  FieldValue x;
  FieldValue y;
  FieldValue zz;
  FieldValue zzz;
  
  __device__ __forceinline__ PointXYZZ() {
  }

  __device__ __forceinline__ PointXYZZ(const FieldValue& xValue, const FieldValue& yValue, const FieldValue& zzValue, const FieldValue& zzzValue) {
    Field::set(x, xValue);
    Field::set(y, yValue);
    Field::set(zz, zzValue);
    Field::set(zzz, zzzValue);
  }
  
  __device__ __forceinline__ void loadSharedXY(uint32_t offset) {
    Field::loadShared(x, offset + Field::bytes*0);
    Field::loadShared(y, offset + Field::bytes*1);
  }
  
  __device__ __forceinline__ void loadSharedZZ(uint32_t offset) {
    Field::loadShared(zz, offset + Field::bytes*0);
    Field::loadShared(zzz, offset + Field::bytes*1);
  }

  __device__ __forceinline__ void loadShared(uint32_t offset) {
    loadSharedXY(offset);
    loadSharedZZ(offset + Field::bytes*2);
  }

  __device__ __forceinline__ void load(void* ptr) {
    Field::load(x, byteOffset(ptr, Field::bytes*0));
    Field::load(y, byteOffset(ptr, Field::bytes*1));
    Field::load(zz, byteOffset(ptr, Field::bytes*2));
    Field::load(zzz, byteOffset(ptr, Field::bytes*3));
  }

  __device__ __forceinline__ void loadUnaligned(void* ptr) {
    Field::loadUnaligned(x, byteOffset(ptr, Field::bytes*0));
    Field::loadUnaligned(y, byteOffset(ptr, Field::bytes*1));
    Field::loadUnaligned(zz, byteOffset(ptr, Field::bytes*2));
    Field::loadUnaligned(zzz, byteOffset(ptr, Field::bytes*3));
  }
  
  __device__ __forceinline__ void store(void* ptr) {
    Field::store(byteOffset(ptr, Field::bytes*0), x);
    Field::store(byteOffset(ptr, Field::bytes*1), y);
    Field::store(byteOffset(ptr, Field::bytes*2), zz);
    Field::store(byteOffset(ptr, Field::bytes*3), zzz);
  }
  
  __device__ __forceinline__ void negate() {
    Field::negateAdd4N(y, y);
  }
  
  __device__ __forceinline__ void reduce() {
    Field::reduce(x, x);
    Field::reduce(y, y);
    Field::reduce(zz, zz);
    Field::reduce(zzz, zzz);
  }

  __device__ __forceinline__ void warpShuffle(uint32_t sourceLane) {
    Field::warpShuffle(x, x, sourceLane);
    Field::warpShuffle(y, y, sourceLane);
    Field::warpShuffle(zz, zz, sourceLane);
    Field::warpShuffle(zzz, zzz, sourceLane);
  }

  __device__ __forceinline__ void fromInternal() {
    Field::fromInternal(x, x);
    Field::fromInternal(y, y);
    Field::fromInternal(zz, zz);
    Field::fromInternal(zzz, zzz);
  }
};


namespace CurveXYZZ {

template<class Field>
class HighThroughput {
  static const uint32_t limbs=Field::limbs;
  static const uint32_t byte=Field::bytes;
  
  typedef typename Field::Value FieldValue;
  typedef typename Field::EvenOdd FieldEvenOdd;

  public:
  FieldValue   x, y, zz, zzz;
  bool         infinity, affine;
  
  __device__ __forceinline__ HighThroughput() {
    infinity=true;
    affine=false;
  }
  
  __device__ __forceinline__ void setZero() {
    infinity=true;
    affine=false;
  }
  
  __device__ __forceinline__ void load(void* ptr) {
    infinity=false;
    affine=false;
    Field::load(x, byteOffset(ptr, Field::bytes*0));
    Field::load(y, byteOffset(ptr, Field::bytes*1));
    Field::load(zz, byteOffset(ptr, Field::bytes*2));
    Field::load(zzz, byteOffset(ptr, Field::bytes*3));
  }    
    
  __device__ __forceinline__ void add(const PointXY<Field> point, bool valid=true) {
    FieldEvenOdd evenOdd;
    FieldValue   N, A, B, T0, T1;
    uint32_t     state;
    bool         carry, done, uniformAffine, dbl=false, square=false;
    
    // INPUT ASSERTIONS:
    //    x<=6*N, y<=4*N, zz<=2*N, zzz<=2*N
    //    point.x<=N, point.y<=N
    
    // OUTPUT ASSERTIONS:
    //    x<=6*N, y<=4*N, zz<=2*N, zzz<=2*N
    
    #define getResult(r) Field::merge(r, evenOdd, carry)
    #define setA(a) Field::setPermuteLow(A, a);
    #define setB(b) Field::setPermuteLow(B, b);
    #define setAB(a, b) { Field::setPermuteLow(A, a); Field::setPermuteLow(B, b); }
    #define setState(s) state=s;
    
    if(infinity) {
      // infinity is uniform across the warp
      infinity=false;
      Field::setPermuteLow(x, point.x);
      Field::setPermuteLow(y, point.y);
      affine=valid;
      if(!valid) 
        Field::setZero(zz);
      return;
    }

    uniformAffine=__all_sync(0xFFFFFFFF, affine);

    done=!valid;
    if(valid && !affine && Field::isZero(zz)) {
      Field::setPermuteLow(x, point.x);
      Field::setPermuteLow(y, point.y);
      affine=true;
      done=true;
    }
 
    if(uniformAffine) {
      Field::sub(T0, point.x, x);
      Field::add6N(T0, T0);  
      Field::sub(T1, point.y, y);
      Field::add4N(T1, T1);
      square=true;
      setA(T0);
      setState(2);
    }
    else {
      if(affine) {
        Field::setOne(zz);
        Field::setOne(zzz);
      }
      setAB(zz, point.x);
      setState(0);
    }
    
    N[0]=0x00000001; N[1]=0x8508C000; N[2]=0x30000000; N[3]=0x170B5D44; N[4]=0xBA094800; N[5]=0x1EF3622F;
    N[6]=0x00F5138F; N[7]=0x1A22D9F3; N[8]=0x6CA1493B; N[9]=0xC63B05C0; N[10]=0x17C510EA; N[11]=0x01AE3A46;

    while(!done) {
      if(!square) 
        Field::mul(evenOdd, carry, A, B, N);
      else {
        Field::sqr(evenOdd, B, carry, A, N);
        square=false;
      }
      switch(state) {
        case 0:
          getResult(T0);
          Field::sub(T0, T0, x);
          Field::add6N(T0, T0);
          setAB(zzz, point.y);
          setState(1);
          break;
        case 1:
          getResult(T1);
          Field::sub(T1, T1, y);
          Field::add4N(T1, T1);
          square=true;
          setA(T0);
          setState(2);
          break;
        case 2:
          dbl=Field::isZero(T0) && Field::isZero(T1);
          getResult(B);
          if(uniformAffine) {
            Field::setPermuteLow(zz, B);
            setA(x); 
            setState(4);
          }
          else {
            setA(zz);
            setState(3);
          }
          break;
        case 3:
          getResult(zz);
          setA(x);
          setState(4);
          break;
        case 4:
          setA(T0);
          getResult(T0);
          setState(5);
          break;
        case 5:
          getResult(B);
          if(uniformAffine) {
            Field::setPermuteLow(zzz,  B);
            setA(y);
            setState(7);
          }
          else {
            setA(zzz);
            setState(6);
          }
          break;
        case 6:
          getResult(zzz);
          setA(y);
          setState(7);
          break;
        case 7:
          getResult(y);
          Field::add(x, B, T0);
          Field::add(x, x, T0);
          square=true;
          setA(T1);
          setState(8);
          break;
        case 8:
          getResult(T1);
          Field::sub(x, T1, x);
          Field::add4N(x, x);
          Field::sub(T0, T0, x);
          Field::add6N(T0, T0);
          setB(T0);
          setState(9);
          break;
        case 9:
          affine=false;
          getResult(T0);
          Field::sub(y, T0, y);
          Field::add2N(y, y);
          done=true;
      }
    }

    if(__all_sync(0xFFFFFFFF, !dbl))
      return;
      
    done=!dbl;
    Field::add(T0, point.y, point.y);
    square=true;
    setA(T0);
    setState(0);
    while(!done) {
      if(!square) 
        Field::mul(evenOdd, carry, A, B, N);
      else {
        Field::sqr(evenOdd, B, carry, A, N);
        square=false;
      }
      switch(state) {
        case 0:
          getResult(zz);
          setB(zz);
          setState(1);
          break;
        case 1:
          getResult(zzz);
          setA(point.x);
          setState(2);
          break;
        case 2:
          getResult(T0);
          setAB(point.y, zzz);
          setState(3);
          break;
        case 3:
          getResult(y);
          square=true;
          setA(point.x);
          setState(4);
          break;
        case 4:
          getResult(T1);
          square=true;
          Field::add(A, T1, T1);
          Field::add(A, A, T1);
          setState(5);
          break;
        case 5:
          getResult(x);
          Field::add(B, T0, T0);
          Field::sub(x, x, B);
          Field::add3N(x, x);
          Field::sub(B, T0, x);
          Field::add5N(B, B);
          setState(6);
          break;
        case 6:
          getResult(T0);
          Field::sub(y, T0, y);
          Field::add2N(y, y);
          done=true;
          break;            
      }
    }

    __syncwarp(0xFFFFFFFF);

    #undef getResult
    #undef setA
    #undef setB
    #undef setAB
  }

  __device__ __forceinline__ void add(const PointXYZZ<Field> point, bool valid=true) {
    FieldEvenOdd evenOdd;
    FieldValue   N, A, B, T0, T1;
    uint32_t     state=0;
    bool         carry, done, dbl, square=false;
    
    #define getResult(r) Field::merge(r, evenOdd, carry)
    #define setA(a) Field::setPermuteLow(A, a);
    #define setB(b) Field::setPermuteLow(B, b);
    #define setAB(a, b) { Field::setPermuteLow(A, a); Field::setPermuteLow(B, b); }
    
    if(infinity) {
      // infinity is uniform across the warp
      infinity=false;      
      affine=false;
      Field::setPermuteLow(x, point.x);
      Field::setPermuteLow(y, point.y);
      #pragma unroll
      for(int32_t i=0;i<12;i++)
        zz[i]=valid ? point.zz[i] : 0;
      Field::setPermuteLow(zzz, point.zzz);
      return;
    }
    
    if(affine) {
      affine=false;
      Field::setOne(zz);
      Field::setOne(zzz);
    }
    
    if(valid && Field::isZero(zz)) {
      Field::setPermuteLow(x, point.x);
      Field::setPermuteLow(y, point.y);
      Field::setPermuteLow(zz, point.zz);
      Field::setPermuteLow(zzz, point.zzz);
      done=true;
    }
    else 
      done=!valid || Field::isZero(point.zz);

    N[0]=0x00000001; N[1]=0x8508C000; N[2]=0x30000000; N[3]=0x170B5D44; N[4]=0xBA094800; N[5]=0x1EF3622F;
    N[6]=0x00F5138F; N[7]=0x1A22D9F3; N[8]=0x6CA1493B; N[9]=0xC63B05C0; N[10]=0x17C510EA; N[11]=0x01AE3A46;

    setAB(zz, point.x);
    while(!done) {
      if(!square) 
        Field::mul(evenOdd, carry, A, B, N);
      else {
        Field::sqr(evenOdd, B, carry, A, N);
        square=false;
      }
      switch(state) {
        case 0:
          getResult(T0);
          setAB(zzz, point.y);
          break;
        case 1:
          getResult(T1);
          setAB(x, point.zz);
          break;
        case 2:
          getResult(x);
          setAB(y, point.zzz);
          break;
        case 3:
          getResult(y);
          Field::sub(T0, T0, x);
          Field::add6N(T0, T0);
          Field::sub(T1, T1, y);
          Field::add4N(T1, T1);       
          square=true;
          setA(T0);
          break;
        case 4:
          dbl=Field::isZero(T0) && Field::isZero(T1);
          getResult(B);
          setA(zz);
          break;
        case 5:
          getResult(zz);
          setA(x);
          break;
        case 6:
          setA(T0);
          getResult(T0);
          break;
        case 7:
          getResult(B);    // PPP is in B
          Field::add(x, B, T0);
          Field::add(x, x, T0);
          setA(zzz);
          break;
        case 8:
          getResult(zzz);
          setA(y);
          break;
        case 9:
          getResult(y);
          setAB(zz, point.zz);
          break;
        case 10:
          getResult(zz);
          setAB(zzz, point.zzz);
          break;
        case 11:
          getResult(zzz);
          square=true;
          setA(T1);
          break;
        case 12:
          getResult(T1);
          Field::sub(x, T1, x);
          Field::add4N(x, x);
          Field::sub(T0, T0, x);
          Field::add6N(T0, T0);
          setB(T0);
          break;
        case 13:
          affine=false;
          getResult(T0);
          Field::sub(y, T0, y);
          Field::add2N(y, y);
          if(dbl) {
            Field::add(T0, point.y, point.y);
            square=true;
            setA(T0);
          }
          else
            done=true;
          break;
        case 14:
          getResult(T0);
          setB(T0);
          break;
        case 15:
          getResult(T1);
          setA(point.zz);
          break;
        case 16:
          getResult(zz);
          setAB(T1, point.zzz);
          break;
        case 17:
          getResult(zzz);
          setB(point.y);
          break;
        case 18:
          getResult(y);
          setAB(point.x, T0);
          break;
        case 19:
          getResult(T0);
          square=true;
          break;
        case 20:
          getResult(T1);
          square=true;
          Field::add(A, T1, T1);
          Field::add(A, A, T1);
          break;
        case 21:
          getResult(x);
          Field::add(T1, T0, T0);
          Field::sub(x, x, T1);
          Field::add3N(x, x);
          Field::sub(B, T0, x);
          Field::add5N(B, B);
          break;
        case 22:
          getResult(T0);
          Field::sub(y, T0, y);
          Field::add2N(y, y);
          done=true;
          break;
      }
      state++;
    }

    __syncwarp(0xFFFFFFFF);

    #undef getResult
    #undef setA
    #undef setB
    #undef setAB
  }
  
  __device__ __forceinline__ void dbl() {
    FieldEvenOdd evenOdd;
    FieldValue   N, A, B, T0, T1;
    uint32_t     state=0;
    bool         carry, done, square;
    
    #define getResult(r) Field::merge(r, evenOdd, carry)
    #define setA(a) Field::setPermuteLow(A, a);
    #define setB(b) Field::setPermuteLow(B, b);
    #define setAB(a, b) { Field::setPermuteLow(A, a); Field::setPermuteLow(B, b); }

    if(affine) {
      affine=false;
      done=false;
      Field::setOne(zz);
      Field::setOne(zzz);
    }
    else
      done=infinity || Field::isZero(zz);

    N[0]=0x00000001; N[1]=0x8508C000; N[2]=0x30000000; N[3]=0x170B5D44; N[4]=0xBA094800; N[5]=0x1EF3622F;
    N[6]=0x00F5138F; N[7]=0x1A22D9F3; N[8]=0x6CA1493B; N[9]=0xC63B05C0; N[10]=0x17C510EA; N[11]=0x01AE3A46;

    square=true;
    Field::add(T1, y, y);
    setA(T1);
    while(!done) {
      if(!square) 
        Field::mul(evenOdd, carry, A, B, N);
      else {
        Field::sqr(evenOdd, B, carry, A, N);
        square=false;
      }
      switch(state) {
        case 0:
          getResult(T0);
          setB(T0);
          break;
        case 1:
          getResult(T1);
          setA(zz);
          break;
        case 2:
          getResult(zz);
          setAB(T1, zzz);
          break;
        case 3:
          getResult(zzz);
          setB(y);
          break;
        case 4:
          getResult(y);
          setAB(x, T0);
          break;
        case 5:
          getResult(T0);
          square=true;
          break;
        case 6:
          getResult(T1);
          square=true;
          Field::add(A, T1, T1);
          Field::add(A, A, T1);
          break;
        case 7:
          getResult(x);
          Field::add(T1, T0, T0);
          Field::sub(x, x, T1);
          Field::add3N(x, x);
          Field::sub(B, T0, x);
          Field::add5N(B, B);
          break;
        case 8:
          getResult(T0);
          Field::sub(y, T0, y);
          Field::add2N(y, y);
          done=true;
          break;
      }
      state++;
    }
    
    #undef getResult
    #undef setA
    #undef setB
    #undef setAB
  }

 /*********************************************************************************************
  * We can build an add routine without using state machines and switches.  But testing shows
  * this to actually be slower, so we stick with the state machine one.   We leave this code
  * here for future testing.
  *
  * __device__ __forceinline__ void addNoCase(const PointXY<Field> point, bool valid=true) {
  *   FieldValue N, T0, T1, T2, T3;
  *   bool       uniformAffine;
  *   
  *   uniformAffine=__all_sync(0xFFFFFFFF, valid && affine);
  *   
  *   if(infinity) {
  *     // infinity is uniform across the warp
  *     infinity=false;
  *     Field::setPermuteLow(x, point.x);
  *     Field::setPermuteLow(y, point.y);
  *     affine=valid;
  *     if(!valid) 
  *       Field::setZero(zz);
  *     return;
  *   }
  * 
  *   if(!valid)
  *     return;
  * 
  *   if(!affine && Field::isZero(zz)) {
  *     Field::setPermuteLow(x, point.x);
  *     Field::setPermuteLow(y, point.y);
  *     affine=true;
  *     return;
  *   }
  * 
  *   if(affine && !uniformAffine) {
  *     Field::setOne(zz);
  *     Field::setOne(zzz);
  *   }
  *       
  *   N[0]=0x00000001; N[1]=0x8508C000; N[2]=0x30000000; N[3]=0x170B5D44; N[4]=0xBA094800; N[5]=0x1EF3622F;
  *   N[6]=0x00F5138F; N[7]=0x1A22D9F3; N[8]=0x6CA1493B; N[9]=0xC63B05C0; N[10]=0x17C510EA; N[11]=0x01AE3A46;
  * 
  *   affine=false;
  *   
  *   if(uniformAffine) {
  *     Field::setPermuteLow(T0, point.x);
  *     Field::setPermuteLow(T1, point.y);
  *   }
  *   else {
  *     Field::mul(T0, point.x, zz, N);
  *     Field::mul(T1, point.y, zzz, N);
  *   }
  *   
  *   Field::sub(T0, T0, x);
  *   Field::sub(T1, T1, y);
  *   Field::add6N(T0, T0);
  *   Field::add4N(T1, T1);
  *   
  *   if(Field::isZero(T0) && Field::isZero(T1)) {
  *     Field::add(T0, point.y, point.y);
  *     Field::sqr(zz, T3, T0, N);
  *     Field::mul(zzz, zz, T0, N);
  *     
  *     Field::mul(T0, x, zz, N);
  *     Field::sqr(x, T3, x, N);
  *     Field::mul(y, y, zzz, N);
  *     
  *     Field::add(T1, x, x);
  *     Field::add(T1, T1, x);
  *     
  *     Field::sqr(x, T3, T1, N);
  *     Field::sub(x, x, T0);
  *     Field::sub(x, x, T0);
  *     Field::add3N(x, x);
  *     Field::sub(T0, T0, x);
  *     Field::add5N(T0, T0);
  *     
  *     Field::mul(T1, T1, T0, N);
  *     Field::sub(y, T1, y);
  *     Field::add2N(y, y);
  *     return;
  *   }
  *  
  *   Field::sqr(T2, T3, T0, N);
  *   Field::mul(T0, T0, T2, N);
  * 
  *   if(uniformAffine) {
  *     Field::setPermuteLow(zz, T2);
  *     Field::setPermuteLow(zzz, T0);
  *   }
  *   else {
  *     Field::mul(zz, zz, T2, N);
  *     Field::mul(zzz, zzz, T0, N);
  *   }
  *   
  *   Field::mul(T2, x, T2, N);
  *   Field::mul(y, y, T0, N);
  *   Field::sqr(x, T3, T1, N);
  *   Field::sub(x, x, T0);
  *   Field::sub(x, x, T2);
  *   Field::sub(x, x, T2);
  *   Field::add4N(x, x);
  *   
  *   Field::sub(T2, T2, x);
  *   Field::add6N(T2, T2);
  *   Field::mul(T2, T1, T2, N);
  *   Field::sub(y, T2, y);
  *   Field::add2N(y, y);
  * }
  *
  *********************************************************************************************/

  __device__ __forceinline__ PointXYZZ<Field> accumulator() {
    if(infinity || (!affine && Field::isZero(zz))) {
      Field::setZero(x);
      Field::setZero(y);
      Field::setZero(zz);
      Field::setZero(zzz);
    }
    else if(affine) {
      Field::setOne(zz);
      Field::setOne(zzz);
    }
    return PointXYZZ<Field>(x, y, zz, zzz);
  }
  
  __device__ __forceinline__ PointXY<Field> normalize() {
    FieldValue N, A, B, T0, T1;
    
    if(infinity || (!affine && Field::isZero(zz))) {
      Field::setZero(x);
      Field::setZero(y);
    }
    else if(!affine) {      
      Field::fromInternal(A, zzz);
      
      N[0]=0x00000001; N[1]=0x8508C000; N[2]=0x30000000; N[3]=0x170B5D44; N[4]=0xBA094800; N[5]=0x1EF3622F;
      N[6]=0x00F5138F; N[7]=0x1A22D9F3; N[8]=0x6CA1493B; N[9]=0xC63B05C0; N[10]=0x17C510EA; N[11]=0x01AE3A46;

      // Finds the inverse of A -- not super fast, but it is simple, and very compact
      mp_copy<limbs>(B, N);   //  A=value, B=N
      mp_zero<limbs>(T0);     //  T0=1, T1=0
      T0[0]=1;
      mp_zero<limbs>(T1);
      while(mp_logical_or<limbs>(A)!=0) {
        if((A[0] & 0x01)!=0) {
          if(mp_comp_gt<limbs>(B, A)) {
            Field::swap(A, B);
            Field::swap(T0, T1);
          }
          mp_sub<limbs>(A, A, B);
          if(!mp_sub_carry<limbs>(T0, T0, T1))
            mp_add<limbs>(T0, T0, N);
        }
        mp_shift_right<limbs>(A, A, 1);
        if((T0[0] & 0x01)!=0)
          mp_add<limbs>(T0, T0, N);
        mp_shift_right<limbs>(T0, T0, 1);
      }
      
      // this will certainly blow up the instruction cache, oh well
      Field::toInternal(T0, T1);
      Field::mul(y, y, T0, N);
      Field::reduce(y, y);
      Field::mul(T0, zz, T0, N);
      Field::mul(T1, T0, T0, N);
      Field::mul(x, x, T1, N);
      Field::reduce(x, x);
      affine=true;
    }
    return PointXY<Field>(x, y);
  }  
  
  __device__ __forceinline__ void fromInternal() {
    Field::fromInternal(x, x);
    Field::fromInternal(y, y);
    Field::fromInternal(zz, zz);
    Field::fromInternal(zzz, zzz);
  }
};

}
