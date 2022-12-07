// finite field definitions

#pragma once

#include "ff_storage.cuh"

namespace bls12_377 {

struct ff_config_base {
  // field structure size = 12 * 32 bit
  static constexpr unsigned limbs_count = 12;
  // modulus = 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
  static constexpr ff_storage<limbs_count> modulus = {0x00000001, 0x8508c000, 0x30000000, 0x170b5d44, 0xba094800, 0x1ef3622f,
                                                      0x00f5138f, 0x1a22d9f3, 0x6ca1493b, 0xc63b05c0, 0x17c510ea, 0x01ae3a46};
  // modulus*2 = 517328852025938188021305467389787067072787025509829321079768525333440936696681645549937776279146720248880642916354
  static constexpr ff_storage<limbs_count> modulus_2 = {0x00000002, 0x0a118000, 0x60000001, 0x2e16ba88, 0x74129000, 0x3de6c45f,
                                                        0x01ea271e, 0x3445b3e6, 0xd9429276, 0x8c760b80, 0x2f8a21d5, 0x035c748c};
  // modulus*4 = 1034657704051876376042610934779574134145574051019658642159537050666881873393363291099875552558293440497761285832708
  static constexpr ff_storage<limbs_count> modulus_4 = {0x00000004, 0x14230000, 0xc0000002, 0x5c2d7510, 0xe8252000, 0x7bcd88be,
                                                        0x03d44e3c, 0x688b67cc, 0xb28524ec, 0x18ec1701, 0x5f1443ab, 0x06b8e918};
  // modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared = {
      0x00000001, 0x0a118000, 0xf0000001, 0x7338d254, 0x2e1bd800, 0x4ada268f, 0x35f1c09a, 0x6bcbfbd2, 0x58638c9d, 0x318324b9, 0x8bb70ae0, 0x460aaaaa,
      0x502a4d6c, 0xc014e712, 0xb90660cd, 0x09d018af, 0x3dda4d5c, 0x1f5e7141, 0xa4aee93f, 0x4bb8b87d, 0xb361263c, 0x2256913b, 0xd0bbaffb, 0x0002d307};
  // 2*modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {
      0x00000002, 0x14230000, 0xe0000002, 0xe671a4a9, 0x5c37b000, 0x95b44d1e, 0x6be38134, 0xd797f7a4, 0xb0c7193a, 0x63064972, 0x176e15c0, 0x8c155555,
      0xa0549ad8, 0x8029ce24, 0x720cc19b, 0x13a0315f, 0x7bb49ab8, 0x3ebce282, 0x495dd27e, 0x977170fb, 0x66c24c78, 0x44ad2277, 0xa1775ff6, 0x0005a60f};
  // 4*modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {
      0x00000004, 0x28460000, 0xc0000004, 0xcce34953, 0xb86f6001, 0x2b689a3c, 0xd7c70269, 0xaf2fef48, 0x618e3275, 0xc60c92e5, 0x2edc2b80, 0x182aaaaa,
      0x40a935b1, 0x00539c49, 0xe4198337, 0x274062be, 0xf7693570, 0x7d79c504, 0x92bba4fc, 0x2ee2e1f6, 0xcd8498f1, 0x895a44ee, 0x42eebfec, 0x000b4c1f};
  // r2 = 66127428376872697816332570116866232405230528984664918319606315420233909940404532140033099444330447428417853902114
  static constexpr ff_storage<limbs_count> r2 = {0x9400cd22, 0xb786686c, 0xb00431b1, 0x0329fcaa, 0x62d6b46d, 0x22a5f111,
                                                 0x827dc3ac, 0xbfdf7d03, 0x41790bf9, 0x837e92f0, 0x1e914b88, 0x006dfccb};
  // inv
  static constexpr uint32_t inv = 0xffffffff;
  // 1 in montgomery form
  static constexpr ff_storage<limbs_count> one = {0xffffff68, 0x02cdffff, 0x7fffffb1, 0x51409f83, 0x8a7d3ff2, 0x9f7db3a9,
                                                  0x6e7c6305, 0x7b4e97b7, 0x803c84e8, 0x4cf495bf, 0xe2fdf49a, 0x008d6661};
  static constexpr unsigned modulus_bits_count = 377;
  static constexpr unsigned B_VALUE = 1;
};

// Can't make this a member of ff_config_p. nvcc does not allow __constant__ on members.
extern __device__ __constant__ uint32_t inv_p;

struct ff_config_scalar {
  // field structure size = 8 * 32 bit
  static constexpr unsigned limbs_count = 8;
  // modulus = 8444461749428370424248824938781546531375899335154063827935233455917409239041
  static constexpr ff_storage<limbs_count> modulus = {0x00000001, 0x0a118000, 0xd0000001, 0x59aa76fe, 0x5c37b001, 0x60b44d1e, 0x9a2ca556, 0x12ab655e};
  // modulus*2 = 16888923498856740848497649877563093062751798670308127655870466911834818478082
  static constexpr ff_storage<limbs_count> modulus_2 = {0x00000002, 0x14230000, 0xa0000002, 0xb354edfd, 0xb86f6002, 0xc1689a3c, 0x34594aac, 0x2556cabd};
  // modulus*4 = 33777846997713481696995299755126186125503597340616255311740933823669636956164
  static constexpr ff_storage<limbs_count> modulus_4 = {0x00000004, 0x28460000, 0x40000004, 0x66a9dbfb, 0x70dec005, 0x82d13479, 0x68b29559, 0x4aad957a};
  // modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared = {0x00000001, 0x14230000, 0xe0000002, 0xc7dd4d2f, 0x8585d003, 0x08ee1bd4,
                                                                   0xe57fc56e, 0x7e7557e3, 0x483a709d, 0x1fdebb41, 0x5678f4e6, 0x8ea77334,
                                                                   0xc19c3ec5, 0xd717de29, 0xe2340781, 0x015c8d01};
  // 2*modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {0x00000002, 0x28460000, 0xc0000004, 0x8fba9a5f, 0x0b0ba007, 0x11dc37a9,
                                                                     0xcaff8adc, 0xfceaafc7, 0x9074e13a, 0x3fbd7682, 0xacf1e9cc, 0x1d4ee668,
                                                                     0x83387d8b, 0xae2fbc53, 0xc4680f03, 0x02b91a03};
  // 4*modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {0x00000004, 0x508c0000, 0x80000008, 0x1f7534bf, 0x1617400f, 0x23b86f52,
                                                                     0x95ff15b8, 0xf9d55f8f, 0x20e9c275, 0x7f7aed05, 0x59e3d398, 0x3a9dccd1,
                                                                     0x0670fb16, 0x5c5f78a7, 0x88d01e07, 0x05723407};
  // r2 = 508595941311779472113692600146818027278633330499214071737745792929336755579
  static constexpr ff_storage<limbs_count> r2 = {0xb861857b, 0x25d577ba, 0x8860591f, 0xcc2c27b5, 0xe5dc8593, 0xa7cc008f, 0xeff1c939, 0x011fdae7};
  // inv
  static constexpr uint32_t inv = 0xffffffff;
  // 1 in montgomery form
  static constexpr ff_storage<limbs_count> one = {0xfffffff3, 0x7d1c7fff, 0x6ffffff2, 0x7257f50f, 0x512c0fee, 0x16d81575, 0x2bbb9a9d, 0x0d4bda32};
  static constexpr unsigned modulus_bits_count = 253;
  // log2 of order of omega
  static constexpr unsigned omega_log_order = 28;
  // k = (modulus - 1) / (2^omega_log_order)
  //   = (modulus - 1) / (2^28)
  //   = 31458071430878231019708607117762962473094088342614709689971929251840
  // omega generator is 22
  static constexpr unsigned omega_generator = 22;
  // omega = generator^k % modulus
  //       = 22^31458071430878231019708607117762962473094088342614709689971929251840 % modulus
  //       = 121723987773120995826174558753991820398372762714909599931916967971437184138
  // omega in montgomery form
  static constexpr ff_storage<limbs_count> omega = {0x06a79893, 0x18edfc4b, 0xd9942118, 0xffe5168f, 0xb4ab2c5c, 0x76b85d81, 0x47c2ecc0, 0x0a439040};
  // inverse of 2 in montgomery form
  static constexpr ff_storage<limbs_count> two_inv = {0xfffffffa, 0xc396ffff, 0x1ffffff9, 0xe6013607, 0xd6b1dff7, 0xbbc63149, 0x62f41ff9, 0x0ffb9fc8};
};

// Can't make this a member of ff_config_q. nvcc does not allow __constant__ on members.
extern __device__ __constant__ uint32_t inv_q;

} // namespace bls12_377

#define ff_config_p bls12_377::ff_config_base
#define ff_config_q bls12_377::ff_config_scalar
#define inv_p bls12_377::inv_p
#define inv_q bls12_377::inv_q
