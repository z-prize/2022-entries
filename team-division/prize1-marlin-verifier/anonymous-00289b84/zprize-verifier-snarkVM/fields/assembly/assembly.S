#if defined(__x86_64) || defined(__x86_64__)
# if defined(__ELF__)
#  if defined(__ADX__) && !defined(__BLST_PORTABLE__)
#   include "ctx_inverse_mod_384-x86_64.s"
#  else
#   include "ctq_inverse_mod_384-x86_64.s"
#  endif
//#  include "elf/add_mod_384-x86_64.s"
//#  include "elf/add_mod_384x384-x86_64.s"
#  if defined(__ADX__) && !defined(__BLST_PORTABLE__)
#   include "mulx_mont_384-x86_64.s"
#   include "mulx_mont_256-x86_64.s"
#  else
#   include "mulq_mont_384-x86_64.s"
#   include "mulq_mont_256-x86_64.s"
#  endif
//#  include "elf/add_mod_256-x86_64.s"
//#  include "elf/ct_inverse_mod_256-x86_64.s"
//#  include "elf/div3w-x86_64.s"
//#  include "elf/ct_is_square_mod_384-x86_64.s"
# endif
//#elif defined(__aarch64__)
//# if defined(__ELF__)
//#  include "elf/sha256-armv8.S"
//#  include "elf/ct_inverse_mod_384-armv8.S"
//#  include "elf/add_mod_384-armv8.S"
//#  define __add_mod_384     __add_mont_384
//#  define __sub_mod_384     __sub_mont_384
//#  include "elf/mul_mont_384-armv8.S"
//#  include "elf/mul_mont_256-armv8.S"
//#  include "elf/add_mod_256-armv8.S"
//#  include "elf/ct_inverse_mod_256-armv8.S"
//#  include "elf/div3w-armv8.S"
//#  include "elf/ct_is_square_mod_384-armv8.S"
//# endif
#elif defined(__BLST_NO_ASM__) || \
      (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__==4)
/* inaccurate way to detect a 32-bit processor, but it's close enough */
#else
# error "unsupported platform"
#endif
