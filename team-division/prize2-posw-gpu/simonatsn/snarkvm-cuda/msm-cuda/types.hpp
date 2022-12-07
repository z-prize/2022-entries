typedef struct {
  uint8_t x[FP_BYTES];
  uint8_t y[FP_BYTES];
} p1_affine_t;

typedef struct {
  uint8_t x[FP_BYTES];
  uint8_t y[FP_BYTES];
  uint8_t z0[FP_BYTES];
  uint8_t z1[FP_BYTES];
} p1_xyzz_t;

typedef struct {
  uint8_t x[FP_BYTES];
  uint8_t y[FP_BYTES];
  uint8_t z[FP_BYTES];
} p1_xyz_t;

typedef struct {
  uint8_t x[FP_BYTES];
  uint8_t y[FP_BYTES];
  uint8_t inf;
  uint8_t pad[7];
} rust_p1_affine_t;

typedef struct {
  uint8_t s[SCALAR_BYTES];
} scalar_t;

void print_value(unsigned int* value, size_t n, const char* name) {
    if (name != NULL)
        printf("%s = 0x", name);
    else
        printf("0x");
    while (n--)
        printf("%08x", value[n]);
    printf("\n");
}

void host_print_affine(p1_affine_t* p) {
  print_value((unsigned int *)p->x, FP_BYTES / sizeof(unsigned int),  "  x:   ");
  print_value((unsigned int *)p->y, FP_BYTES / sizeof(unsigned int),  "  y:   ");
}

void host_print_xyz(p1_xyz_t* p) {
  print_value((unsigned int *)p->x, FP_BYTES / sizeof(unsigned int), "  x:   ");
  print_value((unsigned int *)p->y, FP_BYTES / sizeof(unsigned int), "  y:   ");
  print_value((unsigned int *)p->z, FP_BYTES / sizeof(unsigned int), "  z:   ");
}

void host_print_xyzz(p1_xyzz_t* p) {
  print_value((unsigned int *)p->x, FP_BYTES / sizeof(unsigned int),  "  x:   ");
  print_value((unsigned int *)p->y, FP_BYTES / sizeof(unsigned int),  "  y:   ");
  print_value((unsigned int *)p->z0, FP_BYTES / sizeof(unsigned int), "  zz:  ");
  print_value((unsigned int *)p->z1, FP_BYTES / sizeof(unsigned int), "  zzz: ");
}
