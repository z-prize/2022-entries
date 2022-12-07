// Elliptic curve operations (Short Weierstrass Jacobian form)

#define POINT_ZERO ((POINT_projective){FIELD_ZERO, FIELD_ONE, FIELD_ZERO})

#define POINT_ED_ZERO ((POINT_ed_projective){FIELD_ZERO, FIELD_ONE, FIELD_ZERO, FIELD_ONE})

typedef struct {
  FIELD x;
  FIELD y;
} POINT_affine;

typedef struct {
  FIELD x;
  FIELD y;
  FIELD z;
} POINT_projective;

typedef struct {
  FIELD x;
  FIELD y;
  FIELD t;
} POINT_ed_affine;

typedef struct {
  FIELD x;
  FIELD y;
  FIELD t;
  FIELD z;
} POINT_ed_projective;


// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE POINT_projective POINT_double(POINT_projective inp) {
  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(inp.z, local_zero)) {
      return inp;
  }

  const FIELD a = FIELD_sqr(inp.x); // A = X1^2
  const FIELD b = FIELD_sqr(inp.y); // B = Y1^2
  FIELD c = FIELD_sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  FIELD d = FIELD_add(inp.x, b);
  d = FIELD_sqr(d); d = FIELD_sub(FIELD_sub(d, a), c); d = FIELD_double(d);

  const FIELD e = FIELD_add(FIELD_double(a), a); // E = 3*A
  const FIELD f = FIELD_sqr(e);

  inp.z = FIELD_mul(inp.y, inp.z); inp.z = FIELD_double(inp.z); // Z3 = 2*Y1*Z1
  inp.x = FIELD_sub(FIELD_sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = FIELD_double(c); c = FIELD_double(c); c = FIELD_double(c);
  inp.y = FIELD_sub(FIELD_mul(FIELD_sub(d, inp.x), e), c);

  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE POINT_projective POINT_add_mixed(POINT_projective a, POINT_affine b) {
  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(a.z, local_zero)) {
    const FIELD local_one = FIELD_ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  const FIELD z1z1 = FIELD_sqr(a.z);
  const FIELD u2 = FIELD_mul(b.x, z1z1);
  const FIELD s2 = FIELD_mul(FIELD_mul(b.y, a.z), z1z1);

  if(FIELD_eq(a.x, u2) && FIELD_eq(a.y, s2)) {
      return POINT_double(a);
  }

  const FIELD h = FIELD_sub(u2, a.x); // H = U2-X1
  const FIELD hh = FIELD_sqr(h); // HH = H^2
  FIELD i = FIELD_double(hh); i = FIELD_double(i); // I = 4*HH
  FIELD j = FIELD_mul(h, i); // J = H*I
  FIELD r = FIELD_sub(s2, a.y); r = FIELD_double(r); // r = 2*(S2-Y1)
  const FIELD v = FIELD_mul(a.x, i);

  POINT_projective ret;

  // X3 = r^2 - J - 2*V
  ret.x = FIELD_sub(FIELD_sub(FIELD_sqr(r), j), FIELD_double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = FIELD_mul(a.y, j); j = FIELD_double(j);
  ret.y = FIELD_sub(FIELD_mul(FIELD_sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = FIELD_add(a.z, h); ret.z = FIELD_sub(FIELD_sub(FIELD_sqr(ret.z), z1z1), hh);
  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE POINT_projective POINT_add(POINT_projective a, POINT_projective b) {

  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(a.z, local_zero)) return b;
  if(FIELD_eq(b.z, local_zero)) return a;

  const FIELD z1z1 = FIELD_sqr(a.z); // Z1Z1 = Z1^2
  const FIELD z2z2 = FIELD_sqr(b.z); // Z2Z2 = Z2^2
  const FIELD u1 = FIELD_mul(a.x, z2z2); // U1 = X1*Z2Z2
  const FIELD u2 = FIELD_mul(b.x, z1z1); // U2 = X2*Z1Z1
  FIELD s1 = FIELD_mul(FIELD_mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const FIELD s2 = FIELD_mul(FIELD_mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(FIELD_eq(u1, u2) && FIELD_eq(s1, s2))
    return POINT_double(a);
  else {
    const FIELD h = FIELD_sub(u2, u1); // H = U2-U1
    FIELD i = FIELD_double(h); i = FIELD_sqr(i); // I = (2*H)^2
    const FIELD j = FIELD_mul(h, i); // J = H*I
    FIELD r = FIELD_sub(s2, s1); r = FIELD_double(r); // r = 2*(S2-S1)
    const FIELD v = FIELD_mul(u1, i); // V = U1*I
    a.x = FIELD_sub(FIELD_sub(FIELD_sub(FIELD_sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = FIELD_mul(FIELD_sub(v, a.x), r);
    s1 = FIELD_mul(s1, j); s1 = FIELD_double(s1); // S1 = S1 * J * 2
    a.y = FIELD_sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = FIELD_add(a.z, b.z); a.z = FIELD_sqr(a.z);
    a.z = FIELD_sub(FIELD_sub(a.z, z1z1), z2z2);
    a.z = FIELD_mul(a.z, h);

    return a;
  }
}

// http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#doubling-dbl-2008-hwcd
DEVICE POINT_ed_projective POINT_ed_double(POINT_ed_projective a) {
  const FIELD x1x1 = FIELD_mul(a.x, a.x);		// A = X1 * X1
  const FIELD y1y1 = FIELD_mul(a.y, a.y);		// B = Y1 * Y1

  const FIELD z1z1 = FIELD_mul(a.z, a.z);
  const FIELD c = FIELD_double(z1z1);			// C = 2 * Z1^2

  const FIELD d = FIELD_mul(FIELD_ED_A, x1x1);		// D = a * A

  FIELD e = FIELD_add(a.x, a.y);
  e = FIELD_mul(e, e);
  e = FIELD_sub(e, x1x1);
  e = FIELD_sub(e, y1y1);				// E = (X1 + Y1)^2 - A - B
  
  const FIELD g = FIELD_add(d, y1y1);			// G = D + B
  const FIELD f = FIELD_sub(g, c);			// F = G - C
  const FIELD h = FIELD_sub(d, y1y1);			// H = D - B

  a.x = FIELD_mul(e, f);				// X3 = E * F
  a.y = FIELD_mul(g, h);				// Y3 = G * H
  a.t = FIELD_mul(e, h);				// T3 = E * H
  a.z = FIELD_mul(f, g);				// Z3 = F * G

  return a;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#addition-madd-2008-hwcd-2
// must be distinct points
DEVICE POINT_ed_projective POINT_ed_add_mixed_dedicated(POINT_ed_projective a, POINT_ed_affine b) {
  const FIELD x1x2 = FIELD_mul(a.x, b.x);		// A = X1 * X2
  const FIELD y1y2 = FIELD_mul(a.y, b.y);		// B = Y1 * Y2
  const FIELD z1t2 = FIELD_mul(a.z, b.t);		// C = Z1 * T2
							// D = T1
  const FIELD e = FIELD_add(a.t, z1t2);			// E = D + C
  
  const FIELD x1_sub_y1 = FIELD_sub(a.x, a.y);
  const FIELD x2_add_y2 = FIELD_add(b.x, b.y);
  FIELD f = FIELD_mul(x1_sub_y1, x2_add_y2);
  f = FIELD_add(f, y1y2);
  f = FIELD_sub(f, x1x2);				// F = (X1-Y1) * (X2 + Y2) + B - A

  const FIELD a_A = FIELD_mul(FIELD_ED_A, x1x2);
  const FIELD g = FIELD_add(y1y2, a_A);			// G = B + a * A
  const FIELD h = FIELD_sub(a.t, z1t2);			// H = D - C

  a.x = FIELD_mul(e, f);				// X3 = E * F
  a.y = FIELD_mul(g, h);				// Y3 = G * H
  a.t = FIELD_mul(e, h);				// T3 = E * H
  a.z = FIELD_mul(f, g);				// Z3 = F * G

  return a;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#addition-add-2008-hwcd
DEVICE POINT_ed_projective POINT_ed_add(POINT_ed_projective a, POINT_ed_projective b) {
  const FIELD x1x2 = FIELD_mul(a.x, b.x);		// A = X1 * X2
  const FIELD y1y2 = FIELD_mul(a.y, b.y);		// B = Y1 * Y2
  const FIELD t1t2 = FIELD_mul(a.t, b.t);

  const FIELD c = FIELD_mul(FIELD_ED_D, t1t2);		// C = d * T1 * T2
  const FIELD d = FIELD_mul(a.z, b.z);			// D = Z1 * Z2

  const FIELD x1_add_y1 = FIELD_add(a.x, a.y);
  const FIELD x2_add_y2 = FIELD_add(b.x, b.y);
  FIELD e = FIELD_mul(x1_add_y1, x2_add_y2);
  e = FIELD_sub(e, x1x2);
  e = FIELD_sub(e, y1y2);				// E = (X1 + Y1) * (X2 + Y2) - A - B

  const FIELD f = FIELD_sub(d, c);			// F = D - C

  const FIELD a_A = FIELD_mul(FIELD_ED_A, x1x2);
  const FIELD g = FIELD_add(d, c);			// G = D + C
  const FIELD h = FIELD_sub(y1y2, a_A);			// H = B - a * A

  a.x = FIELD_mul(e, f);				// X3 = E * F
  a.y = FIELD_mul(g, h);				// Y3 = G * H
  a.t = FIELD_mul(e, h);				// T3 = E * H
  a.z = FIELD_mul(f, g);				// Z3 = F * G

  return a;
}

// a = -1
// http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-madd-2008-hwcd-4
DEVICE POINT_ed_projective POINT_ed_neg_one_a_add_mixed_dedicated(POINT_ed_projective a, POINT_ed_affine b) {
  const FIELD y1_sub_x1 = FIELD_sub(a.y, a.x);
  const FIELD y2_add_x2 = FIELD_add(b.y, b.x);
  const FIELD y2_sub_x2 = FIELD_sub(b.y, b.x);
  const FIELD y1_add_x1 = FIELD_add(a.y, a.x);

  const FIELD A = FIELD_mul(y1_sub_x1, y2_add_x2);	// A = (Y1 - X1) * (Y2 + X2)
  const FIELD B = FIELD_mul(y1_add_x1, y2_sub_x2);	// B = (Y1 + X1) * (Y2 - X2)
  FIELD F = FIELD_sub(B, A);				// F = B - A

  FIELD C, D, E, G, H;

  if (FIELD_eq(F, FIELD_ZERO)) {			// use doubling here
    const FIELD A = FIELD_sqr(a.x);
    const FIELD B = FIELD_sqr(a.y);
    C = FIELD_sqr(a.z);
    C = FIELD_double(C);
    D = FIELD_sub(FIELD_ZERO, A);

    E = FIELD_sqr(y1_add_x1);
    E = FIELD_sub(E, A);
    E = FIELD_sub(E, B);

    G = FIELD_add(D, B);
    F = FIELD_sub(G, C);
    H = FIELD_sub(D, B);
  } else {
    C = FIELD_mul(a.z, b.t);
    C = FIELD_double(C);				// C = 2 * Z1 * T2
    D = FIELD_double(a.t);				// D = 2 * T1

    E = FIELD_add(D, C);				// E = D + C
    H = FIELD_sub(D, C);				// H = D - C
    G = FIELD_add(B, A);				// G = B + A
  }
  
  a.x = FIELD_mul(E, F);				// X3 = E * F
  a.y = FIELD_mul(G, H);				// Y3 = G * H
  a.t = FIELD_mul(E, H);				// T3 = E * H
  a.z = FIELD_mul(F, G);				// Z3 = F * G

  return a;
}

// a = -1
// http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-madd-2008-hwcd-4
// Algebraic Structure: G = (x, y, t, z), -G = (-x, y, -t, z)
// a - b = a + (-b)
DEVICE POINT_ed_projective POINT_ed_neg_one_a_add_mixed_dedicated_naf(POINT_ed_projective a, POINT_ed_affine b, bool neg) {
  if (neg) {
    b.x = FIELD_sub(FIELD_ZERO, b.x);
    b.t = FIELD_sub(FIELD_ZERO, b.t);
  }

  const FIELD y1_sub_x1 = FIELD_sub(a.y, a.x);
  const FIELD y2_add_x2 = FIELD_add(b.y, b.x);
  const FIELD y2_sub_x2 = FIELD_sub(b.y, b.x);
  const FIELD y1_add_x1 = FIELD_add(a.y, a.x);

  const FIELD A = FIELD_mul(y1_sub_x1, y2_add_x2);	// A = (Y1 - X1) * (Y2 + X2)
  const FIELD B = FIELD_mul(y1_add_x1, y2_sub_x2);	// B = (Y1 + X1) * (Y2 - X2)
  FIELD F = FIELD_sub(B, A);				// F = B - A

  FIELD C, D, E, G, H;

  if (FIELD_eq(F, FIELD_ZERO)) {			// use doubling here
    const FIELD A = FIELD_sqr(a.x);
    const FIELD B = FIELD_sqr(a.y);
    C = FIELD_sqr(a.z);
    C = FIELD_double(C);
    D = FIELD_sub(FIELD_ZERO, A);

    E = FIELD_sqr(y1_add_x1);
    E = FIELD_sub(E, A);
    E = FIELD_sub(E, B);

    G = FIELD_add(D, B);
    F = FIELD_sub(G, C);
    H = FIELD_sub(D, B);
  } else {
    C = FIELD_mul(a.z, b.t);
    C = FIELD_double(C);				// C = 2 * Z1 * T2
    D = FIELD_double(a.t);				// D = 2 * T1

    E = FIELD_add(D, C);				// E = D + C
    H = FIELD_sub(D, C);				// H = D - C
    G = FIELD_add(B, A);				// G = B + A
  }
  
  a.x = FIELD_mul(E, F);				// X3 = E * F
  a.y = FIELD_mul(G, H);				// Y3 = G * H
  a.t = FIELD_mul(E, H);				// T3 = E * H
  a.z = FIELD_mul(F, G);				// Z3 = F * G

  return a;
}

// a = -1
// http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-3
DEVICE POINT_ed_projective POINT_ed_neg_one_a_add(POINT_ed_projective a, POINT_ed_projective b) {
  const FIELD y1_sub_x1 = FIELD_sub(a.y, a.x);
  const FIELD y2_sub_x2 = FIELD_sub(b.y, b.x);
  const FIELD y1_add_x1 = FIELD_add(a.y, a.x);
  const FIELD y2_add_x2 = FIELD_add(b.y, b.x);

  const FIELD A = FIELD_mul(y1_sub_x1, y2_add_x2);	// A = (Y1 - X1) * (Y2 + X2)
  const FIELD B = FIELD_mul(y1_add_x1, y2_sub_x2);	// B = (Y1 + X1) * (Y2 - X2)

  FIELD F = FIELD_sub(B, A);				// F = B - A
  FIELD C, D, E, G, H;

  if (FIELD_eq(F, FIELD_ZERO)) {			// use doubling here
    const FIELD A = FIELD_sqr(a.x);
    const FIELD B = FIELD_sqr(a.y);
    C = FIELD_sqr(a.z);
    C = FIELD_double(C);
    D = FIELD_sub(FIELD_ZERO, A);

    E = FIELD_sqr(y1_add_x1);
    E = FIELD_sub(E, A);
    E = FIELD_sub(E, B);

    G = FIELD_add(D, B);
    F = FIELD_sub(G, C);
    H = FIELD_sub(D, B);
  } else {
    C = FIELD_mul(a.z, b.t);				// C = 2 * Z1 * T2
    C = FIELD_double(C);

    D = FIELD_mul(a.t, b.z);
    D = FIELD_double(D);				// D = 2 * T1 * Z2

    E = FIELD_add(D, C);				// E = D + C
    H = FIELD_sub(D, C);				// H = D - C
    G = FIELD_add(B, A);				// G = B + A
  }

  a.x = FIELD_mul(E, F);				// X3 = E * F
  a.y = FIELD_mul(G, H);				// Y3 = G * H
  a.t = FIELD_mul(E, H);				// T3 = E * H
  a.z = FIELD_mul(F, G);				// Z3 = F * G

  return a;
}
