#pragma once
#include "ec.cuh"
#include "ff_dispatch_st.cuh"

namespace msm {

typedef ec<fd_p> curve;
typedef curve::storage storage;
typedef curve::field field;
typedef curve::point_affine point_affine;
typedef curve::point_jacobian point_jacobian;
typedef curve::point_xyzz point_xyzz;
typedef curve::point_projective point_projective;

} // namespace msm