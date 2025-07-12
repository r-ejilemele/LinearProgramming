#define MPREAL_HAVE_DYNAMIC_STD_NUMERIC_LIMITS 0
#include "mpreal.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// namespace std {
// template <> class numeric_limits<mpfr::mpreal> {
// public:
//   static constexpr bool is_specialized = true;

//   static mpfr::mpreal min() noexcept { return mpfr::mpreal::min(); }
//   static mpfr::mpreal max() noexcept { return mpfr::mpreal::max(); }

//   static constexpr int digits10 = 50; // or your desired precision
//   static constexpr int max_digits10 = 2 * digits10;
//   static constexpr bool is_signed = true;
//   static constexpr bool is_integer = false;
//   static constexpr bool is_exact = false;
//   static constexpr bool has_infinity = true;
//   static constexpr bool has_quiet_NaN = true;
//   static constexpr bool has_signaling_NaN = false;
//   static constexpr float_denorm_style has_denorm = denorm_absent;
//   static constexpr bool has_denorm_loss = false;
//   static constexpr rounding_style round_style = round_indeterminate;
//   static constexpr bool is_iec559 = false;
//   static constexpr bool is_bounded = false;
//   static constexpr bool traps = false;
//   static constexpr bool tinyness_before = false;

//   static mpfr::mpreal epsilon() noexcept { return mpfr::mpreal::epsilon(); }
//   static mpfr::mpreal round_error() noexcept { return 0.5; }
//   static mpfr::mpreal infinity() noexcept { return mpfr::const_infinity(); }
//   static mpfr::mpreal quiet_NaN() noexcept { return mpfr::const_nan(); }
//   static mpfr::mpreal signaling_NaN() noexcept { return mpfr::const_nan(); }
//   static mpfr::mpreal denorm_min() noexcept { return mpfr::mpreal(0); }
// };
// } // namespace std
