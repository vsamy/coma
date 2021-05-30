/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

/*! \brief Return N!. */
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline constexpr T Factorial(T N) noexcept
{
    return N > 1 ? N * Factorial(N - 1) : 1;
}

/*! \return n choose k. */
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline constexpr T Binomial(T n, T k) noexcept
{
    if (k > n)
        return 0;
    else if (k == 0 || k == n)
        return 1;
    else if (k == 1 || k == n - 1)
        return n;
    else if (2 * k < n)
        return Binomial(n - 1, k - 1) * n / k;
    else
        return Binomial(n - 1, k) * n / (n - k);
}

namespace internal {

template <int Order, int Index>
struct populate_order {
    template <class Fun>
    static constexpr void impl(Fun fun)
    {
        fun(Order, Index);
        populate_order<Order, Index - 1>::impl(fun);
    }
};

template <int Order>
struct populate_order<Order, 0> {
    template <class Fun>
    static constexpr void impl(Fun fun)
    {
        fun(Order, 0);
    }
};

template <int Order, int Index>
struct populate {
    template <class Fun>
    static constexpr void impl(Fun fun)
    {
        populate_order<Index, Index>::impl(fun);
        populate<Order, Index - 1>::impl(fun);
    }
};

template <int Order>
struct populate<Order, 0> {
    template <class Fun>
    static constexpr void impl(Fun fun)
    {
        populate_order<0, 0>::impl(fun);
    }
};

template <typename Scalar, int Order, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
inline constexpr std::array<Scalar, Order*(Order + 1) / 2> make_pascal_factors()
{
    COMA_STATIC_ASSERT(Order > 0, "Order must be > 0");
    std::array<Scalar, Order*(Order + 1) / 2> factors{};
    auto fun = [&factors](int ord, int ind) {
        int curInd = ind + ord * (ord + 1) / 2;
        factors[static_cast<size_t>(curInd)] = static_cast<Scalar>(Binomial(ord, ind));
    };

    populate<Order, Order - 1>::impl(fun);
    return factors;
}

template <int MaxIndex, int Index>
struct generate {
    template <class Fun>
    static constexpr void impl(Fun fun)
    {
        fun(Index);
        generate<MaxIndex, Index + 1>::impl(fun);
    }
};

template <int MaxIndex>
struct generate<MaxIndex, MaxIndex> {
    template <class Fun>
    static constexpr void impl(Fun fun)
    {
        fun(MaxIndex);
    }
};

template <typename Scalar, int Order>
inline constexpr std::array<Scalar, Order> make_factorial_factors()
{
    std::array<Scalar, Order> factors{};
    auto fun = [&factors](size_t ind) {
        factors[ind] = static_cast<Scalar>(Factorial(ind));
    };

    generate<Order - 1, 0>::impl(fun);
    return factors;
}

template <typename Scalar, int Order>
inline constexpr std::array<Scalar, Order> make_inverse_factorial_factors()
{
    COMA_STATIC_ASSERT_IS_FP(Scalar);
    std::array<Scalar, Order> factors{};
    auto fun = [&factors](size_t ind) {
        factors[ind] = Scalar(1) / static_cast<Scalar>(Factorial(ind));
    };

    generate<Order - 1, 0>::impl(fun);
    return factors;
}

} // namespace internal

/*! \brief Binomial coefficient in an array.
 * \tparam Scalar Type to use (int/float/etc...)
 * \tparam Order Max N value of the binomial
 */
template <typename Scalar, int Order>
inline static constexpr auto pascal_factors = internal::make_pascal_factors<Scalar, Order>();
/*! \brief Specialization for dynamic case. Use of big factors. */
template <typename Scalar>
inline static constexpr auto pascal_factors<Scalar, Dynamic> = internal::make_pascal_factors<Scalar, 10>();
/*! \brief Specialization for empty case.  */
template <typename Scalar>
inline static constexpr auto pascal_factors<Scalar, 0> = std::array<Scalar, 0>{};
/*! \brief Factorial factors for the CMTM
 * \tparam Scalar Type to use (int/float/etc...)
 * \tparam Order Order of the CMTM
 */
template <typename Scalar, int Order>
inline static constexpr auto factorial_factors = internal::make_factorial_factors<Scalar, Order>();
/*! \brief Specialization for dynamic case. Use of big factors. */
template <typename Scalar>
inline static constexpr auto factorial_factors<Scalar, Dynamic> = internal::make_factorial_factors<Scalar, 10>();
/*! \brief Specialization for empty case. */
template <typename Scalar>
inline static constexpr auto factorial_factors<Scalar, 0> = std::array<Scalar, 0>{};
/*! \brief Inverse of factorial factors for the CMTM
 * \tparam Scalar Type to use (float/double/etc...)
 * \tparam Order Order of the CMTM
 */
template <typename Scalar, int Order>
inline static constexpr auto inverse_factorial_factors = internal::make_inverse_factorial_factors<Scalar, Order>();
/*! \brief Specialization for dynamic case. Use of big factors. */
template <typename Scalar>
inline static constexpr auto inverse_factorial_factors<Scalar, Dynamic> = internal::make_inverse_factorial_factors<Scalar, 10>();
/*! \brief Specialization for empty case. */
template <typename Scalar>
inline static constexpr auto inverse_factorial_factors<Scalar, 0> = std::array<Scalar, 0>{};

// cross <-> vector

/*! \return \f$\left[v\mkern-2mu\times_3\mkern-2mu\right]\f$ from \f$v\f$.
 * \tparam TVec A 3D vector-like type
 * \param v 3D vector
 */
template <typename TVec>
inline Eigen::Matrix<typename TVec::Scalar, 3, 3> vector3ToCrossMatrix3(const TVec& v) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_VECTOR(TVec, 3);

    Eigen::Matrix<typename TVec::Scalar, 3, 3> out;
    out << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return out;
}
/*! \return \f$v\f$ from \f$\left[v\mkern-2mu\times_3\mkern-2mu\right]\f$.
 * \tparam TMat A 3D matrix-like type
 * \param m 3D matrix
 */
template <typename TMat>
inline Eigen::Matrix<typename TMat::Scalar, 3, 1> crossMatrix3ToVector3(const TMat& m) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_MATRIX(TMat, 3, 3);

    return 0.5 * Eigen::Matrix<typename TMat::Scalar, 3, 1>{ m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1) };
}

namespace internal {

template <typename TVec>
inline Eigen::Matrix<typename TVec::Scalar, 4, 4> eigenVector6ToCrossMatrix4(const TVec& v) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_VECTOR(TVec, 6);

    using Scalar = typename TVec::Scalar;
    Eigen::Matrix<Scalar, 4, 4> out;
    out << vector3ToCrossMatrix3(v.template head<3>()), v.template tail<3>(), Eigen::Matrix<Scalar, 1, 4>::Zero();
    return out;
}

template <typename Scalar>
inline Eigen::Matrix<Scalar, 4, 4> motionVector6ToCrossMatrix4(const MotionVector<Scalar>& m) noexcept
{
    Eigen::Matrix<Scalar, 4, 4> out;
    out << vector3ToCrossMatrix3(m.angular()), m.linear(), Eigen::Matrix<Scalar, 1, 4>::Zero();
    return out;
}

template <typename TVec>
inline Eigen::Matrix<typename TVec::Scalar, 6, 6> eigenVector6ToCrossMatrix6(const TVec& v) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_VECTOR(TVec, 6);

    using Scalar = typename TVec::Scalar;
    Eigen::Matrix<Scalar, 6, 6> out;
    auto w3 = vector3ToCrossMatrix3(v.template head<3>());
    out << w3, Eigen::Matrix<Scalar, 3, 3>::Zero(), vector3ToCrossMatrix3(v.template tail<3>()), w3;
    return out;
}

template <typename Scalar>
inline Eigen::Matrix<Scalar, 6, 6> motionVector6ToCrossMatrix6(const MotionVector<Scalar>& m) noexcept
{
    Eigen::Matrix<Scalar, 6, 6> out;
    auto w3 = vector3ToCrossMatrix3(m.angular());
    out << w3, Eigen::Matrix<Scalar, 3, 3>::Zero(), vector3ToCrossMatrix3(m.linear()), w3;
    return out;
}

template <typename TVec>
inline Eigen::Matrix<typename TVec::Scalar, 6, 6> eigenVector6ToCrossDualMatrix6(const TVec& v) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_VECTOR(TVec, 6);

    using Scalar = typename TVec::Scalar;
    Eigen::Matrix<Scalar, 6, 6> out;
    auto n3 = vector3ToCrossMatrix3(v.template head<3>());
    out << n3, vector3ToCrossMatrix3(v.template tail<3>()), Eigen::Matrix<Scalar, 3, 3>::Zero(), n3;
    return out;
}

template <typename Scalar>
inline Eigen::Matrix<Scalar, 6, 6> motionVector6ToCrossDualMatrix6(const MotionVector<Scalar>& f) noexcept
{
    Eigen::Matrix<Scalar, 6, 6> out;
    auto n3 = vector3ToCrossMatrix3(f.angular());
    out << n3, vector3ToCrossMatrix3(f.linear()), Eigen::Matrix<Scalar, 3, 3>::Zero(), n3;
    return out;
}

template <typename TMat>
inline Eigen::Matrix<typename TMat::Scalar, 6, 1> eigenCrossMatrix4ToVector6(const TMat& m) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_MATRIX(TMat, 4, 4);

    Eigen::Matrix<typename TMat::Scalar, 6, 1> out;
    out << crossMatrix3ToVector3(m.template block<3, 3>(0, 0)), m.col(3).template head<3>();
    return out;
}

template <typename TMat>
inline MotionVector<typename TMat::Scalar> motionCrossMatrix4ToVector6(const TMat& m) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_MATRIX(TMat, 4, 4);

    return MotionVector<typename TMat::Scalar>(coma::crossMatrix3ToVector3(m.template block<3, 3>(0, 0)), m.col(3).template head<3>());
}

template <typename TMat>
inline Eigen::Matrix<typename TMat::Scalar, 6, 1> eigenCrossMatrix6ToVector6(const TMat& m) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_MATRIX(TMat, 6, 6);

    Eigen::Matrix<typename TMat::Scalar, 6, 1> out;
    out << crossMatrix3ToVector3(m.template block<3, 3>(0, 0)), crossMatrix3ToVector3(m.template block<3, 3>(3, 0));
    return out;
}

template <typename TMat>
inline MotionVector<typename TMat::Scalar> motionCrossMatrix6ToVector6(const TMat& m) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_MATRIX(TMat, 6, 6);

    return MotionVector<typename TMat::Scalar>(crossMatrix3ToVector3(m.template block<3, 3>(0, 0)), crossMatrix3ToVector3(m.template block<3, 3>(3, 0)));
}

template <typename TMat>
inline Eigen::Matrix<typename TMat::Scalar, 6, 1> eigenCrossDualMatrix6ToVector6(const TMat& m) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_MATRIX(TMat, 6, 6);

    Eigen::Matrix<typename TMat::Scalar, 6, 1> out;
    out << crossMatrix3ToVector3(m.template block<3, 3>(0, 0)), crossMatrix3ToVector3(m.template block<3, 3>(0, 3));
    return out;
}

template <typename TMat>
inline MotionVector<typename TMat::Scalar> motionCrossDualMatrix6ToVector6(const TMat& m) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_MATRIX(TMat, 6, 6);

    return MotionVector<typename TMat::Scalar>(crossMatrix3ToVector3(m.template block<3, 3>(0, 0)), crossMatrix3ToVector3(m.template block<3, 3>(0, 3)));
}

} // namespace internal

/*! \return \f$\left[v\mkern-2mu\times_4\mkern-2mu\right]\f$ from \f$v\f$.
 * \tparam TVec A 6D vector-like type or MotionVec
 * \param v 6D vector
 */
template <typename TVec>
inline auto vector6ToCrossMatrix4(const TVec& v) noexcept
{
    if constexpr (internal::is_motion_vector<TVec>::value) {
        return internal::motionVector6ToCrossMatrix4(v);
    } else {
        return internal::eigenVector6ToCrossMatrix4(v);
    }
}
/*! \return \f$\left[v\mkern-2mu\times_6\mkern-2mu\right]\f$ from \f$v\f$.
 * \tparam TVec A 6D vector-like type or MotionVec
 * \param v 6D vector
 */
template <typename TVec>
inline auto vector6ToCrossMatrix6(const TVec& v) noexcept
{
    if constexpr (internal::is_motion_vector<TVec>::value) {
        return internal::motionVector6ToCrossMatrix6(v);
    } else {
        return internal::eigenVector6ToCrossMatrix6(v);
    }
}
/*! \return \f$\left[v\mkern-2mu\bar{\times}_6\mkern-2mu\right]\f$ from \f$v\f$
 * \tparam TVec A 6D vector-like type or MotionVec.
 * \param v 6D vector
 */
template <typename TVec>
inline auto vector6ToCrossDualMatrix6(const TVec& v) noexcept
{
    if constexpr (internal::is_motion_vector<TVec>::value) {
        return internal::motionVector6ToCrossDualMatrix6(v);
    } else {
        return internal::eigenVector6ToCrossDualMatrix6(v);
    }
}

/*! \brief Matrix to vector helper.
 * \tparam TVec Should be either a MotionVector or an Eigen::Matrix
 */
template <typename TVec>
struct uncross {
    static constexpr bool is_motion_vec_v = internal::is_motion_vector<TVec>::value; /*!< true if TVec is a MotionVec */
    /*! \return \f$v\f$ from \f$\left[v\mkern-2mu\times_4\mkern-2mu\right]\f$.
     * \tparam TMat A 4D matrix-like type
     * \param m 4D matrix
     */
    template <typename TMat>
    static inline TVec crossMatrix4ToVector6(const TMat& m) noexcept
    {
        if constexpr (is_motion_vec_v) {
            return internal::motionCrossMatrix4ToVector6(m);
        } else {
            return internal::eigenCrossMatrix4ToVector6(m);
        }
    }
    /*! \return \f$v\f$ from \f$\left[v\mkern-2mu\times_6\mkern-2mu\right]\f$.
     * \tparam TMat A 6D matrix-like type
     * \param m 6D matrix
     */
    template <typename TMat>
    static inline TVec crossMatrix6ToVector6(const TMat& m) noexcept
    {
        if constexpr (is_motion_vec_v) {
            return internal::motionCrossMatrix6ToVector6(m);
        } else {
            return internal::eigenCrossMatrix6ToVector6(m);
        }
    }
    /*! \return \f$v\f$ from \f$\left[v\mkern-2mu\bar{\times}_6\mkern-2mu\right]\f$.
     * \tparam TMat A 6D matrix-like type
     * \param m 6D matrix
     */
    template <typename TMat>
    static inline TVec crossDualMatrix6ToVector6(const TMat& m) noexcept
    {
        if constexpr (is_motion_vec_v) {
            return internal::motionCrossDualMatrix6ToVector6(m);
        } else
            return internal::eigenCrossDualMatrix6ToVector6(m);
    }
};

/*! \brief Helper struct to uncross a matrix to a Motion vector.
 * \tparam Scalar Type of matrix (float/double/etc...)
 */
template <typename Scalar>
using uncross_motion = uncross<MotionVector<Scalar>>;
/*! \brief Helper struct to uncross a matrix to a Eigen vector.
 * \tparam Scalar Type of matrix (float/double/etc...)
 */
template <typename Scalar>
using uncross_eigen = uncross<Eigen::Matrix<Scalar, 6, 1>>;

/*! \brief Dummy precision used to compare matrices.
 * The precision value is based on Eigen default value.
 * \tparam Scalar Type of matrix (float/double/etc...)
 * \return Precision value.
 */
template <typename Scalar>
Scalar dummy_precision() noexcept
{
    static Scalar prec = Eigen::NumTraits<Scalar>::dummy_precision();
    return prec;
}

} // namespace coma
