/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

// Exponential map: The exponential map converts any element of the tangent space exactly into a transformation in the group

/*! \brief Exponential map from \f$\mathfrak{so}(3)\f$ to \f$SO(3)\f$.
 * 
 * \tparam Scalar Type to use (float/double/...)
 * \tparam Derived Eigen type that results in a vector of dimension 3
 * \param omega Vector \f$\omega\f$ of dimension 3
 * \return Rotation matrix
 */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> expSO3(const Eigen::MatrixBase<Derived>& omega) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_VECTOR(Derived, 3);

    using Scalar = typename Derived::Scalar;
    using mat3_t = Eigen::Matrix<Scalar, 3, 3>;
    auto norm = omega.norm();
    mat3_t R = mat3_t::Identity();
    if (norm > dummy_precision<Scalar>()) {
        mat3_t wx = vector3ToCrossMatrix3(omega / norm);
        R += std::sin(norm) * wx + (Scalar(1) - std::cos(norm)) * wx * wx;
    }

    return R;
}

/*! \brief Logarithm map from \f$SO(3)\f$ to \f$\mathfrak{so}(3)\f$.
 * 
 * \tparam Scalar Type to use (float/double/...)
 * \tparam Derived Eigen type that results in a matrix of dimension 3x3
 * \param R Rotation matrix \f$R\f$ of dimension 3x3
 * \return Angular velocity
 */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> logSO3(const Eigen::MatrixBase<Derived>& R) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_MATRIX(Derived, 3, 3);
    using Scalar = typename Derived::Scalar;
    auto theta = std::acos((R.trace() - Scalar(1)) / Scalar(2));

    return crossMatrix3ToVector3(R) * ((theta > dummy_precision<Scalar>()) ? theta / std::sin(theta) : Scalar(1));
}

/*! \brief Exponential map from \f$\mathfrak{se}(3)\f$ to \f$SE(3)\f$.
 * 
 * \tparam Derived Eigen type that results in a vector of dimension 6
 * \param nu Motion vector
 * \return A transformation matrix
 */
template <typename Derived>
static Transform<typename Derived::Scalar> expSE3(const Eigen::MatrixBase<Derived>& nu) noexcept
{
    COMA_STATIC_ASSERT_IS_FP(typename Derived::Scalar);
    COMA_STATIC_ASSERT_IS_EIGEN_VECTOR(Derived, 6);
    return { expSO3(nu.template head<3>()), nu.template tail<3>() };
}

/*! \brief Exponential map from \f$\mathfrak{se}(3)\f$ to \f$SE(3)\f$.
 * 
 * \tparam Scalar Type to use (float/double/...)
 * \param nu Motion vector
 * \return A transformation matrix
 */
template <typename Scalar>
static Transform<Scalar> expSE3(const MotionVector<Scalar>& nu) noexcept
{
    COMA_STATIC_ASSERT_IS_FP(Scalar);
    return { expSO3(nu.angular()), nu.linear() };
}

/*! \brief Log map from \f$SE(3)\f$ to \f$\mathfrak{se}(3)\f$.
 * 
 * \tparam ReturnType Type to return: Either a sva motion vector or an Eigen vector of dimension 6
 */
template <typename ReturnType>
struct logSE3 {
    /*! \brief Log map from \f$SE(3)\f$ to \f$\mathfrak{se}(3)\f$.
    * 
    * \tparam Scalar Type used by transformation (float/double/...)
    * \tparam Space Space of transformation.
    * \param tf Transformation \f$X\f$
    * \return \f$\log(X)\f$
    */
    template <typename Scalar>
    static ReturnType map(const Transform<Scalar>& tf) noexcept
    {
        if constexpr (internal::is_motion_vector<ReturnType>::value) {
            return { logSO3(tf.rotation()), tf.translation() };
        } else {
            COMA_STATIC_ASSERT_IS_EIGEN_VECTOR(ReturnType, 6);
            return (ReturnType() << logSO3(tf.rotation()), tf.translation()).finished();
        }
    }
};

/*! \brief Helper struct to log map a matrix into a Motion vector.
 * \tparam Scalar Type of matrix (float/double/etc...)
 */
template <typename Scalar>
using motionLogSE3 = logSE3<MotionVector<Scalar>>;
/*! \brief Helper struct to log map a matrix into a Eigen vector.
 * \tparam Scalar Type of matrix (float/double/etc...)
 */
template <typename Scalar>
using eigenLogSE3 = logSE3<Eigen::Matrix<Scalar, 6, 1>>;

/*! \brief Exponential map of \f$\mathfrak{so}(3)\f$ to \f$SU(2)\f$.
 *
 * See https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf
 * \tparam Derived Eigen type that results in a matrix of dimension 3x3.
 * \param omega Angular velocity.
 * \return Quaternion.
 */
template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> expSU2(const Eigen::MatrixBase<Derived>& omega) noexcept
{
    COMA_STATIC_ASSERT_IS_EIGEN_VECTOR(Derived, 3);

    using Scalar = typename Derived::Scalar;
    Eigen::Quaternion<Scalar> q;
    auto theta = omega.norm();
    Scalar s2;
    if (theta < dummy_precision<Scalar>()) {
        s2 = Scalar(0.5) + std::pow(theta, Scalar(2)) / Scalar(48); // + std::pow(theta, Scalar(4)) / Scalar(3840);
    } else {
        s2 = std::sin(Scalar(0.5) * theta) / theta;
    }

    q.w() = std::cos(Scalar(0.5) * theta);
    q.vec() = omega * s2;
    return q;
}

/*! \brief Logarithm map from \f$SU(2)\f$ to \f$\mathfrak{so}(3)\f$.
 * 
 * https://maxime-tournier.github.io/notes/quaternions
 * \tparam Scalar Type to use (float/double/...)
 * \param q Quaternion
 * \return Angular velocity
 */
template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> logSU2(const Eigen::Quaternion<Scalar>& q) noexcept
{
    auto theta = std::acos(q.w());
    return Scalar(2) * theta * q.vec().normalized();
}

} // namespace coma
