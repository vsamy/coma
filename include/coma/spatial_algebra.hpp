/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

namespace internal {

template <typename _Scalar>
struct traits<SpatialInertia<_Scalar>> {
    using Scalar = _Scalar;
};

template <typename _Scalar>
struct traits<MotionSubspace<_Scalar>> {
    using Scalar = _Scalar;
    using mat_t = Eigen::Matrix<Scalar, 6, Dynamic>;
};

template <typename _Scalar, int _NVec>
struct traits<DiInertia<_Scalar, _NVec>> {
    static constexpr int n_vec = _NVec;
    using Scalar = _Scalar;
    using underlying_t = SpatialInertia<Scalar>;
};

template <typename _Scalar, int _NVec>
struct traits<DiMotionSubspace<_Scalar, _NVec>> {
    static constexpr int n_vec = _NVec;
    using Scalar = _Scalar;
    using underlying_t = MotionSubspace<Scalar>;
};

} // namespace internal

/*! \brief Represent a spatial inertia.
 * Largely inspired from SpaceVecAlg: https://github.com/jrl-umi3218/SpaceVecAlg
 * \tparam Scalar Underlying type to use (float/double/etc...)
 */
template <typename Scalar>
class SpatialInertia {
    COMA_STATIC_ASSERT_IS_FP(Scalar);

public:
    using vec3_t = Eigen::Matrix<Scalar, 3, 1>;
    using mat3_t = Eigen::Matrix<Scalar, 3, 3>;
    using mat6_t = Eigen::Matrix<Scalar, 6, 6>;

public:
    /*! \brief Default constructor. */
    SpatialInertia() = default;
    /*! \brief Constructor..
     *
     * \param m Mass of the link.
     * \param h Linear momentum at link origin.
     * \param I Inertia matrix at link origin.
     */
    SpatialInertia(Scalar m, const vec3_t& h, const mat3_t& I)
        : m_mass(m)
        , m_momentum(h)
        , m_inertia(I)
    {
    }
    /*! \brief Return the link mass.*/
    Scalar mass() const noexcept { return m_mass; }
    /*! \brief Return the link linear momentum at link origin.*/
    const vec3_t& momentum() const noexcept { return m_momentum; }
    /*! \brief Return the link inertia at link origin.*/
    const mat3_t& inertia() const noexcept { return m_inertia; }
    /*! \brief Return the link mass.*/
    Scalar& mass() noexcept { return m_mass; }
    /*! \brief Return the link linear momentum at link origin.*/
    vec3_t& momentum() noexcept { return m_momentum; }
    /*! \brief Return the link mass.*/
    mat3_t& inertia() noexcept { return m_inertia; }
    /*! \brief Return the number of rows of the spatial inertia.*/
    constexpr Index rows() const noexcept { return 6; }
    /*! \brief Return the number of columns of the spatial inertia.*/
    constexpr Index cols() const noexcept { return 6; }
    /*! \brief Return the spatial inertia matrix.*/
    mat6_t matrix() const
    {
        mat6_t out;
        mat3_t hx = vector3ToCrossMatrix3(m_momentum);
        out << m_inertia, hx, hx.transpose(), mat3_t::Identity() * m_mass;
        return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                Operator                                //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Operator between two spatial inertias.
     *
     * \param lhs A Spatial inertia \f$I_1\f$
     * \param rhs A Spatial inertia \f$I_2\f$
     * \return \f$I_1 + I_2\f$
     */
    friend SpatialInertia operator+(const SpatialInertia& lhs, const SpatialInertia& rhs)
    {
        return { lhs.m_mass + rhs.m_mass, lhs.m_momentum + rhs.m_momentum, lhs.m_inertia + rhs.m_inertia };
    }
    /*! \brief Operator between two spatial inertias.
     *
     * \param lhs A Spatial inertia \f$I_1\f$
     * \param rhs A Spatial inertia \f$I_2\f$
     * \return \f$I_1 - I_2\f$
     */
    friend SpatialInertia operator-(const SpatialInertia& lhs, const SpatialInertia& rhs)
    {
        return { lhs.m_mass - rhs.m_mass, lhs.m_momentum - rhs.m_momentum, lhs.m_inertia - rhs.m_inertia };
    }
    /*! \brief Unary minus operator.
     *
     * \param lhs A Spatial inertia \f$I\f$
     * \return \f$-I\f$
     */
    friend SpatialInertia operator-(const SpatialInertia& rhs)
    {
        return { -rhs.m_mass, -rhs.m_momentum, -rhs.m_inertia };
    }
    /*! \brief Operator between two SpatialInertia.
     *
     * \param rhs A Spatial inertia \f$I_2\f$
     * \return \f$I\f$ after \f$I += I_2\f$
     */
    SpatialInertia& operator+=(const SpatialInertia& rhs)
    {
        m_inertia += rhs.m_inertia;
        m_mass += rhs.m_mass;
        m_momentum += rhs.m_momentum;
        return *this;
    }
    /*! \brief Operator between two SpatialInertia.
     *
     * \param rhs A Spatial inertia \f$I_2\f$
     * \return \f$I\f$ after \f$I -= I_2\f$
     */
    SpatialInertia& operator-=(const SpatialInertia& rhs)
    {
        m_inertia -= rhs.m_inertia;
        m_mass -= rhs.m_mass;
        m_momentum -= rhs.m_momentum;
        return *this;
    }
    /*! \brief Operator between a spatial inertia and a scalar.
     *
     * \param lhs A Spatial inertia \f$I\f$
     * \param rhs A scalar \f$v\f$
     * \return \f$I * v\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend SpatialInertia operator*(const SpatialInertia& lhs, T rhs)
    {
        return { rhs * lhs.m_mass, rhs * lhs.m_momentum, rhs * lhs.m_inertia };
    }
    /*! \brief Operator between a spatial inertia and a scalar.
     *
     * \param lhs A scalar \f$v\f$
     * \param rhs A Spatial inertia \f$I\f$
     * \return \f$v * I\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend SpatialInertia operator*(T lhs, const SpatialInertia& rhs)
    {
        return rhs * lhs;
    }
    /*! \brief Operator between a spatial inertia and a scalar.
     *
     * \param rhs A scalar \f$v\f$
     * \return \f$I\f$ after \f$I *= v\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    SpatialInertia& operator*=(T rhs)
    {
        m_mass *= rhs;
        m_momentum *= rhs;
        m_inertia *= rhs;
        return *this;
    }
    /*! \brief Check if two SpatialInertia are equals.
     *
     * \warning This performs a strict comparison.
     * To compare two SpatialInertia operators with a precision limit, use SpatialInertia::isApprox.
     * \param lhs A SpatialInertia \f$I_1\f$
     * \param rhs A SpatialInertia \f$I_2\f$
     * \return true if equals.
     */
    friend bool operator==(const SpatialInertia& lhs, const SpatialInertia& rhs) noexcept
    {
        return lhs.m_mass == rhs.m_mass && lhs.m_momentum == rhs.m_momentum && lhs.m_inertia == rhs.m_inertia;
    }
    /*! \brief Check if two SpatialInertia are different.
     *
     * \warning This performs a strict comparison.
     * To compare two SpatialInertia operators with a precision limit, use SpatialInertia::isApprox.
     * \param lhs A SpatialInertia \f$I_1\f$
     * \param rhs A SpatialInertia \f$I_2\f$
     * \return true if different.
     */
    friend bool operator!=(const SpatialInertia& lhs, const SpatialInertia& rhs) noexcept
    {
        return lhs.m_mass != rhs.m_mass || lhs.m_momentum != rhs.m_momentum || lhs.m_inertia != rhs.m_inertia;
    }
    /*! \brief Compare two SpatialInertia with a precision limit.
     *
     * \param rhs A SpatialInertia \f$s\f$
     * \param prec The precision limit. Default is base on Eigen limit
     * \return true if approximately equals.
     */
    bool isApprox(const SpatialInertia& rhs, Scalar prec = dummy_precision<Scalar>()) const noexcept
    {
        bool success = std::abs(m_mass - rhs.m_mass) < prec;
        success = success && m_momentum.isApprox(rhs.m_momentum, prec);
        success = success && m_inertia.isApprox(rhs.m_inertia, prec);
        return success;
        // return std::abs(m_mass - rhs.m_mass) < prec && m_momentum.isApprox(rhs.m_momentum, prec) && m_inertia.isApprox(rhs.m_inertia, prec);
    }

private:
    Scalar m_mass; /*!< Mass */
    vec3_t m_momentum; /*!< Momentum at link origin */
    mat3_t m_inertia; /*!< Inertia at link origin */
};

/*! \brief Represent a Joint motion subspace.
 * \tparam Scalar Underlying type to use (float/double/etc...)
 */
template <typename Scalar>
class MotionSubspace {
    COMA_STATIC_ASSERT_IS_FP(Scalar);

public:
    using mat_t = Eigen::Matrix<Scalar, Dynamic, Dynamic>;

public:
    /*! \brief Default constructor. */
    MotionSubspace() = default;
    /*! \brief Constructor.
     *
     * \tparam Derived An Eigen type.
     * \param S Motion subsbspace matrix.
     */
    template <typename Derived>
    MotionSubspace(const Eigen::MatrixBase<Derived>& S)
        : m_S(S)
    {}
    /*! \brief Transpose of the Motion subspace. */
    MotionSubspace transpose() const noexcept { return { m_S.transpose() }; }
    /*! \brief Return the motion subspace matrix */
    const mat_t& matrix() const noexcept { return m_S; }
    /*! \brief Return the motion subspace matrix */
    mat_t& matrix() noexcept { return m_S; }
    /*! \brief Return the number of rows of the motion subspace matrix */
    Index rows() const noexcept { return m_S.rows(); }
    /*! \brief Return the number of cols of the motion subspace matrix */
    Index cols() const noexcept { return m_S.cols(); }
    /*! \brief Check if two MotionSubspace are equals.
     *
     * \warning This performs a strict comparison.
     * To compare two MotionSubspace operators with a precision limit, use MotionSubspace::isApprox.
     * \param lhs A MotionSubspace \f$S_1\f$
     * \param rhs A MotionSubspace \f$S_2\f$
     * \return true if equals.
     */
    friend bool operator==(const MotionSubspace& lhs, const MotionSubspace& rhs) noexcept { return lhs.m_S == rhs.m_S; }
    /*! \brief Check if two MotionSubspace are different.
     *
     * \warning This performs a strict comparison.
     * To compare two MotionSubspace operators with a precision limit, use SpatialVectorX::isApprox.
     * \param lhs A MotionSubspace \f$S_1\f$
     * \param rhs A MotionSubspace \f$S_2\f$
     * \return true if different.
     */
    friend bool operator!=(const MotionSubspace& lhs, const MotionSubspace& rhs) noexcept { return !(lhs == rhs); }
    /*! \brief Compare two MotionSubspace with a precision limit.
     *
     * \param rhs A MotionSubspace \f$S\f$
     * \param prec The precision limit. Default is base on Eigen limit
     * \return true if approximately equals.
     */
    bool isApprox(const MotionSubspace& rhs, Scalar prec = dummy_precision<Scalar>()) const noexcept { return m_S.isApprox(rhs.matrix(), prec); }

private:
    mat_t m_S; /*!< Motion subspace matrix */
};

/*! \brief Diagonal inertia matrix.
 *
 * This class represents a block-diagonal matrix of Spatial inertia.
 * \tparam Scalar Underlying type (float/double/...)
 * \tparam NVec Number of matrix blocks
 */
template <typename Scalar, int NVec>
class DiInertia : public DiBlockT<DiInertia<Scalar, NVec>> {
    COMA_STATIC_ASSERT_IS_FP(Scalar);

public:
    /*! \brief Default constructor. */
    DiInertia() = default;
    /*! \brief Constructor.
     *
     * \param I The spatial inertia matrix.
     */
    DiInertia(SpatialInertia<Scalar> I)
        : DiBlockT<DiInertia<Scalar, NVec>>::DiBlockT(std::move(I))
    {}
};

/*! \brief Diagonal Motion subspace matrix.
 *
 * This class represents a block-diagonal matrix of motion subspace.
 * \tparam Scalar Underlying type (float/double/...)
 * \tparam NVec Number of matrix blocks
 */
template <typename Scalar, int NVec>
class DiMotionSubspace : public DiBlockT<DiMotionSubspace<Scalar, NVec>> {
    COMA_STATIC_ASSERT_IS_FP(Scalar);

public:
    /*! \brief Default constructor. */
    DiMotionSubspace() = default;
    /*! \brief Constructor.
     *
     * \param S The motion subspace matrix.
     */
    DiMotionSubspace(MotionSubspace<Scalar> S)
        : DiBlockT<DiMotionSubspace<Scalar, NVec>>::DiBlockT(std::move(S))
    {}
    /*! \brief Return the transpose of the motion subspace. */
    DiMotionSubspace<Scalar, NVec> transpose() const { return { this->m_block.transpose() }; }
};

} // namespace coma
