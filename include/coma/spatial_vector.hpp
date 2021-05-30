/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

namespace internal {

template <typename _Scalar>
struct traits<MotionVector<_Scalar>> {
    using Scalar = _Scalar;
    using vec3_t = Eigen::Matrix<Scalar, 3, 1>;
    using vec6_t = Eigen::Matrix<Scalar, 6, 1>;
};

template <typename _Scalar>
struct traits<ForceVector<_Scalar>> {
    using Scalar = _Scalar;
    using vec3_t = Eigen::Matrix<Scalar, 3, 1>;
    using vec6_t = Eigen::Matrix<Scalar, 6, 1>;
};

template <typename _Scalar, int _NVec>
struct traits<MotionVectorX<_Scalar, _NVec>> {
    static constexpr int n_vec = _NVec;
    using Scalar = _Scalar;
    using spatial_vector_t = MotionVector<Scalar>;
    using storage_t = Storage<spatial_vector_t, n_vec>;
    using vector_t = Eigen::Matrix<Scalar, (n_vec == Dynamic ? Dynamic : 6 * n_vec), 1>;
};

template <typename _Scalar, int _NVec>
struct traits<ForceVectorX<_Scalar, _NVec>> {
    static constexpr int n_vec = _NVec;
    using Scalar = _Scalar;
    using spatial_vector_t = ForceVector<Scalar>;
    using storage_t = Storage<spatial_vector_t, n_vec>;
    using vector_t = Eigen::Matrix<Scalar, (n_vec == Dynamic ? Dynamic : 6 * n_vec), 1>;
};

template <typename Derived, int S>
struct redux {
    static constexpr bool is_mvx = is_mvx_vector<Derived>::value;
    using Scalar = typename traits<Derived>::Scalar;
    using type = std::conditional_t<is_mvx, MotionVectorX<Scalar, S>, ForceVectorX<Scalar, S>>;
};

template <class Derived, int S>
using redux_t = typename redux<Derived, S>::type;

} // namespace internal

/*! \brief Represent a CMTM tangent vector.
 *
 * A Tangent vector is the representation of the Motion/Force at given point.
 * Under the hood,it is composed of a size-constrained vector of MotionVector or ForceVector
 * and is fixed at compile-time.
 * \tparam Derived Child class (CRTP)
 */
template <typename Derived>
class SpatialVector {
    friend Derived;
    using traits = internal::traits<Derived>;

public:
    using Scalar = typename traits::Scalar;
    using vec3_t = typename traits::vec3_t;
    using vec6_t = typename traits::vec6_t;

public:
    /*! \brief Default constructor. */
    SpatialVector() = default;
    /*! \brief Constructor.
     * \param angular Angular part of the vector.
     * \param linear Linear part of the vector.
     */
    SpatialVector(const vec3_t& angular, const vec3_t& linear)
        : m_angular(angular)
        , m_linear(linear)
    {
    }
    /*! \brief Constructor.
     * \param v Full representation of the vector.
     */
    SpatialVector(const vec6_t& v)
        : m_angular(v.template head<3>())
        , m_linear(v.template tail<3>())
    {
    }
    /*! \brief Set the vector to zero. */
    Derived& setZero() noexcept
    {
        m_angular.setZero();
        m_linear.setZero();
        return this->derived();
    }

    /*! \brief Return the angular part of the vector. */
    const vec3_t& angular() const noexcept { return m_angular; }
    /*! \brief Return the angular part of the vector. */
    vec3_t& angular() noexcept { return m_angular; }
    /*! \brief Return the linear part of the vector. */
    const vec3_t& linear() const noexcept { return m_linear; }
    /*! \brief Return the linear part of the vector. */
    vec3_t& linear() noexcept { return m_linear; }
    /*! \brief Return the full vector. */
    vec6_t vector() const noexcept { return (vec6_t() << m_angular, m_linear).finished(); }

    ////////////////////////////////////////////////////////////////////////////
    //                                 Static                                 //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Return a zero vector. 
     *
     * Only available for fixed-n_vec class.
     */
    static Derived Zero()
    {
        return { vec3_t::Zero(), vec3_t::Zero() };
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                Operator                                //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Unary minus operator.
     *
     * \param rhs A spatial vector \f$s\f$
     * return \f$-s\f$
     */
    friend Derived operator-(const Derived& rhs)
    {
        return { -rhs.m_angular, -rhs.m_linear };
    }
    /*! \brief Return the value at index i. */
    Scalar operator()(Index i) noexcept
    {
        COMA_ASSERT(i >= 0 && i < 6, "Out-of-order index");
        return i < 3 ? m_angular(i) : m_linear(i - 3);
    }
    /*! \brief Operation between two SpatialVector.
     *
     * \param lhs A SpatialVector \f$s_1\f$
     * \param rhs A SpatialVector \f$s_2\f$
     * \return \f$s_1 + s_2\f$
     */
    friend Derived operator+(Derived lhs, const SpatialVector& rhs)
    {
        lhs += rhs;
        return lhs;
    }
    /*! \brief Operation between two SpatialVector.
     *
     * \param lhs A SpatialVector \f$s_1\f$
     * \param rhs A SpatialVector \f$s_2\f$
     * \return \f$s_1 + s_2\f$
     */
    friend Derived operator-(Derived lhs, const SpatialVector& rhs)
    {
        lhs -= rhs;
        return lhs;
    }
    /*! \brief Operation between a SpatialVector and a scalar.
     *
     * \param lhs A SpatialVector \f$s\f$
     * \param rhs A scalar \f$v\f$
     * \return \f$s \cdot v\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived operator*(Derived lhs, T rhs)
    {
        lhs *= rhs;
        return lhs;
    }
    /*! \brief Operation between a scalar and a SpatialVector.
     *
     * \param lhs A scalar \f$v\f$
     * \param rhs A SpatialVector \f$s\f$
     * \return \f$v \cdot s\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived operator*(T lhs, const Derived& rhs)
    {
        return rhs * lhs;
    }
    /*! \brief Operation between a SpatialVector and a scalar.
     *
     * \param lhs A SpatialVector \f$s\f$
     * \param rhs A scalar \f$v\f$
     * \return \f$s / v\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived operator/(Derived lhs, T rhs)
    {
        lhs /= rhs;
        return lhs;
    }
    /*! \brief Operation between two SpatialVector.
     *
     * \param rhs A SpatialVector \f$s_2\f$
     * \return \f$s_1\f$ after \f$s_1 += s_2\f$
     */
    Derived& operator+=(const SpatialVector& rhs)
    {
        m_angular += rhs.angular();
        m_linear += rhs.linear();
        return this->derived();
    }
    /*! \brief Operation between two SpatialVector.
     *
     * \param rhs A SpatialVector \f$s_2\f$
     * \return \f$s_1\f$ after \f$s_1 -= s_2\f$
     */
    Derived& operator-=(const SpatialVector& rhs)
    {
        m_angular -= rhs.angular();
        m_linear -= rhs.linear();
        return this->derived();
    }
    /*! \brief Operation between a SpatialVector and a scalar.
     *
     * \param rhs A scalar \f$v\f$
     * \return \f$s\f$ after \f$s *= v\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    Derived& operator*=(T rhs)
    {
        m_angular *= rhs;
        m_linear *= rhs;
        return this->derived();
    }
    /*! \brief Operation between a SpatialVector and a scalar.
     *
     * \param rhs A scalar \f$v\f$
     * \return \f$s\f$ after \f$s /= v\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    Derived& operator/=(T rhs)
    {
        m_angular /= rhs;
        m_linear /= rhs;
        return this->derived();
    }
    /*! \brief Check if two SpatialVector are equals.
     *
     * \warning This performs a strict comparison.
     * To compare two SpatialVector operators with a precision limit, use SpatialVector::isApprox.
     * \param lhs A SpatialVector \f$s_1\f$
     * \param rhs A SpatialVector \f$s_2\f$
     * \return true if equals.
     */
    friend bool operator==(const SpatialVector& lhs, const SpatialVector& rhs) noexcept
    {
        return lhs.angular() == rhs.angular() && lhs.linear() == rhs.linear();
    }
    /*! \brief Check if two SpatialVector are different.
     *
     * \warning This performs a strict comparison.
     * To compare two SpatialVector operators with a precision limit, use SpatialVector::isApprox.
     * \param lhs A SpatialVector \f$s_1\f$
     * \param rhs A SpatialVector \f$s_2\f$
     * \return true if different.
     */
    friend bool operator!=(const SpatialVector& lhs, const SpatialVector& rhs) noexcept
    {
        return !(lhs == rhs);
    }
    /*! \brief Compare two SpatialVector with a precision limit.
     *
     * \param rhs A SpatialVector \f$s\f$
     * \param prec The precision limit. Default is base on Eigen limit
     * \return true if approximately equals.
     */
    bool isApprox(const SpatialVector& rhs, Scalar prec = dummy_precision<Scalar>()) const noexcept
    {
        return m_angular.isApprox(rhs.angular(), prec) && m_linear.isApprox(rhs.linear(), prec);
    }

private:
    /*! \brief Return derived class. */
    Derived& derived() noexcept { return static_cast<Derived&>(*this); }
    /*! \brief Return derived class. */
    const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

private:
    vec3_t m_angular; /*!< Angular part of the spatial vector */
    vec3_t m_linear; /*!< Linear part of the spatial vector */
};

/*! \brief Represent a CMTM tangent vector.
 *
 * A Tangent vector is the representation of the Motion/Force at given point.
 * Under the hood,it is composed of a size-constrained vector of MotionVector or ForceVector
 * and is fixed at compile-time.
 * \tparam Derived Child class (CRTP)
 */
template <typename Derived>
class SpatialVectorX {
    friend Derived;
    using traits = internal::traits<Derived>; /*!< Regroup all SpatialVectorX underlying types and constexpr values */

public:
    using Scalar = typename traits::Scalar; /*!< Underlying scalar type (float/double/etc...) */
    using storage_t = typename traits::storage_t; /*!< Underlying storage type of tangents */
    using spatial_vector_t = typename traits::spatial_vector_t; /*!< Underlying tangent type (MotionVector/ForceVector/etc...) */
    using vector_t = typename traits::vector_t; /*!< Representation of the full vector */

public:
    /*! \brief Default constructor. */
    SpatialVectorX() = default;

    /*! \brief Generic set SuperVector.
     *
     * There can be N parameters or 1 parameter.
     * If only 1 paramater is provided, it must be a underlying_t or assignable to an spatial_vector_t (for 0-n_vec vector).
     * If N parameters are provided, all must be assignable to an spatial_vector_t.
     * \tparam T1 First parameter type.
     * \tparam ...Args spatial_vector_t assignable types.
     * \param a1 First parameter.
     * \param ...args spatial_vector_t assignable arguments.
     */
    template <typename... Args>
    void set(Args&&... args)
    {
        COMA_STATIC_ASSERT(sizeof...(Args) > 0, "Number of parameters should be non-zero");
        if constexpr (sizeof...(Args) == 1) {
            setUnique(std::forward<Args>(args)...);
        } else {
            m_vector.set(std::forward<Args>(args)...);
        }
    }

    /*! \brief Return the number of Motion/Force vectors */
    Index nVec() const noexcept { return m_vector.size(); }
    /*! \brief Return the size of the vector */
    Index size() const noexcept { return 6 * m_vector.size(); }
    /*! \brief Return the i-th underlying sub-vector with bound checking */
    const spatial_vector_t& at(Index index) const { return m_vector.at(index); }
    /*! \brief Return the i-th underlying sub-vector with bound checking */
    spatial_vector_t& at(Index index) { return m_vector.at(index); }
    /*! \brief Return the i-th underlying sub-vector */
    const spatial_vector_t& operator[](Index index) const noexcept { return m_vector[index]; }
    /*! \brief Return the i-th underlying sub-vector */
    spatial_vector_t& operator[](Index index) noexcept { return m_vector[index]; }
    /*! \brief Return the full vector */
    vector_t vector() const noexcept
    {
        vector_t out(size());
        for (Index i = 0; i < nVec(); ++i)
            out.template segment<6>(6 * i) = m_vector[i].vector();

        return out;
    }
    /*! \brief Resize the vector.
     *
     * Only available for dynamic-n_vec class
     * \param nVec New vector sub-vector number
     */
    void resize(Index nVec)
    {
        COMA_ASSERT_IS_RESIZABLE_TO(Derived, nVec);
        COMA_ASSERT(nVec >= 0, "Number of tangent vector must be positive or 0");
        m_vector.resize(nVec);
    }
    /*! \brief Set the vector to zero. */
    Derived& setZero() noexcept
    {
        for (Index i = 0; i < nVec(); ++i)
            m_vector[i].setZero();

        return this->derived();
    }
    /*! \brief Resize and set the vector to zero.
     *
     * Only available for dynamic-n_vec class.
     * \param nVector New vector sub-vector number
     */
    Derived& setZero(Index nVector) noexcept
    {
        COMA_ASSERT_IS_RESIZABLE_TO(Derived, nVector);
        resize(nVector);
        for (Index i = 0; i < nVec(); ++i)
            m_vector[i].setZero();

        return this->derived();
    }
    /*! \brief Get a subvector for the spatial vector.
     *
     * \tparam Size Size of the new sub-vector.
     * \param fromPos Position at which the new vector start.
     * \return A SpatialVectorX extracted from position fromPos and of size Size
     */
    template <int Size>
    internal::redux_t<Derived, Size> subVector(Index fromPos)
    {
        return subVector<internal::redux_t<Derived, Size>>(fromPos, Size);
    }
    // Static/Dynamic implementation
    /*! \brief Get a subvector for the spatial vector.
     *
     * \tparam Return type. Default is Derived.
     * \param fromPos Position at which the new vector start.
     * \param size Size Size of the new sub-vector.
     * \return A SpatialVectorX extracted from position fromPos and of size size
     */
    template <typename T = Derived>
    T subVector(Index fromPos, Index size)
    {
        COMA_ASSERT(fromPos >= 0 && size >= 0 && fromPos + size <= nVec(), "Wrong size");

        T out(size);
        for (Index i = 0; i < size; ++i) {
            out[i] = m_vector[fromPos + i];
        }

        return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                 Static                                 //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Return a zero vector. 
     *
     * Only available for fixed-n_vec class.
     * \return A Zero-initialized SpatialVectorX
     */
    static Derived Zero()
    {
        COMA_STATIC_ASSERT_IS_FIXED(Derived);
        return Zero(traits::n_vec);
    }
    /*! \brief Return a zero vector. 
     *
     * Only available for dynamic-class of fixed-class of size nVec class.
     * \param nVec Number of sub-vectors
     * \return A Zero-initialized SpatialVectorX of nVec sub-vectors
     */
    static Derived Zero(Index nVec)
    {
        COMA_ASSERT_IS_RESIZABLE_TO(Derived, nVec);
        COMA_ASSERT(nVec >= 0, "Order of tangent vector must be positive or 0");
        Derived out;
        out.setZero(nVec);
        return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                Operator                                //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Return the value at index i. */
    Scalar operator()(Index i) noexcept
    {
        COMA_ASSERT(i >= 0 && i < 6 * nVec(), "Out-of-order index");
        return m_vector[i / 6](i % 6);
    }
    /*! \brief Unary minus operator.
     *
     * \param rhs A spatial vector \f$s\f$
     * return \f$-s\f$
     */
    friend Derived operator-(const Derived& rhs)
    {
        Derived out(rhs.nVec());
        for (Index i = 0; i < rhs.m_vector.size(); ++i)
            out.m_vector[i] = -rhs.m_vector[i];

        return out;
    }
    /*! \brief Operation between two SpatialVectorX.
     *
     * \param lhs A SpatialVectorX \f$s_1\f$
     * \param rhs A SpatialVectorX \f$s_2\f$
     * \return \f$s_1 + s_2\f$
     */
    friend Derived operator+(Derived lhs, const SpatialVectorX& rhs)
    {
        lhs += rhs;
        return lhs;
    }
    /*! \brief Operation between two SpatialVectorX.
     *
     * \param lhs A SpatialVectorX \f$s_1\f$
     * \param rhs A SpatialVectorX \f$s_2\f$
     * \return \f$s_1 - s_2\f$
     */
    friend Derived operator-(Derived lhs, const SpatialVectorX& rhs)
    {
        lhs -= rhs;
        return lhs;
    }
    /*! \brief Operation between a SpatialVectorX and a scalar.
     *
     * \param lhs A SpatialVectorX \f$s\f$
     * \param rhs A scalar \f$v\f$
     * \return \f$s \cdot v\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived operator*(Derived lhs, T rhs)
    {
        lhs *= rhs;
        return lhs;
    }
    /*! \brief Operation between a scalar and a SpatialVectorX.
     *
     * \param lhs A scalar \f$v\f$
     * \param rhs A SpatialVectorX \f$s\f$
     * \return \f$v \cdot s\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived operator*(T lhs, const Derived& rhs)
    {
        return rhs * lhs;
    }
    /*! \brief Operation between a SpatialVectorX and a scalar.
     *
     * \param lhs A SpatialVectorX \f$s\f$
     * \param rhs A scalar \f$v\f$
     * \return \f$s / v\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived operator/(Derived lhs, T rhs)
    {
        lhs /= rhs;
        return lhs;
    }
    /*! \brief Operation between two SpatialVectorX.
     *
     * \param rhs A SpatialVectorX \f$s_2\f$
     * \return \f$s_1\f$ after \f$s_1 += s_2\f$
     */
    Derived& operator+=(const SpatialVectorX& rhs)
    {
        COMA_ASSERT(nVec() == rhs.nVec(), "Order of tangent vectors mismatched");
        for (Index i = 0; i < nVec(); ++i)
            m_vector[i] += rhs[i];

        return this->derived();
    }
    /*! \brief Operation between two SpatialVectorX.
     *
     * \param rhs A SpatialVectorX \f$s_2\f$
     * \return \f$s_1\f$ after \f$s_1 -= s_2\f$
     */
    Derived& operator-=(const SpatialVectorX& rhs)
    {
        COMA_ASSERT(nVec() == rhs.nVec(), "Order of tangent vectors mismatched");
        for (Index i = 0; i < nVec(); ++i)
            m_vector[i] -= rhs[i];

        return this->derived();
    }
    /*! \brief Operation between a SpatialVectorX and a scalar.
     *
     * \param rhs A scalar \f$v\f$
     * \return \f$s\f$ after \f$s *= v\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    Derived& operator*=(T rhs)
    {
        for (Index i = 0; i < nVec(); ++i)
            m_vector[i] *= rhs;

        return this->derived();
    }
    /*! \brief Operation between a SpatialVectorX and a scalar.
     *
     * \param rhs A scalar \f$v\f$
     * \return \f$s\f$ after \f$s /= v\f$
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    Derived& operator/=(T rhs)
    {
        for (Index i = 0; i < nVec(); ++i)
            m_vector[i] /= rhs;

        return this->derived();
    }
    /*! \brief Check if two SpatialVectorX are equals.
     *
     * \warning This performs a strict comparison.
     * To compare two SpatialVectorX operators with a precision limit, use SpatialVectorX::isApprox.
     * \param lhs A SpatialVectorX \f$s_1\f$
     * \param rhs A SpatialVectorX \f$s_2\f$
     * \return true if equals.
     */
    friend bool operator==(const SpatialVectorX& lhs, const SpatialVectorX& rhs) noexcept
    {
        if constexpr (traits::n_vec == Dynamic) { // Always true for fixed-size
            if (lhs.nVec() != rhs.nVec()) return false;
        }

        bool isSame = true;
        for (Index i = 0; i < lhs.nVec(); ++i)
            isSame = isSame && (lhs[i] == rhs[i]); // Force short-circuit

        return isSame;
    }
    /*! \brief Check if two SpatialVectorX are different.
     *
     * \warning This performs a strict comparison.
     * To compare two SpatialVectorX operators with a precision limit, use SpatialVectorX::isApprox.
     * \param lhs A SpatialVectorX \f$s_1\f$
     * \param rhs A SpatialVectorX \f$s_2\f$
     * \return true if different.
     */
    friend bool operator!=(const SpatialVectorX& lhs, const SpatialVectorX& rhs) noexcept
    {
        return !(lhs == rhs);
    }
    /*! \brief Compare two SpatialVectorX with a precision limit.
     *
     * \param rhs A SpatialVectorX \f$s\f$
     * \param prec The precision limit. Default is base on Eigen limit
     * \return true if approximately equals.
     */
    bool isApprox(const SpatialVectorX& rhs, Scalar prec = dummy_precision<Scalar>()) const noexcept
    {
        if constexpr (traits::n_vec == Dynamic) { // Always true for fixed size
            if (nVec() != rhs.nVec()) return false;
        }

        bool isSame = true;
        for (Index i = 0; i < nVec(); ++i)
            isSame = isSame && m_vector[i].isApprox(rhs[i], prec); // Force short-circuit

        return isSame;
    }

private:
    void setUnique(typename storage_t::underlying_t&& s) { m_vector.values = std::move(s); }
    void setUnique(const typename storage_t::underlying_t& s) { m_vector.values = s; }
    void setUnique(storage_t&& s) { m_vector = std::move(s); }
    void setUnique(const storage_t& s) { m_vector = s; }
    void setUnique(spatial_vector_t&& s) { m_vector[0] = std::move(s); }
    void setUnique(const spatial_vector_t& s) { m_vector[0] = s; }
    void setUnique(SpatialVectorX&& s) { m_vector = std::move(s.m_vector); }
    void setUnique(const SpatialVectorX& s) { m_vector = s.m_vector; }
    void setUnique(const vector_t& v)
    {
        COMA_ASSERT(v.size() % 6 == 0, "the vector must be a multiple of 6");
        Index nv = v.size() / 6;
        resize(nv);
        for (Index i = 0; i < nv; ++i) {
            m_vector[i] = spatial_vector_t{ v.template segment<6>(6 * i) };
        }
    }

    /*! \brief Return derived class. */
    Derived& derived() noexcept { return static_cast<Derived&>(*this); }
    /*! \brief Return derived class. */
    const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

private:
    storage_t m_vector; /*!< Set of spatial sub-vectors */
};

/*! \brief Spatial Force .
 *
 * \tparam Scalar Underlying type (float/double/etc...)
 * \see SpatialVector
 */
template <typename Scalar>
class ForceVector : public SpatialVector<ForceVector<Scalar>>, public Formatter<ForceVector<Scalar>> {
    COMA_STATIC_ASSERT_IS_FP(Scalar);
    using traits = internal::traits<ForceVector>; /*!< Regroup all SpatialVector underlying types and constexpr values */

public:
    using Base = SpatialVector<ForceVector>; /*!< Base class type */
    using vec3_t = typename traits::vec3_t;
    using vec6_t = typename traits::vec6_t;

public:
    /*! \see SpatialVector<Scalar>::SpatialVector() */
    ForceVector() = default;
    /*! \see SpatialVector<Scalar>::SpatialVector(vec3_t, vec3_t) */
    ForceVector(const vec3_t& angular, const vec3_t& linear)
        : Base::SpatialVector(angular, linear)
    {
    }
    /*! \see SpatialVector<Scalar>::SpatialVector(vec6_t) */
    ForceVector(const vec6_t& v)
        : Base::SpatialVector(v)
    {
    }
};

/*! \brief Spatial motion.
 *
 * \tparam Scalar Underlying type (float/double/etc...)
 * \see SpatialVector
 */
template <typename Scalar>
class MotionVector : public SpatialVector<MotionVector<Scalar>>, public Formatter<MotionVector<Scalar>> {
    COMA_STATIC_ASSERT_IS_FP(Scalar);
    using traits = internal::traits<MotionVector>; /*!< Regroup all SpatialVector underlying types and constexpr values */

public:
    using Base = SpatialVector<MotionVector>; /*!< Base class type */
    using vec3_t = typename Base::vec3_t;
    using vec6_t = typename Base::vec6_t;

public:
    /*! \see SpatialVector<Scalar>::SpatialVector() */
    MotionVector() = default;
    /*! \see SpatialVector<Scalar>::SpatialVector(vec3_t, vec3_t) */
    MotionVector(const vec3_t& angular, const vec3_t& linear)
        : Base::SpatialVector(angular, linear)
    {
    }
    /*! \see SpatialVector<Scalar>::SpatialVector(vec6_t) */
    MotionVector(const vec6_t& v)
        : Base::SpatialVector(v)
    {
    }

    /*! \brief Cross operator betwen two spatial vector.
     *
     * \param rhs MotionVector \f$\nu_2\f$
     * \return \mathline{\op{\nu}\nu_2}
     */
    MotionVector cross(const MotionVector& rhs) const
    {
        return { this->m_angular.cross(rhs.angular()), this->m_angular.cross(rhs.linear()) + this->m_linear.cross(rhs.angular()) };
    }
    /*! \brief Cross dual operator betwen two spatial vector.
     *
     * \param rhs ForceVector \f$f\f$
     * \return \mathline{\opd{\nu}f}
     */
    ForceVector<Scalar> crossDual(const ForceVector<Scalar>& rhs) const
    {
        return { this->m_angular.cross(rhs.angular()) + this->m_linear.cross(rhs.linear()), this->m_angular.cross(rhs.linear()) };
    }
    /*! \brief Dot operator betwen two spatial vector.
     *
     * \param rhs ForceVector \f$f\f$
     * \return \f$\nu \cdot f\f$
     */
    Scalar dot(const ForceVector<Scalar>& rhs) const
    {
        return this->m_angular.dot(rhs.angular()) + this->m_linear.dot(rhs.linear());
    }
};

/*! \brief Concatenation of spatial force.
 *
 * \tparam Scalar Underlying type (float/double/etc...)
 * \tparam NVec Number of sub-vector
 * \see SpatialVectorX
 */
template <typename Scalar, int NVec>
class ForceVectorX : public SpatialVectorX<ForceVectorX<Scalar, NVec>>, public Formatter<ForceVectorX<Scalar, NVec>> {
    COMA_STATIC_ASSERT_IS_FP(Scalar);
    using traits = internal::traits<ForceVectorX>; /*!< Regroup all SpatialVectorX underlying types and constexpr values */

public:
    static constexpr int n_vec = traits::n_vec;
    using Base = SpatialVectorX<ForceVectorX>; /*!< Base class type */
    using spatial_vector_t = typename Base::spatial_vector_t;
    using storage_t = typename Base::storage_t;
    using vector_t = typename Base::vector_t;

public:
    ForceVectorX() = default;
    template <typename... Args, typename = std::enable_if_t<!(std::is_base_of_v<ForceVectorX, std::decay_t<Args>>, ...)>>
    ForceVectorX(Args&&... args)
    {
        Base::set(std::forward<Args>(args)...);
    }
};

/*! \brief Concatenation of spatial motion.
 *
 * \tparam Scalar Underlying type (float/double/etc...)
 * \tparam NVec Number of sub-vector
 * \see SpatialVectorX
 */
template <typename Scalar, int NVec>
class MotionVectorX : public SpatialVectorX<MotionVectorX<Scalar, NVec>>, public Formatter<MotionVectorX<Scalar, NVec>> {
    COMA_STATIC_ASSERT_IS_FP(Scalar);
    using traits = internal::traits<MotionVectorX>; /*!< Regroup all SpatialVectorX underlying types and constexpr values */

public:
    static constexpr int n_vec = traits::n_vec;
    using Base = SpatialVectorX<MotionVectorX>; /*!< Base class type */
    using spatial_vector_t = typename Base::spatial_vector_t;
    using storage_t = typename Base::storage_t;
    using vector_t = typename Base::vector_t;

public:
    MotionVectorX() = default;
    template <typename... Args, typename = std::enable_if_t<!(std::is_base_of_v<MotionVectorX, std::decay_t<Args>>, ...)>>
    MotionVectorX(Args&&... args)
    {
        Base::set(std::forward<Args>(args)...);
    }
};

} // namespace coma
