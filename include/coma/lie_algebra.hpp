/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

namespace internal {

template <typename _Scalar>
struct traits<Transform<_Scalar>> {
    using Scalar = _Scalar;
    using rotation_t = Eigen::Matrix<Scalar, 3, 3>;
    using translation_t = Eigen::Matrix<Scalar, 3, 1>;
};

template <typename _Scalar>
struct traits<Cross<_Scalar>> {
    using Scalar = _Scalar;
    using mv_t = MotionVector<Scalar>;
};

template <typename _Scalar, int _NVec>
struct traits<CrossN<_Scalar, _NVec>> {
    static constexpr int n_vec = _NVec;
    using Scalar = _Scalar;
    using mvx_t = MotionVectorX<Scalar, n_vec>;
};

} // namespace internal

/*! \brief Basic Transformation matrix.
 *
 * This class represents the Lie group SE(3) which provides in the end
 * either an Homogeneous matrix (4D) or a spatial transformation (6D) and its dual form.
 * In the documentation, the transformation is represented by the letter \f$T\f$.
 * \tparam Scalar Underlying type (float/double/etc...)
 * \note The transformation matrix uses trigonomic orientation (!= Featherstones who use anti-trigonomic rotation).
 */
template <typename Scalar>
class Transform : public Formatter<Transform<Scalar>> {
    using traits = internal::traits<Transform>; /*!< Regroup all Transform underlying types and constexpr values */

public:
    using rotation_t = typename traits::rotation_t;
    using translation_t = typename traits::translation_t;
    using quat_t = Eigen::Quaternion<Scalar>;
    using vec3_t = Eigen::Matrix<Scalar, 3, 1>;
    using mat3_t = Eigen::Matrix<Scalar, 3, 3>;
    using mat4_t = Eigen::Matrix<Scalar, 4, 4>;
    using mat6_t = Eigen::Matrix<Scalar, 6, 6>;
    using mv_t = MotionVector<Scalar>;
    using fv_t = ForceVector<Scalar>;

public:
    /*! \brief Default constructor */
    Transform()
        : Transform(mat3_t::Identity(), vec3_t::Zero())
    {}
    /*! \brief Constructor from rotation matrix and translation.
     *
     * \tparam Derived1 A rotation-like matrix
     * \tparam Derived2 A translation-like vector
     * \param R A 3D rotation matrix
     * \param p A 3D translation vector
     */
    template <typename Derived1, typename Derived2>
    Transform(const Eigen::MatrixBase<Derived1>& R, const Eigen::MatrixBase<Derived2>& p)
    {
        set(R, p);
    }
    /*! \brief Constructor from quaternion and translation.
     *
     * \param q A normalized quaternion
     * \param p A 3D translation vector
     */
    Transform(const quat_t& q, const vec3_t& p)
        : Transform(q.toRotationMatrix(), p)
    {}
    /*! \brief Constructor from a representation matrix.
     *
     * The function extracts the rotation and translation part of the matrix.
     * \param H An homogeneous transformation matrix
     */
    Transform(const mat4_t& H)
    {
        m_R = H.template topLeftCorner<3, 3>();
        m_p = H.template topRightCorner<3, 1>();
    }
    /*! \brief Constructor from a representation matrix.
     *
     * The function extracts the rotation and translation part of the matrix.
     * \param X A spatial transformation matrix
     */
    Transform(const mat6_t& X)
    {
        m_R = X.template topLeftCorner<3, 3>();
        m_p = crossMatrix3ToVector3(X.template bottomLeftCorner<3, 3>() * m_R.transpose());
    }

    /*! \brief Set copy the Transformation from rotation and translation
     *
     * \tparam TR A rotation-like matrix
     * \tparam TP A translation-like vector
     * \param R A rotation matrix
     * \param p A translation vector
     */
    template <typename Derived1, typename Derived2>
    void set(const Eigen::MatrixBase<Derived1>& R, const Eigen::MatrixBase<Derived2>& p)
    {
        COMA_STATIC_ASSERT_ASSIGNABLE(Eigen::MatrixBase<Derived1>, rotation_t);
        COMA_STATIC_ASSERT_ASSIGNABLE(Eigen::MatrixBase<Derived2>, translation_t);
        m_R = R;
        m_p = p;
    }
    /*! \brief Return the homogeneous representation matrix of the transformation. */
    mat4_t homogeneousMatrix() const noexcept
    {
        return (mat4_t() << m_R, m_p, Eigen::Matrix<Scalar, 1, 3>::Zero(), Scalar(1)).finished();
    }
    /*! \brief Return the spatial representation matrix of the transformation. */
    mat6_t matrix() const noexcept
    {
        return (mat6_t() << m_R, mat3_t::Zero(), vector3ToCrossMatrix3(m_p) * m_R, m_R).finished();
    }
    /*! \brief Return the representation matrix of the dual transformation. */
    mat6_t dualMatrix() const noexcept
    {
        return (mat6_t() << m_R, vector3ToCrossMatrix3(m_p) * m_R, mat3_t::Zero(), m_R).finished();
    }
    /*! \brief Return the rotation matrix. */
    const rotation_t& rotation() const noexcept { return m_R; }
    /*! \brief Return the rotation matrix. */
    rotation_t& rotation() noexcept { return m_R; }
    /*! \brief Return the translation vector. */
    const translation_t& translation() const noexcept { return m_p; }
    /*! \brief Return the translation vector. */
    translation_t& translation() noexcept { return m_p; }
    /*! \brief Return the rotation as a quaternion. */
    quat_t rotationAsQuat() const noexcept { return quat_t{ m_R }; }
    /*! \brief Return the transformation inverse.
     *
     * This does not perform a matrix inversion but exploits the transformation definition instead.
     * It makes this function very stable and fast.
     */
    Transform inverse() const { return { m_R.transpose(), -m_R.transpose() * m_p }; }
    /*! \brief Set the transformation to identity. */
    Transform& setIdentity() noexcept
    {
        m_R.setIdentity();
        m_p.setZero();
        return *this;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                 Static                                 //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Return an identity transformation. */
    static Transform Identity() noexcept
    {
        return Transform(mat3_t::Identity(), vec3_t::Zero());
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                Operator                                //
    ////////////////////////////////////////////////////////////////////////////

    // /*! \brief Operation between two transformation.
    //  *
    //  * \param rhs A transformation matrix \f$T\f$
    //  * \return *this
    //  */
    Transform& operator*=(const Transform& rhs)
    {
        m_p += m_R * rhs.m_p; // m_p + m_R * rhs.m_p
        m_R *= rhs.m_R; // m_R * rhs.m_R
        return *this;
    }
    /*! \brief Operation between two transformation.
     *
     * \param lhs A transformation matrix \f$T_1\f$
     * \param rhs A transformation matrix \f$T_2\f$
     * \return \f$T_1 \cdot T_2\f$
     */
    friend Transform operator*(const Transform& lhs, const Transform& rhs)
    {
        return { lhs.m_R * rhs.m_R, lhs.m_R * rhs.m_p + lhs.m_p };
    }
    /*! \brief Operation between a 6D transformation and a spatial motion vector.
     *
     * \param lhs A transformation matrix \f$T\f$
     * \param rhs A motion vector \f$\nu\f$
     * \return \f$T \cdot \nu\f$
     */
    friend mv_t operator*(const Transform& lhs, const mv_t& rhs)
    {
        vec3_t wOut = lhs.m_R * rhs.angular();
        return { wOut, lhs.m_p.cross(wOut) + lhs.m_R * rhs.linear() };
    }
    /*! \brief Operation between a 6D transformation and a spatial force vector.
     *
     * \param rhs A force vector \f$f\f$
     * \return \f$\bar{T} \cdot f\f$
     */
    fv_t dualMul(const fv_t& rhs) const
    {
        vec3_t Rf = m_R * rhs.linear();
        return { m_R * rhs.angular() + m_p.cross(Rf), Rf };
    }
    /*! \brief Operation between a transformation inverse and a transformation.
     *
     * It is the same as writing \code{.cpp} this->inverse() * rhs \endcode but faster.
     * \param rhs A transformation matrix \f$T_2\f$
     * \return \mathline{T\inv \cdot T_2}
     */
    Transform invMul(const Transform& rhs) const
    {
        return { m_R.transpose() * rhs.m_R, m_R.transpose() * (rhs.m_p - m_p) };
    }
    /*! \brief Operation between a 6D transformation inverse and a motion vector.
     *
     * It is the same as writing X.inverse() * rhs but faster.
     * \param rhs A spatial motion vector \f$\nu\f$
     * \return \mathline{T\inv \cdot \nu}
     */
    mv_t invMul(const mv_t& rhs) const
    {
        const vec3_t& a = rhs.angular();
        const vec3_t& l = rhs.linear();
        return { m_R.transpose() * a, m_R.transpose() * (l - m_p.cross(a)) };
    }
    /*! \brief Check if two transforms are equals.
     *
     * \warning This performs a strict comparison.
     * To compare two transformations with a precision limit, use Transform::isApprox.
     * \param lhs A transformation matrix \f$T_1\f$
     * \param rhs A transformation matrix \f$T_2\f$
     * \return true if equals.
     */
    friend bool operator==(const Transform& lhs, const Transform& rhs) noexcept
    {
        return lhs.m_R == rhs.m_R && lhs.m_p == rhs.m_p;
    }
    /*! \brief Check if two transforms are different.
     *
     * \warning This performs a strict comparison.
     * To compare two transformations with a precision limit, use Transform::isApprox.
     * \param lhs A transformation matrix \f$T_1\f$
     * \param rhs A transformation matrix \f$T_2\f$
     * \return true if different.
     */
    friend bool operator!=(const Transform& lhs, const Transform& rhs) noexcept
    {
        return !(lhs == rhs);
    }
    /*! \brief Compare two transformations with a precision limit.
     *
     * \param rhs A transformation matrix \f$T\f$
     * \param prec The precision limit. Default is base on Eigen limit
     * \return true if approximately equals.
     */
    bool isApprox(const Transform& rhs, Scalar prec = dummy_precision<Scalar>()) const noexcept
    {
        return m_R.isApprox(rhs.m_R, prec) && m_p.isApprox(rhs.m_p, prec);
    }

private:
    mat3_t m_R; /*!< Rotation matrix */
    vec3_t m_p; /*!< Translation vector */
};

/*! \brief Cross operator.
 *
 * This class represents the cross operator of a spatial motion vector \f$\nu\f$ such that it is \mathline{\op{\nu}}
 * where \f$\times\f$ is either for space 4 or 6.
 * \tparam Scalar Underlying type (float/double/etc...)
 * \note The cross operator uses Plucker rotation, so 6D motion vector should be of the form \f$[\omega^T\ v^T]^T\f$.
 */
template <typename Scalar>
class Cross : public Formatter<Cross<Scalar>> {
    using traits = internal::traits<Cross>; /*!< Regroup all Cross underlying types and constexpr values */

public:
    using mv_t = typename traits::mv_t;
    using mat3_t = Eigen::Matrix<Scalar, 3, 3>;
    using vec3_t = Eigen::Matrix<Scalar, 3, 1>;
    using mat4_t = Eigen::Matrix<Scalar, 4, 4>;
    using mat6_t = Eigen::Matrix<Scalar, 6, 6>;
    using fv_t = ForceVector<Scalar>;
    using transform_t = Transform<Scalar>; /*!< Transformation type */

public:
    /*! \brief Default constructor. */
    Cross() = default;
    /*! \brief Constructor from a motion vector. */
    Cross(const mv_t& m)
        : m_motion(m)
    {}
    /*! \brief Set cross operator from a spatial motion vector. */
    void set(const mv_t& m)
    {
        m_motion = m;
    }
    /*! \brief Constructor from a moved spatial motion vector. */
    Cross(mv_t&& m)
        : m_motion(std::move(m))
    {}
    /*! \brief Set cross operator from a motion vector. */
    void set(mv_t&& m)
    {
        m_motion = std::move(m);
    }

    /*! \brief Return angular part \f$\omega\f$. */
    const vec3_t& angular() const noexcept { return m_motion.angular(); }
    /*! \brief Return angular part \f$\omega\f$. */
    vec3_t& angular() noexcept { return m_motion.angular(); }
    /*! \brief Return angular part \mathline{\op{\omega}}. */
    mat3_t angularMat() const noexcept { return vector3ToCrossMatrix3(m_motion.angular()); }
    /*! \brief Return linear part \f$v\f$. * */
    const vec3_t& linear() const noexcept { return m_motion.linear(); }
    /*! \brief Return linear part \f$v\f$. * */
    vec3_t& linear() noexcept { return m_motion.linear(); }
    /*! \brief Return linear part \mathline{\op{\nu}}. */
    mat3_t linearMat() const noexcept { return vector3ToCrossMatrix3(m_motion.linear()); }
    /*! \return Motion vector. */
    const mv_t& motion() const noexcept { return m_motion; }
    /*! \return Motion vector. */
    mv_t& motion() noexcept { return m_motion; }
    /*! \brief Return the differential operator matrix representation of the homogeneous matrix. */
    mat4_t homogeneousDifferentiator() const noexcept { return vector6ToCrossMatrix4(m_motion); }
    /*! \brief Return the differential operator matrix representation of the spatial matrix. */
    mat6_t matrix() const noexcept { return vector6ToCrossMatrix6(m_motion); }
    /*! \brief Return the differential operator matrix representation of the dual of the spatial matrix */
    mat6_t dualMatrix() const noexcept { return vector6ToCrossDualMatrix6(m_motion); }
    /*! \brief Set the cross operator to zero. */
    Cross& setZero() noexcept
    {
        m_motion.setZero();
        return *this;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                 Static                                 //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Return a zero cross operator. */
    static Cross Zero() noexcept { return Cross{ mv_t::Zero() }; }

    ////////////////////////////////////////////////////////////////////////////
    //                                Operator                                //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Operation between two cross operators.
     *
     * \param rhs A cross \mathline{\op{\nu_2}}
     * \return \mathline{\op{\nu}} after \mathline{\op{\nu} += \op{\nu_2}}
     */
    Cross& operator+=(const Cross& rhs)
    {
        m_motion.angular() += rhs.m_motion.angular();
        m_motion.linear() += rhs.m_motion.linear();
        return *this;
    }
    /*! \brief Operation between two cross operators.
     *
     * \param rhs A cross \mathline{\op{\nu_2}}
     * \return \mathline{\op{\nu}} after  \mathline{\op{\nu} -= \op{\nu_2}}
     */
    Cross& operator-=(const Cross& rhs)
    {
        m_motion.angular() -= rhs.m_motion.angular();
        m_motion.linear() -= rhs.m_motion.linear();
        return *this;
    }
    /*! \brief Operation between a cross operator and a scalar.
     *
     * \param rhs A scalar \f$v\f$
     * \return \mathline{\op{\nu}} after \mathline{\op{\nu} \cdot = v}
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    Cross& operator*=(T rhs)
    {
        m_motion *= rhs;
        return *this;
    }
    /*! \brief Operation between a cross operator and a scalar.
     *
     * \param rhs A scalar \f$v\f$
     * \return \mathline{\op{\nu}} after \mathline{\op{\nu} /= v}
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    Cross& operator/=(T rhs)
    {
        m_motion /= rhs;
        return *this;
    }
    /*! \brief Operation between two cross operator.
     *
     * \param lhs A cross operator \mathline{\op{\nu_1}}
     * \param rhs A cross operator \mathline{\op{\nu_2}}
     * \return \mathline{\op{\nu_1} + \op{\nu_2}}
     */
    friend Cross operator+(const Cross& lhs, const Cross& rhs)
    {
        return { lhs.m_motion + rhs.m_motion };
    }
    /*! \brief Operation between two cross operator.
     *
     * \param lhs A cross operator \mathline{\op{\nu_1}}
     * \param rhs A cross operator \mathline{\op{\nu_2}}
     * \return \mathline{\op{\nu_1} - \op{\nu_2}}
     */
    friend Cross operator-(const Cross& lhs, const Cross& rhs)
    {
        return { lhs.m_motion - rhs.m_motion };
    }
    /*! \brief Unary minus operation.
     *
     * \param rhs A cross operator \mathline{\op{\nu}}
     * \return \mathline{-\op{\nu}}
     */
    friend Cross operator-(const Cross& rhs)
    {
        return { -rhs.m_motion };
    }
    /*! \brief Operation between a cross operator and a scalar.
     *
     * \param lhs A cross operator \mathline{\op{\nu}}
     * \param rhs A scalar \f$v\f$
     * \return \mathline{\op{\nu} \cdot v}
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Cross operator*(const Cross& lhs, T rhs)
    {
        return { lhs.m_motion * rhs };
    }
    /*! \brief Operation between a scalar and a cross operator.
     *
     * \param lhs A scalar \f$v\f$
     * \param rhs A cross operator \mathline{\op{\nu}}
     * \return \mathline{v \cdot \op{\nu}}
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Cross operator*(T lhs, const Cross& rhs)
    {
        return rhs * lhs;
    }
    /*! \brief Operation between a cross operator and a scalar.
     *
     * \param lhs A cross operator \mathline{\op{\nu}}
     * \param rhs A scalar \f$v\f$
     * \return \mathline{\op{\nu} / v}
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Cross operator/(const Cross& lhs, T rhs)
    {
        return { lhs.m_motion / rhs };
    }
    /*! \brief Operation between a cross operator and a motion vector (space 6 only).
     *
     * \param lhs A spatial motion vector \mathline{\op{\nu_1}}
     * \param rhs A spatial motion vector \f$\nu_2\f$
     * \return \mathline{\op{\nu_1} \nu_2}
     */
    friend mv_t operator*(const Cross& lhs, const mv_t& rhs)
    {
        return lhs.m_motion.cross(rhs);
    }
    /*! \brief Dual operation between a cross operator and a force vector (space 6 only).
     *
     * \param rhs A force vector \f$f\f$
     * \return \mathline{\op{\nu} f}
     */
    fv_t dualMul(const fv_t& rhs) const
    {
        return m_motion.crossDual(rhs);
    }
    /*! \brief Check if two cross operators are equals.
     *
     * \warning This performs a strict comparison.
     * To compare two cross operators with a precision limit, use Cross::isApprox.
     * \param lhs A cross operator \mathline{\op{\nu_1}}
     * \param rhs A cross operator \mathline{\op{\nu_2}}
     * \return true if equals.
     */
    friend bool operator==(const Cross& lhs, const Cross& rhs) noexcept
    {
        return lhs.m_motion == rhs.m_motion;
    }
    /*! \brief Check if two cross operator are different.
     *
     * \warning This performs a strict comparison.
     * To compare two cross operators with a precision limit, use Cross::isApprox.
     * \param lhs A cross operator \mathline{\op{\nu_1}}
     * \param rhs A cross operator \mathline{\op{\nu_2}}
     * \return true if different.
     */
    friend bool operator!=(const Cross& lhs, const Cross& rhs) noexcept
    {
        return !(lhs == rhs);
    }
    /*! \brief Compare two cross operators with a precision limit.
     *
     * \param rhs A cross operator matrix \mathline{\op{\nu}}
     * \param prec The precision limit. Default is base on Eigen limit
     * \return true if approximately equals.
     */
    bool isApprox(const Cross& rhs, Scalar prec = dummy_precision<Scalar>()) const noexcept
    {
        return m_motion.isApprox(rhs.m_motion, prec);
    }

private:
    mv_t m_motion;
};

/*! \brief N-order cross operator vector.
 *
 * This class represents the N-order cross operator of space 4 or 6.
 * It is the derivative operator \mathline{\op{\xi}}
 * The class embedded the tangent vector, it applies the factorial p_factors but these are not visible outside of the class.
 * \tparam Scalar Underlying type (float/double/etc...)
 * \tparam Space The crossN operator space (either 4 or 6)
 * \tparam NVec Number of sub-matrix cross
 */
template <typename Scalar, int NVec>
class CrossN : public Formatter<CrossN<Scalar, NVec>> {
    using traits = internal::traits<CrossN>; /*!< Regroup all CrossN underlying types and constexpr values */
    static inline const auto& p_factors = pascal_factors<Scalar, NVec>;
    static inline const auto& f_factors = factorial_factors<Scalar, NVec>;

public:
    static inline constexpr int n_vec = NVec;
    using mvx_t = typename traits::mvx_t;
    using cross_t = Cross<Scalar>;
    using mv_t = MotionVector<Scalar>;
    using fv_t = ForceVector<Scalar>;
    using fvx_t = ForceVectorX<Scalar, n_vec>;
    using mat4_t = Eigen::Matrix<Scalar, (n_vec == Dynamic ? Dynamic : n_vec * 4), (n_vec == Dynamic ? Dynamic : n_vec * 4)>;
    using mat6_t = Eigen::Matrix<Scalar, (n_vec == Dynamic ? Dynamic : n_vec * 6), (n_vec == Dynamic ? Dynamic : n_vec * 6)>;
    using matX_t = Eigen::Matrix<Scalar, Dynamic, Dynamic>;

public:
    /*! \brief Default constructor. */
    CrossN() = default;
    /*! \brief Constructor.
     *
     * Only available for dynamic-order class.
     * \param nVec New size of the sub-matrix vector
     */
    CrossN(Index nVec)
    {
        resize(nVec);
    }
    /*! \brief Generic constructor.
     *
     * There can be N+1 parameters or 1 parameter.
     * If only 1 paramater is provided, it must be a MotionVectorX or a std::array<Cross>.
     * If N parameters are provided, all must be assignable to a Cross.
     * \tparam ...Args MotionVectorX or Cross assignable types of size N-Order.
     * \param ...args MotionVectorX or Cross assignable arguments of size N-Order.
     * \see Cross, MotionVectorX
     */
    template <typename... Args, typename = std::enable_if_t<!(std::is_base_of_v<CrossN, std::decay_t<Args>>, ...)>>
    CrossN(Args... args)
    {
        set(std::forward<Args>(args)...);
    }

    /*! \brief Generic set crossN.
     *
     * There can be N parameters or 1 parameter.
     * If only 1 paramater is provided, it must be a MotionVectorX or an underlying_t.
     * If N parameters are provided, all must be assignable to a Cross.
     * \tparam ...Args MotionVectorX or Cross assignable types.
     * \param ...args MotionVectorX or Cross assignable arguments.
     * \see Cross, MotionVectorX
     */
    template <typename... Args>
    void set(Args... args)
    {
        m_motion.set(std::forward<Args>(args)...);
    }

    /*! \brief Return the underlying motion. */
    const mvx_t& motion() const noexcept { return m_motion; }
    /*! \brief Return the underlying motion. */
    mvx_t& motion() noexcept { return m_motion; }
    /*! \brief Return the i-th cross operator. */
    const mv_t& motion(Index i) const noexcept { return m_motion[i]; }
    /*! \brief Return the i-th cross operator. */
    mv_t& motion(Index i) noexcept { return m_motion[i]; }
    /*! \brief Return the i-th cross operator with bounds checking. */
    const mv_t& motionAt(Index i) const noexcept { return m_motion.at(i); }
    /*! \brief Return the i-th cross operator with bounds checking. */
    mv_t& motionAt(Index i) noexcept { return m_motion.at(i); }
    /*! \brief Return the i-th cross operator. */
    cross_t operator[](Index i) const noexcept { return cross_t{ m_motion[i] }; }
    /*! \brief Return the i-th cross operator with bounds checking. */
    cross_t at(Index i) const { return cross_t{ m_motion.at(i) }; }
    /*! \brief Return the number of underlying vector of cross matrix. */
    Index nVec() const noexcept { return m_motion.nVec(); }
    /*! \brief Resize .the CrossN.
     *
     * Only available for dynamic-order class.
     * \param nVec New CrossN sub-matrix vector
     */
    void resize(Index nVec)
    {
        COMA_ASSERT_IS_RESIZABLE_TO(CrossN, nVec);
        m_motion.resize(nVec);
    }
    /*! \brief Set all cross operator to zero. */
    CrossN& setZero() noexcept
    {
        m_motion.setZero();
        return *this;
    }
    /*! \brief Resize and set the cross N operator to 0.
     *
     * Only available for dynamic-order class.
     * \param nVec New CrossN sub-matrix vector
     */
    CrossN& setZero(Index nVec) noexcept
    {
        COMA_ASSERT_IS_RESIZABLE_TO(CrossN, nVec);
        m_motion.setZero(nVec);
        return *this;
    }
    /*! \brief Return the CrossN representation matrix until NewNVec for CrossN. */
    template <int NewNVec>
    matX_t matrix() const noexcept
    {
        COMA_STATIC_ASSERT_IS_FIXED(CrossN);
        COMA_STATIC_ASSERT(NewNVec >= 0 && NewNVec <= traits::n_vec, "Wrong template parameter");
        return generateMatrix(NewNVec);
    }
    /*! \brief Return the CrossN representation matrix until NewNVec for Dynamic CrossN. */
    matX_t matrix(Index newNVec = traits::n_vec) const noexcept
    {
        COMA_ASSERT(newNVec >= Dynamic && newNVec <= nVec(), "Wrong matrix size");
        if (newNVec == Dynamic) newNVec = nVec();
        return generateMatrix(newNVec);
    }
    /*! \brief Return the CrossN dual representation matrix until NewNVec for Fixed CrossN. */
    template <int NewNVec>
    matX_t dualMatrix() const noexcept
    {
        COMA_STATIC_ASSERT_IS_FIXED(CrossN);
        COMA_STATIC_ASSERT(NewNVec >= 0 && NewNVec <= traits::n_vec, "Wrong template parameter");
        return generateDualMatrix(NewNVec);
    }
    /*! \brief Return the CrossN dual representation matrix (space 6 only). */
    matX_t dualMatrix(Index newNVec = traits::n_vec) const noexcept
    {
        COMA_ASSERT(newNVec >= Dynamic && newNVec <= nVec(), "Wrong matrix size");
        if (newNVec == Dynamic) newNVec = nVec();
        return generateDualMatrix(newNVec);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                 Static                                 //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Return a zero crossN operator.
     *
     * Only available for fixed-order class.
     */
    static CrossN Zero() noexcept
    {
        COMA_STATIC_ASSERT_IS_FIXED(CrossN);
        return { mvx_t::Zero() };
    }
    /*! \brief Return a zero crossN operator.
     *
     * Only available for dynamic-order class.
     */
    static CrossN Zero(Index nVec) noexcept
    {
        COMA_ASSERT_IS_RESIZABLE_TO(CrossN, nVec);
        return { mvx_t::Zero(nVec) };
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                Operator                                //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Operation between two crossN operators.
     *
     * \param rhs A crossN operator \mathline{\op{\xi_2}}
     * \return \mathline{\op{\xi}} after \mathline{\op{\xi} += \op{\xi_2}}
     */
    CrossN& operator+=(const CrossN& rhs)
    {
        COMA_ASSERT(nVec() == rhs.nVec(), "Order of CrossN mismatched");
        m_motion += rhs.motion();
        return *this;
    }
    /*! \brief Operation between two crossN operators.
     *
     * \param rhs A crossN operator \mathline{\op{\xi_2}}
     * \return \mathline{\op{\xi}} after \mathline{\op{\xi} -= \op{\xi_2}}
     */
    CrossN& operator-=(const CrossN& rhs)
    {
        COMA_ASSERT(nVec() == rhs.nVec(), "Order of CrossN mismatched");
        m_motion -= rhs.motion();
        return *this;
    }
    /*! \brief Operation between a crossN operator and a scalar.
     *
     * \param rhs A scalar \f$v\f$
     * \return \mathline{\op{\xi}} after \mathline{\op{\xi} \cdot = v}
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    CrossN& operator*=(T rhs)
    {
        m_motion *= rhs;
        return *this;
    }
    /*! \brief Operation between a crossN operator and a scalar.
     *
     * \param rhs A scalar \f$v\f$
     * \return \mathline{\op{\xi}} after \mathline{\op{\xi} /= v}
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    CrossN& operator/=(T rhs)
    {
        m_motion /= rhs;
        return *this;
    }
    /*! \brief Check if two crossN operators are equals.
     *
     * \warning This performs a strict comparison.
     * To compare two crossN operators with a precision limit, use CrossN::isApprox.
     * \param lhs A crossN operator \mathline{\op{\xi_1}}
     * \param rhs A crossN operator \mathline{\op{\xi_2}}
     * \return true if equals.
     */
    friend bool operator==(const CrossN& lhs, const CrossN& rhs) noexcept
    {
        if constexpr (traits::n_vec == Dynamic) { // Always true for fixed size
            if (lhs.nVec() != rhs.nVec()) return false;
        }

        return lhs.motion() == rhs.motion();
    }
    /*! \brief Check if two crossN operators are different.
         *
         * \warning This performs a strict comparison.
         * To compare two crossN operators with a precision limit, use CrossN::isApprox.
         * \param lhs A crossN operator \mathline{\op{\xi_1}}
         * \param rhs A crossN operator \mathline{\op{\xi_2}}
         * \return true if different.
         */
    friend bool operator!=(const CrossN& lhs, const CrossN& rhs) noexcept
    {
        return !(lhs == rhs);
    }
    /*! \brief Operation between a crossN operator and MotionVectorX.
     *
     * \param rhs A CrossN operator \mathline{\op{\xi}}
     * \param rhs A MotionVectorX \f$\xi\f$
     * \return \mathline{\op{\xi} \xi}
     * \see MotionVectorX
     */
    friend mvx_t operator*(const CrossN& lhs, const mvx_t& rhs)
    {
        COMA_ASSERT(lhs.nVec() == rhs.nVec(), "Order of CrossN and tangent vector mismatched");
        mvx_t out = mvx_t::Zero(lhs.nVec());
        size_t pos = 0;
        for (Index i = 0; i < lhs.nVec(); ++i)
            for (Index j = 0; j <= i; ++j)
                out[i] += p_factors[pos++] * lhs.m_motion[i - j].cross(rhs[j]);

        return out;
    }
    /*! \brief Operation between a crossN operator and ForceVectorX.
     *
     * \param rhs A scalar \f$v\f$
     * \return \mathline{\opd{\xi} \cdot v}
     * \see ForceVectorX
     */
    fvx_t dualMul(const fvx_t& rhs) const
    {
        COMA_ASSERT(nVec() == rhs.nVec(), "Order of CrossN and tangent vector mismatched");
        fvx_t out = fvx_t::Zero(nVec());
        size_t pos = 0;
        for (Index i = 0; i < nVec(); ++i) {
            for (Index j = 0; j <= i; ++j)
                out[i] += p_factors[pos++] * m_motion[i - j].crossDual(rhs[j]);
        }

        return out;
    }

    /*! \brief Operation between a crossN operator and a scalar.
     *
     * \param lhs A crossN operator \mathline{\op{\xi}}
     * \param rhs A scalar \f$v\f$
     * \return \mathline{\op{\xi} \cdot v}
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend CrossN operator*(const CrossN& lhs, T rhs)
    {
        return { lhs.m_motion * rhs };
    }
    /*! \brief Operation between a crossN operator and a scalar.
     *
     * \param lhs A scalar \f$v\f$
     * \param rhs A crossN operator \mathline{\op{\xi}}
     * \return \mathline{v \cdot \op{\xi}}
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend CrossN operator*(T lhs, const CrossN& rhs)
    {
        return rhs * lhs;
    }
    /*! \brief Operation between a crossN operator and a scalar.
     *
     * \param lhs A crossN operator \mathline{\op{\xi}}
     * \param rhs A scalar \f$v\f$
     * \return \mathline{\op{\xi} / v}
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend CrossN operator/(const CrossN& lhs, T rhs)
    {
        return { lhs.m_motion / rhs };
    }
    /*! \brief Operation between two crossN.
     *
     * \param lhs A crossN operator \mathline{\op{\xi_1}}
     * \param rhs A crossN operator \mathline{\op{\xi_2}}
     * \return \mathline{\op{\xi_1} + \op{\xi_2}}
     */
    friend CrossN operator+(const CrossN& lhs, const CrossN& rhs)
    {
        return { lhs.m_motion + rhs.m_motion };
    }
    /*! \brief Operation between a crossN operator and a scalar.
     *
     * \param lhs A crossN operator \mathline{\op{\xi_1}}
     * \param rhs A crossN operator \mathline{\op{\xi_2}}
     * \return \mathline{\op{\xi_1} - \op{\xi_2}}
     */
    friend CrossN operator-(const CrossN& lhs, const CrossN& rhs)
    {
        return { lhs.m_motion - rhs.m_motion };
    }
    /*! \brief Unary minus operation.
     *
     * \param rhs A cross operator \mathline{\op{\xi}}
     * \return \mathline{-\op{\xi}}
     */
    friend CrossN operator-(const CrossN& rhs)
    {
        return { -rhs.motion() };
    }

    /*! \brief Compare two crossN operators with a precision limit.
         *
         * \param rhs A crossN operator matrix \mathline{\op{\xi_2}}
         * \param prec The precision limit. Default is base on Eigen limit
         * \return true if approximately equals.
         */
    bool isApprox(const CrossN& rhs, Scalar prec = dummy_precision<Scalar>()) const noexcept
    {
        if constexpr (traits::n_vec == Dynamic) { // Always true for fixed size
            if (nVec() != rhs.nVec()) return false;
        }

        return m_motion.isApprox(rhs.motion(), prec);
    }

private:
    matX_t generateMatrix(Index newNVec) const noexcept
    {
        matX_t out = matX_t::Zero(6 * newNVec, 6 * newNVec);
        for (Index i = 0; i < newNVec; ++i) {
            matX_t mat = cross_t{ m_motion[i] }.matrix() / f_factors[static_cast<size_t>(i)];
            for (Index j = 0; j < newNVec - i; ++j) {
                out.template block<6, 6>(6 * (i + j), 6 * j) = mat;
            }
        }

        return out;
    }
    matX_t generateDualMatrix(Index newNVec) const noexcept
    {
        matX_t out = matX_t::Zero(6 * newNVec, 6 * newNVec);
        for (Index i = 0; i < newNVec; ++i) {
            matX_t mat = cross_t{ m_motion[i] }.dualMatrix() / f_factors[static_cast<size_t>(i)];
            for (Index j = 0; j < newNVec - i; ++j) {
                out.template block<6, 6>(6 * (i + j), 6 * j) = mat;
            }
        }

        return out;
    }

private:
    mvx_t m_motion; /*!< Vector of motions */
};

} // namespace coma