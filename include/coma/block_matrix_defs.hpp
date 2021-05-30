/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

namespace internal {

template <typename _Scalar>
struct traits<BiBlock33<_Scalar>> {
    using Scalar = _Scalar;
    using blockT1_t = Eigen::Matrix<Scalar, 3, 3>;
    using blockT2_t = Eigen::Matrix<Scalar, 3, 3>;
    using mat_t = Eigen::Matrix<Scalar, 6, 6>;
};

template <typename _Scalar>
struct traits<BiBlock31<_Scalar>> {
    using Scalar = _Scalar;
    using blockT1_t = Eigen::Matrix<Scalar, 3, 3>;
    using blockT2_t = Eigen::Matrix<Scalar, 3, 1>;
    using mat_t = Eigen::Matrix<Scalar, 4, 4>;
};

} // namespace internal

/*! \brief CRTP base class of compact representation matrix for block diagonal matrix
 *
 * This class is a N-block diagonal matrix where N is the number of block of a repeated sub-matrix.
 * The repeated block matrix can be of any type but must have a function matrix() that can convert it to an Eigen matrix.
 * If D is a matrix, then this class becomes
 * \f{
 *   \left[
 *   \begein{array}{cccc}
 *     D      & 0      & \dots  & 0 \\
 *     0      & \ddots & \ddots & \vdots \\
 *     \vdots & \ddots & \ddots & \vdots \\
 *     0      & \dots  & 0      & D
 *   \end{array}
 *   \right]
 * \f}
 * \tparam Derived CRTP Derived class
 */
template <typename Derived>
class DiBlockT : public Formatter<Derived> {
    friend Derived;
    using traits = internal::traits<Derived>;

public:
    using Scalar = typename traits::Scalar;
    using underlying_t = typename traits::underlying_t;
    using mat_t = Eigen::Matrix<Scalar, Dynamic, Dynamic>;

public:
    /*! \brief Default constructor. */
    DiBlockT() = default;
    /*! \brief Sub-matrix assignment at construction.
     * \tparam T Type that can assign/move to underlying_t
     */
    template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, DiBlockT>>>
    DiBlockT(T&& block)
        : m_block(std::forward<T>(block))
    {
        COMA_STATIC_ASSERT_ASSIGNABLE(T, underlying_t);
    }
    /*! \brief Provide the full matrix representation.
     * \param nMat Number of block matrix repetition (default to static value n_vec of Derived class)
     * \return Eigen dynamic matrix
     */
    mat_t matrix(Index nMat = traits::n_vec) const
    {
        COMA_ASSERT(nMat > 0, "nMat must be positive");
        mat_t block = m_block.matrix();
        Index rows = block.rows();
        Index cols = block.cols();
        mat_t out = mat_t::Zero(nMat * rows, nMat * cols);
        for (Index i = 0; i < nMat; ++i) {
            out.block(i * rows, i * cols, rows, cols) = block;
        }

        return out;
    }
    /*! \brief Return the transpose matrix. */
    Derived transpose() const noexcept { return { m_block.transpose().eval() }; }
    /*! \brief Get the block matrix. */
    const underlying_t& block() const noexcept { return m_block; }
    /*! \brief Get the block matrix. */
    underlying_t& block() noexcept { return m_block; }
    /*! \brief Set block to zero and return itself. */
    Derived& setZero() noexcept
    {
        COMA_STATIC_ASSERT((internal::has_setZero<underlying_t, underlying_t&()>::value), "Underlying type has no setZero() function");
        m_block.setZero();
        return this->derived();
    }
    /*! \brief Call block::setZero(rows, cols) and return itself. */
    Derived& setZero(Index rows, Index cols)
    {
        COMA_STATIC_ASSERT((internal::has_setZero<underlying_t, underlying_t&(Index, Index)>::value), "Underlying type has no setZero(Index) function");
        m_block.setZero(rows, cols);
        return this->derived();
    }
    /*! \brief Set block to identity and return itself. */
    Derived& setIdentity() noexcept
    {
        COMA_STATIC_ASSERT((internal::has_setIdentity<underlying_t, underlying_t&()>::value), "Underlying type has no setIdentity() function");
        m_block.setIdentity();
        return this->derived();
    }
    /*! \brief Call block::setIdentity(rows, cols) and return itself. */
    Derived& setIdentity(Index rows, Index cols)
    {
        COMA_STATIC_ASSERT((internal::has_setIdentity<underlying_t, underlying_t&(Index, Index)>::value), "Underlying type has no setIdentity(Index) function");
        m_block.setIdentity(rows, cols);
        return this->derived();
    }
    /*! \brief Return true if two block are equal. */
    friend bool operator==(const Derived& lhs, const Derived& rhs) noexcept { return lhs.m_block == rhs.m_block; }
    /*! \brief Return true if two block are not equal. */
    friend bool operator!=(const Derived& lhs, const Derived& rhs) noexcept { return !(lhs == rhs); }
    /*! \brief Return true if two block are almost equal. */
    bool isApprox(const Derived& rhs, Scalar prec = dummy_precision<Scalar>()) const noexcept { return m_block.isApprox(rhs.block(), prec); }

private:
    inline Derived& derived() noexcept { return static_cast<Derived&>(*this); }
    inline const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

private:
    underlying_t m_block; /*!< Underlying block that is repeated several time. */
};

/*! \brief CRTP base class of compact representation matrix for coma bi/tri block matrix.
 *
 * This class is a bi-block matrix.
 * The repeated block matrix can be of any type but must have a function matrix() that can convert it to an Eigen matrix.
 * This is class has 2 to 3 meaning depending of the space we work in.
 * If we work in space 4 and B1 and B2 are two matrices
 * \f{
 *   \left[
 *   \begein{array}{cc}
 *     B1 & B2  \\
 *     0  & 0/1 \\
 *   \end{array}
 *   \right]
 * \f}, it takes 0 for differentiator matrix and 1 for transformation matrix.
 * For 6-space, we have
 * \f{
 *   \left[
 *   \begein{array}{cc}
 *     B1 & 0  \\
 *     B2 & B1 \\
 *   \end{array}
 *   \right]
 * \f}.
 * The class does not define any of the representation, the very definition is delayed until an operation occurs.
 * \tparam Derived CRTP Derived class
 */
template <typename Derived>
class BiBlockTT : public Formatter<Derived> {
    friend Derived;
    using traits = typename internal::traits<Derived>;
    using Scalar = typename traits::Scalar;
    using blockT1_t = typename traits::blockT1_t;
    using blockT2_t = typename traits::blockT2_t;

public:
    /*! \brief Default constructor. */
    BiBlockTT() = default;
    /*! \brief Block-copy constructor. */
    BiBlockTT(const blockT1_t& b1, const blockT2_t& b2)
        : m_blockT1(b1)
        , m_blockT2(b2)
    {}
    /*! \brief Block-move constructor. */
    BiBlockTT(blockT1_t&& b1, blockT2_t&& b2)
        : m_blockT1(std::move(b1))
        , m_blockT2(std::move(b2))
    {}

    /*! \brief Get the first block matrix. */
    const blockT1_t& blockT1() const noexcept { return m_blockT1; }
    /*! \brief Get the first block matrix. */
    blockT1_t& blockT1() noexcept { return m_blockT1; }
    /*! \brief Get the second block matrix. */
    const blockT2_t& blockT2() const noexcept { return m_blockT2; }
    /*! \brief Get the second block matrix. */
    blockT2_t& blockT2() noexcept { return m_blockT2; }
    /*! \brief Get motion from matrix if exist.
     * \warning You cannot get a motion from these matrices whenever you want.
     * It has to have a mathematical meaning (it has to become a coma::Cross for example).
     */
    MotionVector<Scalar> toMotion() const noexcept { return derived().toMotion(); }
    /*! \brief Unary minus operator */
    friend Derived operator-(const Derived& rhs)
    {
        return { -rhs.m_blockT1, -rhs.m_blockT2 };
    }
    /*! \brief Operator += */
    Derived& operator+=(const Derived& rhs)
    {
        m_blockT1 += rhs.m_blockT1;
        m_blockT2 += rhs.m_blockT2;
        return this->derived();
    }
    /*! \brief Operator -= */
    Derived& operator-=(const Derived& rhs)
    {
        m_blockT1 -= rhs.m_blockT1;
        m_blockT2 -= rhs.m_blockT2;
        return this->derived();
    }
    /*! \brief Operator + */
    friend Derived operator+(const Derived& lhs, const Derived& rhs)
    {
        return { lhs.m_blockT1 + rhs.m_blockT1, lhs.m_blockT2 + rhs.m_blockT2 };
    }
    /*! \brief Operator - */
    friend Derived operator-(const Derived& lhs, const Derived& rhs)
    {
        return { lhs.m_blockT1 - rhs.m_blockT1, lhs.m_blockT2 - rhs.m_blockT2 };
    }
    /*! \brief Operator * */
    friend Derived operator*(Derived lhs, const Derived& rhs)
    {
        lhs *= rhs;
        return lhs;
    }
    /*! \brief Operator * */
    friend Derived operator*(Derived lhs, const Transform<Scalar>& rhs)
    {
        lhs *= rhs;
        return lhs;
    }
    /*! \brief Operator * */
    friend Derived operator*(Derived lhs, const Cross<Scalar>& rhs)
    {
        lhs *= rhs;
        return lhs;
    }
    /*! \brief Operator * */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived operator*(const Derived& lhs, T rhs)
    {
        return { lhs.m_blockT1 * rhs, lhs.m_blockT2 * rhs };
    }
    /*! \brief Operator * */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived operator*(T lhs, const Derived& rhs)
    {
        return rhs * lhs;
    }
    /*! \brief Operator / */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived operator/(const Derived& lhs, T rhs)
    {
        return { lhs.m_blockT1 / rhs, lhs.m_blockT2 / rhs };
    }
    /*! \brief Operator *= */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived& operator*=(Derived& lhs, T rhs)
    {
        lhs.m_blockT1 *= rhs;
        lhs.m_blockT2 *= rhs;
        return lhs.derived();
    }
    /*! \brief Operator /= */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    friend Derived& operator/=(Derived& lhs, T rhs)
    {
        lhs.m_blockT1 /= rhs;
        lhs.m_blockT2 /= rhs;
        return lhs.derived();
    }
    /*! \brief Operator == */
    friend bool operator==(const Derived& lhs, const Derived& rhs) noexcept
    {
        return lhs.m_blockT1 == rhs.m_blockT1 && lhs.m_blockT2 == rhs.m_blockT2;
    }
    /*! \brief Operator != */
    friend bool operator!=(const Derived& lhs, const Derived& rhs) noexcept
    {
        return !(lhs == rhs);
    }
    /*! \brief Operator isApprox */
    bool isApprox(const Derived& rhs, Scalar prec = dummy_precision<Scalar>()) const noexcept
    {
        return m_blockT1.isApprox(rhs.m_blockT1, prec) && m_blockT2.isApprox(rhs.m_blockT2, prec);
    }
    /*! \brief Set blocks to zero and return. */
    Derived& setZero() noexcept
    {
        COMA_STATIC_ASSERT((internal::has_setZero<blockT1_t, blockT1_t&()>::value), "blockT1_t type has no setZero() function");
        COMA_STATIC_ASSERT((internal::has_setZero<blockT2_t, blockT2_t&()>::value), "blockT2_t type has no setIdentity() function");
        m_blockT1.setZero();
        m_blockT2.setZero();
        return this->derived();
    }
    /*! \brief Set B1 to Identity and B2 to zero and return. */
    Derived& setIdentity() noexcept
    {
        COMA_STATIC_ASSERT((internal::has_setIdentity<blockT1_t, blockT1_t&()>::value), "blockT1_t type has no setIdentity() function");
        COMA_STATIC_ASSERT((internal::has_setZero<blockT2_t, blockT2_t&()>::value), "blockT2_t type has no setZero() function");
        m_blockT1.setIdentity();
        m_blockT2.setZero();
        return this->derived();
    }

private:
    inline Derived& derived() noexcept { return static_cast<Derived&>(*this); }
    inline const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

private:
    blockT1_t m_blockT1; /*!< Block B1 */
    blockT2_t m_blockT2; /*!< Block B2 */
};

/*! \brief Bi-block 3x3 matrix.
 * \see BiBlockTT for details.
 */
template <typename Scalar>
class BiBlock33 : public BiBlockTT<BiBlock33<Scalar>> {
    using blockT1_t = Eigen::Matrix<Scalar, 3, 3>;
    using blockT2_t = Eigen::Matrix<Scalar, 3, 3>;
    using mat_t = Eigen::Matrix<Scalar, 6, 6>;

public:
    using BiBlockTT<BiBlock33<Scalar>>::BiBlockTT;
    /*! \brief Get the full matrix form. */
    mat_t matrix() const
    {
        mat_t out;
        out << this->m_blockT1, blockT2_t::Zero(), this->m_blockT2, this->m_blockT1;
        return out;
    }
    /*! \brief Get the full matrix form with dual layout. */
    mat_t dualMatrix() const
    {
        mat_t out;
        out << this->m_blockT1, this->m_blockT2, blockT2_t::Zero(), this->m_blockT1;
        return out;
    }
    /*! \brief Operator * */
    MotionVector<Scalar> operator*(const MotionVector<Scalar>& rhs) const
    {
        return { this->m_blockT1 * rhs.angular(), this->m_blockT1 * rhs.linear() + this->m_blockT2 * rhs.angular() };
    }
    /*! \brief Operator dualMul */
    ForceVector<Scalar> dualMul(const ForceVector<Scalar>& rhs) const
    {
        return { this->m_blockT1 * rhs.angular() + this->m_blockT2 * rhs.linear(), this->m_blockT1 * rhs.linear() };
    }
    /*! \brief Operator *= */
    BiBlock33& operator*=(const BiBlock33& rhs)
    {
        this->m_blockT2 *= rhs.m_blockT1; // m_linear * rhs.m_angular + m_angular * rhs.m_linear
        this->m_blockT2 += this->m_blockT1 * rhs.m_blockT2;
        this->m_blockT1 *= rhs.m_blockT1; // m_angular * rhs.m_angular
        return *this;
    }
    /*! \brief Operator *= */
    BiBlock33& operator*=(const Cross<Scalar>& rhs)
    {
        this->m_blockT2 *= rhs.angularMat(); // m_linear * rhs.m_angular + m_angular * rhs.m_linear
        this->m_blockT2 += this->m_blockT1 * rhs.linearMat();
        this->m_blockT1 *= rhs.angularMat(); // m_angular * rhs.m_angular
        return *this;
    }
    /*! \brief Operator *= */
    BiBlock33& operator*=(const Transform<Scalar>& rhs)
    {
        this->m_blockT2 *= rhs.rotation(); // m_linear * rhs.m_rotation + m_angular *cross(rhs.m_translation) * rhs.m_rotation
        this->m_blockT2 += this->m_blockT1 * vector3ToCrossMatrix3(rhs.translation()) * rhs.rotation();
        this->m_blockT1 *= rhs.rotation(); // m_angular * rhs.m_angular
        return *this;
    }
    /*! \brief Get motion from matrix if exist.
     * \warning You cannot get a motion from these matrices whenever you want.
     * It has to have a mathematical meaning (it has to become a coma::Cross for example).
     */
    MotionVector<Scalar> toMotion() const { return { crossMatrix3ToVector3(this->m_blockT1), crossMatrix3ToVector3(this->m_blockT2) }; }
};

/*! \brief Bi-block 3x3 and 3x1 matrix.
 * \see BiBlockTT for details.
 */
template <typename Scalar>
class BiBlock31 : public BiBlockTT<BiBlock31<Scalar>> {
    using blockT1_t = Eigen::Matrix<Scalar, 3, 3>;
    using blockT2_t = Eigen::Matrix<Scalar, 3, 1>;
    using mat_t = Eigen::Matrix<Scalar, 4, 4>;

public:
    using BiBlockTT<BiBlock31<Scalar>>::BiBlockTT;
    /*! \brief Get the full matrix form. */
    mat_t matrix() const
    {
        mat_t out;
        out << this->m_blockT1, this->m_blockT2, Eigen::Matrix<Scalar, 1, 4>::Zero();
        return out;
    }
    /*! \brief Operator *= */
    BiBlock31& operator*=(const BiBlock31& rhs)
    {
        this->m_blockT2 = this->m_blockT1 * rhs.m_blockT2; // m_angular * rhs.m_linear
        this->m_blockT1 *= rhs.m_blockT1; // m_angular * rhs.m_angular
        return *this;
    }
    /*! \brief Operator *= */
    BiBlock31& operator*=(const Cross<Scalar>& rhs)
    {
        this->m_blockT2 = this->m_blockT1 * rhs.linear(); // m_angular * rhs.m_linear
        this->m_blockT1 *= rhs.angularMat(); // m_angular * rhs.m_angular
        return *this;
    }
    /*! \brief Operator *= */
    BiBlock31& operator*=(const Transform<Scalar>& rhs)
    {
        this->m_blockT2 += this->m_blockT1 * rhs.translation(); // m_angular * rhs.m_translation + m_linear
        this->m_blockT1 *= rhs.rotation(); // m_angular * rhs.m_rotation
        return *this;
    }
    /*! \brief Get motion from matrix if exist.
     * \warning You cannot get a motion from these matrices whenever you want.
     * It has to have a mathematical meaning (it has to become a coma::Cross for example).
     */
    MotionVector<Scalar> toMotion() const { return { crossMatrix3ToVector3(this->m_blockT1), this->m_blockT2 }; }
};

} // namespace coma
