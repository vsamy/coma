/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

namespace internal {

template <typename _Scalar, int _Space, int _Order>
struct traits<CMTM<_Scalar, _Space, _Order>> {
    static inline constexpr int order = _Order;
    static inline constexpr int space = _Space;
    static inline constexpr int n_vec = (order == Dynamic ? Dynamic : order + 1);
    using Scalar = _Scalar;
    using motion_vector_X_t = MotionVectorX<Scalar, order>;
    using transform_t = Transform<Scalar>;
    using block_mat_t = std::conditional_t<space == 6, BiBlock33<Scalar>, BiBlock31<Scalar>>;
    using sub_matrix_t = Storage<block_mat_t, order>;
};

} // namespace internal

/*! \brief N-order Comprehensive Motion Transformation Matrix (CMTM).
 *
 * This class represents an N-order CMTM of space 4 or 6.
 * \tparam Scalar Underlying type (float/double/etc...)
 * \tparam Space The CMTM space (either 4 or 6)
 * \tparam Order N-order of the CMTM
 */
template <typename Scalar, int Space, int Order>
class CMTM : public Formatter<CMTM<Scalar, Space, Order>> {
    COMA_STATIC_ASSERT(Space == 4 || Space == 6, "This class can only be of space 4 or 6");
    using traits = internal::traits<CMTM>; /*!< Regroup all CMTM underlying types and constexpr values */
    using matX_t = Eigen::Matrix<Scalar, -1, -1>;
    static inline const auto& p_factors = pascal_factors<Scalar, traits::n_vec>;
    static inline const auto& f_factors = factorial_factors<Scalar, traits::n_vec>;

public:
    using mvx_t = typename traits::motion_vector_X_t; /*!< CMTM underlying tangent motion type */
    using transform_t = typename traits::transform_t; /*!< CMTM underlying Transformation type */
    using sub_matrix_t = typename traits::sub_matrix_t; /*!< CMTM underlying vector of sub-matrix type */
    using fvx_t = ForceVectorX<Scalar, traits::order>; /*!< CMTM underlying force type */
    using block_mat_t = typename traits::block_mat_t;
    using cross_t = Cross<Scalar>; /*!< CMTM underlying operator \mathline{\op{\cdot}} type */

public:
    /*! \brief Default constructor */
    CMTM()
        : m_needDeconstruction(false)
    {
    }
    /*! \brief Constructor.
     *
     * Only available for dynamic-class or fixed-class of same order.
     * \param order New order of the vector
     */
    CMTM(Index order)
        : m_needDeconstruction(false)
    {
        resize(order);
    }
    /*! \brief Generic constructor.
     *
     * If N-Order paramater are given, they must be MotionVector constructible.
     * \tparam TF Transform assignable type
     * \tparam ...Args MotionVector constructible type.
     * \param tf Transform assignable param
     * \param ...args N-Order MotionVector or 1 MotionVectorX assignable parameter
     * \see Transform, MotionVectorX
     */
    template <typename TF, typename... Args, typename = std::enable_if_t<!std::is_arithmetic_v<std::decay_t<TF>> && !std::is_base_of_v<CMTM, std::decay_t<TF>>>>
    CMTM(TF&& tf, Args&&... args) // TODO: Change to initializer_list<T>!!
    : m_needDeconstruction(false)
    {
        set(std::forward<TF>(tf), std::forward<Args>(args)...);
    }
    /*! \brief Generic set (copy or move).
     *
     * If N-Order paramater are given, they must be MotionVector constructible.
     * \tparam TF Transform assignable type
     * \tparam ...Args MotionVector constructible type.
     * \param tf Transform assignable param
     * \param ...args N-Order MotionVector or 1 MotionVectorX assignable parameter
     * \see Transform, MotionVectorX
     */
    template <typename TF, typename... Args>
    void set(TF&& tf, Args&&... args)
    {
        setNoConstruct(std::forward<TF>(tf), std::forward<Args>(args)...);
        construct();
    }
    /*! \brief Same as set but do not call the construct method. */
    template <typename TF, typename... Args>
    void setNoConstruct(TF&& tf, Args&&... args)
    {
        COMA_STATIC_ASSERT_ASSIGNABLE(TF, transform_t);
        m_tf = std::forward<TF>(tf);
        setMotion(std::forward<Args>(args)...);
    }

    /*! \brief Return the order of the CMTM. */
    Index order() const noexcept { return m_motion.nVec(); }
    /*! \brief Return the size of underlying vector of CMTM matrix. */
    Index nMat() const noexcept { return order() + 1; } // order() + transform
    /*! \brief Return the number of rows of the CMTM representation matrix. */
    Index rows() const noexcept { return traits::space * nMat(); }
    /*! \brief Return the number of cols of the CMTM representation matrix. */
    Index cols() const noexcept { return traits::space * nMat(); }
    /*! \brief Return the size of the CMTM representation matrix. */
    Index size() const noexcept { return rows() * cols(); }
    /*! \brief Return the CMTM transformation \f$C_{\{0\}}\f$. */
    const transform_t& transform() const noexcept { return m_tf; }
    /*! \brief Return the CMTM transformation \f$C_{\{0\}}\f$. */
    transform_t& transform() noexcept { return m_tf; }
    /*! \brief Return CMTM underlying motions.
     *
     * This function will compute automatically the underlying motion vector if the CMTM is a result of a multiplication.
     * It will return directly the user-provided motion vector otherwise (no computation are made).
     */
    const mvx_t& motion() const
    {
        if (m_needDeconstruction) const_cast<CMTM*>(this)->deconstruct();
        return m_motion;
    }
    /*! \brief Return CMTM underlying motions.
     *
     * This function will compute automatically the motion vector if the CMTM is a result of a multiplication.
     * It will return directly the user-provided motion otherwise (no computation are made).
     */
    mvx_t& motion()
    {
        if (m_needDeconstruction) deconstruct();
        return m_motion;
    }
    /*! \brief Return the CMTM sub-matrices without \f$C_{\{0\}}\f$. */
    const sub_matrix_t& subMatrices() const noexcept { return m_subMatrices; }
    /*! \brief Return the (i+1)-th CMTM sub-matrix. */
    const block_mat_t& operator[](Index i) const noexcept { return m_subMatrices[i]; }
    /*! \brief Return the (i+1)-th CMTM sub-matrix with bound checking. */
    const block_mat_t& at(Index i) const { return m_subMatrices.at(i); }
    /*! \brief Return inverse of CMTM while preserving the state of the CMTM. */
    CMTM inverse() const
    {
        if constexpr (traits::space == 6) {
            return m_needDeconstruction ? invC() : invMotion();
        } else {
            return invC(); // invMotion can not be performed for space 4
        }
    }
    /*! \brief Return the CMTM representation matrix of order NewNMat-1 for Fixed CMTM.
     * \warning Don't forget that this matrix includes factorial p_factors.
     */
    template <int NewNMat>
    matX_t matrix() const
    {
        COMA_STATIC_ASSERT_IS_FIXED(CMTM);
        COMA_STATIC_ASSERT(NewNMat >= 0 && NewNMat <= traits::n_vec, "Wrong template parameter");
        return generateMatrix(NewNMat);
    }
    /*! \brief Return the CMTM representation matrix of order newNMat-1 for Dynamic CMTM. */
    matX_t matrix(Index newNMat = traits::n_vec) const
    {
        COMA_ASSERT(newNMat >= Dynamic && newNMat <= nMat(), "Wrong matrix size");
        if (newNMat == Dynamic) newNMat = nMat();
        return generateMatrix(newNMat);
    }
    /*! \brief Return the CMTM dual representation matrix of order NewNMat-1 for Fixed CMTM. */
    template <int NewNMat>
    matX_t dualMatrix() const
    {
        COMA_STATIC_ASSERT_IS_FIXED(CMTM);
        COMA_STATIC_ASSERT(NewNMat >= 0 && NewNMat <= traits::n_vec, "Wrong template parameter");
        return generateDualMatrix(NewNMat);
    }
    /*! \brief Return the CMTM dual representation matrix (space 6 only) of order newNMat-1. */
    matX_t dualMatrix(Index newNMat = traits::n_vec) const
    {
        COMA_ASSERT(newNMat >= Dynamic && newNMat <= nMat(), "Wrong matrix size");
        if (newNMat == Dynamic) newNMat = nMat();
        return generateDualMatrix(newNMat);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                 Static                                 //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Return an identity CMTM. */
    static CMTM Identity() noexcept
    {
        COMA_STATIC_ASSERT_IS_FIXED(CMTM);
        return CMTM{ transform_t::Identity(), mvx_t::Zero() };
    }
    /*! \brief Return a CMTM as Identity matrix.
     *
     * Only available for dynamic-class or fixed-class of same order.
     */
    static CMTM Identity(Index order)
    {
        COMA_ASSERT_IS_RESIZABLE_TO(CMTM, order + 1);
        COMA_ASSERT(order >= 0, "Order must be a positive number or 0");
        return CMTM{ transform_t::Identity(), mvx_t::Zero(order) };
    }

    ////////////////////////////////////////////////////////////////////////////
    //                                Operator                                //
    ////////////////////////////////////////////////////////////////////////////

    /*! \brief Operation between two CMTM.
     *
     * \param rhs A CMTM \f$C_1\f$
     * \param rhs A CMTM \f$C_2\f$
     * \return \f$C_1 \cdot C_2\f$
     */
    friend CMTM operator*(const CMTM& lhs, const CMTM& rhs)
    {
        COMA_ASSERT(lhs.order() == rhs.order(), "Orders of CMTM mismatched");
        CMTM out(lhs.order());

        out.m_tf = lhs.m_tf * rhs.m_tf; // * 1
        size_t pos = 0;
        for (Index i = 0; i < lhs.order(); ++i) {
            pos += 2;
            out.m_subMatrices[i] = lhs.m_tf * rhs.m_subMatrices[i] + lhs.m_subMatrices[i] * rhs.m_tf;
            for (Index j = 0; j < i; ++j)
                out.m_subMatrices[i] += p_factors[pos++] * lhs.m_subMatrices[j] * rhs.m_subMatrices[i - 1 - j];
        }

        out.m_needDeconstruction = true;
        return out;
    }
    /*! \brief Operation between a CMTM and a spatial motion vector.
     *
     * The spatial motion vector order must be inferior or equal to nVec().
     * Thus, this function computes \mathline{C \cdot \xi}
     * \param rhs A MotionVectorX \f$\xi\f$
     * \return \mathline{C \cdot \xi}
     * \see MotionVectorX
     */
    template <int MotionVectorXSize>
    MotionVectorX<Scalar, std::min(MotionVectorXSize, traits::n_vec)> operator*(const MotionVectorX<Scalar, MotionVectorXSize>& rhs) const
    {
        COMA_STATIC_ASSERT_IS_SPACE_6(CMTM);
        constexpr int new_n_vec = std::min(MotionVectorXSize, traits::n_vec);
        if constexpr (new_n_vec != Dynamic) { // Catch error at compile-time
            COMA_STATIC_ASSERT(MotionVectorXSize <= traits::n_vec, "Wrong motion vector X size");
        }

        COMA_ASSERT(rhs.nVec() <= nMat(), "The motion vector size must lesser or equal to the CMTM size");
        MotionVectorX<Scalar, new_n_vec> out(rhs.nVec());

        size_t pos = 0;
        for (Index i = 0; i < rhs.nVec(); ++i) {
            out[i] = p_factors[pos++] * (m_tf * rhs[i]);
            for (Index j = 0; j < i; ++j)
                out[i] += p_factors[pos++] * (m_subMatrices[j] * rhs[i - 1 - j]);
        }

        return out;
    }
    /*! \brief Dual operation between a CMTM and a spatial force vector.
     *
     * The spatial force vector order must be inferior or equal to nMat().
     * Thus, this function computes \mathline{\bar{C} \cdot f}
     * \param rhs A ForceVectorX \f$f\f$
     * \return \mathline{\bar{C} \cdot f}
     * \see ForceVectorX
     */
    template <int ForceVectorXSize>
    ForceVectorX<Scalar, std::min(ForceVectorXSize, traits::n_vec)> dualMul(const ForceVectorX<Scalar, ForceVectorXSize>& rhs) const
    {
        COMA_STATIC_ASSERT_IS_SPACE_6(CMTM);
        constexpr int new_n_vec = std::min(ForceVectorXSize, traits::n_vec);
        if constexpr (new_n_vec != Dynamic) { // Catch error at compile-time
            COMA_STATIC_ASSERT(ForceVectorXSize <= traits::n_vec, "Size mismatched");
        }

        COMA_ASSERT(rhs.nVec() <= nMat(), "The tangent vector size must lesser or equal to the CMTM size");
        ForceVectorX<Scalar, new_n_vec> out(rhs.nVec());
        size_t pos = 0;
        for (Index i = 0; i < rhs.nVec(); ++i) {
            out[i] = p_factors[pos++] * m_tf.dualMul(rhs[i]);
            for (Index j = 0; j < i; ++j)
                out[i] += p_factors[pos++] * m_subMatrices[j].dualMul(rhs[i - 1 - j]);
        }

        return out;
    }
    /*! \brief Check if two CMTM are equals.
     *
     * Depending of the state of the CMTM, the comparison is on the CMTM sub-matrices or MotionVectorX.
     * \warning This performs a strict comparison.
     * To compare two CMTM with a precision limit, use CMTM::isApprox.
     * 
     * \param lhs A CMTM \f$C_1\f$
     * \param rhs A CMTM \f$C_2\f$
     * \return true if equals.
     */
    friend bool operator==(const CMTM& lhs, const CMTM& rhs) noexcept
    {
        if constexpr (traits::order == Dynamic) { // Always true for fixed size
            if (lhs.order() != rhs.order()) return false;
        }

        return (lhs.m_needDeconstruction || rhs.m_needDeconstruction) ? lhs.m_tf == rhs.m_tf && lhs.m_subMatrices == rhs.m_subMatrices
                                                                      : lhs.m_tf == rhs.m_tf && lhs.m_motion == rhs.m_motion;
    }
    /*! \brief Check if two CMTM are different.
     * 
     * Depending of the state of the CMTM, the comparison is on the CMTM sub-matrices or MotionVectorX.
     * \warning This performs a strict comparison.
     * To compare two CMTM with a precision limit, use CMTM::isApprox.
     *
     * \param lhs A CMTM \f$C_1\f$
     * \param rhs A CMTM \f$C_2\f$
     * \return true if different.
     */
    friend bool operator!=(const CMTM& lhs, const CMTM& rhs) noexcept
    {
        return !(lhs == rhs);
    }
    /*! \brief Compare two CMTM with a precision limit.
     * 
     * Depending of the state of the CMTM, the comparison is on the CMTM sub-matrices or MotionVectorX.
     *
     * \param rhs A CMTM \f$C_2\f$
     * \param prec The precision limit. Default is base on Eigen limit
     * \return true if approximately equals.
     */
    bool isApprox(const CMTM& rhs, Scalar prec = dummy_precision<Scalar>()) const noexcept
    {
        if constexpr (traits::order == Dynamic) { // Always true for fixed size
            if (order() != rhs.order()) return false;
        }

        if (m_needDeconstruction || rhs.m_needDeconstruction) {
            bool isSame = true;
            for (Index i = 0; i < m_subMatrices.size(); ++i)
                isSame = isSame && m_subMatrices[i].isApprox(rhs.m_subMatrices[i], prec);

            return isSame && m_tf.isApprox(rhs.m_tf, prec);
        } else {
            return m_tf.isApprox(rhs.m_tf, prec) && m_motion.isApprox(rhs.m_motion, prec);
        }
    }
    /*! \brief Resize the CMTM.
     *
     * Only available for dynamic-class or fixed-class of same order.
     * \param order New CMTM order
     */
    void resize(Index order)
    {
        COMA_ASSERT_IS_RESIZABLE_TO(CMTM, order + 1);
        COMA_ASSERT(order >= 0, "Order of tangent vector must be positive or 0");
        m_motion.resize(order);
        m_subMatrices.resize(order);
    }
    /*! \brief Set the CMTM to identity. */
    CMTM& setIdentity() noexcept
    {
        m_needDeconstruction = false;
        m_tf.setIdentity();
        m_motion.setZero();
        for (Index i = 0; i < m_subMatrices.size(); ++i)
            m_subMatrices[i].setZero();

        return *this;
    }
    /*! \brief Resize and set the CMTM to identity.
     *
     * Only available for dynamic-class or fixed-class of same order.
     * \param order New CMTM order
     */
    CMTM& setIdentity(Index order) noexcept
    {
        COMA_ASSERT_IS_RESIZABLE_TO(CMTM, order + 1);
        resize(order);
        setIdentity();
        return *this;
    }
    /*! \brief Construct the CMTM underlying sub-matrices.
     * 
     * This computes \mathline{\forall p\in\\{0..N-1\\},\ C_N=\sum\limits_{k=0}^{p}\binom{p}{k}C_{p-k}\op{m^{(k)}}}.
     */
    void construct()
    {
        Storage<cross_t, traits::order> vx(m_motion.nVec());
        size_t pos = 0;
        for (Index i = 0; i < m_motion.nVec(); ++i) {
            vx[i] = cross_t{ m_motion[i] };
            if constexpr (traits::space == 4) {
                m_subMatrices[i] = p_factors[pos++] * mul4(m_tf, vx[i]); // we need the call of the special operator
            } else {
                m_subMatrices[i] = p_factors[pos++] * (m_tf * vx[i]);
            }

            for (Index j = 0; j < i; ++j)
                m_subMatrices[i] += p_factors[pos++] * (m_subMatrices[j] * vx[i - j - 1]);
        }
    }
    /*! \brief Deconstruct the CMTM sub-matrices.
     *
     * This computes \mathline{\forall p\in\\{0..N-1\\},\ \op{m^{(p)}} = C_0\inv\left(C_{p+1} - \sum\limits_{k=0}^{p-1}\binom{p}{k}C_{p-k}\op{m^{(k)}}\right)}.
     */
    void deconstruct()
    {
        sub_matrix_t vx(order());
        transform_t AInv = m_tf.inverse();
        size_t pos = 0;
        for (Index i = 0; i < order(); ++i) {
            auto Ci = m_subMatrices[i];
            for (Index k = 0; k < i; ++k)
                Ci -= p_factors[pos++] * (m_subMatrices[i - 1 - k] * vx[k]);

            vx[i] = AInv * Ci; // p_factor is always 1 here
            m_motion[i] = vx[i].toMotion();
            pos++;
        }

        m_needDeconstruction = false;
    }

private:
    template <typename A1, typename... Args>
    void setMotion(A1&& a1, Args&&... args)
    {
        if constexpr (sizeof...(Args) == 0 && std::is_same_v<mvx_t, std::decay_t<A1>>) { // Same type and not dynamic
            setFromMotionVectorX(std::forward<A1>(a1));
        } else { // Any other cases
            setFromMultipleArgs(std::forward<A1>(a1), std::forward<Args>(args)...);
        }
    }
    void setMotion() const noexcept { return; }
    template <typename T>
    void setFromMotionVectorX(T&& mvx)
    {
        constexpr int n_vec = internal::traits<std::decay_t<T>>::n_vec;
        resize(mvx.nVec());
        if constexpr (traits::order != Dynamic) {
            COMA_STATIC_ASSERT(n_vec >= traits::order, "The motion vector must be at least of size N");
        }

        if constexpr (n_vec == traits::order) {
            m_motion = std::forward<T>(mvx);
        } else {
            for (Index i = 0; i < order(); ++i)
                m_motion[i] = mvx[i];
        }
    }
    template <typename... Args>
    void setFromMultipleArgs(Args&&... args)
    {
        m_motion.set(std::forward<Args>(args)...);
        if constexpr (traits::order == Dynamic) {
            m_subMatrices.resize(sizeof...(Args));
        }
    }
    CMTM invC() const
    {
        CMTM out(m_motion.nVec());
        out.m_tf = m_tf.inverse();
        size_t pos = 0;
        for (Index i = 0; i < m_subMatrices.size(); ++i) {
            pos += 2;
            out.m_subMatrices[i] = m_subMatrices[i] * out.m_tf; // * 1
            for (Index k = 0; k < i; ++k)
                out.m_subMatrices[i] += p_factors[pos++] * m_subMatrices[k] * out.m_subMatrices[i - 1 - k];

            out.m_subMatrices[i] = -(out.m_tf * out.m_subMatrices[i]);
        }

        if (m_needDeconstruction) {
            out.m_needDeconstruction = true;
        } else {
            out.deconstruct();
        }

        return out;
    }
    CMTM invMotion() const
    {
        COMA_STATIC_ASSERT_IS_SPACE_6(CMTM);
        CMTM out(m_motion.nVec());
        out.m_needDeconstruction = false;
        out.m_tf = m_tf.inverse();
        if (m_motion.nVec() == 0) {
            return out;
        }

        out.m_motion = -((*this) * m_motion); // m^j_i = -^iC_j m^i_j
        out.construct();
        return out;
    }
    matX_t generateMatrix(Index newNMat) const
    {
        constexpr int space = traits::space;
        matX_t out = matX_t::Zero(space * newNMat, space * newNMat);
        matX_t mat;
        if constexpr (space == 6) {
            mat = m_tf.matrix();
        } else {
            mat = m_tf.homogeneousMatrix();
        }
        for (Index i = 0; i < newNMat; ++i)
            out.template block<space, space>(i * space, i * space) = mat;
        for (Index i = 1; i < newNMat; ++i) {
            mat = m_subMatrices[i - 1].matrix() / f_factors[static_cast<size_t>(i)];
            for (Index j = 0; j < newNMat - i; ++j)
                out.template block<space, space>((i + j) * space, j * space) = mat;
        }

        return out;
    }
    /*! \brief Return the CMTM dual representation matrix \f$\bar{C}\f$ (space 6 only). */
    matX_t generateDualMatrix(Index newNMat) const
    {
        COMA_STATIC_ASSERT_IS_SPACE_6(CMTM);
        constexpr int space = traits::space;
        matX_t out = matX_t::Zero(space * newNMat, space * newNMat);

        matX_t mat = m_tf.dualMatrix();
        for (Index i = 0; i < newNMat; ++i)
            out.template block<space, space>(i * space, i * space) = mat;
        for (Index i = 1; i < newNMat; ++i) {
            mat = m_subMatrices[i - 1].dualMatrix() / f_factors[static_cast<size_t>(i)];
            for (Index j = 0; j < newNMat - i; ++j)
                out.template block<space, space>((i + j) * space, j * space) = mat;
        }

        return out;
    }

private:
    bool m_needDeconstruction; /*!< Flag that describes the state of the CMTM */
    mvx_t m_motion; /*!< tangent motion of the CMTM */
    transform_t m_tf; /*!< Transformation */
    sub_matrix_t m_subMatrices; /*!< Sub-matrices */
};

} // namespace coma
