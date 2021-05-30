/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "coma/Core"
#include "macros.hpp"
#include "doctest/doctest.h"
#include <tuple>

struct OrderF {
    static constexpr int n_vec = 7;
};

struct OrderD {
    static constexpr int n_vec = coma::Dynamic;
};

template <typename T1, typename T2>
struct TypePair
{
    using first_type = T1;
    using second_type = T2;
};

#define test_pairs \
    TypePair<float, OrderF>, \
    TypePair<double, OrderF>, \
    TypePair<float, OrderD>, \
    TypePair<double, OrderD>

TEST_CASE_TEMPLATE("Transform", Scalar, float, double)
{
    using namespace coma;
    using qt_t = Eigen::Quaternion<Scalar>;
    using m3_t = Eigen::Matrix<Scalar, 3, 3>;
    using v3_t = Eigen::Matrix<Scalar, 3, 1>;
    using tf_t = Transform<Scalar>;

    DISABLE_CONVERSION_WARNING_BEGIN
    qt_t q = qt_t::UnitRandom();
    DISABLE_CONVERSION_WARNING_END

    m3_t R = q.toRotationMatrix();
    v3_t p = v3_t::Random();
    Eigen::Matrix<Scalar, 4, 4> H;
    Eigen::Matrix<Scalar, 6, 6> A;
    H << R, p, Eigen::Matrix<Scalar, 1, 3>::Zero(), Scalar(1);
    A << R, m3_t::Zero(), vector3ToCrossMatrix3(p) * R, R;

    auto requireRpEq = [R, p](const tf_t& transf) {
        REQUIRE(transf.rotation() == R);
        REQUIRE(transf.translation() == p);
    };

    {
        // Identity
        tf_t tf = tf_t::Identity();
        REQUIRE(tf.rotation().isApprox(m3_t::Identity()));
        REQUIRE(tf.translation().isApproxToConstant(0));
    }
    {
        // {R, p} ctor
        tf_t tf{ R, p };
        requireRpEq(tf);
    }
    {
        // {q, p} ctor
        tf_t tf{ q, p };
        requireRpEq(tf);
    }
    {
        // {H} ctor
        tf_t tf{ H };
        REQUIRE(tf.rotation() == R);
        REQUIRE(tf.translation().isApprox(p));

        tf_t tf2{ A };
        REQUIRE(tf2.rotation() == R);
        REQUIRE(tf2.translation().isApprox(p));
    }
    {
        // Copy-ctor
        tf_t tf1{ R, p };
        tf_t tf2{ tf1 };
        requireRpEq(tf2);
    }
    {
        // move-ctor
        tf_t tf1{ R, p };
        tf_t tf2{ std::move(tf1) };
        requireRpEq(tf2);
    }
    {
        // assign-op
        tf_t tf1{ R, p };
        tf_t tf2;
        tf2 = tf1;
        requireRpEq(tf2);
    }
    {
        // move-assign-op
        tf_t tf1{ R, p };
        tf_t tf2;
        tf2 = std::move(tf1);
        requireRpEq(tf2);
    }
    {
        // set
        tf_t tf;
        tf.set(R, p);
        requireRpEq(tf);
    }
    {
        // set R, p solo
        tf_t tf;
        tf.rotation() = R;
        tf.translation() = p;
        requireRpEq(tf);
        qt_t res = tf.rotationAsQuat();
        REQUIRE((res.isApprox(q) || res.isApprox(qt_t(-q.w(), -q.x(), -q.y(), -q.z()))));
    }
    {
        // setIdentity
        tf_t tf{ R, p };
        tf.setIdentity();
        REQUIRE(tf == tf_t::Identity());
    }
    {
        // homogeneousMatrix()/matrix()
        tf_t tf{ R, p };
        REQUIRE(H.isApprox(tf.homogeneousMatrix()));
        REQUIRE(A.isApprox(tf.matrix()));
    }
    {
        // dual matrix()
        tf_t tf{ R, p };
        Eigen::Matrix<Scalar, 6, 6> AD;
        AD << R, vector3ToCrossMatrix3(p) * R, m3_t::Zero(), R;
        REQUIRE(AD.isApprox(tf.dualMatrix()));
    }
    {
        // operator*=(Transform)
        tf_t tf1{ R, p };
        tf_t tf2{ R, p };
        tf1 *= tf2;
        REQUIRE(tf1.homogeneousMatrix().isApprox(H * H));
        REQUIRE(tf1.matrix().isApprox(A * A));
    }
    {
        // operator*(Transform)
        tf_t tf1{ R, p };
        tf_t tf2{ R, p };
        tf_t tf3 = tf1 * tf2;
        REQUIRE(tf3.homogeneousMatrix().isApprox(H * H));
        REQUIRE(tf3.matrix().isApprox(A * A));
    }
    {
        // operator*(MotionVector)
        tf_t tf{ R, p };
        MotionVector<Scalar> m{ v3_t::Random(), v3_t::Random() };
        MotionVector<Scalar> res = tf * m;
        REQUIRE(res.vector().isApprox(tf.matrix() * m.vector()));
    }
    {
        // dualMul(ForceVector)
        tf_t tf{ R, p };
        ForceVector<Scalar> f{ v3_t::Random(), v3_t::Random() };
        ForceVector<Scalar> res = tf.dualMul(f);
        REQUIRE(res.vector().isApprox(tf.dualMatrix() * f.vector()));
    }
    {
        // invMul(Transform)
        tf_t tf1{ R, p };
        tf_t tf2{ qt_t::UnitRandom(), v3_t::Random() };
        tf_t tf3 = tf1.invMul(tf2);
        REQUIRE(tf3.homogeneousMatrix().isApprox(tf1.homogeneousMatrix().inverse() * tf2.homogeneousMatrix()));
        REQUIRE(tf3.matrix().isApprox(tf1.matrix().inverse() * tf2.matrix()));
    }
    {
        // invMul(MotionVector)
        tf_t tf{ R, p };
        MotionVector<Scalar> m1{ v3_t::Random(), v3_t::Random() };
        MotionVector<Scalar> m2 = tf.invMul(m1);
        REQUIRE(m2.vector().isApprox(tf.matrix().inverse() * m1.vector()));
    }
    {
        // inverse()
        tf_t tf1{ R, p };
        tf_t tf2 = tf1.inverse();
        REQUIRE(tf2.homogeneousMatrix().isApprox(tf1.homogeneousMatrix().inverse()));
        REQUIRE(tf2.matrix().isApprox(tf1.matrix().inverse()));
    }
    {
        // operator==()
        tf_t tf1{ R, p };
        tf_t tf2{ R, p };
        REQUIRE(tf1 == tf2);
    }
    {
        // operator!=()
        tf_t tf1{ R, p };
        tf_t tf2{ qt_t::UnitRandom(), v3_t::Random() };
        REQUIRE(tf1 != tf2);
    }
    {
        // isApprox
        v3_t v_eps = v3_t::Ones() * 0.1 * dummy_precision<Scalar>();
        m3_t m_eps = m3_t::Ones() * 0.1 * dummy_precision<Scalar>();
        tf_t tf1{ R, p };
        tf_t tf2{ R + m_eps, p };
        REQUIRE(tf1 != tf2);
        REQUIRE(tf1.isApprox(tf2));
        tf_t tf3{ R, p + v_eps };
        REQUIRE(tf1 != tf3);
        REQUIRE(tf1.isApprox(tf3));
    }
}

TEST_CASE_TEMPLATE("Cross", Scalar, float, double)
{
    using namespace coma;
    using m3_t = Eigen::Matrix<Scalar, 3, 3>;
    using m4_t = Eigen::Matrix<Scalar, 4, 4>;
    using m6_t = Eigen::Matrix<Scalar, 6, 6>;
    using v3_t = Eigen::Matrix<Scalar, 3, 1>;
    using mv_t = MotionVector<Scalar>;
    using fv_t = ForceVector<Scalar>;
    using cx_t = Cross<Scalar>;
    v3_t w = v3_t::Random();
    v3_t v = v3_t::Random();
    mv_t m{ w, v };
    fv_t f{ w, v };

    {
        // Zero
        cx_t cx = cx_t::Zero();
        REQUIRE(cx.angularMat() == m3_t::Zero());
        REQUIRE(cx.angular() == v3_t::Zero());
        REQUIRE(cx.linearMat() == m3_t::Zero());
        REQUIRE(cx.linear() == v3_t::Zero());
    }
    {
        // { m } ctor
        cx_t cx{ m };
        REQUIRE(cx.angularMat() == vector3ToCrossMatrix3(w));
        REQUIRE(cx.linearMat() == vector3ToCrossMatrix3(v));
        REQUIRE(cx.angular() == w);
        REQUIRE(cx.linear() == v);
    }
    {
        // Copy-ctor
        cx_t cx1{ m };
        cx_t cx2{ cx1 };
        REQUIRE(cx2.angularMat() == vector3ToCrossMatrix3(w));
        REQUIRE(cx2.linearMat() == vector3ToCrossMatrix3(v));
    }
    {
        // Move-ctor
        cx_t cx1{ m };
        cx_t cx2{ std::move(cx1) };
        REQUIRE(cx2.angularMat() == vector3ToCrossMatrix3(w));
        REQUIRE(cx2.linearMat() == vector3ToCrossMatrix3(v));
    }
    {
        // Assign-op
        cx_t cx1{ m };
        cx_t cx2;
        cx2 = cx1;
        REQUIRE(cx2.angularMat() == vector3ToCrossMatrix3(w));
        REQUIRE(cx2.linearMat() == vector3ToCrossMatrix3(v));
    }
    {
        // Move-assign-op
        cx_t cx1{ m };
        cx_t cx2;
        cx2 = std::move(cx1);
        REQUIRE(cx2.angularMat() == vector3ToCrossMatrix3(w));
        REQUIRE(cx2.linearMat() == vector3ToCrossMatrix3(v));
    }
    {
        // matrix()
        cx_t cx{ m };
        REQUIRE(cx.homogeneousDifferentiator() == vector6ToCrossMatrix4(m));
        REQUIRE(cx.matrix() == vector6ToCrossMatrix6(m));
        REQUIRE(cx.dualMatrix() == vector6ToCrossDualMatrix6(m));
    }
    {
        // motion()
        cx_t cx{ m };
        REQUIRE(cx.motion() == m);
    }
    {
        // setZero
        cx_t cx{ m };
        cx.setZero();
        REQUIRE(cx.angularMat() == m3_t::Zero());
        REQUIRE(cx.linearMat() == m3_t::Zero());
    }
    {
        // unary operator-
        cx_t cx{ m };
        auto cxInv = -cx;
        REQUIRE(cxInv.motion() == -cx.motion());
    }
    {
        // operator+=(cx_t)
        cx_t cx{ m };
        cx += cx;
        REQUIRE(cx.angularMat() == vector3ToCrossMatrix3(w + w));
        REQUIRE(cx.linearMat() == vector3ToCrossMatrix3(v + v));
    }
    {
        // operator-=(cx_t)
        cx_t cx1{ m };
        cx_t cx2{ m };
        cx1 -= cx2;
        REQUIRE(cx1.angularMat() == m3_t::Zero());
        REQUIRE(cx1.linearMat() == m3_t::Zero());
    }
    {
        // operator*=(tf)
        cx_t cx{ m };
        m4_t mat4 = vector6ToCrossMatrix4(m);
        m6_t mat6 = vector6ToCrossMatrix6(m);
        Scalar s = 5;
        cx *= s;
        REQUIRE(cx.homogeneousDifferentiator() == mat4 * s);
        REQUIRE(cx.matrix() == mat6 * s);
    }
    {
        // operator/=(tf)
        cx_t cx{ m };
        m4_t mat4 = vector6ToCrossMatrix4(m);
        m6_t mat6 = vector6ToCrossMatrix6(m);
        Scalar s = 5;
        cx /= s;
        REQUIRE(cx.homogeneousDifferentiator() == mat4 / s);
        REQUIRE(cx.matrix() == mat6 / s);
    }
    {
        // operator+(cx_t)
        cx_t cx1{ m };
        cx_t cx2{ m };
        cx_t cx3 = cx1 + cx2;
        REQUIRE(cx3.angularMat() == vector3ToCrossMatrix3(w + w));
        REQUIRE(cx3.linearMat() == vector3ToCrossMatrix3(v + v));
    }
    {
        // operator-(cx_t)
        cx_t cx1{ m };
        cx_t cx2{ m };
        cx_t cx3 = cx1 - cx2;
        REQUIRE(cx3.angularMat() == m3_t::Zero());
        REQUIRE(cx3.linearMat() == m3_t::Zero());
    }
    {
        // operator*(tf)
        cx_t cx{ m };
        m4_t mat4 = vector6ToCrossMatrix4(m);
        m6_t mat6 = vector6ToCrossMatrix6(m);
        auto s = Scalar(5);
        REQUIRE((s * cx).homogeneousDifferentiator() == s * mat4);
        REQUIRE((cx * s).homogeneousDifferentiator() == mat4 * s);
        REQUIRE((s * cx).matrix() == s * mat6);
        REQUIRE((cx * s).matrix() == mat6 * s);
    }
    {
        // operator*(mv_t)
        mv_t m2{ v3_t::Random(), v3_t::Random() };
        cx_t cx{ m };
        REQUIRE((cx * m2).vector().isApprox(m.cross(m2).vector()));
    }
    {
        // dualMul(fv_t)
        cx_t cx{ m };
        REQUIRE((cx.dualMul(f)).vector().isApprox(m.crossDual(f).vector()));
    }
    {
        // operator/(tf)
        cx_t cx{ m };
        m4_t mat4 = vector6ToCrossMatrix4(m);
        m6_t mat6 = vector6ToCrossMatrix6(m);
        auto s = Scalar(5);
        REQUIRE((cx / s).homogeneousDifferentiator() == mat4 / s);
        REQUIRE((cx / s).matrix() == mat6 / s);
    }
    {
        // operator==()
        cx_t cx1{ m };
        cx_t cx2{ m };
        REQUIRE(cx1 == cx2);
    }
    {
        // operator!=()
        cx_t cx1{ m };
        cx_t cx2{ mv_t{ v3_t::Random(), v3_t::Random() } };
        REQUIRE(cx1 != cx2);
    }
    {
        // isApprox
        v3_t eps = v3_t::Ones() * 0.1 * dummy_precision<Scalar>();
        cx_t cx1{ m };
        mv_t m2{ w + eps, v };
        cx_t cx2{ m2 };
        REQUIRE(cx1 != cx2);
        REQUIRE(cx1.isApprox(cx2));
        mv_t m3{ w, v + eps };
        cx_t cx3{ m3 };
        REQUIRE(cx1 != cx3);
        REQUIRE(cx1.isApprox(cx3));
    }
}

TEST_CASE_TEMPLATE("blockMat operators", Scalar, float, double)
{
    using namespace coma;
    using qt_t = Eigen::Quaternion<Scalar>;
    using m3_t = Eigen::Matrix<Scalar, 3, 3>;
    using m4_t = Eigen::Matrix<Scalar, 4, 4>;
    using m6_t = Eigen::Matrix<Scalar, 6, 6>;
    using v3_t = Eigen::Matrix<Scalar, 3, 1>;
    using tf_t = Transform<Scalar>;
    using mv_t = MotionVector<Scalar>;
    using cx_t = Cross<Scalar>;
    using bm31_t = BiBlock31<Scalar>;
    using bm33_t = BiBlock33<Scalar>;
    v3_t w = v3_t::Random();
    v3_t v = v3_t::Random();
    mv_t m{ w, v };

    DISABLE_CONVERSION_WARNING_BEGIN
    qt_t q = qt_t::UnitRandom();
    DISABLE_CONVERSION_WARNING_END

    m3_t R = q.toRotationMatrix();
    v3_t p = v3_t::Random();

    {
        // unary operator-
        bm31_t bm31{ vector3ToCrossMatrix3(w), v };
        auto bm31Inv = -bm31;
        REQUIRE(bm31Inv.blockT1() == -bm31.blockT1());
        REQUIRE(bm31Inv.blockT2() == -bm31.blockT2());
        bm33_t bm33{ vector3ToCrossMatrix3(w), vector3ToCrossMatrix3(v) };
        auto bm33Inv = -bm33;
        REQUIRE(bm33Inv.blockT1() == -bm33.blockT1());
        REQUIRE(bm33Inv.blockT2() == -bm33.blockT2());
    }
    {
        // operator*(cx_t, cx_t)
        cx_t cx{ m };
        m6_t mat6 = vector6ToCrossMatrix6(m);
        REQUIRE((cx * cx).matrix().isApprox(mat6 * mat6));
    }
    {
        // mul4(cx_t, cx_t)
        cx_t cx{ m };
        m4_t mat4 = vector6ToCrossMatrix4(m);
        REQUIRE(mul4(cx, cx).matrix() == mat4 * mat4);
    }
    {
        // bm33 *= cx_t
        bm33_t bm{ vector3ToCrossMatrix3(w), vector3ToCrossMatrix3(v) };
        cx_t cx{ m };
        m6_t mat = bm.matrix();
        bm *= cx;
        REQUIRE(bm.matrix().isApprox(mat * cx.matrix()));
    }
    {
        // bm31 *= cx_t
        bm31_t bm{ vector3ToCrossMatrix3(w), v };
        cx_t cx{ m };
        m4_t mat = bm.matrix();
        bm *= cx;
        REQUIRE(bm.matrix() == mat * cx.homogeneousDifferentiator());
    }
    {
        // operator*(tf_t, cx_t)
        cx_t cx{ m };
        tf_t tf{ R, p };
        auto res = tf * cx;
        REQUIRE(res.matrix().isApprox(tf.matrix() * cx.matrix()));
    }
    {
        // mul4(tf_t, cx_t)
        cx_t cx{ m };
        tf_t tf{ R, p };
        auto res = mul4(tf, cx);
        REQUIRE(res.matrix().isApprox(tf.homogeneousMatrix() * cx.homogeneousDifferentiator()));
    }
    {
        // operator*(cx_t, tf_t)
        tf_t tf{ R, p };
        cx_t cx{ m };
        auto res = cx * tf;
        REQUIRE(res.matrix().isApprox(cx.matrix() * tf.matrix()));
    }
    {
        // mul4(cx_t, tf_t)
        tf_t tf{ R, p };
        cx_t cx{ m };
        auto res = mul4(cx, tf);
        REQUIRE(res.matrix() == cx.homogeneousDifferentiator() * tf.homogeneousMatrix());
    }
    {
        // operator*=(bm33_t, tf_t)
        bm33_t bm{ vector3ToCrossMatrix3(w), vector3ToCrossMatrix3(v) };
        tf_t tf{ R, p };
        m6_t mat = bm.matrix();
        bm *= tf;
        REQUIRE(bm.matrix().isApprox(mat * tf.matrix()));
    }
    {
        // operator*=(bm31_t, tf_t)
        bm31_t bm{ vector3ToCrossMatrix3(w), v };
        tf_t tf{ R, p };
        m4_t mat = bm.matrix();
        bm *= tf;
        REQUIRE(bm.matrix() == mat * tf.homogeneousMatrix());
    }
    {
        // operator*=(tf_t, bm33_t)
        bm33_t bm{ vector3ToCrossMatrix3(w), vector3ToCrossMatrix3(v) };
        tf_t tf{ R, p };
        auto res = tf * bm;
        REQUIRE(res.matrix().isApprox(tf.matrix() * bm.matrix()));
    }
    {
        // operator*=(tf_t, bm31_t)
        bm31_t bm{ vector3ToCrossMatrix3(w), v };
        tf_t tf{ R, p };
        auto res = tf * bm;
        REQUIRE(res.matrix().isApprox(tf.homogeneousMatrix() * bm.matrix()));
    }
    {
        // operator*=(cx_t, bm33_t)
        bm33_t bm{ vector3ToCrossMatrix3(w), vector3ToCrossMatrix3(v) };
        cx_t cx{ m };
        auto res = cx * bm;
        REQUIRE(res.matrix().isApprox(cx.matrix() * bm.matrix()));
    }
    {
        // operator*=(cx_t, bm31_t)
        bm31_t bm{ vector3ToCrossMatrix3(w), v };
        cx_t cx{ m };
        auto res = cx * bm;
        REQUIRE(res.matrix() == cx.homogeneousDifferentiator() * bm.matrix());
    }
    {
        // operator*=(bm31_t, T)
        bm31_t bm{ vector3ToCrossMatrix3(w), v };
        Scalar s = 5;
        m4_t mat = bm.matrix();
        bm *= s;
        REQUIRE(bm.matrix() == mat * s);
    }
    {
        // operator*=(bm33_t, T)
        bm33_t bm{ vector3ToCrossMatrix3(w), vector3ToCrossMatrix3(v) };
        Scalar s = 5;
        m6_t mat = bm.matrix();
        bm *= s;
        REQUIRE(bm.matrix() == mat * s);
    }
    {
        // operator*=(bm31_t, T)
        bm31_t bm{ vector3ToCrossMatrix3(w), v };
        Scalar s = 5;
        auto res = bm * s;
        REQUIRE(res.matrix() == bm.matrix() * s);
    }
    {
        // operator*=(bm33_t, T)
        bm33_t bm{ vector3ToCrossMatrix3(w), vector3ToCrossMatrix3(v) };
        Scalar s = 5;
        auto res = bm * s;
        REQUIRE(res.matrix() == bm.matrix() * s);
    }
    {
        // operator*=(bm31_t, T)
        bm31_t bm{ vector3ToCrossMatrix3(w), v };
        Scalar s = 5;
        auto res = s * bm;
        REQUIRE(res.matrix() == s * bm.matrix());
    }
    {
        // operator*=(bm33_t, T)
        bm33_t bm{ vector3ToCrossMatrix3(w), vector3ToCrossMatrix3(v) };
        Scalar s = 5;
        auto res = s * bm;
        REQUIRE(res.matrix() == s * bm.matrix());
    }
}

TEST_CASE_TEMPLATE("CrossN", T, test_pairs)
{
    using namespace coma;
    using Scalar = typename T::first_type;
    constexpr int n_vec = T::second_type::n_vec;
    constexpr int dynNVec = OrderF::n_vec; // dynamic tests
    using v3_t = Eigen::Matrix<Scalar, 3, 1>;
    using v6_t = Eigen::Matrix<Scalar, 6, 1>;
    using mv_t = MotionVector<Scalar>;
    using fv_t = ForceVector<Scalar>;
    using mvx_t = MotionVectorX<Scalar, n_vec>;
    using fvx_t = ForceVectorX<Scalar, n_vec>;
    using cx_t = Cross<Scalar>;
    using cn_t = CrossN<Scalar, n_vec>;
    using vecX = Eigen::Matrix<Scalar, -1, 1>;
    mv_t m{ v6_t::Random() };
    cx_t cx{ m };

    {
        // Zero
        cn_t cn = cn_t::Zero(dynNVec);
        for (int i = 0; i < cn.nVec(); ++i)
            REQUIRE(cn[i] == cx_t::Zero());

        if constexpr (n_vec != Dynamic) {
            cn = cn_t::Zero();
            for (int i = 0; i < cn.nVec(); ++i)
                REQUIRE(cn[i] == cx_t::Zero());
        }
    }
    {
        // default ctor
        cn_t cn;
        if constexpr (n_vec == Dynamic) {
            REQUIRE(cn.nVec() == 0);
            cn_t cn2{ dynNVec };
            REQUIRE(cn2.nVec() == dynNVec);
        } else {
            REQUIRE(cn.nVec() == n_vec);
        }
    }
    {
        // motion vector ctor
        mvx_t mv{ m, m, m, m, m, m, m };
        cn_t cn{ mv };
        for (int i = 0; i < cn.nVec(); ++i) {
            REQUIRE(cn[i] == cx);
            REQUIRE(cn.motion(i) == m);
        }
    }
    {
        // move-motion vector ctor
        mvx_t mv{ m, m, m, m, m, m, m };
        cn_t cn{ std::move(mv) };
        for (int i = 0; i < cn.nVec(); ++i) {
            REQUIRE(cn[i] == cx);
            REQUIRE(cn.motion(i) == m);
        }
    }
    {
        // vectors ctor
        cn_t cn{ m, m, m, m, m, m, m };
        for (int i = 0; i < cn.nVec(); ++i) {
            REQUIRE(cn.at(i) == cx);
            REQUIRE(cn.motionAt(i) == m);
        }
    }
    {
        // move-vectors ctor
        cn_t cn{ mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero() };
        for (int i = 0; i < cn.nVec(); ++i) {
            REQUIRE(cn.at(i) == cx_t::Zero());
            REQUIRE(cn.motionAt(i) == mv_t::Zero());
        }
    }
    {
        // indirect vectors ctor
        cn_t cn;
        cn.set(m, m, m, m, m, m, m);
        for (int i = 0; i < cn.nVec(); ++i) {
            REQUIRE(cn.at(i) == cx);
            REQUIRE(cn.motionAt(i) == m);
        }

        if constexpr (n_vec == Dynamic) {
            // Re-sizing on place
            cn.set(m, m, m);
            for (int i = 0; i < cn.nVec(); ++i) {
                REQUIRE(cn.at(i) == cx);
                REQUIRE(cn.motionAt(i) == m);
            }
            REQUIRE_THROWS_AS(cn.at(10), std::out_of_range);
        }
    }
    {
        // indirect motion vector ctor
        mvx_t mv;
        mv.set(m, m, m, m, m, m, m);
        cn_t cn{ mv };
        for (int i = 0; i < cn.nVec(); ++i) {
            REQUIRE(cn[i] == cx);
            REQUIRE(cn.motion(i) == m);
        }

        if constexpr (n_vec == Dynamic) {
            // Re-sizing on place
            mvx_t tmv2 = { m, m, m };
            cn.set(tmv2);
            for (int i = 0; i < cn.nVec(); ++i) {
                REQUIRE(cn.at(i) == cx);
                REQUIRE(cn.motionAt(i) == m);
            }
            REQUIRE_THROWS_AS(cn.at(10), std::out_of_range);
        }
    }
    {
        // indirect move-ctor
        cn_t cn;
        cn.set(mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero());
        for (int i = 0; i < cn.nVec(); ++i) {
            REQUIRE(cn[i] == cx_t::Zero());
            REQUIRE(cn.motion(i) == mv_t::Zero());
        }

        if constexpr (n_vec == Dynamic) {
            // Re-sizing on place
            cn.set(mv_t::Zero(), mv_t::Zero(), mv_t::Zero());
            for (int i = 0; i < cn.nVec(); ++i) {
                REQUIRE(cn.at(i) == cx_t::Zero());
                REQUIRE(cn.motionAt(i) == mv_t::Zero());
            }
            REQUIRE_THROWS_AS(cn.at(10), std::out_of_range);
        }
    }
    {
        // copy ctor
        cn_t cn1{ m, m, m, m, m, m, m };
        cn_t cn2{ cn1 };
        for (int i = 0; i < cn1.nVec(); ++i) {
            REQUIRE(cn2[i].motion() == m);
            REQUIRE(cn2.motion(i) == m);
        }
    }
    {
        // move ctor
        cn_t cn1{ m, m, m, m, m, m, m };
        cn_t cn2{ std::move(cn1) };
        for (int i = 0; i < cn1.nVec(); ++i) {
            REQUIRE(cn2[i].motion() == m);
            REQUIRE(cn2.motion(i) == m);
        }
    }
    {
        // assign-op
        cn_t cn1{ m, m, m, m, m, m, m };
        cn_t cn2;
        cn2 = cn1;
        for (int i = 0; i < cn1.nVec(); ++i) {
            REQUIRE(cn2[i].motion() == m);
            REQUIRE(cn2.motion(i) == m);
        }
    }
    {
        // move-assign-op
        cn_t cn1{ m, m, m, m, m, m, m };
        cn_t cn2;
        cn2 = std::move(cn1);
        for (int i = 0; i < cn1.nVec(); ++i) {
            REQUIRE(cn2[i].motion() == m);
            REQUIRE(cn2.motion(i) == m);
        }
    }
    {
        // operator==()
        cn_t cn1{ m, m, m, m, m, m, m };
        cn_t cn2{ m, m, m, m, m, m, m };
        REQUIRE(cn1 == cn2);
    }
    {
        // operator!=()
        cn_t cn1{ m, m, m, m, m, m, m };
        cn_t cn2 = cn1;
        if constexpr (n_vec == Dynamic) {
            cn2.set(m, m, m);
        } else {
            cn2.setZero();
        }
        REQUIRE(cn1 != cn2);
    }
    {
        // setZero
        cn_t cn;
        cn.setZero(dynNVec);
        REQUIRE(cn == cn_t::Zero(dynNVec));
        if constexpr (n_vec != Dynamic) {
            cn.motion(0) = m;
            cn.setZero();
            REQUIRE(cn == cn_t::Zero());
        }
    }
    {
        // unary operator-
        cn_t cn{ m, m, m, m, m, m, m };
        auto cnInv = -cn;
        REQUIRE(cnInv.motion() == -cn.motion());
    }
    {
        // operator+=
        cn_t cn1{ m, m, m, m, m, m, m };
        cn_t cn2{ m, m, m, m, m, m, m };
        mv_t m2 = m + m;
        cn_t sol{ m2, m2, m2, m2, m2, m2, m2 };
        cn1 += cn2;
        REQUIRE(cn1 == sol);
        if constexpr (n_vec == Dynamic) {
            cn2.set(m, m, m);
            REQUIRE_THROWS_AS(cn1 += cn2, std::runtime_error);
        }
    }
    {
        // operator-=
        cn_t cn1{ m, m, m, m, m, m, m };
        cn_t cn2{ m, m, m, m, m, m, m };
        cn1 -= cn2;
        if constexpr (n_vec == Dynamic) {
            REQUIRE(cn1 == cn_t::Zero(dynNVec));
            cn2.set(m, m, m);
            REQUIRE_THROWS_AS(cn1 -= cn2, std::runtime_error);
        } else {
            REQUIRE(cn1 == cn_t::Zero());
        }
    }
    {
        // operator*=
        cn_t cn{ m, m, m, m, m, m, m };
        Scalar v = 5;
        mv_t m2 = v * m;
        cn *= v;
        cn_t sol{ m2, m2, m2, m2, m2, m2, m2 };
        REQUIRE(cn == sol);
    }
    {
        // operator/=
        cn_t cn{ m, m, m, m, m, m, m };
        Scalar v = 5;
        mv_t m2 = m / v;
        cn /= v;
        cn_t sol{ m2, m2, m2, m2, m2, m2, m2 };
        REQUIRE(cn == sol);
    }
    {
        // operator+
        cn_t cn1{ m, m, m, m, m, m, m };
        cn_t cn2{ m, m, m, m, m, m, m };
        mv_t m2 = m + m;
        cn_t sol{ m2, m2, m2, m2, m2, m2, m2 };
        cn_t res = cn1 + cn2;
        REQUIRE(res == sol);
        if constexpr (n_vec == Dynamic) {
            cn2.set(m, m, m);
            REQUIRE_THROWS_AS(cn1 + cn2, std::runtime_error);
        }
    }
    {
        // operator-
        cn_t cn1{ m, m, m, m, m, m, m };
        cn_t cn2{ m, m, m, m, m, m, m };
        cn_t res = cn1 - cn2;
        if constexpr (n_vec == Dynamic) {
            REQUIRE(res == cn_t::Zero(dynNVec));
            cn2.set(m, m, m);
            REQUIRE_THROWS_AS(cn1 - cn2, std::runtime_error);
        } else {
            REQUIRE(res == cn_t::Zero());
        }
    }
    {
        // operator*(Scalar)
        cn_t cn{ m, m, m, m, m, m, m };
        Scalar v = 5;
        mv_t m2 = v * m;
        cn_t res = v * cn;
        cn_t sol{ m2, m2, m2, m2, m2, m2, m2 };
        REQUIRE(res == sol);
        res = cn * v;
        REQUIRE(res == sol);
    }
    // {
    //     // TODO: Add CrossN * CrossN
    //     cn_t cn1{ m, m, m, m, m, m, m };
    //     cn_t cn2{ m, m, m, m, m, m, m };
    //     REQUIRE((cn1 * cn2).matrix().isApprox(cn1.matrix() * cn2.matrix()));
    // }
    {
        // operator*(mvx_t)
        mv_t m2{ v6_t::Random() };
        mvx_t mv{ m2, m2, m2, m2, m2, m2, m2 };
        cn_t cn{ m, m, m, m, m, m, m };
        auto res = cn * mv;
        vecX mvf = mv.vector();
        const auto& factors = factorial_factors<Scalar, n_vec>;
        for (int i = 0; i < dynNVec; ++i) {
            mvf.template segment<6>(6 * i) /= factors[static_cast<size_t>(i)];
        }
        vecX resVec = cn.matrix() * mvf;
        for (int i = 0; i < dynNVec; ++i) {
            resVec.template segment<6>(6 * i) *= factors[static_cast<size_t>(i)];
        }
        REQUIRE(res.vector().isApprox(resVec));
        if constexpr (n_vec == Dynamic) {
            auto shortMat = cn.matrix(dynNVec / 2);
            REQUIRE(shortMat == cn.matrix().template topLeftCorner<6 * (dynNVec / 2), 6 * (dynNVec / 2)>());
            mv.set(m2, m2, m2);
            REQUIRE_THROWS_AS(cn * mv, std::runtime_error);
        } else {
            auto shortMat = cn.template matrix<n_vec / 2>();
            REQUIRE(shortMat == cn.matrix().template topLeftCorner<6 * (n_vec / 2), 6 * (n_vec / 2)>());
        }
    }
    {
        // dualMul
        fv_t f{ v6_t::Random() };
        fvx_t fv{ f, f, f, f, f, f, f };
        cn_t cn{ m, m, m, m, m, m, m };
        auto res = cn.dualMul(fv);
        vecX fvf = fv.vector();
        const auto& factors = factorial_factors<Scalar, n_vec>;
        for (int i = 0; i < dynNVec; ++i) {
            fvf.template segment<6>(6 * i) /= factors[static_cast<size_t>(i)];
        }
        vecX resVec = cn.dualMatrix() * fvf;
        for (int i = 0; i < dynNVec; ++i) {
            resVec.template segment<6>(6 * i) *= factors[static_cast<size_t>(i)];
        }
        REQUIRE(res.vector().isApprox(resVec));
        if constexpr (n_vec == Dynamic) {
            auto shortMat = cn.dualMatrix(dynNVec / 2);
            REQUIRE(shortMat == cn.dualMatrix().template topLeftCorner<6 * (dynNVec / 2), 6 * (dynNVec / 2)>());
            fv.set(f, f, f);
            REQUIRE_THROWS_AS(cn.dualMul(fv), std::runtime_error);
        } else {
            auto shortMat = cn.template dualMatrix<n_vec / 2>();
            REQUIRE(shortMat == cn.dualMatrix().template topLeftCorner<6 * (n_vec / 2), 6 * (n_vec / 2)>());
        }
    }
    {
        // operator/
        cn_t cn{ m, m, m, m, m, m, m };
        Scalar v = 5;
        mv_t m2 = m / v;
        cn_t res = cn / v;
        cn_t sol{ m2, m2, m2, m2, m2, m2, m2 };
        REQUIRE(res == sol);
    }
    {
        // isApprox
        v3_t v_eps = v3_t::Ones() * 0.1 * dummy_precision<Scalar>();
        cn_t cn1{ m, m, m, m, m, m, m };
        for (int i = 0; i < cn1.nVec(); ++i) {
            cn_t cn2{ m, m, m, m, m, m, m };
            cn2.motion(i).angular() += v_eps;
            cn2.motion(i).linear() += v_eps;
            REQUIRE(cn1 != cn2);
            REQUIRE(cn1.isApprox(cn2));
        }
        if constexpr (n_vec == Dynamic) {
            cn_t cn2{ m, m, m };
            REQUIRE(!cn1.isApprox(cn2));
        }
    }
}

TEST_CASE("CrossN 0-n_vec")
{
    using namespace coma;
    using cn0_t = CrossN<double, 0>;
    using cnd_t = CrossN<double, Dynamic>;

    {
        // Init
        cn0_t cn0{};
        cnd_t cnd{ 0 };
        REQUIRE(cn0.nVec() == 0);
        REQUIRE(cnd.nVec() == 0);
    }
    {
        // Resize
        cn0_t cn0{};
        cnd_t cnd;
        cn0.resize(0);
        cnd.resize(1);
        REQUIRE(cnd.nVec() == 1);
    }
}
