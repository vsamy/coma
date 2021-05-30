/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "coma/Core"
#include "doctest/doctest.h"

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

TEST_CASE_TEMPLATE("SpatialVector", Scalar, float, double)
{
    using namespace coma;
    using mv_t = MotionVector<Scalar>;
    using fv_t = ForceVector<Scalar>;
    using v3_t = Eigen::Matrix<Scalar, 3, 1>;
    using v6_t = Eigen::Matrix<Scalar, 6, 1>;
    v3_t v3 = v3_t::Random();
    v6_t v6;
    v6 << v3, v3;

    {
        // ctor
        mv_t mv1(v3, v3);
        REQUIRE(mv1.vector() == v6);
        mv_t mv2(v6);
        REQUIRE(mv2.vector() == v6);
        fv_t fv1(v3, v3);
        REQUIRE(fv1.vector() == v6);
        fv_t fv2(v6);
        REQUIRE(fv2.vector() == v6);
    }
    {
        // copy-ctor
        mv_t mv1(v6);
        mv_t mv2(mv1);
        REQUIRE(mv2.vector() == v6);
        fv_t fv1(v6);
        fv_t fv2(fv1);
        REQUIRE(fv2.vector() == v6);
    }
    {
        // move-ctor
        mv_t mv1(v6);
        mv_t mv2(std::move(mv1));
        REQUIRE(mv2.vector() == v6);
        fv_t fv1(v6);
        fv_t fv2(std::move(fv1));
        REQUIRE(fv2.vector() == v6);
    }
    {
        // assign-op
        mv_t mv1(v6);
        mv_t mv2;
        mv2 = mv1;
        REQUIRE(mv2.vector() == v6);
        fv_t fv1(v6);
        fv_t fv2;
        fv2 = fv1;
        REQUIRE(fv2.vector() == v6);
    }
    {
        // move-assign-op
        mv_t mv1(v6);
        mv_t mv2;
        mv2 = std::move(mv1);
        REQUIRE(mv2.vector() == v6);
        fv_t fv1(v6);
        fv_t fv2;
        fv2 = std::move(fv1);
        REQUIRE(fv2.vector() == v6);
    }
    {
        // set vector
        mv_t mv;
        fv_t fv;
        mv.angular() = v3;
        mv.linear() = v3;
        fv.angular() = v3;
        fv.linear() = v3;
        REQUIRE(mv.vector() == v6);
        REQUIRE(fv.vector() == v6);
    }
    {
        // setZero
        mv_t mv;
        mv.setZero();
        REQUIRE(mv.vector() == v6_t::Zero());
        fv_t fv;
        fv.setZero();
        REQUIRE(fv.vector() == v6_t::Zero());
    }
    {
        // cross
        mv_t m1(v6);
        mv_t m2(v6_t::Random());
        auto res = m1.cross(m2);
        REQUIRE(res.vector().isApprox(vector6ToCrossMatrix6(m1) * m2.vector()));
    }
    {
        // crossDual
        mv_t m(v6);
        fv_t f(v6_t::Random());
        auto res = m.crossDual(f);
        REQUIRE(res.vector().isApprox(vector6ToCrossDualMatrix6(m) * f.vector()));
    }
    {
        // dot
        mv_t m(v6);
        fv_t f(v6_t::Random());
        auto res = m.dot(f);
        REQUIRE(std::abs(res - m.vector().dot(f.vector())) < dummy_precision<Scalar>());
    }
    {
        // operator()
        mv_t m(v6);
        for (int i = 0; i < 6; ++i) {
            REQUIRE(m(i) == v6(i));
        }
    }
    {
        // unary operator-
        mv_t m(v6);
        auto mInv = -m;
        REQUIRE(mInv.vector() == -m.vector());
        fv_t f(v6_t::Random());
        auto fInv = -f;
        REQUIRE(fInv.vector() == -f.vector());
    }
    {
        // operator+=
        mv_t mv1(v6), mv2;
        mv2.setZero();
        mv2 += mv1;
        REQUIRE(mv2.vector() == v6);
        fv_t fv1(v6), fv2;
        fv2.setZero();
        fv2 += fv1;
        REQUIRE(fv2.vector() == v6);
    }
    {
        // operator+
        mv_t mv1(v6), mv2(v6);
        mv_t mv3 = mv1 + mv2;
        REQUIRE(mv3.vector() == v6 + v6);
        fv_t fv1(v6), fv2(v6);
        fv_t fv3 = fv1 + fv2;
        REQUIRE(fv3.vector() == v6 + v6);
    }
    {
        // operator-=
        mv_t mv1(v6), mv2;
        mv2.setZero();
        mv2 -= mv1;
        REQUIRE(mv2.vector() == -v6);
        fv_t fv1(v6), fv2;
        fv2.setZero();
        fv2 -= fv1;
        REQUIRE(fv2.vector() == -v6);
    }
    {
        // operator-
        mv_t mv1(v6), mv2(v6);
        mv_t mv3 = mv1 - mv2;
        REQUIRE(mv3.vector() == v6 - v6);
        fv_t fv1(v6), fv2(v6);
        fv_t fv3 = fv1 - fv2;
        REQUIRE(fv3.vector() == v6 - v6);
    }
    {
        // operator*=
        Scalar s = Scalar(2);
        mv_t mv(v6);
        mv *= s;
        REQUIRE(mv.vector() == s * v6);
        fv_t fv(v6);
        fv *= s;
        REQUIRE(fv.vector() == s * v6);
    }
    {
        // operator*
        Scalar s = Scalar(2);
        mv_t mv1(v6);
        mv_t mv2 = mv1 * s;
        REQUIRE(mv2.vector() == s * v6);
        mv2 = s * mv1;
        REQUIRE(mv2.vector() == s * v6);
        fv_t fv1(v6);
        fv_t fv2 = fv1 * s;
        REQUIRE(fv2.vector() == s * v6);
        fv2 = s * fv1;
        REQUIRE(fv2.vector() == s * v6);
    }
    {
        // operator/=
        Scalar s = Scalar(2);
        mv_t mv(v6);
        mv /= s;
        REQUIRE(mv.vector() == v6 / s);
        fv_t fv(v6);
        fv /= s;
        REQUIRE(fv.vector() == v6 / s);
    }
    {
        // operator/
        Scalar s = Scalar(2);
        mv_t mv1(v6);
        mv_t mv2 = mv1 / s;
        REQUIRE(mv2.vector() == v6 / s);
        fv_t fv1(v6);
        fv_t fv2 = fv1 / s;
        REQUIRE(fv2.vector() == v6 / s);
    }
    {
        // operator==
        mv_t mv1(v6), mv2(v6);
        REQUIRE(mv1 == mv2);
        fv_t fv1(v6), fv2(v6);
        REQUIRE(fv1 == fv2);
    }
    {
        // operator!=
        mv_t mv1(v6_t::Random());
        mv_t mv2(v6_t::Random());
        REQUIRE(mv1 != mv2);
        fv_t fv1(v6_t::Random());
        fv_t fv2(v6_t::Random());
        REQUIRE(fv1 != fv2);
    }
    {
        // isApprox
        v6_t eps = v6_t::Ones() * 0.1 * dummy_precision<Scalar>();
        mv_t mv1(v6), mv2(v6 + eps);
        REQUIRE(mv1.isApprox(mv2));
        fv_t fv1(v6), fv2(v6 + eps);
        REQUIRE(fv1.isApprox(fv2));
    }
    {
        // Zero
        mv_t mv = mv_t::Zero();
        REQUIRE(mv.vector() == v6_t::Zero());
        fv_t fv = fv_t::Zero();
        REQUIRE(fv.vector() == v6_t::Zero());
    }
}

TEST_CASE_TEMPLATE("SpatialVectorX", T, test_pairs)
{
    using namespace coma;
    using Scalar = typename T::first_type;
    constexpr int n_vec = T::second_type::n_vec;
    constexpr int dynNVec = OrderF::n_vec; // dynamic tests
    using mv_t = MotionVector<Scalar>;
    using fv_t = ForceVector<Scalar>;
    using mvx_t = MotionVectorX<Scalar, n_vec>;
    using fvx_t = ForceVectorX<Scalar, n_vec>;
    using v6_t = Eigen::Matrix<Scalar, 6, 1>;
    using sm_t = typename mvx_t::storage_t::underlying_t;
    using sf_t = typename fvx_t::storage_t::underlying_t;
    mv_t m{ v6_t::Random() };
    fv_t f{ v6_t::Random() };

    {
        // ctor
        mvx_t mv;
        REQUIRE(mv.nVec() == (n_vec == Dynamic ? 0 : n_vec));
        fvx_t fv;
        REQUIRE(fv.nVec() == (n_vec == Dynamic ? 0 : n_vec));
    }
    {
        // Multi-vector copy ctor
        mvx_t mv{ m, m, m, m, m, m, m };
        for (int i = 0; i < mv.nVec(); ++i)
            REQUIRE(mv[i] == m);
        fvx_t fv{ f, f, f, f, f, f, f };
        for (int i = 0; i < fv.nVec(); ++i)
            REQUIRE(fv[i] == f);
    }
    {
        // Multi-vector move ctor
        mvx_t mv{ mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero(), mv_t::Zero() };
        for (int i = 0; i < mv.nVec(); ++i)
            REQUIRE(mv[i] == mv_t::Zero());
        fvx_t fv{ fv_t::Zero(), fv_t::Zero(), fv_t::Zero(), fv_t::Zero(), fv_t::Zero(), fv_t::Zero(), fv_t::Zero() };
        for (int i = 0; i < fv.nVec(); ++i)
            REQUIRE(fv[i] == fv_t::Zero());
    }
    {
        // vector copy ctor
        sm_t sm{ m, m, m, m, m, m, m };
        mvx_t mv{ sm };
        for (int i = 0; i < mv.nVec(); ++i)
            REQUIRE(mv.at(i) == m);
        sf_t sf{ f, f, f, f, f, f, f };
        fvx_t fv{ sf };
        for (int i = 0; i < fv.nVec(); ++i)
            REQUIRE(fv.at(i) == f);
    }
    {
        // vector move ctor
        sm_t sm{ m, m, m, m, m, m, m };
        mvx_t mv{ std::move(sm) };
        for (int i = 0; i < mv.nVec(); ++i)
            REQUIRE(mv.at(i) == m);
        sf_t sf{ f, f, f, f, f, f, f };
        fvx_t fv{ std::move(sf) };
        for (int i = 0; i < fv.nVec(); ++i)
            REQUIRE(fv.at(i) == f);
    }
    {
        // copy-ctor
        mvx_t mv{ m, m, m, m, m, m, m };
        mvx_t mv2{ mv };
        REQUIRE(mv == mv2);
        fvx_t fv{ f, f, f, f, f, f, f };
        fvx_t fv2{ fv };
        REQUIRE(fv == fv2);
    }
    {
        // move-ctor
        mvx_t mv{ m, m, m, m, m, m, m };
        mvx_t mv2{ std::move(mv) };
        for (int i = 0; i < mv.nVec(); ++i)
            REQUIRE(mv2[i] == m);
        fvx_t fv{ f, f, f, f, f, f, f };
        fvx_t fv2{ std::move(fv) };
        for (int i = 0; i < fv.nVec(); ++i)
            REQUIRE(fv2[i] == f);
    }
    {
        // assign-op
        mvx_t mv{ m, m, m, m, m, m, m };
        mvx_t mv2 = mv;
        REQUIRE(mv == mv2);
        fvx_t fv{ f, f, f, f, f, f, f };
        fvx_t fv2 = fv;
        REQUIRE(fv == fv2);
    }
    {
        // move-assign-op
        mvx_t mv{ m, m, m, m, m, m, m };
        mvx_t mv2 = std::move(mv);
        for (int i = 0; i < mv.nVec(); ++i)
            REQUIRE(mv2[i] == m);
        fvx_t fv{ f, f, f, f, f, f, f };
        fvx_t fv2 = std::move(fv);
        for (int i = 0; i < fv.nVec(); ++i)
            REQUIRE(fv2[i] == f);
    }
    {
        // set move-vector
        sm_t sm = { m, m, m, m, m, m, m };
        mvx_t mv;
        mv.set(std::move(sm));
        for (int i = 0; i < mv.nVec(); ++i)
            REQUIRE(mv.at(i) == m);
        sf_t sf = { f, f, f, f, f, f, f };
        fvx_t fv;
        fv.set(std::move(sf));
        for (int i = 0; i < fv.nVec(); ++i)
            REQUIRE(fv.at(i) == f);
    }
    {
        // set vector
        sm_t sm = { m, m, m, m, m, m, m };
        mvx_t mv;
        mv.set(sm);
        for (int i = 0; i < mv.nVec(); ++i)
            REQUIRE(mv.at(i) == m);
        sf_t sf = { f, f, f, f, f, f, f };
        fvx_t fv;
        fv.set(sf);
        for (int i = 0; i < fv.nVec(); ++i)
            REQUIRE(fv.at(i) == f);
    }
    {
        // set variadic
        mvx_t mv;
        mv.set(m, m, m, m, m, m, m);
        for (int i = 0; i < mv.nVec(); ++i)
            REQUIRE(mv.at(i) == m);
        fvx_t fv;
        fv.set(f, f, f, f, f, f, f);
        for (int i = 0; i < fv.nVec(); ++i)
            REQUIRE(fv.at(i) == f);
    }
    {
        // Zero
        mvx_t mv = mvx_t::Zero(dynNVec);
        for (int i = 0; i < n_vec; ++i)
            REQUIRE(mv[i] == mv_t::Zero());

        if constexpr (n_vec != Dynamic) {
            mv = mvx_t::Zero();
            for (int i = 0; i < n_vec; ++i)
                REQUIRE(mv[i] == mv_t::Zero());
        }
    }
    {
        // operator()
        mvx_t mv{ m, m, m, m, m, m, m };
        fvx_t fv{ f, f, f, f, f, f, f };
        for (int i = 0; i < mv.size(); ++i) {
            REQUIRE(mv(i) == m(i % 6));
            REQUIRE(fv(i) == f(i % 6));
        }
    }
    {
        // unary operator-
        mvx_t mv{ m, m, m, m, m, m, m };
        auto mvInv = -mv;
        REQUIRE(mvInv.vector() == -mv.vector());
        fvx_t fv{ f, f, f, f, f, f, f };
        auto fvInv = -fv;
        REQUIRE(fvInv.vector() == -fv.vector());
    }
    {
        // operator+=
        mvx_t mv1{ m, m, m, m, m, m, m };
        mvx_t mv2{ m, m, m, m, m, m, m };
        mv_t m2 = m + m;
        mvx_t sol1{ m2, m2, m2, m2, m2, m2, m2 };
        mv1 += mv2;
        REQUIRE(mv1 == sol1);
        fvx_t fv1{ f, f, f, f, f, f, f };
        fvx_t fv2{ f, f, f, f, f, f, f };
        fv_t f2 = f + f;
        fvx_t sol2{ f2, f2, f2, f2, f2, f2, f2 };
        fv1 += fv2;
        REQUIRE(fv1 == sol2);
        if constexpr (n_vec == Dynamic) {
            mv2.set(m, m, m);
            REQUIRE_THROWS_AS(mv1 += mv2, std::runtime_error);
            fv2.set(f, f, f);
            REQUIRE_THROWS_AS(fv1 += fv2, std::runtime_error);
        }
    }
    {
        // operator-=
        mvx_t mv1{ m, m, m, m, m, m, m };
        mvx_t mv2{ m, m, m, m, m, m, m };
        mv1 -= mv2;
        REQUIRE(mv1 == mvx_t::Zero(dynNVec));
        fvx_t fv1{ f, f, f, f, f, f, f };
        fvx_t fv2{ f, f, f, f, f, f, f };
        fv1 -= fv2;
        REQUIRE(fv1 == fvx_t::Zero(dynNVec));
        if constexpr (n_vec == Dynamic) {
            mv2.set(m, m, m);
            REQUIRE_THROWS_AS(mv1 -= mv2, std::runtime_error);
            fv2.set(f, f, f);
            REQUIRE_THROWS_AS(fv1 -= fv2, std::runtime_error);
        }
    }
    {
        // operator*=
        mvx_t mv{ m, m, m, m, m, m, m };
        Scalar v = 5;
        mv_t m2 = v * m;
        mv *= v;
        mvx_t sol1{ m2, m2, m2, m2, m2, m2, m2 };
        REQUIRE(mv == sol1);
        fvx_t fv{ f, f, f, f, f, f, f };
        fv_t f2 = v * f;
        fv *= v;
        fvx_t sol2{ f2, f2, f2, f2, f2, f2, f2 };
        REQUIRE(fv == sol2);
    }
    {
        // operator+
        mvx_t mv1{ m, m, m, m, m, m, m };
        mvx_t mv2{ m, m, m, m, m, m, m };
        mv_t m2 = m + m;
        mvx_t sol1{ m2, m2, m2, m2, m2, m2, m2 };
        mvx_t res = mv1 + mv2;
        REQUIRE(res == sol1);
        fvx_t fv1{ f, f, f, f, f, f, f };
        fvx_t fv2{ f, f, f, f, f, f, f };
        fv_t f2 = f + f;
        fvx_t sol2{ f2, f2, f2, f2, f2, f2, f2 };
        fvx_t res2 = fv1 + fv2;
        REQUIRE(res2 == sol2);
    }
    {
        // operator-
        mvx_t mv1{ m, m, m, m, m, m, m };
        mvx_t mv2{ m, m, m, m, m, m, m };
        mvx_t res1 = mv1 - mv2;
        REQUIRE(res1 == mvx_t::Zero(dynNVec));
        fvx_t fv1{ f, f, f, f, f, f, f };
        fvx_t fv2{ f, f, f, f, f, f, f };
        fvx_t res2 = fv1 - fv2;
        REQUIRE(res2 == fvx_t::Zero(dynNVec));
    }
    {
        // operator*
        mvx_t mv{ m, m, m, m, m, m, m };
        Scalar v = 5;
        mv_t m2 = v * m;
        mvx_t res1 = v * mv;
        mvx_t sol1{ m2, m2, m2, m2, m2, m2, m2 };
        REQUIRE(res1 == sol1);
        res1 = mv * v;
        REQUIRE(res1 == sol1);
        fvx_t fv{ f, f, f, f, f, f, f };
        fv_t f2 = v * f;
        fvx_t res2 = v * fv;
        fvx_t sol2{ f2, f2, f2, f2, f2, f2, f2 };
        REQUIRE(res2 == sol2);
        res2 = fv * v;
        REQUIRE(res2 == sol2);
    }
    {
        // operator/
        mvx_t mv{ m, m, m, m, m, m, m };
        Scalar v = 5;
        mv_t m2 = m / v;
        mvx_t res1 = mv / v;
        mvx_t sol1{ m2, m2, m2, m2, m2, m2, m2 };
        REQUIRE(res1 == sol1);
        fvx_t fv{ f, f, f, f, f, f, f };
        fv_t f2 = f / v;
        fvx_t res2 = fv / v;
        fvx_t sol2{ f2, f2, f2, f2, f2, f2, f2 };
        REQUIRE(res2 == sol2);
    }
    {
        // operator/
        mvx_t mv{ m, m, m, m, m, m, m };
        Scalar v = 5;
        mv_t m2 = m / v;
        mv /= v;
        mvx_t sol1{ m2, m2, m2, m2, m2, m2, m2 };
        REQUIRE(mv == sol1);
        fvx_t fv{ f, f, f, f, f, f, f };
        fv_t f2 = f / v;
        fv /= v;
        fvx_t sol2{ f2, f2, f2, f2, f2, f2, f2 };
        REQUIRE(fv == sol2);
    }
    {
        // operator==
        mvx_t mv1{ m, m, m, m, m, m, m };
        mvx_t mv2{ m, m, m, m, m, m, m };
        REQUIRE(mv1 == mv2);
        fvx_t fv1{ f, f, f, f, f, f, f };
        fvx_t fv2{ f, f, f, f, f, f, f };
        REQUIRE(fv1 == fv2);
        if constexpr (n_vec == Dynamic) {
            mvx_t mv3;
            fvx_t fv3;
            REQUIRE(!(mv1 == mv3));
            REQUIRE(!(fv1 == fv3));
        }
    }
    {
        // operator!=
        mv_t m2{ v6_t::Random() };
        mvx_t mv1{ m, m, m, m, m, m, m };
        mvx_t mv2{ m2, m2, m2, m2, m2, m2, m2 };
        REQUIRE(mv1 != mv2);
        fv_t f2{ v6_t::Random() };
        fvx_t fv1{ f, f, f, f, f, f, f };
        fvx_t fv2{ f2, f2, f2, f2, f2, f2, f2 };
        REQUIRE(fv1 != fv2);
        if constexpr (n_vec == Dynamic) {
            mvx_t mv3;
            fvx_t fv3;
            REQUIRE(mv1 != mv3);
            REQUIRE(fv1 != fv3);
        }
    }
    {
        // isApprox
        v6_t eps = v6_t::Ones() * 0.1 * dummy_precision<Scalar>();
        mvx_t mv1{ m, m, m, m, m, m, m };
        for (int i = 0; i < mv1.nVec(); ++i) {
            mvx_t mv2{ m, m, m, m, m, m, m };
            mv2[i] = mv_t{ m.vector() + eps };
            REQUIRE(mv1 != mv2);
            REQUIRE(mv1.isApprox(mv2));
        }
        fvx_t fv1{ f, f, f, f, f, f, f };
        for (int i = 0; i < fv1.nVec(); ++i) {
            fvx_t fv2{ f, f, f, f, f, f, f };
            fv2[i] = fv_t{ f.vector() + eps };
            REQUIRE(fv1 != fv2);
            REQUIRE(fv1.isApprox(fv2));
        }
        if constexpr (n_vec == Dynamic) {
            mvx_t mv2{ m, m, m };
            REQUIRE(!mv1.isApprox(mv2));
            fvx_t fv2{ f, f, f };
            REQUIRE(!fv1.isApprox(fv2));
        }
    }
    {
        // setZero
        mvx_t mv;
        fvx_t fv;
        mv.setZero(7); // Ok - Same size than fixed
        REQUIRE(mv == mvx_t::Zero(dynNVec));
        fv.setZero(7);
        REQUIRE(fv == fvx_t::Zero(dynNVec));
        if constexpr (n_vec != Dynamic) {
            mv[0] = m;
            mv.setZero();
            REQUIRE(mv == mvx_t::Zero());
            fv[0] = f;
            fv.setZero();
            REQUIRE(fv == fvx_t::Zero());
        }
    }
    {
        // vector
        mvx_t mv{ m, m, m, m, m, m, m };
        auto vec1 = mv.vector();
        for (int i = 0; i < mv.nVec(); ++i)
            REQUIRE(vec1.template segment<6>(6 * i) == mv[i].vector());
        fvx_t fv{ f, f, f, f, f, f, f };
        auto vec2 = fv.vector();
        for (int i = 0; i < fv.nVec(); ++i)
            REQUIRE(vec2.template segment<6>(6 * i) == fv[i].vector());
    }
}

TEST_CASE("SpatialVectorX 0-n_vec")
{
    using namespace coma;
    using mv0_t = MotionVectorX<double, 0>;
    using mvd_t = MotionVectorX<double, Dynamic>;
    using fv0_t = ForceVectorX<double, 0>;
    using fvd_t = ForceVectorX<double, Dynamic>;

    {
        // Init
        mv0_t mv0{};
        mvd_t mvd{ 0 };
        REQUIRE(mv0.nVec() == 0);
        REQUIRE(mvd.nVec() == 0);
        REQUIRE(mv0.size() == 0);
        REQUIRE(mvd.size() == 0);
        fv0_t fv0{};
        fvd_t fvd{ 0 };
        REQUIRE(fv0.nVec() == 0);
        REQUIRE(fvd.nVec() == 0);
        REQUIRE(fv0.size() == 0);
        REQUIRE(fvd.size() == 0);
    }
    {
        // Resize
        mv0_t mv0{};
        mv0.resize(0); // OK, the size is unchanged
        mvd_t mvd;
        mvd.resize(1);
        REQUIRE(mvd.nVec() == 1);
        REQUIRE(mvd.size() == 6);
        fvd_t fvd;
        fvd.resize(1);
        REQUIRE(fvd.nVec() == 1);
        REQUIRE(fvd.size() == 6);
        fvd.setZero(2);
        REQUIRE(fvd.nVec() == 2);
        REQUIRE(fvd.size() == 12);
        REQUIRE(fvd.vector() == Eigen::VectorXd::Zero(12));
    }
}