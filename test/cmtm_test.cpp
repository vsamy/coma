/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "coma/Core"
#include "macros.hpp"
#include "doctest/doctest.h"
#include <Eigen/QR>
#include <tuple>

struct Space4 {
    static constexpr int space = 4;
};

struct Space6 {
    static constexpr int space = 6;
};

struct OrderF {
    static constexpr int order = 5;
};

struct OrderD {
    static constexpr int order = coma::Dynamic;
};

template <typename T1, typename T2, typename T3>
struct TypeTriple
{
    using first_type = T1;
    using second_type = T2;
    using third_type = T3;
};

#define test_triples \
    TypeTriple<float, Space4, OrderF>, \
    TypeTriple<double, Space4, OrderF>, \
    TypeTriple<float, Space6, OrderF>, \
    TypeTriple<double, Space6, OrderF>, \
    TypeTriple<float, Space4, OrderD>, \
    TypeTriple<double, Space4, OrderD>, \
    TypeTriple<float, Space6, OrderD>, \
    TypeTriple<double, Space6, OrderD>

class CMTM24 {
    using PMat = coma::Transform<double>;
    using MVd = coma::MotionVector<double>;

public:
    CMTM24(const PMat& pt, const MVd& n, const MVd& dn, const MVd& ddn)
        : A(pt)
        , nu(n)
        , dnu(dn)
        , ddnu(ddn)
    {
    }

    friend CMTM24 operator*(const CMTM24& lhs, const CMTM24& rhs)
    {
        MVd Anu = rhs.A.invMul(lhs.nu);
        MVd Adnu = rhs.A.invMul(lhs.dnu);

        PMat A = lhs.A * rhs.A;
        MVd nu = Anu + rhs.nu;
        MVd dnu = Adnu + rhs.dnu + Anu.cross(rhs.nu);
        MVd ddnu = rhs.A.invMul(lhs.ddnu) + rhs.ddnu + (Anu.cross(rhs.nu) + 2. * Adnu).cross(rhs.nu) + Anu.cross(rhs.dnu);
        return CMTM24{ A, nu, dnu, ddnu };
    }

public:
    PMat A;
    MVd nu;
    MVd dnu;
    MVd ddnu;
};

TEST_CASE_TEMPLATE("CMTM", T, test_triples)
{
    using namespace coma;
    using Scalar = typename T::first_type;
    constexpr int space = T::second_type::space;
    constexpr int order = T::third_type::order;
    constexpr int dynOrder = OrderF::order;
    using qt_t = Eigen::Quaternion<Scalar>;
    using v3_t = Eigen::Matrix<Scalar, 3, 1>;
    using v6_t = Eigen::Matrix<Scalar, 6, 1>;
    using m3_t = Eigen::Matrix<Scalar, 3, 3>;
    using mv_t = MotionVector<Scalar>;
    using cmtm_t = CMTM<Scalar, space, order>;
    using transform_t = typename cmtm_t::transform_t;
    using mvx_t = typename cmtm_t::mvx_t;

    DISABLE_CONVERSION_WARNING_BEGIN
    qt_t q = qt_t::UnitRandom();
    DISABLE_CONVERSION_WARNING_END

    m3_t R = q.toRotationMatrix();
    v3_t p = v3_t::Random();

    transform_t A{ R, p };
    mv_t m{ v6_t::Random() };
    mvx_t mx{ m, m, m, m, m };

    {
        // default ctor
        cmtm_t cmtm;
        if constexpr (order == Dynamic) {
            cmtm_t cmtm2{ dynOrder };
            REQUIRE(cmtm2.order() == dynOrder);
            REQUIRE(cmtm2.nMat() == dynOrder + 1);
            REQUIRE(cmtm2.rows() == space * (dynOrder + 1));
            REQUIRE(cmtm2.rows() == cmtm2.cols());
            REQUIRE(cmtm2.size() == static_cast<int>(std::pow((dynOrder + 1) * space, 2)));
            REQUIRE(cmtm2.matrix().size() == cmtm2.size());
            REQUIRE(cmtm2.matrix(dynOrder / 2) == cmtm2.matrix().template topLeftCorner<space*(dynOrder / 2), space*(dynOrder / 2)>());
            if constexpr (space == 6) {
                REQUIRE(cmtm2.dualMatrix().size() == cmtm2.size());
                REQUIRE(cmtm2.dualMatrix(dynOrder / 2) == cmtm2.dualMatrix().template topLeftCorner<6 * (dynOrder / 2), 6 * (dynOrder / 2)>());
            }
        } else {
            REQUIRE(cmtm.order() == order);
            REQUIRE(cmtm.nMat() == order + 1);
            REQUIRE(cmtm.rows() == space * (order + 1));
            REQUIRE(cmtm.rows() == cmtm.cols());
            REQUIRE(cmtm.size() == static_cast<int>(std::pow((order + 1) * space, 2)));
            REQUIRE(cmtm.matrix().size() == cmtm.size());
            REQUIRE(cmtm.template matrix<order / 2>() == cmtm.matrix().template topLeftCorner<space*(order / 2), space*(order / 2)>());
            if constexpr (space == 6) {
                REQUIRE(cmtm.dualMatrix().size() == cmtm.size());
                REQUIRE(cmtm.template dualMatrix<order / 2>() == cmtm.dualMatrix().template topLeftCorner<6 * (order / 2), 6 * (order / 2)>());
            }
        }
    }
    {
        // Identity
        cmtm_t cmtm;
        cmtm = cmtm_t::Identity(5);
        REQUIRE(cmtm.transform() == transform_t::Identity());
        REQUIRE(cmtm.motion() == mvx_t::Zero(5));
        if constexpr (order != Dynamic) {
            cmtm = cmtm_t::Identity();
            REQUIRE(cmtm.transform() == transform_t::Identity());
            REQUIRE(cmtm.motion() == mvx_t::Zero());
        }
    }
    {
        // setIdentity
        cmtm_t cmtm;
        cmtm.setIdentity(dynOrder);
        REQUIRE(cmtm == cmtm_t::Identity(dynOrder));
        if constexpr (order != Dynamic) {
            cmtm.transform() = A;
            cmtm.motion()[0] = m;
            cmtm.setIdentity();
            REQUIRE(cmtm == cmtm_t::Identity());
        }
    }
    {
        // lvalues ctor with tangent vector
        cmtm_t cmtm{ A, mx };
        REQUIRE(cmtm.transform() == A);
        REQUIRE(cmtm.motion() == mx);
    }
    {
        // lvalues ctor without tangent vector
        cmtm_t cmtm{ A, m, m, m, m, m };
        REQUIRE(cmtm.transform() == A);
        REQUIRE(cmtm.motion() == mx);
    }
    {
        // rvalues ctor with tangent vector
        cmtm_t cmtm{ transform_t{ q, p }, mvx_t{ m, m, m, m, m } };
        REQUIRE(cmtm.transform() == A);
        REQUIRE(cmtm.motion() == mx);
    }
    {
        // rvalues ctor without tangent vector
        mvx_t smi{ v6_t::Ones(), v6_t::Ones(), v6_t::Ones(), v6_t::Ones(), v6_t::Ones() };
        cmtm_t cmtm{ transform_t{ q, p }, v6_t::Ones(), v6_t::Ones(), v6_t::Ones(), v6_t::Ones(), v6_t::Ones() };
        REQUIRE(cmtm.transform() == A);
        REQUIRE(cmtm.motion() == smi);
    }
    {
        // copy ctor
        cmtm_t cmtm1{ A, m, m, m, m, m };
        cmtm_t cmtm2{ cmtm1 };
        REQUIRE(cmtm2.transform() == A);
        REQUIRE(cmtm2.motion() == mx);
    }
    {
        // move ctor
        cmtm_t cmtm1{ A, m, m, m, m, m };
        cmtm_t cmtm2{ std::move(cmtm1) };
        REQUIRE(cmtm2.transform() == A);
        REQUIRE(cmtm2.motion() == mx);
    }
    {
        // assign op
        cmtm_t cmtm1{ A, m, m, m, m, m };
        cmtm_t cmtm2;
        cmtm2 = cmtm1;
        REQUIRE(cmtm2.transform() == A);
        REQUIRE(cmtm2.motion() == mx);
    }
    {
        // move-assign op
        cmtm_t cmtm1{ A, m, m, m, m, m };
        cmtm_t cmtm2;
        cmtm2 = std::move(cmtm1);
        REQUIRE(cmtm2.transform() == A);
        REQUIRE(cmtm2.motion() == mx);
    }
    {
        // lvalues set with tangent vector
        cmtm_t cmtm;
        cmtm.set(A, mx);
        REQUIRE(cmtm.transform() == A);
        REQUIRE(cmtm.motion() == mx);
    }
    {
        // lvalues set without tangent vector
        cmtm_t cmtm;
        cmtm.set(A, m, m, m, m, m);
        REQUIRE(cmtm.transform() == A);
        REQUIRE(cmtm.motion() == mx);
    }
    {
        // rvalues ctor with tangent vector
        cmtm_t cmtm;
        cmtm.set(transform_t{ q, p }, mvx_t{ m, m, m, m, m });
        REQUIRE(cmtm.transform() == A);
        REQUIRE(cmtm.motion() == mx);
    }
    {
        // rvalues ctor without tangent vector
        mvx_t smi{ v6_t::Ones(), v6_t::Ones(), v6_t::Ones(), v6_t::Ones(), v6_t::Ones() };
        cmtm_t cmtm;
        cmtm.set(transform_t{ q, p }, v6_t::Ones(), v6_t::Ones(), v6_t::Ones(), v6_t::Ones(), v6_t::Ones());
        REQUIRE(cmtm.transform() == A);
        REQUIRE(cmtm.motion() == smi);
    }
    {
        // deconstruction
        cmtm_t cmtm{ A, mx };
        cmtm.deconstruct();
        REQUIRE(cmtm.transform() == A);
        REQUIRE(cmtm.motion().isApprox(mx));
    }
    {
        // operator==(CMTM6n)
        cmtm_t cmtm1{ A, m, m, m, m, m };
        cmtm_t cmtm2{ A, m, m, m, m, m };
        REQUIRE(cmtm1 == cmtm2);
    }
    {
        // operator!=(CMTM6n)
        cmtm_t cmtm1{ A, m, m, m, m, m };
        cmtm_t cmtm2{ transform_t::Identity(), m, m, m, m, m };
        REQUIRE(cmtm1 != cmtm2);
        for (int i = 0; i < order; ++i) {
            mvx_t smt{ m, m, m, m, m };
            smt[i] = mv_t::Zero();
            cmtm_t cmtm3{ A, smt };
            REQUIRE(cmtm1 != cmtm3);
        }
    }
    {
        // isApprox(CMTM6n)
        m3_t m_eps = m3_t::Ones() * 0.1 * dummy_precision<Scalar>();
        v6_t v_eps = v6_t::Ones() * 0.1 * dummy_precision<Scalar>();
        cmtm_t cmtm1{ A, m, m, m, m, m };
        cmtm_t cmtm2{ transform_t{ R + m_eps, p }, m, m, m, m, m };
        REQUIRE(cmtm1 != cmtm2);
        REQUIRE(cmtm1.isApprox(cmtm2));
        for (int i = 0; i < order; ++i) {
            mvx_t smt{ m, m, m, m, m };
            smt[i] = mv_t{ m.vector() + v_eps };
            cmtm_t cmtm3{ A, smt };
            REQUIRE(cmtm1 != cmtm3);
            REQUIRE(cmtm1.isApprox(cmtm2));
        }
    }
}

TEST_CASE("CMTM 0-order")
{
    using namespace coma;
    using cmtm0_t = CMTM<double, 4, 0>;
    using cmtmd_t = CMTM<double, 4, Dynamic>;

    {
        // Init
        cmtm0_t cmtm0{};
        cmtmd_t cmtmd{ 0 };
        REQUIRE(cmtm0.order() == 0);
        REQUIRE(cmtmd.order() == 0);
        REQUIRE(cmtm0.nMat() == 1);
        REQUIRE(cmtmd.nMat() == 1);
        REQUIRE(cmtm0.rows() == 4);
        REQUIRE(cmtmd.rows() == 4);
        REQUIRE(cmtm0.cols() == 4);
        REQUIRE(cmtmd.cols() == 4);
        REQUIRE(cmtm0.size() == 16);
        REQUIRE(cmtmd.size() == 16);
        REQUIRE(cmtm0.matrix().size() == cmtm0.size());
        REQUIRE(cmtmd.matrix().size() == cmtmd.size());
    }
    {
        // Resize
        cmtm0_t cmtm0{};
        cmtmd_t cmtmd;
        cmtm0.resize(0);
        cmtmd.resize(1);
        REQUIRE(cmtmd.order() == 1);
        REQUIRE(cmtmd.nMat() == 2);
        REQUIRE(cmtmd.rows() == 8);
        REQUIRE(cmtmd.cols() == 8);
        REQUIRE(cmtmd.size() == 64);
        REQUIRE(cmtmd.matrix().size() == cmtmd.size());
    }
}

TEST_CASE_TEMPLATE("CMTM operations", T, Space4, Space6)
{
    using namespace coma;
    constexpr int space = T::space;
    constexpr int order = 3;
    Eigen::Matrix3d R = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    Eigen::Vector3d p = Eigen::Vector3d::Random();
    Transform<double> A{ R, p };
    MotionVector<double> nu{ Eigen::Vector6d::Random() };
    MotionVector<double> dnu{ Eigen::Vector6d::Random() };
    MotionVector<double> ddnu{ Eigen::Vector6d::Random() };

    CMTM24 c1{ Transform<double>{ R, p }, nu, dnu, ddnu };
    CMTM24 c2{ Transform<double>{ R, p }, nu, dnu, ddnu };
    MotionVectorX<double, order> mvx1{ nu, dnu, ddnu };
    MotionVectorX<double, order> mvx2{ nu, dnu, ddnu };
    CMTM<double, space, order> c3{ A, mvx1 };
    CMTM<double, space, order> c4{ A, mvx2 };

    auto c12 = c1 * c2;
    auto c34 = c3 * c4;
    REQUIRE(c34.order() == order);

    auto checkEq = [](auto c1, auto c2) {
        REQUIRE(c1.angular().isApprox(c2.angular()));
        REQUIRE(c1.linear().isApprox(c2.linear()));
    };

    auto motions34 = c34.motion();

    // Multiplication test
    REQUIRE(c12.A.rotation() == c34.transform().rotation());
    REQUIRE(c12.A.translation() == c34.transform().translation());
    checkEq(c12.nu, motions34[0]);
    checkEq(c12.dnu, motions34[1]);
    checkEq(c12.ddnu, motions34[2]);

    // Matrix test
    REQUIRE(c34.matrix().isApprox(c3.matrix() * c4.matrix()));

    // Dual matrix test
    if constexpr (space == 6) {
        REQUIRE(c34.dualMatrix().isApprox(c3.dualMatrix() * c4.dualMatrix()));
        Eigen::Vector6d randVec = Eigen::Vector6d::Random();
        ForceVector<double> fv{ randVec };
        MotionVector<double> mv{ randVec };

        const auto& factors = factorial_factors<double, order + 1>;
        MotionVectorX<double, order + 1> v{ mv, mv, mv, mv };
        ForceVectorX<double, order + 1> f{ fv, fv, fv, fv };
        Eigen::Matrix<double, 6 * (order + 1), 1> v2, f2;
        v2 << randVec, randVec, randVec, randVec;
        f2 << randVec, randVec, randVec, randVec;
        for (int i = 0; i < order + 1; ++i) {
            size_t ui = static_cast<size_t>(i);
            v2.segment<6>(6 * i) /= factors[ui];
            f2.segment<6>(6 * i) /= factors[ui];
        }
        Eigen::VectorXd vr = c3.matrix() * v2;
        Eigen::VectorXd fr = c3.dualMatrix() * f2;
        for (int i = 0; i < order + 1; ++i) {
            size_t ui = static_cast<size_t>(i);
            vr.segment<6>(6 * i) *= factors[ui];
            fr.segment<6>(6 * i) *= factors[ui];
        }

        auto c3v = c3 * v;
        auto c3f = c3.dualMul(f);
        REQUIRE(c3v.vector().isApprox(vr));
        REQUIRE(c3f.vector().isApprox(fr));
    }

    // Inverse test
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{ c3.rows(), c3.cols() };
    Eigen::MatrixXd c3Inv = c3.inverse().matrix();
    Eigen::MatrixXd m3Inv = qr.compute(c3.matrix()).inverse();
    Eigen::MatrixXd c34Inv = c34.inverse().matrix();
    Eigen::MatrixXd m34Inv = qr.compute(c34.matrix()).inverse();

    REQUIRE(c3.inverse().matrix().isApprox(qr.compute(c3.matrix()).inverse(), dummy_precision<double>()));
    REQUIRE(c34.inverse().matrix().isApprox(qr.compute(c34.matrix()).inverse(), dummy_precision<double>()));

    // test for Dynamic
    MotionVectorX<double, Dynamic> mvxd{ nu, dnu, ddnu };
    CMTM<double, space, Dynamic> c5{ A, mvxd };
    CMTM<double, space, Dynamic> c6;
    c6.set(A, nu, dnu, ddnu);

    auto c56 = c5 * c6;

    auto motions56 = c56.motion();

    // Test
    REQUIRE(c12.A.rotation() == c56.transform().rotation());
    REQUIRE(c12.A.translation() == c56.transform().translation());
    checkEq(c12.nu, motions56[0]);
    checkEq(c12.dnu, motions56[1]);
    checkEq(c12.ddnu, motions56[2]);

    c6.set(A, nu, dnu, ddnu, MotionVector<double>{ Eigen::Vector6d::Random() });
    REQUIRE_THROWS_AS(c5 * c6, std::runtime_error);
}
