/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "coma/Core"
#include "doctest/doctest.h"

struct OrderF {
    static constexpr int n_vec = 5;
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


TEST_CASE_TEMPLATE("Spatial Inertia", Scalar, float, double)
{
    using namespace coma;
    using spi_t = SpatialInertia<Scalar>;
    using v3_t = typename spi_t::vec3_t;
    using m3_t = typename spi_t::mat3_t;
    using m6_t = typename spi_t::mat6_t;
    Scalar m = Eigen::Matrix<Scalar, 1, 1>::Random()(0);
    v3_t h = v3_t::Random();
    m3_t I = m3_t::Random();

    auto checkEq = [](const spi_t si, Scalar m2, const v3_t& h2, const m3_t& I2) {
        REQUIRE(si.mass() == m2);
        REQUIRE(si.momentum() == h2);
        REQUIRE(si.inertia() == I2);
    };

    {
        // Constructor
        spi_t si{ m, h, I };
        checkEq(si, m, h, I);
    }
    {
        // Indirect constructor
        spi_t si;
        si.mass() = m;
        si.momentum() = h;
        si.inertia() = I;
        checkEq(si, m, h, I);
    }
    {
        // operator+
        spi_t si1{ m, h, I };
        spi_t si2{ m, h, I };
        auto res = si1 + si2;
        checkEq(res, m + m, h + h, I + I);
    }
    {
        // operator-
        spi_t si1{ m, h, I };
        spi_t si2{ m, h, I };
        auto res = si1 - si2;
        REQUIRE(res.matrix() == m6_t::Zero());
    }
    {
        // unary operator-
        spi_t si1{ m, h, I };
        auto res = -si1;
        checkEq(res, -m, -h, -I);
    }
    {
        // operator+=
        spi_t si1{ m, h, I };
        spi_t res{ m, h, I };
        res += si1;
        checkEq(res, m + m, h + h, I + I);
    }
    {
        // operator-=
        spi_t si1{ m, h, I };
        spi_t res{ m, h, I };
        res -= si1;
        REQUIRE(res.matrix() == m6_t::Zero());
    }
    {
        // operator*
        spi_t si{ m, h, I };
        Scalar s = 5;
        auto res = si * s;
        checkEq(res, m * s, h * s, I * s);
        res = s * si;
        checkEq(res, m * s, h * s, I * s);
    }
    {
        // operator*=
        spi_t si{ m, h, I };
        Scalar s = 5;
        si *= s;
        checkEq(si, m * s, h * s, I * s);
    }
    {
        // operator ==
        spi_t si1{ m, h, I };
        spi_t si2{ m, h, I };
        REQUIRE(si1 == si2);
    }
    {
        // operator !=
        spi_t si1{ m, h, I };
        spi_t si2{ m + 1, h, I };
        REQUIRE(si1 != si2);
    }
    {
        // isApprox
        v3_t eps = Scalar(0.1) * v3_t::Ones() * dummy_precision<Scalar>();
        spi_t si1{ m, h, I };
        spi_t si2{ m, h + eps, I };
        REQUIRE(si1 != si2);
        REQUIRE(si1.isApprox(si2));
    }
}

TEST_CASE_TEMPLATE("Spatial operators", T, test_pairs)
{
    using namespace coma;
    using Scalar = typename T::first_type;
    constexpr int n_vec = T::second_type::n_vec;
    int dyn_n_vec = OrderF::n_vec;
    using m3_t = Eigen::Matrix<Scalar, 3, 3>;
    using mv_t = MotionVector<Scalar>;
    m3_t mat = m3_t::Random();
    Eigen::Matrix<Scalar, 6, 2> SMat = Eigen::Matrix<Scalar, 6, 2>::Random();
    mv_t m{ Eigen::Matrix<Scalar, 6, 1>::Random() };
    SpatialInertia<Scalar> I{ m.angular()[0], m.linear(), mat };

    {
        MotionSubspace<Scalar> S;
        S.matrix() = SMat;
        REQUIRE(S.matrix() == SMat);
    }
    {
        MotionSubspace<Scalar> S{ SMat };
        REQUIRE(S.matrix() == SMat);
    }
    {
        MotionSubspace<Scalar> S{ SMat };
        Eigen::Matrix<Scalar, -1, -1> dq = Eigen::Matrix<Scalar, 2, 1>::Random();
        auto res = S * dq;
        REQUIRE(res.vector() == SMat * dq);
    }
    {
        DiInertia<Scalar, n_vec> diagI;
        DiMotionSubspace<Scalar, n_vec> G;
        MotionSubspace<Scalar> S{ SMat };
        diagI.block() = I;
        G.block() = S;
        REQUIRE(I == diagI.block());
        REQUIRE(S == G.block());

        Scalar eps = dummy_precision<Scalar>() * Scalar(0.5);
        diagI.block().mass() += eps;
        REQUIRE(I != diagI.block());
        REQUIRE(I.isApprox(diagI.block()));
        G.block().matrix()(0, 0) += eps;
        REQUIRE(S != G.block());
        REQUIRE(S.isApprox(G.block()));
    }
    {
        MotionSubspace<Scalar> S{ SMat };
        DiInertia<Scalar, n_vec> diagI{ I };
        DiMotionSubspace<Scalar, n_vec> G{ S };
        REQUIRE(I == diagI.block());
        REQUIRE(S == G.block());
    }
    {
        MotionSubspace<Scalar> S{ SMat };
        DiInertia<Scalar, n_vec> diagI{ I };
        DiMotionSubspace<Scalar, n_vec> G{ S };
        REQUIRE(diagI.matrix(4).template bottomRightCorner<6, 6>() == I.matrix());
        REQUIRE(G.matrix(4).bottomRightCorner(SMat.rows(), SMat.cols()) == SMat);
        if constexpr (n_vec >= 0) {
            REQUIRE(diagI.matrix().size() == 36 * n_vec * n_vec);
            REQUIRE(G.matrix().size() == 12 * n_vec * n_vec);
        } else {
            REQUIRE(diagI.matrix(dyn_n_vec).size() == 36 * dyn_n_vec * dyn_n_vec);
            REQUIRE(G.matrix(dyn_n_vec).size() == 12 * dyn_n_vec * dyn_n_vec);
        }
    }
    {
        MotionSubspace<Scalar> S{ SMat };
        DiInertia<Scalar, n_vec> diagI{ I };
        DiMotionSubspace<Scalar, n_vec> G{ S };
        MotionVectorX<Scalar, n_vec> mx{ m, m, m, m, m };
        ForceVectorX<Scalar, n_vec> fx = diagI * mx;
        Eigen::Matrix<Scalar, -1, 1> dq2 = Eigen::Matrix<Scalar, 10, 1>::Random();
        MotionVectorX<Scalar, n_vec> mx2 = G * dq2;
        if constexpr (n_vec >= 0) {
            REQUIRE(fx.vector().isApprox(diagI.matrix() * mx.vector()));
            REQUIRE(mx2.vector() == G.matrix() * dq2);
        } else {
            REQUIRE(fx.vector().isApprox(diagI.matrix(dyn_n_vec) * mx.vector()));
            REQUIRE(mx2.vector() == G.matrix(dyn_n_vec) * dq2);
        }
    }
}