/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "coma/Core"
#include "doctest/doctest.h"

TEST_CASE("math utility tools")
{
    using namespace coma;
    constexpr int max_test_val = 10;
    {
        // Check Factorial function
        int factor = 1;
        REQUIRE(Factorial(0) == 1);
        for (int i = 1; i < max_test_val; ++i) {
            factor *= i;
            REQUIRE(Factorial(i) == factor);
        }
    }
    {
        // Check binomial function
        for (int n = 0; n < max_test_val; ++n)
            for (int k = 0; k <= n; ++k)
                REQUIRE(Binomial(n, k) == Factorial(n) / (Factorial(k) * Factorial(n - k)));
    }

    // Check factors
    {
        // Check Pascal factors
        size_t pos = 0;
        const auto& pFactors = pascal_factors<int, max_test_val>;
        REQUIRE(pFactors.size() == max_test_val * (max_test_val + 1) / 2);
        for (int n = 0; n < max_test_val; ++n) {
            for (int k = 0; k <= n; ++k) {
                REQUIRE(pFactors[pos++] == Binomial(n, k));
            }
        }

        // Check Factorial factors
        const auto& fFactors = factorial_factors<size_t, max_test_val>;
        REQUIRE(fFactors.size() == max_test_val);
        for (size_t n = 0; n < max_test_val; ++n)
            REQUIRE(fFactors[n] == Factorial(n));
    }
}

TEST_CASE_TEMPLATE("Cross utility tools", T, float, double)
{
    using namespace coma;
    using v3_t = Eigen::Matrix<T, 3, 1>;
    using v4_t = Eigen::Matrix<T, 4, 1>;
    using v6_t = Eigen::Matrix<T, 6, 1>;
    using m3_t = Eigen::Matrix<T, 3, 3>;
    using m4_t = Eigen::Matrix<T, 4, 4>;
    using m6_t = Eigen::Matrix<T, 6, 6>;
    using m8_t = Eigen::Matrix<T, 8, 8>;
    using m12_t = Eigen::Matrix<T, 12, 12>;
    using mv_t = MotionVector<T>;
    mv_t n{ v3_t::Random(), v3_t::Random() };
    v4_t v4 = v4_t::Random();
    v6_t v6 = v6_t::Random();

    auto cross4 = [](const auto& v1, const auto& v2) {
        v4_t out;
        out << v1.template head<3>().cross(v2.template head<3>()) + v1.template tail<3>() * v2(3), 0;
        return out;
    };
    auto cross6 = [](const auto& v1, const auto& v2) {
        v6_t out;
        out << v1.template head<3>().cross(v2.template head<3>()), v1.template head<3>().cross(v2.template tail<3>()) + v1.template tail<3>().cross(v2.template head<3>());
        return out;
    };
    auto cross6D = [](const auto& v1, const auto& v2) {
        v6_t out;
        out << v1.template head<3>().cross(v2.template head<3>()) + v1.template tail<3>().cross(v2.template tail<3>()), v1.template head<3>().cross(v2.template tail<3>());
        return out;
    };

    {
        // Check [vx_3]v
        v3_t v1 = v6.template head<3>();
        v3_t v2 = v6.template tail<3>();
        REQUIRE((vector3ToCrossMatrix3(v1) * v2).isApprox(v1.cross(v2))); // Eigen Vector style
        REQUIRE((vector3ToCrossMatrix3(v6.template head<3>()) * v6.template tail<3>()).isApprox(v1.cross(v2))); // Eigen block Style
    }
    {
        // Check [vx_4]v
        Eigen::Matrix<T, 8, 1> v8 = Eigen::Matrix<T, 8, 1>::Random();
        REQUIRE((vector6ToCrossMatrix4(n) * v4).isApprox(cross4(n.vector(), v4))); // MotionVec style
        REQUIRE((vector6ToCrossMatrix4(v6) * v4).isApprox(cross4(v6, v4))); // Eigen Vector style
        REQUIRE((vector6ToCrossMatrix4(v8.template head<6>()) * v4).isApprox(cross4(v8.template head<6>(), v4))); // Eigen block Style
    }
    {
        // Check [vx_6]v
        Eigen::Matrix<T, 8, 1> v8 = Eigen::Matrix<T, 8, 1>::Random();
        REQUIRE((vector6ToCrossMatrix6(n) * v6).isApprox(cross6(n.vector(), v6))); // MotionVec style
        REQUIRE((vector6ToCrossMatrix6(v6) * v8.template head<6>()).isApprox(cross6(v6, v8.template head<6>()))); // Eigen Vector style
        REQUIRE((vector6ToCrossMatrix6(v8.template head<6>()) * v6).isApprox(cross6(v8.template head<6>(), v6))); // Eigen block Style
    }
    {
        // Check [vx^*_6]v
        Eigen::Matrix<T, 8, 1> v8 = Eigen::Matrix<T, 8, 1>::Random();
        REQUIRE((vector6ToCrossDualMatrix6(n) * v6).isApprox(cross6D(n.vector(), v6))); // ForceVec style
        REQUIRE((vector6ToCrossDualMatrix6(v6) * v8.template head<6>()).isApprox(cross6D(v6, v8.template head<6>()))); // Eigen Vector style
        REQUIRE((vector6ToCrossDualMatrix6(v8.template head<6>()) * v6).isApprox(cross6D(v8.template head<6>(), v6))); // Eigen block Style
    }
    {
        // Check [vx_3] <-> v
        v3_t v3 = v6.template head<3>();
        m3_t m3 = vector3ToCrossMatrix3(v3);
        m6_t m6;
        m6 << m3, m3, m3, m3;
        REQUIRE(crossMatrix3ToVector3(m3) == v3);
        REQUIRE(crossMatrix3ToVector3(m6.template block<3, 3>(0, 0)) == v3); // Eigen block Style
    }
    {
        // Check [vx_4] <-> v
        m4_t m4 = vector6ToCrossMatrix4(n);
        m8_t m8;
        m8 << m4, m4, m4, m4;
        REQUIRE(uncross<mv_t>::crossMatrix4ToVector6(m4) == n); // Eigen Vector Style to MotionVec
        REQUIRE(uncross<mv_t>::crossMatrix4ToVector6(m8.template block<4, 4>(0, 0)) == n); // Eigen block Style to MotionVec
        REQUIRE(uncross<v6_t>::crossMatrix4ToVector6(m4) == n.vector()); // Eigen Vector Style to Eigen Vector
        REQUIRE(uncross<v6_t>::crossMatrix4ToVector6(m8.template block<4, 4>(0, 0)) == n.vector()); // Eigen block Style to Eigen Vector
    }
    {
        // Check [vx_6] <-> v
        m6_t m6 = vector6ToCrossMatrix6(n);
        m12_t m12;
        m12 << m6, m6, m6, m6;
        REQUIRE(uncross<mv_t>::crossMatrix6ToVector6(m6) == n); // Eigen Vector Style to MotionVec
        REQUIRE(uncross<mv_t>::crossMatrix6ToVector6(m12.template block<6, 6>(0, 0)) == n); // Eigen block Style to MotionVec
        REQUIRE(uncross<v6_t>::crossMatrix6ToVector6(m6) == n.vector()); // Eigen Vector Style to Eigen Vector
        REQUIRE(uncross<v6_t>::crossMatrix6ToVector6(m12.template block<6, 6>(0, 0)) == n.vector()); // Eigen block Style to Eigen Vector
    }
    {
        // Check [vx^*_6] <-> v
        m6_t m6 = vector6ToCrossDualMatrix6(n);
        m12_t m12;
        m12 << m6, m6, m6, m6;
        REQUIRE(uncross<mv_t>::crossDualMatrix6ToVector6(m6) == n); // Eigen Vector Style to ForceVec
        REQUIRE(uncross<mv_t>::crossDualMatrix6ToVector6(m12.template block<6, 6>(0, 0)) == n); // Eigen block Style to ForceVec
        REQUIRE(uncross<v6_t>::crossDualMatrix6ToVector6(m6) == n.vector()); // Eigen Vector Style to Eigen Vector
        REQUIRE(uncross<v6_t>::crossDualMatrix6ToVector6(m12.template block<6, 6>(0, 0)) == n.vector()); // Eigen block Style to Eigen Vector
    }
}