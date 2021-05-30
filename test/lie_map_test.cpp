/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "coma/Core"
#include "macros.hpp"
#include "doctest/doctest.h"

TEST_CASE_TEMPLATE("mapSO3", T, float, double)
{
    using namespace coma;
    using qt_t = Eigen::Quaternion<T>;
    using m3_t = Eigen::Matrix<T, 3, 3>;
    using v3_t = Eigen::Matrix<T, 3, 1>;

    DISABLE_CONVERSION_WARNING_BEGIN
    qt_t q = qt_t::UnitRandom();
    DISABLE_CONVERSION_WARNING_END

    m3_t R = q.toRotationMatrix();
    {
        // Check random
        v3_t omega = logSO3(R);
        m3_t R2 = expSO3(omega);
        REQUIRE(R.isApprox(R2));
    }
    {
        // Check identity
        v3_t omega = logSO3(m3_t::Identity());
        m3_t R2 = expSO3(omega);
        REQUIRE(R2.isApprox(m3_t::Identity()));
    }
}

TEST_CASE_TEMPLATE("mapSE3", T, float, double)
{
    using namespace coma;
    using qt_t = Eigen::Quaternion<T>;
    using v3_t = Eigen::Matrix<T, 3, 1>;
    using tf_t = Transform<T>;

    DISABLE_CONVERSION_WARNING_BEGIN
    qt_t q = qt_t::UnitRandom();
    DISABLE_CONVERSION_WARNING_END

    tf_t tf{ q.toRotationMatrix(), v3_t::Random() };
    {
        // Check random on sva
        auto m = motionLogSE3<T>::map(tf);
        tf_t tf2 = expSE3(m);
        REQUIRE(tf.isApprox(tf2));
    }
    {
        // Check random on eigen
        auto m = eigenLogSE3<T>::map(tf);
        tf_t tf2 = expSE3(m);
        REQUIRE(tf.isApprox(tf2));
    }
    {
        // Check identity
        auto m = motionLogSE3<T>::map(tf_t::Identity());
        auto tf2 = expSE3(m);
        REQUIRE(tf2.isApprox(tf_t::Identity()));
    }
}

TEST_CASE_TEMPLATE("mapS3", T, float, double)
{
    using namespace coma;
    using qt_t = Eigen::Quaternion<T>;
    using v3_t = Eigen::Matrix<T, 3, 1>;

    DISABLE_CONVERSION_WARNING_BEGIN
    qt_t q = qt_t::UnitRandom();
    DISABLE_CONVERSION_WARNING_END

    {
        // Check random on eigen
        v3_t w = logSU2(q);
        qt_t q2 = expSU2(w);
        REQUIRE(q.isApprox(q2));
    }
    {
        // Check identity
        v3_t w = logSU2(qt_t::Identity());
        REQUIRE(w == v3_t::Zero());
        qt_t q2 = expSU2(v3_t::Zero());
        REQUIRE(q2.angularDistance(qt_t::Identity()) == T(0));
    }
}
