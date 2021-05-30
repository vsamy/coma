/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "benchmark/benchmark.h"
#include <coma/Core>

template <int Order>
static void BM_STATIC_CMTM_MUL(benchmark::State& state)
{
    coma::CMTM<double, 6, Order> C1, C2, C;
    C1.transform() = coma::Transform<double>(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
    C2.transform() = coma::Transform<double>(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
    for (int i = 0; i < Order; ++i) {
        C1.motion()[i] = coma::MotionVector<double>(Eigen::Vector6d::Random());
        C2.motion()[i] = coma::MotionVector<double>(Eigen::Vector6d::Random());
    }

    // Code inside this loop is measured repeatedly
    C1.construct();
    C2.construct();
    for (auto _ : state) {
        benchmark::DoNotOptimize(C = C1 * C2);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 0);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 1);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 2);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 3);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 4);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 5);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 6);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 7);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 8);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 9);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL, 10);

static void BM_DYNAMIC_CMTM_MUL(benchmark::State& state)
{
    auto order = state.range(0);
    coma::CMTM<double, 6, coma::Dynamic> C1(order), C2(order), C(order);
    C1.transform() = coma::Transform<double>(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
    C2.transform() = coma::Transform<double>(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
    for (int i = 0; i < order; ++i) {
        C1.motion()[i] = coma::MotionVector<double>(Eigen::Vector6d::Random());
        C2.motion()[i] = coma::MotionVector<double>(Eigen::Vector6d::Random());
    }

    // Code inside this loop is measured repeatedly
    C1.construct();
    C2.construct();
    for (auto _ : state) {
        benchmark::DoNotOptimize(C = C1 * C2);
        benchmark::ClobberMemory();
    }
}
// Register the function as a benchmark
BENCHMARK(BM_DYNAMIC_CMTM_MUL)->DenseRange(0, 10, 1);

template <int Order>
static void BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT(benchmark::State& state)
{
    coma::CMTM<double, 6, Order> C1, C2, C;
    coma::Transform<double> T1(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()), T2(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
    coma::MotionVectorX<double, Order> m1, m2, res;
    for (int i = 0; i < Order; ++i) {
        m1[i] = coma::MotionVector<double>(Eigen::Vector6d::Random());
        m2[i] = coma::MotionVector<double>(Eigen::Vector6d::Random());
    }

    // Code inside this loop is measured repeatedly
    for (auto _ : state) {
        C1.set(T1, m1);
        C2.set(T2, m2);
        C = C1 * C2;
        benchmark::DoNotOptimize(res = C.motion()); // Call deconstruct
        benchmark::ClobberMemory();
    }
}
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 0);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 1);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 2);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 3);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 4);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 5);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 6);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 7);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 8);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 9);
BENCHMARK_TEMPLATE(BM_STATIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT, 10);

static void BM_DYNAMIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT(benchmark::State& state)
{
    auto order = state.range(0);
    coma::CMTM<double, 6, coma::Dynamic> C1(order), C2(order), C(order);
    coma::Transform<double> T1(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()), T2(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
    coma::MotionVectorX<double, coma::Dynamic> m1(order), m2(order), res(order);
    for (int i = 0; i < order; ++i) {
        m1[i] = coma::MotionVector<double>(Eigen::Vector6d::Random());
        m2[i] = coma::MotionVector<double>(Eigen::Vector6d::Random());
    }

    // Code inside this loop is measured repeatedly
    for (auto _ : state) {
        C1.set(T1, m1);
        C2.set(T2, m2);
        C = C1 * C2;
        benchmark::DoNotOptimize(res = C.motion()); // Call deconstruct
        benchmark::ClobberMemory();
    }
}
// Register the function as a benchmark
BENCHMARK(BM_DYNAMIC_CMTM_MUL_CONSTRUCT_DECONSTRUCT)->DenseRange(0, 10, 1);

// Run the benchmark
BENCHMARK_MAIN();