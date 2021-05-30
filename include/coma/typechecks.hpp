/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

namespace internal {

// Based on https://stackoverflow.com/a/41449096/9774052
template <typename T>
std::true_type is_mv_impl(MotionVector<T>);
std::false_type is_mv_impl(...);
template <typename T, int N>
std::true_type is_mvx_impl(MotionVectorX<T, N>);
std::false_type is_mvx_impl(...);

template <typename T>
using is_motion_vector = decltype(is_mv_impl(std::declval<T>()));

template <typename T>
using is_mvx_vector = decltype(is_mvx_impl(std::declval<T>()));

template <typename EigenType, int Row, int Col>
constexpr bool is_eigen_matrix()
{
    using T = std::decay_t<EigenType>;
    return T::RowsAtCompileTime == Row && T::ColsAtCompileTime == Col;
}

template <typename EigenType, int Row>
constexpr bool is_eigen_row_vector() { return is_eigen_matrix<EigenType, Row, 1>(); }

template <typename EigenType, int Col>
constexpr bool is_eigen_col_vector() { return is_eigen_matrix<EigenType, 1, Col>(); }

template <typename EigenType, int Size>
constexpr bool is_eigen_vector() { return is_eigen_row_vector<EigenType, Size>() || is_eigen_col_vector<EigenType, Size>(); }

template <typename, typename T>
struct has_setZero {
    static_assert(std::integral_constant<T, false>::value, "Second template parameter needs to be of function type.");
};

template <typename, typename T>
struct has_setIdentity {
    static_assert(std::integral_constant<T, false>::value, "Second template parameter needs to be of function type.");
};

template <typename C, typename Ret, typename... Args>
struct has_setZero<C, Ret(Args...)> {
private:
    template <typename T>
    static constexpr auto check(T*) -> typename std::is_same<decltype(std::declval<T>().setZero(std::declval<Args>()...)), Ret>::type;

    template <typename>
    static constexpr std::false_type check(...);

    using type = decltype(check<C>(0));

public:
    static constexpr bool value = type::value;
};

template <typename C, typename Ret, typename... Args>
struct has_setIdentity<C, Ret(Args...)> {
private:
    template <typename T>
    static constexpr auto check(T*) -> typename std::is_same<decltype(std::declval<T>().setIdentity(std::declval<Args>()...)), Ret>::type;

    template <typename>
    static constexpr std::false_type check(...);

    using type = decltype(check<C>(0));

public:
    static constexpr bool value = type::value;
};

} // namespace internal

} // namespace coma
