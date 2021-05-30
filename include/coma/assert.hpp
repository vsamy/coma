/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#ifdef COMA_THROW_ON_ASSERT_FAIL
namespace coma {

//see http://stackoverflow.com/questions/37181621/easy-way-of-constructing-information-message-for-throwing-stdexception-using-p
template <class E>
[[noreturn]] inline void fancy_throw(std::string msg, char const* file, char const* function, std::size_t line)
{
    throw E(std::string("In file: ") + file + "(line " + std::to_string(line) + "): [In function: " + function + "]\n" + msg);
}

} // namespace coma
#define COMA_EXCEPTION(TYPE, MESSAGE) fancy_throw<TYPE>(MESSAGE, __FILE__, __func__, __LINE__)
#define COMA_ASSERT(CHECK, MESSAGE) \
    if (!(CHECK)) COMA_EXCEPTION(std::runtime_error, MESSAGE)
#else
#define COMA_ASSERT(CHECK, MESSAGE) assert((CHECK) && (MESSAGE))
#endif
#define COMA_ASSERT_IS_RESIZABLE_TO(TYPE, N) COMA_ASSERT((internal::traits<TYPE>::n_vec == N && internal::traits<TYPE>::n_vec != Dynamic) || internal::traits<TYPE>::n_vec == Dynamic, "This function is only for dynamic-sized class or fixed-size class of size " #N)

#define COMA_STATIC_ASSERT(X, MESSAGE) static_assert((X), MESSAGE ": " #X)
#define COMA_STATIC_ASSERT_IS_FP(TYPE) COMA_STATIC_ASSERT(std::is_floating_point_v<TYPE>, "This is only for floating point based variables")
#define COMA_STATIC_ASSERT_IS_FIXED(TYPE) COMA_STATIC_ASSERT(internal::traits<TYPE>::n_vec != Dynamic, "This function is disable for dynamic-order class")
#define COMA_STATIC_ASSERT_IS_SPACE_6(TYPE) COMA_STATIC_ASSERT(internal::traits<TYPE>::space == 6, "This function is available only for 6-space class")
#define COMA_STATIC_ASSERT_IS_EIGEN_VECTOR(TYPE, SIZE) COMA_STATIC_ASSERT((internal::is_eigen_vector<TYPE, SIZE>()), "The type " #TYPE " must be an eigen vector of size " #SIZE)
#define COMA_STATIC_ASSERT_IS_EIGEN_MATRIX(TYPE, ROWS, COLS) COMA_STATIC_ASSERT((internal::is_eigen_matrix<TYPE, ROWS, COLS>()), "The type " #TYPE " must be an eigen matrix " #ROWS "-" #COLS)
#define COMA_STATIC_ASSERT_ASSIGNABLE(FROM_TYPE, TO_TYPE) COMA_STATIC_ASSERT((std::is_assignable_v<TO_TYPE&, FROM_TYPE>), #FROM_TYPE " is not assignable to " #TO_TYPE)