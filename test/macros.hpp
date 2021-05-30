/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#if defined(_MSC_VER)
#define DCW_BEGIN           \
    __pragma(warning(push)) \
        __pragma(warning(disable : 4305))
#define DCW_END __pragma(warning(pop))
#elif defined(__GNUC__) || defined(__clang__)
#define DCW_BEGIN                                                \
    _Pragma("GCC diagnostic push")                               \
        _Pragma("GCC diagnostic ignored \"-Wfloat-conversion\"") \
            _Pragma("GCC diagnostic ignored \"-Wconversion\"") // Only needed for clang
#define DCW_END _Pragma("GCC diagnostic pop")
#endif

#ifdef DCW_BEGIN
#define DISABLE_CONVERSION_WARNING_BEGIN DCW_BEGIN
#define DISABLE_CONVERSION_WARNING_END DCW_END
#else
#define DISABLE_CONVERSION_WARNING_BEGIN
#define DISABLE_CONVERSION_WARNING_END
#endif
