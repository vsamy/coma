/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

/*! \brief Allow formatting coma types.
 * 
 * This should be used along the format() function common to all coma classes.
 */
enum class FormatType {
    Compact, /*!< Print the class in a compact form */
    Full, /*!< Print the class in its Full form */
    Dual /*!< Print the class in its dual form (print full if dual does not exist) */
};

namespace internal {

template <typename T>
std::ostream& PrintCompact(std::ostream& os, const T& rhs, const Eigen::IOFormat& fmt);

template <typename T>
std::ostream& PrintFull(std::ostream& os, const T& rhs, const Eigen::IOFormat& fmt);

template <typename T>
std::ostream& PrintDual(std::ostream& os, const T& rhs, const Eigen::IOFormat& fmt);

template <typename Scalar>
std::ostream& PrintCompact(std::ostream& os, const MotionVector<Scalar>& sv, const Eigen::IOFormat& fmt)
{
    os << "ang=" << sv.angular().transpose().format(fmt) << ", lin=" << sv.linear().transpose().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintCompact(std::ostream& os, const ForceVector<Scalar>& sv, const Eigen::IOFormat& fmt)
{
    os << "ang=" << sv.angular().transpose().format(fmt) << ", lin=" << sv.linear().transpose().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintFull(std::ostream& os, const MotionVector<Scalar>& sv, const Eigen::IOFormat& fmt)
{
    os << sv.vector().transpose().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintFull(std::ostream& os, const ForceVector<Scalar>& sv, const Eigen::IOFormat& fmt)
{
    os << sv.vector().transpose().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintDual(std::ostream& os, const MotionVector<Scalar>& sv, const Eigen::IOFormat& fmt)
{
    os << sv.vector().transpose().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintDual(std::ostream& os, const ForceVector<Scalar>& sv, const Eigen::IOFormat& fmt)
{
    os << sv.vector().transpose().format(fmt);
    return os;
}

template <typename Scalar, int Order>
std::ostream& PrintCompact(std::ostream& os, const MotionVectorX<Scalar, Order>& svx, const Eigen::IOFormat& fmt)
{
    const char* s = "\n";
    for (Index i = 0; i < svx.nVec(); ++i) {
        if (i == svx.nVec() - 1) s = "";
        os << "sv_" << i << ": " << svx[i].format(FormatType::Compact, fmt) << s;
    }

    return os;
}

template <typename Scalar, int Order>
std::ostream& PrintCompact(std::ostream& os, const ForceVectorX<Scalar, Order>& svx, const Eigen::IOFormat& fmt)
{
    const char* s = "\n";
    for (Index i = 0; i < svx.nVec(); ++i) {
        if (i == svx.nVec() - 1) s = "";
        os << "sv_" << i << ": " << svx[i].format(FormatType::Compact, fmt) << s;
    }

    return os;
}

template <typename Scalar, int Order>
std::ostream& PrintFull(std::ostream& os, const MotionVectorX<Scalar, Order>& svx, const Eigen::IOFormat& fmt)
{
    os << svx.vector().format(fmt);
    return os;
}

template <typename Scalar, int Order>
std::ostream& PrintFull(std::ostream& os, const ForceVectorX<Scalar, Order>& svx, const Eigen::IOFormat& fmt)
{
    os << svx.vector().format(fmt);
    return os;
}

template <typename Scalar, int Order>
std::ostream& PrintDual(std::ostream& os, const MotionVectorX<Scalar, Order>& svx, const Eigen::IOFormat& fmt)
{
    os << svx.vector().format(fmt);
    return os;
}

template <typename Scalar, int Order>
std::ostream& PrintDual(std::ostream& os, const ForceVectorX<Scalar, Order>& svx, const Eigen::IOFormat& fmt)
{
    os << svx.vector().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintCompact(std::ostream& os, const Transform<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    os << "p: " << rhs.translation().transpose().format(fmt) << "'\nR:\n"
       << rhs.rotation().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintFull(std::ostream& os, const Transform<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.homogeneousMatrix().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintDual(std::ostream& os, const Transform<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.dualMatrix().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintCompact(std::ostream& os, const Cross<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    return PrintCompact(os, rhs.motion(), fmt);
}

template <typename Scalar>
std::ostream& PrintFull(std::ostream& os, const Cross<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.homogeneousDifferentiator().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintDual(std::ostream& os, const Cross<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.dualMatrix().format(fmt);
    return os;
}

template <typename Scalar, int Order>
std::ostream& PrintCompact(std::ostream& os, const CrossN<Scalar, Order>& rhs, const Eigen::IOFormat& fmt)
{
    return PrintCompact(os, rhs.motion(), fmt);
}

template <typename Scalar, int Order>
std::ostream& PrintFull(std::ostream& os, const CrossN<Scalar, Order>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.matrix().format(fmt);
    return os;
}

template <typename Scalar, int Order>
std::ostream& PrintDual(std::ostream& os, const CrossN<Scalar, Order>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.dualMatrix().format(fmt);
    return os;
}

template <typename Scalar, int Space, int Order>
std::ostream& PrintCompact(std::ostream& os, const CMTM<Scalar, Space, Order>& rhs, const Eigen::IOFormat& fmt)
{
    std::string s = "\n";
    os << "C_0:\n"
       << rhs.transform().format(FormatType::Compact, fmt) << s;
    for (Index i = 0; i < Order; ++i) {
        if (i == Order - 1) s = "";
        os << "C_" << i + 1 << ":\n"
           << rhs[i].format(FormatType::Compact, fmt) << s;
    }
    return os;
}

template <typename Scalar, int Space, int Order>
std::ostream& PrintFull(std::ostream& os, const CMTM<Scalar, Space, Order>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.matrix().format(fmt);
    return os;
}

template <typename Scalar, int Order>
std::ostream& PrintDual(std::ostream& os, const CMTM<Scalar, 4, Order>& /* rhs */, const Eigen::IOFormat& /* fmt */)
{
    COMA_ASSERT(false, "4-space CMTM does not have a dual part");
    return os;
}

template <typename Scalar, int Order>
std::ostream& PrintDual(std::ostream& os, const CMTM<Scalar, 6, Order>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.dualMatrix().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintCompact(std::ostream& os, const BiBlock33<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    os << "block 1:\n"
       << rhs.blockT1().format(fmt) << "\nBlock 2:\n"
       << rhs.blockT2().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintFull(std::ostream& os, const BiBlock33<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.matrix().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintDual(std::ostream& os, const BiBlock33<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.dualMatrix().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintCompact(std::ostream& os, const BiBlock31<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    os << "block 1:\n"
       << rhs.blockT1().format(fmt) << "\nBlock 2: " << rhs.blockT2().transpose().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintFull(std::ostream& os, const BiBlock31<Scalar>& rhs, const Eigen::IOFormat& fmt)
{
    os << rhs.matrix().format(fmt);
    return os;
}

template <typename Scalar>
std::ostream& PrintDual(std::ostream& os, const BiBlock31<Scalar>& /* rhs */, const Eigen::IOFormat& /* fmt */)
{
    COMA_ASSERT(false, "BiBlock31 class does not have a dual part");
    return os;
}

// Proxy class to format
template <typename T>
struct WithFormat {
    const T& value;
    FormatType globalFormat;
    Eigen::IOFormat eigenFormat;

    friend std::ostream& operator<<(std::ostream& os, const WithFormat<T>& f)
    {
        switch (f.globalFormat) {
        case FormatType::Dual:
            return internal::PrintDual(os, f.value, f.eigenFormat);
        case FormatType::Full:
            return internal::PrintFull(os, f.value, f.eigenFormat);
        default:
        case FormatType::Compact:
            return internal::PrintCompact(os, f.value, f.eigenFormat);
        }
    }
};

} // namespace internal

/*! \brief Formatter class inherited by all coma classes.
 * 
 * This class exposes all its child class to be printable in a standard output.
 * \tparam Derived The type of the child class
 */
template <typename Derived>
class Formatter {
    friend Derived;
    inline static Eigen::IOFormat default_format = {};

public:
    /*! \brief Default constructor. */
    Formatter() = default;

    /*! \brief Format the class to output.
     * 
     * The global formatting can be either Compact, Full or Dual.
     * Note that the Dual format is only for space==6 types.
     * You can also tweak the output using Eigen own formatting system.
     * This function returns itself so one can write, for example 
     * \code{.cpp}
     * std::cout << comaType.format(...) << std::endl;
     * \endcode
     * \param fmtType Global formatting type
     * \param fmt Eigen formatting to tweak the output
     * \return Derived class
     */
    internal::WithFormat<Derived> format(FormatType fmtType, const Eigen::IOFormat& fmt = default_format) const noexcept
    {
        return { static_cast<const Derived&>(*this), fmtType, fmt };
    }
    /*! \brief Output stream operator.
     * 
     * This declares the output operator for all classes in the coma library.
     * \param os Stream
     * \param f Derived class
     * \return Stream
     */
    friend std::ostream& operator<<(std::ostream& os, const Derived& f)
    {
        os << internal::WithFormat<Derived>({ f, FormatType::Compact, default_format });
        return os;
    }

    static void SetDefaultFormat(const Eigen::IOFormat& fmt) noexcept { default_format = fmt; }

}; // namespace coma

} // namespace coma
