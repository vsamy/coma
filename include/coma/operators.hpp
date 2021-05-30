/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

///////////////////////////////////////////////////////////////////////////////
//                               Cross x Cross                               //
///////////////////////////////////////////////////////////////////////////////

// Cross x Cross does not return a Cross
template <typename Scalar>
BiBlock33<Scalar> operator*(const Cross<Scalar>& lhs, const Cross<Scalar>& rhs)
{
    return { lhs.angularMat() * rhs.angularMat(), lhs.angularMat() * rhs.linearMat() + lhs.linearMat() * rhs.angularMat() };
}

template <typename Scalar>
BiBlock31<Scalar> mul4(const Cross<Scalar>& lhs, const Cross<Scalar>& rhs)
{
    return { lhs.angularMat() * rhs.angularMat(), lhs.angularMat() * rhs.linear() };
}

///////////////////////////////////////////////////////////////////////////////
//                             Transform x Cross                             //
///////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
BiBlock33<Scalar> operator*(const Cross<Scalar>& lhs, const Transform<Scalar>& rhs)
{
    // { [wx]R, [vx]R+[wx][px]R }
    return { lhs.angularMat() * rhs.rotation(), lhs.linearMat() * rhs.rotation() + lhs.angularMat() * vector3ToCrossMatrix3(rhs.translation()) * rhs.rotation() };
}

template <typename Scalar>
BiBlock31<Scalar> mul4(const Cross<Scalar>& lhs, const Transform<Scalar>& rhs)
{
    // { [wx]R, [wx]p + v}
    return { lhs.angularMat() * rhs.rotation(), lhs.linear() + lhs.angularMat() * rhs.translation() };
}

template <typename Scalar>
BiBlock33<Scalar> operator*(const Transform<Scalar>& lhs, const Cross<Scalar>& rhs)
{
    // { R[wx], [px]R[wx]+R[vx] }
    return { lhs.rotation() * rhs.angularMat(), vector3ToCrossMatrix3(lhs.translation()) * lhs.rotation() * rhs.angularMat() + lhs.rotation() * rhs.linearMat() };
}

template <typename Scalar>
BiBlock31<Scalar> mul4(const Transform<Scalar>& lhs, const Cross<Scalar>& rhs)
{
    // { R[wx], Rv}
    return { lhs.rotation() * rhs.angularMat(), lhs.rotation() * rhs.linear() };
}

///////////////////////////////////////////////////////////////////////////////
//                           Transform x BlockMat33                          //
///////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
BiBlock33<Scalar> operator*(const Transform<Scalar>& lhs, const BiBlock33<Scalar>& rhs)
{
    // { R[wx], [px]R[wx]+R[vx] }
    return { lhs.rotation() * rhs.blockT1(), vector3ToCrossMatrix3(lhs.translation()) * lhs.rotation() * rhs.blockT1() + lhs.rotation() * rhs.blockT2() };
}

template <typename Scalar>
BiBlock31<Scalar> operator*(const Transform<Scalar>& lhs, const BiBlock31<Scalar>& rhs)
{
    // { R[wx], Rv}
    return { lhs.rotation() * rhs.blockT1(), lhs.rotation() * rhs.blockT2() };
}

///////////////////////////////////////////////////////////////////////////////
//                         Transform x MotionSubspace                        //
///////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
Eigen::MatrixXd operator*(const Transform<Scalar>& lhs, const MotionSubspace<Scalar>& rhs)
{
    Eigen::MatrixXd out(6, rhs.cols());
    auto S = rhs.matrix();
    Eigen::MatrixXd RSw = lhs.rotation() * S.template topRows<3>();
    out.template topRows<3>() = RSw;
    out.template bottomRows<3>() = vector3ToCrossMatrix3(lhs.translation()) * RSw + lhs.rotation() * S.template bottomRows<3>();
    return out;
}

///////////////////////////////////////////////////////////////////////////////
//                          Block33 x MotionSubspace                         //
///////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
Eigen::MatrixXd operator*(const BiBlock33<Scalar>& lhs, const MotionSubspace<Scalar>& rhs)
{
    Eigen::MatrixXd out(6, rhs.cols());
    auto S = rhs.matrix();
    out.template topRows<3>() = lhs.blockT1() * S.template topRows<3>();
    out.template bottomRows<3>() = lhs.blockT2() * S.template topRows<3>() + lhs.blockT1() * S.template bottomRows<3>();
    return out;
}

////////////////////////////////////////////////////////////////////////////////
//                             Cross x BlockMat33                             //
////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
BiBlock33<Scalar> operator*(const Cross<Scalar>& lhs, const BiBlock33<Scalar>& rhs)
{
    return { lhs.angularMat() * rhs.blockT1(), lhs.angularMat() * rhs.blockT2() + lhs.linearMat() * rhs.blockT1() };
}

template <typename Scalar>
BiBlock31<Scalar> operator*(const Cross<Scalar>& lhs, const BiBlock31<Scalar>& rhs)
{
    return { lhs.angularMat() * rhs.blockT1(), lhs.angularMat() * rhs.blockT2() };
}

////////////////////////////////////////////////////////////////////////////////
//                              Motion x Inertia                              //
////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
ForceVector<Scalar> operator*(const SpatialInertia<Scalar>& lhs, const MotionVector<Scalar>& rhs)
{
    return { lhs.inertia() * rhs.angular() + lhs.momentum().cross(rhs.linear()), lhs.mass() * rhs.linear() - lhs.momentum().cross(rhs.angular()) };
}

//////////////////////////////////////////////////////////////////////////////
//                          MotionSuspace x Motion                          //
//////////////////////////////////////////////////////////////////////////////

template <typename Scalar, typename Derived>
MotionVector<Scalar> operator*(const MotionSubspace<Scalar>& lhs, const Eigen::MatrixBase<Derived>& rhs)
{
    COMA_STATIC_ASSERT((std::is_same_v<Scalar, typename Derived::Scalar>), "Wrong Eigen type");
    COMA_ASSERT(rhs.cols() == 1 && lhs.cols() == rhs.rows(), "Matrices size mismatch");
    const auto& S = lhs.matrix();
    return { S.matrix() * rhs };
}

//////////////////////////////////////////////////////////////////////////////
//                          MotionSuspace x Force                           //
//////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
Eigen::Matrix<Scalar, Dynamic, 1> operator*(const MotionSubspace<Scalar>& lhs, const ForceVector<Scalar>& rhs)
{
    COMA_ASSERT(lhs.cols() == 6, "Wrong number of columns");
    return lhs.matrix() * rhs.vector();
}

///////////////////////////////////////////////////////////////////////////////
//                      DiMotionSubspace x EigenVectorX                      //
///////////////////////////////////////////////////////////////////////////////

template <typename Scalar, int NVec, typename Derived>
MotionVectorX<Scalar, NVec> operator*(const DiMotionSubspace<Scalar, NVec>& lhs, const Eigen::MatrixBase<Derived>& rhs)
{
    COMA_STATIC_ASSERT((std::is_same_v<Scalar, typename Derived::Scalar>), "Wrong Eigen type");

    const Index dof = lhs.block().cols();
    const Index newNVec = NVec != Dynamic ? NVec : rhs.rows() / dof;
    COMA_ASSERT(rhs.cols() == 1 && rhs.rows() % dof == 0, "rhs must be a vector which size is a multiple dof");
    COMA_ASSERT(NVec == Dynamic ? true : rhs.rows() % NVec == 0, "Matrices size mismatch");
    const auto& S = lhs.block();
    MotionVectorX<Scalar, NVec> out(newNVec);
    for (Index i = 0; i < newNVec; ++i)
        out[i] = S * rhs.segment(i * dof, dof);

    return out;
}

///////////////////////////////////////////////////////////////////////////////
//                         DiMotionSubspace x ForceX                         //
///////////////////////////////////////////////////////////////////////////////

template <typename Scalar, int NVec>
Eigen::Matrix<Scalar, Dynamic, 1> operator*(const DiMotionSubspace<Scalar, NVec>& lhs, const ForceVectorX<Scalar, NVec>& rhs)
{
    const auto& S = lhs.block();
    const Index dof = S.rows();
    Eigen::Matrix<Scalar, Dynamic, 1> out(dof * rhs.nVec());
    for (Index i = 0; i < rhs.nVec(); ++i)
        out.segment(i * dof, dof) = S * rhs[i];

    return out;
}

////////////////////////////////////////////////////////////////////////////////
//                            DiInertia x MotionX                             //
////////////////////////////////////////////////////////////////////////////////

template <typename Scalar, int NVec>
ForceVectorX<Scalar, NVec> operator*(const DiInertia<Scalar, NVec>& lhs, const MotionVectorX<Scalar, NVec>& rhs)
{
    const auto& I = lhs.block();
    ForceVectorX<Scalar, NVec> out;
    out.resize(rhs.nVec());
    for (Index i = 0; i < rhs.nVec(); ++i)
        out[i] = I * rhs[i];

    return out;
}

} // namespace coma
