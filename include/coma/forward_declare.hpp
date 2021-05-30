/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

template <typename T, int Size>
struct Storage;

template <typename Derived>
class SpatialVector;

template <typename Scalar>
class MotionVector;

template <typename Scalar>
class ForceVector;

template <typename Derived>
class SpatialVectorX;

template <typename Scalar, int Order>
class MotionVectorX;

template <typename Scalar, int Order>
class ForceVectorX;

template <typename Scalar>
class Transform;

template <typename Scalar>
class Cross;

template <typename Scalar>
class BiBlock33;

template <typename Scalar>
class BiBlock31;

template <typename Scalar, int Order>
class CrossN;

template <typename Scalar, int Order, int Space>
class CMTM;

template <typename Scalar>
class SpatialInertia;

template <typename Scalar>
class MotionSubspace;

template <typename Scalar, int NVec>
class DiInertia;

template <typename Scalar, int NVec>
class DiMotionSubspace;

namespace internal {

template <typename T>
struct traits;

} // namespace internal

} // namespace coma
