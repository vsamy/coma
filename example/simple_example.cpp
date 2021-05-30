/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "coma/Core"
#include <iostream>

int main()
{
    coma::Transformd T;

    // Use CMTM-defined types
    coma::Transformd::rotation_t R1 = coma::Transformd::quat_t::UnitRandom().toRotationMatrix();
    coma::Transformd::translation_t p1 = coma::Transformd::translation_t::Random();

    // Or Eigen types directly
    Eigen::Matrix3d R2 = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    Eigen::Vector3d p2 = Eigen::Vector3d::Random();
    // Set both rotation or translation
    T.set(R1, p1);
    // Or set it individually
    T.rotation() = R2;
    T.translation() = p2;
    // Or construct it directly
    coma::Transformd TDummy{ R1, p1 };

    // Print Matrix
    std::cout << "A compact print:\n"
              << T << std::endl;
    // Print Matrix with special format
    std::cout << "A Full print:\n"
              << T.format(coma::FormatType::Full) << std::endl;
    std::cout << "A Dual print:\n"
              << T.format(coma::FormatType::Dual) << std::endl;
    // Print Matrix with special format and more tweaking
    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
    std::cout << "A more tweaked print:\n"
              << T.format(coma::FormatType::Full, fmt) << std::endl;

    // Use Spatial vectors
    coma::MotionVectord nu{ Eigen::Vector6d::Random() };
    coma::ForceVectord h{ Eigen::Vector6d::Random() };

    // Create a motion vector N of size N=4
    coma::MotionVectorNd<4> mvn{ nu, nu, nu, nu };
    // mvn{ nu, nu } and mvn{ nu, nu, nu, nu, nu, nu, nu, nu } won't compile
    // You can also set it afterwards
    coma::ForceVectorNd<4> fvn;
    fvn.set(h, h, h, h);
    std::cout << "Force vector 4:\n"
              << fvn.format(coma::FormatType::Full) << std::endl;

    // Create a CMTM of order 4
    coma::CMTM4Nd<4> c4_1{ T, nu, nu, nu, nu }; // Needs 4 nu since order is 4
    coma::CMTM4Nd<4> c4_2;
    coma::CMTM6Nd<4> c6_1{ T, nu, nu, nu, nu };
    coma::CMTM6Nd<4> c6_2{ T, nu, nu, nu, nu };
    // Or set it indirectly
    c4_2.set(T, nu, nu, nu, nu);
    // Or set it through a any tangent motion vector of order at least equal to N.
    c4_2.set(T, mvn);
    coma::CMTM4Nd<4> c4_3{ T, mvn };

    // Compute 2 multiplication CMTM
    auto c4_res = c4_1 * c4_2;
    auto c6_res = c6_1 * c6_2;

    // Get motion
    auto motion = c4_res.motion();
    // Note that, as long as the CMTM is not a result of a CMTM product, .motion() is instantaneous

    // Print motions
    std::cout << "Print motion:\n"
              << motion << std::endl;

    // Print CMTM sub-matrices
    std::cout << "Print CMTM sub-matrices:\n"
              << c4_res << std::endl;

    // Print full matrix
    std::cout << "Print full CMTM:\n"
              << c6_res.format(coma::FormatType::Full) << std::endl;

    // Print CMTM sub-dual matrices
    std::cout << "Print Dual compact CMTM:\n"
              << c6_res.format(coma::FormatType::Dual) << std::endl;

    // Other stuffs
    // You can use Cross class to represent lie algebra [vx]
    coma::Crossd vx{ nu };
    // A Cross can be multiply by a motion vector
    auto nu2 = vx * nu;
    std::cout << nu2 << std::endl;
    // Or a force vector
    auto h2 = vx.dualMul(h);
    std::cout << h2 << std::endl;
    // You can use CrossN class to represent a vector of lie algebra [vx]
    coma::CrossNd<4> cx{ nu, nu, nu, nu };
    // A CrossN can be multiply by a tangent motion vector
    auto mvn2 = cx * mvn;
    std::cout << mvn2 << std::endl;
    // Or a tangent force vector
    auto fvn2 = cx.dualMul(fvn);
    std::cout << fvn2 << std::endl;
}