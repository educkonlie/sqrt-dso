#include "util/NumType.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "util/NumType.h"
#include "util/IndexThreadReduce.h"
#include "OptimizationBackend/EnergyFunctional.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"
namespace dso {
//#include "common.h"
//! 要将三列数据全部QR化，需要第一列，第二列，第三列，依次进行
//  1 2 3
//  2 4 6
//  4 5 6
//  7 8 9
// =>
//  a x x
//  0 b x
//  0 0 c
//  0 0 0
// Jp 和 Jr合成Jp

    void EnergyFunctional::qr(MatXXc &Jp, MatXXc &Jl) {
        MatXXc temp1, temp2;
        int nres = Jl.rows();
        int cols = Jl.cols();
        assert(nres > 3);
        // i: row
        // j: col
        for (int j = 0; j < cols; j++) {
            rkf_scalar pivot = Jl(j, j);
            for (int i = j + 1; i < nres; i++) {
                if (std::abs(Jl(i, j)) < 1e-15)
                    continue;
                rkf_scalar a = Jl(i, j);
                rkf_scalar r = sqrt(pivot * pivot + a * a);
                rkf_scalar c = pivot / r;
                rkf_scalar s = a / r;
                pivot = r;
                assert(std::isfinite(r));
                assert(std::abs(r) > 1e-15);
// 变0的，先到temp
                temp1 = -s * Jl.row(j) + c * Jl.row(i);
                temp2 = -s * Jp.row(j) + c * Jp.row(i);
// 变大的.  j是pivot，在上面，i在下面
                Jl.row(j) = c * Jl.row(j) + s * Jl.row(i);
                Jp.row(j) = c * Jp.row(j) + s * Jp.row(i);
// 变0的, temp => i
                Jl.row(i) = temp1;
                Jp.row(i) = temp2;

                Jl(j, j) = pivot = r;
                Jl(i, j) = 0;
            }
        }
    }
    void EnergyFunctional::qr3(MatXXcr &Jp, MatXXcr &Jl, VecXc &Jr) {
        MatXXcr temp1, temp2;
        VecXc temp3;
        int nres = Jl.rows();
        int cols = Jl.cols();
        assert(nres > 3);
        // i: row
        // j: col
        for (int j = 0; j < cols; j++) {
            rkf_scalar pivot = Jl(j, j);
            for (int i = j + 1; i < nres; i++) {
                if (std::abs(Jl(i, j)) < 1e-15)
                    continue;
                rkf_scalar a = Jl(i, j);
                rkf_scalar r = sqrt(pivot * pivot + a * a);
                rkf_scalar c = pivot / r;
                rkf_scalar s = a / r;
                pivot = r;
                assert(std::isfinite(r));
                assert(std::abs(r) > 1e-15);
// 变0的，先到temp
                temp1 = -s * Jp.row(j) + c * Jp.row(i);
                temp2 = -s * Jl.row(j) + c * Jl.row(i);
                temp3 = -s * Jr.row(j) + c * Jr.row(i);
// 变大的.  j是pivot，在上面，i在下面
                Jp.row(j) = c * Jp.row(j) + s * Jp.row(i);
                Jl.row(j) = c * Jl.row(j) + s * Jl.row(i);
                Jr.row(j) = c * Jr.row(j) + s * Jr.row(i);
// 变0的, temp => i
                Jp.row(i) = temp1;
                Jl.row(i) = temp2;
                Jr.row(i) = temp3;

                Jl(j, j) = pivot = r;
                Jl(i, j) = 0;
            }
        }
    }
    void EnergyFunctional::qr3_householder(MatXXc &Jp, MatXXc &Jl, VecXc &r)
    {
        int m = Jl.rows();
        int n = Jl.cols();
//    MatXX Q = MatXX::Identity(m, m);

        for (int j = 0; j < n; j++) {
            VecXc R_j = Jl.col(j).bottomRows(m - j);
            rkf_scalar normx = R_j.norm();
            if (std::abs(normx) <= 0.0)
                continue;
            int s = (Jl(j, j) > 0) ? -1 : 1;
            rkf_scalar u1 = Jl(j, j) - s * normx;
            VecXc w = R_j * (1 / u1);
            w(0) = 1;
            rkf_scalar tau = -s * u1 / normx;
            Jl.bottomRows(m - j) = Jl.bottomRows(m - j) -
                                   (tau * w) * (w.transpose() * Jl.bottomRows(m - j));
            Jp.bottomRows(m - j) = Jp.bottomRows(m - j) -
                                   (tau * w) * (w.transpose() * Jp.bottomRows(m - j));
            r.bottomRows(m - j) = r.bottomRows(m - j) -
                                  (tau * w) * (w.transpose() * r.bottomRows(m - j));
        }
//        for (int i = 0; i < m; i++)
//            for (int j = 0; j < n; j++)
//                if (std::abs(Jl(i, j)) < 1e-10)
//                    Jl(i, j) = 0.0;
    }
    void EnergyFunctional::qr3f(MatXXfr &Jp, VecXf &Jl, VecXf &Jr) {
        MatXXfr temp1;
        VecXf temp2, temp3;
        int nres = Jl.rows();
        int cols = Jl.cols();
        assert(nres > 3);
        // i: row
        // j: col
        for (int j = 0; j < cols; j++) {
            rkf_scalar pivot = Jl(j, j);
            for (int i = j + 1; i < nres; i++) {
                if (std::abs(Jl(i, j)) < 1e-15)
                    continue;
                rkf_scalar a = Jl(i, j);
                rkf_scalar r = sqrt(pivot * pivot + a * a);
                rkf_scalar c = pivot / r;
                rkf_scalar s = a / r;
                pivot = r;
                assert(std::isfinite(r));
                assert(std::abs(r) > 1e-15);
// 变0的，先到temp
                temp2 = -s * Jl.row(j) + c * Jl.row(i);
                temp1 = -s * Jp.row(j) + c * Jp.row(i);
                temp3 = -s * Jr.row(j) + c * Jr.row(i);
// 变大的.  j是pivot，在上面，i在下面
                Jl.row(j) = c * Jl.row(j) + s * Jl.row(i);
                Jp.row(j) = c * Jp.row(j) + s * Jp.row(i);
                Jr.row(j) = c * Jr.row(j) + s * Jr.row(i);
// 变0的, temp => i
                Jl.row(i) = temp2;
                Jp.row(i) = temp1;
                Jr.row(i) = temp3;

                Jl(j, j) = pivot = r;
                Jl(i, j) = 0;
            }
        }
    }
    void EnergyFunctional::qr3f_householder(MatXXf &Jp, VecXf &Jl, VecXf &r)
    {
        int m = Jl.rows();
        int n = Jl.cols();
//    MatXX Q = MatXX::Identity(m, m);

        for (int j = 0; j < n; j++) {
            VecXf R_j = Jl.col(j).bottomRows(m - j);
            rkf_scalar normx = R_j.norm();
            if (std::abs(normx) <= 0.0)
                continue;
            int s = (Jl(j, j) > 0) ? -1 : 1;
            rkf_scalar u1 = Jl(j, j) - s * normx;
            VecXf w = R_j * (1 / u1);
            w(0) = 1;
            rkf_scalar tau = -s * u1 / normx;
            Jl.bottomRows(m - j) = Jl.bottomRows(m - j) -
                                   (tau * w) * (w.transpose() * Jl.bottomRows(m - j));
            Jp.bottomRows(m - j) = Jp.bottomRows(m - j) -
                                   (tau * w) * (w.transpose() * Jp.bottomRows(m - j));
            r.bottomRows(m - j) = r.bottomRows(m - j) -
                                  (tau * w) * (w.transpose() * r.bottomRows(m - j));
        }
//        for (int i = 0; i < m; i++)
//            for (int j = 0; j < n; j++)
//                if (std::abs(Jl(i, j)) < 1e-10)
//                    Jl(i, j) = 0.0;
    }
    void EnergyFunctional::qr2(MatXXcr &Jl) {
        MatXXcr temp1, temp2;
        int nres = Jl.rows();
        int cols = Jl.cols();
        assert(nres > 3);
        // i: row
        // j: col
        for (int j = 0; j < cols; j++) {
            rkf_scalar pivot = Jl(j, j);
            for (int i = j + 1; i < nres; i++) {
                if (std::abs(Jl(i, j)) < 1e-15)
//                if (Jl(i, j) == 0)
                    continue;
                rkf_scalar a = Jl(i, j);
                rkf_scalar r = sqrt(pivot * pivot + a * a);
                rkf_scalar c = pivot / r;
                rkf_scalar s = a / r;
                pivot = r;
                assert(std::isfinite(r));
                assert(std::abs(r) > 1e-15);
// 变0的，先到temp
                temp1 = -s * Jl.row(j) + c * Jl.row(i);
// 变大的.  j是pivot，在上面，i在下面
                Jl.row(j) = c * Jl.row(j) + s * Jl.row(i);
// 变0的, temp => i
                Jl.row(i) = temp1;

                Jl(j, j) = pivot = r;
                Jl(i, j) = 0;
            }
        }
    }
    void EnergyFunctional::qr2_householder(MatXXc &R)
    {
        int m = R.rows();
        int n = R.cols();
//    MatXX Q = MatXX::Identity(m, m);

        int k = 0;
        for (int j = 0; j < n; j++) {
            VecXc R_j = R.col(j).bottomRows(m - k);
            double normx = R_j.norm();
            if (std::abs(normx) <= 0.0)
                continue;
            int s = (R(k, j) > 0) ? -1 : 1;
            double u1 = R(k, j) - s * normx;
            VecXc w = R_j * (1 / u1);
            w(0) = 1;
            double tau = -s * u1 / normx;
            R.bottomRows(m - k) = R.bottomRows(m - k) -
                    (tau * w) * (w.transpose() * R.bottomRows(m - k));
            k++;
        }
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (std::abs(R(i, j)) < 1.0)
                    R(i, j) = 0.0;
    }
// struct of JM rM
//  JM_marg   JM_remained   JM_CPARS  rM
// 然后将上面整体进行QR分解，rM会比JM多一行非零，但这个是正常现象
// 最后将JM_marg对应的行列删除，将底下为0的行删除
// 如果不做marg，那么仍旧可以通过求解(J.transpose() * JM).ldlt().solve(J.transpose() * rM)
// 来比较，如果解不变，则化简有效
#if 0
    void EnergyFunctional::marg_frame(MatXXc &J, VecXc &r, MatXXc &J_new, VecXc &r_new,
                                      int nframes, int idx)
    {
        MatXXc Jr = MatXXc::Zero(J.rows(), J.cols() + 1);
        //! 组合Jr，将CPARS从最左边移为J_pose J_cpars r的结构
        Jr.leftCols(nframes * 8) = J.middleCols(CPARS, J.cols() - CPARS);
        Jr.middleCols(nframes * 8, CPARS) = J.leftCols(CPARS);
        Jr.rightCols(1) = r;

//! 将需要marg的帧移到第0帧
        if (idx != 0) {
            MatXXc temp1 = Jr.leftCols(idx * 8);
            MatXXc temp2 = Jr.middleCols(idx * 8, 8);
            Jr.leftCols(8) = temp2;
            Jr.middleCols(8, idx * 8) = temp1;
        }

        //! qr分解，以及化简，即删除多余的零行
        qr2(Jr);
        MatXXc Jr_temp = Jr.topRows(Jr.cols());

        //! 去掉前8列，和上8行 (marg)
        Jr = Jr_temp.bottomRightCorner(Jr_temp.rows() - 8, Jr_temp.cols() - 8);

        //! 将化简后的Jr分为J, r
        J = Jr.leftCols(CPARS + nframes * 8 - 8);
        r = Jr.rightCols(1);

        //! 输出为J_new, r_new
        J_new = MatXXc::Zero(J.rows(), J.cols());
        r_new = r;
        //! 再把CPARS换回头部
        J_new.leftCols(CPARS) = J.middleCols(nframes * 8 - 8, CPARS);
        J_new.middleCols(CPARS, nframes * 8 - 8) = J.leftCols(nframes * 8 - 8);
    }
#endif
    void EnergyFunctional::marg_frame(MatXXc &J, VecXc &r, int idx)
    {
        MatXXcr J_new = MatXXcr::Zero(J.rows(), J.cols() - 8);
        MatXXcr Jl = MatXXcr::Zero(J.rows(), 8);

//! 将需要marg的帧移到Jl
//        if (idx != 0) {
        J_new.leftCols(CPARS + idx * 8) = J.leftCols(CPARS + idx * 8);
        Jl = J.middleCols(CPARS + idx * 8, 8);
        J_new.rightCols(J_new.cols() - CPARS - idx * 8)
                    = J.rightCols(J.cols() - CPARS - idx * 8 - 8);
//        }

//        std::cout << "J_new:\n" << J_new << std::endl;

        //! qr分解
        qr3(J_new, Jl, r);
//        qr3_householder(J_new, Jl, r);

//        std::cout << "J_new:\n" << J_new << std::endl;

        //! 去掉上8行 (marg)
        J = J_new.bottomRows(J_new.rows() - 8);
        VecXc r_new = r;
        r = r_new.bottomRows(r_new.rows() - 8);

//        std::cout << "J_new:\n" << (J.transpose() * J).ldlt().solve(J.transpose() * r).transpose() << std::endl;
//        std::cout << "J_new2:\n" << (J2.transpose() * J2).ldlt().solve(J2.transpose() * r2).transpose() << std::endl;
    }

#if 0
    void EnergyFunctional::no_marg_frame(MatXXc &J, VecXc &r, MatXXc &J_new, VecXc &r_new, int nframes)
    {
        MatXXc Jr = MatXXc::Zero(J.rows(), J.cols() + 1);
        //! 组Jr，把CPARS放到后面
        Jr.leftCols(nframes * 8) = J.middleCols(CPARS, J.cols() - CPARS);
        Jr.middleCols(nframes * 8, CPARS) = J.leftCols(CPARS);
        Jr.rightCols(1) = r;

        //! qr分解，以及化简，即删除多余的零行
        qr2(Jr);
        Jr.conservativeResize(Jr.cols(), Jr.cols());
//    MatXX temp = Jr.topRows(Jr.cols());
//    Jr = temp;

        //! 将化简后的Jr分为J, r
        J = Jr.leftCols(CPARS + nframes * 8);
        r = Jr.rightCols(1);

        //! 输出为J_new, r_new
        J_new = MatXXc::Zero(J.rows(), J.cols());
        r_new = r;
        //! 把CPARS换回头部
        J_new.leftCols(CPARS) = J.middleCols(nframes * 8, CPARS);
        J_new.middleCols(CPARS, nframes * 8) = J.leftCols(nframes * 8);
    }
#endif

    void EnergyFunctional::compress_Jr(MatXXc &J, VecXc &r)
    {
        if (J.rows() <  1 * J.cols() + 1)
            return;
//        MatXXc Jr = MatXXc::Zero(J.rows(), J.cols() + 1);
        MatXXcr Jr = MatXXcr::Zero(J.rows(), J.cols() + 1);
        //! 组Jr
        Jr.leftCols(J.cols()) = J;
        Jr.rightCols(1) = r;
//        MatXXc Jr2 = Jr;

        //! qr分解，以及化简，即删除多余的零行
        qr2(Jr);
//        qr2_householder(Jr);

        Jr.conservativeResize(Jr.cols(), Jr.cols());

//        std::cout << "qr2:\n" << Jr << std::endl;
//        std::cout << "qr2_householder:\n" << Jr << std::endl;

        //! 将化简后的Jr分为J, r
        J = Jr.leftCols(Jr.cols() - 1);
        r = Jr.rightCols(1);

//        std::cout << "r :\n" << r.transpose() << std::endl;
//        std::cout << "r2:\n" << r2.transpose() << std::endl;

//        std::cout << "qr2:\n"
//                << (J.transpose() * J).ldlt().solve(J.transpose() * r).transpose() << std::endl;
//        std::cout << "qr2_householder:\n"
//                << (J2.transpose() * J2).ldlt().solve(J2.transpose() * r2).transpose() << std::endl;
    }
    void EnergyFunctional::compress_Jr_reductor(std::vector<MatXXc> *JJsp, std::vector<VecXc> *rrsp,
                                                int min, int max, Vec10 *stat, int tid)
    {
        if (tid == -1)
            tid = 0;
        for (int i = min; i < max; i++)
            compress_Jr((*JJsp)[i], (*rrsp)[i]);
    }
    void EnergyFunctional::compress_JrMT(IndexThreadReduce<Vec10> *red,
                                         std::vector<MatXXc > *JJsp, std::vector<VecXc > *rrsp,
                                         EnergyFunctional const * const EF,
                                         bool MT)
    {
        if (!MT) {
            compress_Jr_reductor(JJsp, rrsp, 0, (*JJsp).size(), NULL, -1);
        } else {
            red->reduce(boost::bind(&EnergyFunctional::compress_Jr_reductor,
                                    this, JJsp, rrsp, _1, _2, _3, _4),
                        0, (*JJsp).size(), 0);
        }
    }
    void EnergyFunctional::add_lambda_frame(MatXXc &J, VecXc &r, int idx, Vec8c Lambda, Vec8c alpha)
    {
        int old_rows = J.rows();
        //! 扩展底下8行
        J.conservativeResize(J.rows() + 8, J.cols());
        r.conservativeResize(r.rows() + 8);
//    J.bottomRows(8) = MatXX::Zero(8, J.cols());
        J.bottomRows(8) = MatXXc::Zero(8, J.cols());
        r.bottomRows(8) = VecXc::Zero(8);
        //! 底下idx对应的8行8列设置为Lambda，r底下设置为 Lambda * alpha
        J.block(old_rows, CPARS + idx * 8, 8, 8) = Lambda.asDiagonal();
        r.bottomRows(8) = Lambda.asDiagonal() * alpha;
//        r.bottomRows(8) = Lambda.cwiseProduct(alpha);
    }
#if 0
    void EnergyFunctional::test_marg_frame() {
        int num_of_frames = 25;
        int idx = 2;
        MatXXc J = MatXXc::Random(2000, CPARS + num_of_frames * 8);
        VecXc r = VecXc::Random(2000);
        Vec8c Lambda = Vec8c::Random(8);
        Vec8c alpha = Vec8c::Random(8);
        add_lambda_frame(J, r, 2, Lambda, alpha);
        std::cout << "J\n" << J << std::endl;
        std::cout << "r\n" << r.transpose() << std::endl;

        MatXXc J_new;
        VecXc r_new;
        std::cout << "x    : "
                  << (J.transpose() * J).ldlt().solve(J.transpose() * r).transpose()
                  << std::endl;
//    marg_frame(J, r, J_new, r_new, num_of_frames, 2);
        no_marg_frame(J, r, J_new, r_new, num_of_frames);

        std::cout << "new x: "
                  << (J_new.transpose() * J_new).ldlt().solve(J_new.transpose() * r_new).transpose()
                  << std::endl;

        std::cout << "J.rows(): " << J_new.rows() << std::endl;
        std::cout << "J.cols(): " << J_new.cols() << std::endl;
        std::cout << "r.rows(): " << r_new.rows() << std::endl;
    }
#endif

    void EnergyFunctional::test_qr()
    {
//    MatXX Jp = MatXX::Random(10, 10);
        MatXXc I = MatXXc::Identity(10, 10);
        MatXXc A = MatXXc::Random(10, 3);

        MatXXc Qt = I;
        MatXXc R = A;

        qr(Qt, R);
        std::cout << "A\n" << A << std::endl;
        std::cout << "R\n" << R << std::endl;
        std::cout << "Q * R\n" << Qt.transpose() * R << std::endl;
        return;

        std::cout << "test 2............" << std::endl;
        A = MatXXc::Random(20, 3 * 3);
        MatXXc Al = A.block(0, 0, 20, 3);
        MatXXc Ap = A.block(0, 3, 20, 3 * 2);
        qr(Ap, Al);

        std::cout << "A:\n" << A << std::endl;
        std::cout << "Ap:\n" << Ap << std::endl;
        std::cout << "Al:\n" << Al << std::endl;
    }

}