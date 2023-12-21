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

#if 0
        void EnergyFunctional::pcgReductor(VecXc AAq[], const std::vector<MatXXc> &A, const VecXc &q,
                                           int min, int max, Vec10 *stat, int tid)
        {
//    std::cout << "tid: " << tid << std::endl;
//    std::cout << "min-max: " << min << " " << max << std::endl;
            if (tid == -1)
                tid = 0;
            for (int j = min; j < max; j++) {
                VecXc Aq = A[j] * q;
                AAq[tid] += (A[j].transpose() * Aq);
            }
//    std::cout << AAq[tid].transpose() << std::endl;
        }

        //! A[]里可以存放指针，每个指针都是一个矩阵的地址
        void EnergyFunctional::pcgMT(IndexThreadReduce<Vec10> *red,
                                     const std::vector<MatXXc> &A, const std::vector<VecXc> &b,
                                     EnergyFunctional const * const EF,
                                     /*int num_of_A,*/ VecXc &x,
                                     rkf_scalar tor, int maxiter, bool MT)
        {
            x = VecXc::Zero(A[0].cols());
            VecXc lambda = VecXc::Zero(A[0].cols());
            VecXc r = VecXc::Zero(A[0].cols());

            std::cout << "A size: " << A.size() << std::endl;
            std::cout << "b size: " << b.size() << std::endl;

            for (int j = 0; j < A.size(); j++) {
//                std::cout << "A[j]: " << j << std::endl;
//                std::cout << A[j] << std::endl;
//                std::cout << "rkf 115 " << A.size() << std::endl;
                assert(A[0].cols() == A[j].cols());
                for (int i = 0; i < A[j].cols(); i++)
                    lambda(i) += (A[j].col(i).transpose() * A[j].col(i));
//                std::cout << "rkf 116 " << A[j].cols() << std::endl;
                r = r + A[j].transpose() * b[j];
            }
            for (int i = 0; i < A[0].cols(); i++)
                lambda(i) = 1.0 / lambda(i);

            r = lambda.asDiagonal() * r;
            VecXc q = r;
            rkf_scalar rr_old = r.transpose() * r;

            for (int i = 0; i < maxiter; i++) {
                VecXc AAq = VecXc::Zero(A[0].cols());
                if (!MT) {
//                    for (int j = 0; j < A.size(); j++) {
//                        AAq = AAq + (A[j].transpose() * (A[j] * q));
//                    }
                    pcgReductor(&AAq, A, q, 0, A.size(), NULL, -1);
                } else {
                    VecXc AAqs[NUM_THREADS];
                    for (int k = 0; k < NUM_THREADS; k++) {
                        AAqs[k] = VecXc::Zero(A[0].cols());
                    }
                    red->reduce(boost::bind(&EnergyFunctional::pcgReductor,
                                            this, AAqs, A, q, _1, _2, _3, _4),
                                0, A.size(), 0);
                    AAq = AAqs[0];
                    for (int k = 1; k < NUM_THREADS; k++)
                        AAq.noalias() += AAqs[k];
                }

                AAq = lambda.asDiagonal() * AAq;

                rkf_scalar alpha = rr_old / (q.transpose() * AAq);
                x += alpha * q;
                r -= alpha * AAq;
                rkf_scalar rr_new = r.transpose() * r;

                if (std::sqrt(rr_new) < tor) {
                    std::cout << "iter: " << i << std::endl;
                    break;
                }

                q = r + (rr_new / rr_old) * q;
                rr_old = rr_new;
            }
            std::cout << "maxiter........" << maxiter << std::endl;
        }
#endif
    void EnergyFunctional::pcgReductor(VecXc q[], std::vector<MatXXc > *A, VecXc d,
                                       int min, int max, Vec10 *stat, int tid)
    {
//    std::cout << "tid: " << tid << std::endl;
//    std::cout << "min-max: " << min << " " << max << std::endl;
        if (tid == -1)
            tid = 0;
        for (int j = min; j < max; j++) {
//            VecXc Ad = (*A)[j] * d;
            q[tid] = q[tid] + (*A)[j].transpose() * ((*A)[j] * d);
        }
//    std::cout << AAq[tid].transpose() << std::endl;
    }

    void EnergyFunctional::leastsquare_pcg_origMT(IndexThreadReduce<Vec10> *red,
                                 std::vector<MatXXc > *A, std::vector<VecXc > *b,
                                 EnergyFunctional const * const EF,
                                 VecXc &x, rkf_scalar tor, int maxiter, bool MT)
    {
            int i = 0;
        x = VecXc::Zero((*A)[0].cols());
        VecXc lambda_point_inv = VecXc::Zero((*A)[0].cols());
        MatXXc lambda_inv = MatXXc::Zero((*A)[0].cols(), (*A)[0].cols());

        VecXc b_total = VecXc::Zero((*A)[0].cols());

        bool block_jacobi = true;

        int nframes = ((*A)[0].cols() - CPARS) / 8;

        //! 在滑窗中，因为频繁的优化，这里也可以进行并行化
        //! 其实lambda的计算和A.transpose * b的计算可以放到addPoint里
        //! 可以改为block-jacobi
        //! block-jacobi即对角线上的CPARS * CPARS, 8 * 8, ..., 8 * 8小块取逆
        for (int j = 0; j < (*A).size(); j++) {
            if (!block_jacobi) {
                for (int k = 0; k < (*A)[0].cols(); k++)
                    lambda_point_inv(k) += ((*A)[j].col(k).transpose() * (*A)[j].col(k));
            } else {
                MatXXc temp = (*A)[j].leftCols(CPARS);
                lambda_inv.topLeftCorner(CPARS, CPARS) +=
                        (temp.transpose() * temp);
                for (int k = 0; k < nframes; k++) {
                    temp = (*A)[j].middleCols(CPARS + k * 8, 8);
                    lambda_inv.block(CPARS + k * 8, CPARS + k * 8, 8, 8) +=
                            (temp.transpose() * temp);
                }
            }
            b_total += (*A)[j].transpose() * (*b)[j];
        }
        if (!block_jacobi) {
//            VecXc lambda_point = VecXc::Zero((*A)[0].cols());
            for (int k = 0; k < (*A)[0].cols(); k++) {
                lambda_point_inv(k) = 1.0 / lambda_point_inv(k);
            }
        } else {
//            MatXXc temp = MatXXc::Identity(CPARS + 8 * nframes, CPARS + 8 * nframes);
            MatXXc temp = lambda_inv.topLeftCorner(CPARS, CPARS).inverse();
//            temp.topLeftCorner(CPARS, CPARS) =
//                    lambda_inv.topLeftCorner(CPARS, CPARS).inverse();
            lambda_inv.topLeftCorner(CPARS, CPARS) = temp;
//            for (int i = CPARS; i < (*A)[0].cols(); i++) {
//                if (lambda_inv(i, i) == 0)
//                    temp(i, i) = 1;
//                else
//                    temp(i, i) = 1.0 / lambda_inv(i, i);
//            }
            for (int k = 0; k < nframes; k++) {
                temp = lambda_inv.block(CPARS + k * 8, CPARS + k * 8, 8, 8).inverse();
                lambda_inv.block(CPARS + k * 8, CPARS + k * 8, 8, 8) = temp;
            }
//            lambda_inv.bottomRightCorner((*A)[0].cols() - CPARS, (*A)[0].cols() - CPARS) =
//                    MatXXc::Identity((*A)[0].cols() - CPARS, (*A)[0].cols() - CPARS);
//            lambda_inv = temp;
        }

        VecXc r = b_total;

        VecXc d;
        if (!block_jacobi)
            d = lambda_point_inv.asDiagonal() * r;
        else
            d = lambda_inv * r;
        rkf_scalar delta_new = r.transpose() * d;
//        std::cout << "block-jacobi d: " << d.transpose() << std::endl;
//        std::cout << "point-jacobi  : " << lambda_point_inv.transpose() << std::endl;
//        std::cout << "point-jacobi d: " << (lambda_point.asDiagonal() * r).transpose() << std::endl;
        rkf_scalar delta_0 = delta_new;

        std::cout << "delta_0: " << delta_0 << std::endl;

        while (i < maxiter && delta_new > tor * tor * delta_0) {
            VecXc q = VecXc::Zero((*A)[0].cols());
            if (!MT) {
//                for (int j = 0; j < (*A).size(); j++) {
//                    q = q + ((*A)[j].transpose() * ((*A)[j] * d));
//                }
               //! 如果i取余50,可以加上计算Ax
                pcgReductor(&q, A, d, 0, (*A).size(), NULL, -1);
            }
#if 1
            else {
                VecXc qs[NUM_THREADS];
                for (int k = 0; k < NUM_THREADS; k++) {
                    qs[k] = VecXc::Zero((*A)[0].cols());
                }
                red->reduce(boost::bind(&EnergyFunctional::pcgReductor,
                                        this, qs, A, d, _1, _2, _3, _4),
                            0, (*A).size(), 0);
                q = qs[0];
                for (int k = 1; k < NUM_THREADS; k++)
                    q.noalias() += qs[k];
            }
#endif
            rkf_scalar alpha = delta_new / (d.transpose() * q);
            x = x + alpha * d;
//            VecXc temp_total = VecXc::Zero((*A)[0].cols());
//            if (i % 50 == 0) {
//                for (int j = 0; j < (*A).size(); j++) {
//                    temp_total = temp_total + ((*A)[j].transpose() * ((*A)[j] * x));
//                }
//                r = b_total - (delta_new / (d.transpose() * q)) * q;
//            } else {
                r = r - alpha * q;
//            }
            VecXc s;
            if (!block_jacobi)
                s = lambda_point_inv.asDiagonal() * r;
            else
                s = lambda_inv * r;
            rkf_scalar delta_old = delta_new;
            delta_new = r.transpose() * s;
            rkf_scalar beta = delta_new / delta_old;
            d = s + beta * d;
            i = i + 1;
        }
        std::cout << "iter: " << i << std::endl;
    }

    int num_of_iter = 0;
    void EnergyFunctional::cg_orig(MatXXc &A, VecXc &b, VecXc &x, rkf_scalar tor, int maxiter)
    {
        int i = 0;
        x = VecXc::Zero(A.cols());
        VecXc r = b - A * x;
        VecXc d = r;
        rkf_scalar delta_new = r.transpose() * r;
        rkf_scalar delta_0 = delta_new;
        rkf_scalar delta_old;
        rkf_scalar alpha;
        rkf_scalar beta;
        while (i < maxiter && delta_new > tor * tor * delta_0) {
            VecXc q = A * d;
            alpha = delta_new / (d.transpose() * q);
            x = x + alpha * d;
            if (i % 50 == 0)
                r = b - A * x;
            else
                r = r - alpha * q;
            delta_old = delta_new;
            delta_new = r.transpose() * r;

//            std::cout << "cg delta_new: " << delta_new << std::endl;
//            num_of_iter++;
            beta = delta_new / delta_old;
            d = r + beta * d;
            i++;
        }
        std::cout << "iters " << i << std::endl;
    }
    void EnergyFunctional::leastsquare_cg_orig(MatXXc &A, VecXc &b, VecXc &x, rkf_scalar tor, int maxiter)
    {
        int i = 0;
        x = VecXc::Zero(A.cols());
        VecXc r = A.transpose() * b - A.transpose() * (A * x);
        VecXc d = r;
        rkf_scalar delta_new = r.transpose() * r;
        rkf_scalar delta_0 = delta_new;
        rkf_scalar delta_old;
        rkf_scalar alpha;
        rkf_scalar beta;
        while (i < maxiter && delta_new > tor * tor * delta_0) {
            VecXc q = A.transpose() * (A * d);
            alpha = delta_new / (d.transpose() * q);
            x = x + alpha * d;
            if (i % 50 == 0)
                r = A.transpose() * b - A.transpose() * (A * x);
            else
                r = r - alpha * q;
            delta_old = delta_new;
            delta_new = r.transpose() * r;

            std::cout << "cg delta_new: " << delta_new << std::endl;
            num_of_iter++;
            beta = delta_new / delta_old;
            d = r + beta * d;
            i++;
        }
        std::cout << i << std::endl;
    }
    void EnergyFunctional::pcg_orig(MatXXc &A, VecXc &b, VecXc &x, rkf_scalar tor, int maxiter)
    {
        int i = 0;
        MatXXc M_inv = MatXXc::Zero(A.rows(), A.cols());
        M_inv.block(0, 0, A.rows() / 2, A.cols() / 2)
                = A.block(0, 0, A.rows() / 2, A.cols() / 2);
        M_inv.block(A.rows() / 2, A.cols() / 2, A.rows() - A.rows() / 2, A.cols() - A.cols() / 2)
        = A.block(A.rows() / 2, A.cols() / 2, A.rows() - A.rows() / 2, A.cols() - A.cols() / 2);
        M_inv = M_inv.inverse();

//        MatXXc M_inv = A.diagonal().asDiagonal().inverse();
        x = VecXc::Zero(A.cols());
        VecXc r = b - A * x;
        VecXc d = M_inv * r;
        rkf_scalar delta_new = r.transpose() * d;
        rkf_scalar delta_0 = delta_new;
        rkf_scalar delta_old;
        rkf_scalar alpha;
        rkf_scalar beta;
        std::cout << "cg delta_0: " << delta_0 << std::endl;
        while (i < maxiter && delta_new > tor * tor * delta_0) {
            VecXc q = A * d;
            alpha = delta_new / (d.transpose() * q);
            x = x + alpha * d;
            VecXc r1 = b - A * x;
            VecXc r2 = r - alpha * q;
//            assert (r1.transpose() * r1 == r2.transpose() * r2);
//            std::cout << "r1^2: " << r1.transpose() * r1 << std::endl;
//            std::cout << "r2^2: " << r2.transpose() * r2 << std::endl;
            if (i % 50 == 0)
                r = b - A * x;
            else
                r = r - alpha * q;

            VecXc s = M_inv * r;
            delta_old = delta_new;
            delta_new = r.transpose() * s;

//            std::cout << "cg delta_new: " << delta_new << std::endl;
            num_of_iter++;

            beta = delta_new / delta_old;
            d = s + beta * d;
            i++;
        }
        std::cout << "pcg iters: " << i << std::endl;
    }
    void EnergyFunctional::leastsquare_pcg_orig(MatXXc &A, VecXc &b, VecXc &x, rkf_scalar tor, int maxiter)
    {
        int i = 0;
//        MatXXc M_inv = (A.transpose() * A).diagonal().asDiagonal().inverse();

        VecXc lambda = VecXc::Zero(A.cols());
        for (int i = 0; i < A.cols(); i++)
            lambda(i) = 1.0 / (A.col(i).transpose() * A.col(i));
        MatXXc M_inv = lambda.asDiagonal();
        VecXc Atb = A.transpose() * b;
//        std::cout << "M_inv: " << lambda.transpose() << std::endl;

//        for (int i = 0; i < M_inv.cols(); i++)
//            assert(lambda(i) == M_inv(i, i));

        x = VecXc::Zero(A.cols());
        VecXc r = Atb; // - A.transpose() * (A * x);
        VecXc d = lambda.asDiagonal() * r;
        rkf_scalar delta_new = r.transpose() * d;
        rkf_scalar delta_0 = delta_new;
        rkf_scalar delta_old;
        rkf_scalar alpha;
        rkf_scalar beta;

//        MatXXc AtA = A.transpose() * A;
        std::cout << "cg delta_0: " << delta_0 << std::endl;
        while (i < maxiter && delta_new > tor * tor * delta_0) {
            VecXc q = A.transpose() * (A * d);
//            VecXc q = AtA * d;
            alpha = delta_new / (d.transpose() * q);
            x = x + alpha * d;
//            VecXc r1 = b - A * x;
//            VecXc r2 = r - alpha * q;
//            assert (r1.transpose() * r1 == r2.transpose() * r2);
//            std::cout << "r1^2: " << r1.transpose() * r1 << std::endl;
//            std::cout << "r2^2: " << r2.transpose() * r2 << std::endl;
            if (i % 50 == 0)
                r = Atb - A.transpose() * (A * x);
            else
                r = r - alpha * q;

            VecXc s = lambda.asDiagonal() * r;
            delta_old = delta_new;
            delta_new = r.transpose() * s;

//            std::cout << "cg delta_new: " << delta_new << std::endl;
            num_of_iter++;

            beta = delta_new / delta_old;
            d = s + beta * d;
            i++;
        }
        std::cout << i << std::endl;
    }

}