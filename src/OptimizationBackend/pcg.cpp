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
    void EnergyFunctional::pcgReductor(VecXc AAq[], std::vector<MatXXc > *A, VecXc q,
                                       int min, int max, Vec10 *stat, int tid)
    {
//    std::cout << "tid: " << tid << std::endl;
//    std::cout << "min-max: " << min << " " << max << std::endl;
        if (tid == -1)
            tid = 0;
        for (int j = min; j < max; j++) {
            VecXc Aq = (*A)[j] * q;
            AAq[tid] = AAq[tid] + ((*A)[j].transpose() * Aq);
        }
//    std::cout << AAq[tid].transpose() << std::endl;
    }

    void EnergyFunctional::pcgMT(IndexThreadReduce<Vec10> *red,
                                 std::vector<MatXXc > *A, std::vector<VecXc > *b,
                                 EnergyFunctional const * const EF,
                                 VecXc &x, rkf_scalar tor, int maxiter, bool MT)
    {
        x = VecXc::Zero((*A)[0].cols());
        VecXc lambda_inv = VecXc::Zero((*A)[0].cols());
        VecXc r = VecXc::Zero((*A)[0].cols());

        for (int j = 0; j < (*A).size(); j++) {
            for (int i = 0; i < (*A)[0].cols(); i++)
                lambda_inv(i) += ((*A)[j].col(i).transpose() * (*A)[j].col(i));
            r += (*A)[j].transpose() * (*b)[j];
        }
        VecXc lambda = VecXc::Zero((*A)[0].cols());
        for (int i = 0; i < (*A)[0].cols(); i++) {
            lambda(i) = 1.0 / lambda_inv(i);
        }
//        std::cout << lambda.asDiagonal() * (lambda_inv.asDiagonal().toDenseMatrix()) << std::endl;
//        std::cout << lambda_inv.asDiagonal() << std::endl;
//        exit(0);

        r = lambda.asDiagonal() * r;
        VecXc q = r;
        rkf_scalar rr_old = r.transpose() * r;
//        std::cout << "rr_old: " << rr_old << std::endl;

        for (int i = 0; i < maxiter; i++) {
            VecXc AAq = VecXc::Zero((*A)[0].cols());
            if (!MT) {
                for (int j = 0; j < (*A).size(); j++) {
                    AAq = AAq + ((*A)[j].transpose() * ((*A)[j] * q));
                }
//                pcgReductor(&AAq, A, q, 0, (*A).size(), NULL, -1);
            } else {
                VecXc AAqs[NUM_THREADS];
                for (int k = 0; k < NUM_THREADS; k++) {
                    AAqs[k] = VecXc::Zero((*A)[0].cols());
                }
                red->reduce(boost::bind(&EnergyFunctional::pcgReductor,
                                        this, AAqs, A, q, _1, _2, _3, _4),
                            0, (*A).size(), 0);
                AAq = AAqs[0];
                for (int k = 1; k < NUM_THREADS; k++)
                    AAq.noalias() += AAqs[k];
            }
            AAq = lambda.asDiagonal() * AAq;

            rkf_scalar alpha = rr_old / (q.transpose() * AAq);
            x = x + alpha * q;
            r = r - alpha * AAq;
            rkf_scalar rr_new = r.transpose() * r;

//            std::cout << "rr_new: " << rr_new << std::endl;

//            if (std::sqrt(rr_new) < tor) {
            if (std::sqrt(rr_new) < tor * std::sqrt(rr_new)) {
                std::cout << "iter: " << i << std::endl;
                break;
            }
            if (rr_new > rr_old) {
                x = x - alpha * q;
                break;
            }
            q = r + (rr_new / rr_old) * q;
            rr_old = rr_new;
        }
    }
    void EnergyFunctional::cg(MatXXc &A, VecXc &b, VecXc &x, rkf_scalar tor, int maxiter)
    {
        x = VecXc::Zero(A.cols());
        VecXc r = A.transpose() * b;
        VecXc q = r;
        rkf_scalar rr_old = r.transpose() * r;
        double real_tor = tor * tor * rr_old;
        for (int i = 0; i < maxiter; i++) {
            VecXc Aq = A * q;
            VecXc AAq = A.transpose() * Aq;
//            VecXc AAq = (A.transpose() * A) * q;

            rkf_scalar alpha = rr_old / (q.transpose() * AAq);
            x = x + alpha * q;
            r = r - alpha * AAq;
            rkf_scalar rr_new = r.transpose() * r;
//            assert(rr_new >= 0);

            std::cout << "cg rr_new: " << rr_new << std::endl;
//            if (std::sqrt(rr_new) < tor) {
            if (rr_new < real_tor) {
                std::cout << "iter: " << i << std::endl;
                break;
            }
//            if (rr_new > rr_old) {
//                std::cout << "rr_new > rr_old" << std::endl;
//                break;
//            }

            q = r + (rr_new / rr_old) * q;
            rr_old = rr_new;
        }
    }

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

            std::cout << "cg delta_new: " << delta_new << std::endl;
            beta = delta_new / delta_old;
            d = r + beta * d;
        }
    }
    void EnergyFunctional::pcg_orig(MatXXc &A, VecXc &b, VecXc &x, rkf_scalar tor, int maxiter)
    {
        int i = 0;
        MatXXc M_inv = A.diagonal().asDiagonal().inverse();
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
            if (i % 50 == 0)
                r = b - A * x;
            else
                r = r - alpha * q;
            VecXc s = M_inv * r;
            delta_old = delta_new;
            delta_new = r.transpose() * s;

            std::cout << "cg delta_new: " << delta_new << std::endl;
            beta = delta_new / delta_old;
            d = s + beta * d;
        }
    }
    void EnergyFunctional::pcg(MatXXc &A, VecXc &b, VecXc &x, rkf_scalar tor, int maxiter)
    {
        x = VecXc::Zero(A.cols());
        VecXc lambda = VecXc::Zero(A.cols());
        for (int i = 0; i < A.cols(); i++)
            lambda(i) = 1.0 / (A.col(i).transpose() * A.col(i));

        VecXc r =  (A.transpose() * b);

        r = lambda.asDiagonal() * r;

        VecXc q = r;
        rkf_scalar rr_old = r.transpose() * r;

        int i = 0;
        for (i = 0; i < maxiter; i++) {
            VecXc AAq = lambda.asDiagonal() * (A.transpose() * (A * q));

            rkf_scalar alpha = rr_old / (q.transpose() * AAq);
            x = x + alpha * q;
            r = r - alpha * AAq;
            rkf_scalar rr_new = r.transpose() * r;

            std::cout << "pcg rr_new: " << rr_new << std::endl;
            if (std::sqrt(rr_new) < tor * std::sqrt(rr_new)) {
                std::cout << "iter: " << i << std::endl;
                break;
            }
            if (rr_new > rr_old) {
                std::cout << "                rr_new > rr_old" << std::endl;
//                break;
            }

            q = r + (rr_new / rr_old) * q;
            rr_old = rr_new;
        }
        std::cout << "break iter: " << i << std::endl;
    }
}