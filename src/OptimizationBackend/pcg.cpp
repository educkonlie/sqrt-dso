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

        void EnergyFunctional::pcgReductor(VecXc AAq[], MatXXc A[], VecXc &q,
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
        void EnergyFunctional::pcgMT(IndexThreadReduce<Vec10> *red, MatXXc A[], VecXc b[],
                                     EnergyFunctional const * const EF,
                                     int num_of_A, VecXc &x,
                                     rkf_scalar tor, int maxiter, bool MT)
        {
            x = VecXc::Zero(A[0].cols());
            VecXc lambda = VecXc::Zero(A[0].cols());
            VecXc r = VecXc::Zero(A[0].cols());

            for (int j = 0; j < num_of_A; j++) {
                for (int i = 0; i < A[0].cols(); i++)
                    lambda(i) += (A[j].col(i).transpose() * A[j].col(i));
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
                    for (int j = 0; j < num_of_A; j++) {
                        AAq = AAq + (A[j].transpose() * (A[j] * q));
                    }
//            pcgReductor(&AAq, A, q, 0, num_of_A, NULL, -1);
                } else {
                    VecXc AAqs[NUM_THREADS];
                    for (int k = 0; k < NUM_THREADS; k++) {
                        AAqs[k] = VecXc::Zero(A[0].cols());
                    }
                    red->reduce(boost::bind(&EnergyFunctional::pcgReductor,
                                            this, AAqs, A, q, _1, _2, _3, _4),
                                0, num_of_A, 0);
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
        }
}