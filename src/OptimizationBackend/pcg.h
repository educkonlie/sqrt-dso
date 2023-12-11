#include <iostream>
#include "common.h"
#include "sophus/se3.hpp"

#include "IndexThreadReduce.h"

#include <ctime>
#include <cstdlib>
#include <chrono>
#include <cstdio>
//#include <pthread.h>

using namespace Eigen;
using namespace Sophus;
using namespace std;

class my {
public:
    inline my() {}
    inline ~my() {}
//    void pcgMT(IndexThreadReduce<Vec10> *red, MatXX A[], VectorXd b[], int num_of_A, VectorXd &x, double tor,
//               int maxiter, bool MT);
//    void pcgReductor(VectorXd AAq[], MatXX A[], VectorXd &q, int min, int max, Vec10 *stat, int tid);
//    void pcg(MatXX &A, VectorXd &b, VectorXd &x, double tor, int maxiter);
//    void cg(MatXX &A, VectorXd &b, VectorXd &x, double tor, int maxiter);

    void cg(MatXX &A, VectorXd &b, VectorXd &x, double tor, int maxiter)
    {
        x = VectorXd::Zero(A.cols());
//    VectorXd Ax = A * x;
//    VectorXd r = A.transpose() * b - A.transpose() * Ax;
        VectorXd r = A.transpose() * b;
        VectorXd q = r;
        double rr_old = r.transpose() * r;
        for (int i = 0; i < maxiter; i++) {
//        timer_ACC2.tic();
            VectorXd Aq = A * q;
            VectorXd AAq = A.transpose() * Aq;
//        VectorXd AAq = A.transpose() * (A * q);
//        times_ACC2 += timer_ACC2.toc();

            double alpha = rr_old / (q.transpose() * AAq);
            x += alpha * q;
            r -= alpha * AAq;
            double rr_new = r.transpose() * r;

            if (std::sqrt(rr_new) < tor) {
                std::cout << "iter: " << i << std::endl;
                break;
            }

            q = r + (rr_new / rr_old) * q;
            rr_old = rr_new;
        }
    }

    void pcg(MatXX &A, VectorXd &b, VectorXd &x, double tor, int maxiter)
    {
        x = VectorXd::Zero(A.cols());
        VectorXd lambda = VectorXd::Zero(A.cols());
        for (int i = 0; i < A.cols(); i++)
            lambda(i) = 1.0 / (A.col(i).transpose() * A.col(i));

//    std::cout << "lambda: " << lambda.transpose() << std::endl;
//
//    VectorXd Ax = A * x;
//    VectorXd r = A.transpose() * b - A.transpose() * Ax;
        VectorXd r =  (A.transpose() * b);
//    std::cout << "r1: " << r.transpose() << std::endl;

        r = lambda.asDiagonal() * r;
//    std::cout << "r2: " << r.transpose() << std::endl;

        VectorXd q = r;
        double rr_old = r.transpose() * r;

        for (int i = 0; i < maxiter; i++) {
//        timer_ACC2.tic();
//        VectorXd Aq = A * q;
            VectorXd AAq = lambda.asDiagonal() * (A.transpose() * (A * q));
//        std::cout << "AAq: " << AAq.transpose() << std::endl;
//        AAq = lambda.asDiagonal() * AAq;
//        times_ACC2 += timer_ACC2.toc();

            double alpha = rr_old / (q.transpose() * AAq);
            x += alpha * q;
            r -= alpha * AAq;
            double rr_new = r.transpose() * r;

            if (std::sqrt(rr_new) < tor) {
                std::cout << "iter: " << i << std::endl;
                break;
            }

            q = r + (rr_new / rr_old) * q;
            rr_old = rr_new;
        }
    }

    void pcgReductor(VectorXd AAq[], MatXX A[], VectorXd &q, int min, int max, Vec10 *stat, int tid)
    {
//    std::cout << "tid: " << tid << std::endl;
//    std::cout << "min-max: " << min << " " << max << std::endl;
        if (tid == -1)
            tid = 0;
        for (int j = min; j < max; j++) {
            VectorXd Aq = A[j] * q;
            AAq[tid] += (A[j].transpose() * Aq);
        }
//    std::cout << AAq[tid].transpose() << std::endl;
    }

    void pcgMT(IndexThreadReduce<Vec10> *red, MatXX A[], VectorXd b[], int num_of_A, VectorXd &x,
                   double tor, int maxiter, bool MT)
    {
        x = VectorXd::Zero(A[0].cols());
        VectorXd lambda = VectorXd::Zero(A[0].cols());
        VectorXd r = VectorXd::Zero(A[0].cols());

        for (int j = 0; j < num_of_A; j++) {
            for (int i = 0; i < A[0].cols(); i++)
                lambda(i) += (A[j].col(i).transpose() * A[j].col(i));
            r = r + A[j].transpose() * b[j];
        }
        for (int i = 0; i < A[0].cols(); i++)
            lambda(i) = 1.0 / lambda(i);

        r = lambda.asDiagonal() * r;
        VectorXd q = r;
        double rr_old = r.transpose() * r;

        for (int i = 0; i < maxiter; i++) {
            VectorXd AAq = VectorXd::Zero(A[0].cols());
            if (!MT) {
                for (int j = 0; j < num_of_A; j++) {
                    AAq = AAq + (A[j].transpose() * (A[j] * q));
                }
//            pcgReductor(&AAq, A, q, 0, num_of_A, NULL, -1);
            } else {
                VectorXd AAqs[NUM_THREADS];
                for (int k = 0; k < NUM_THREADS; k++) {
                    AAqs[k] = VectorXd::Zero(A[0].cols());
                }
                red->reduce(boost::bind(&my::pcgReductor,
                                        this, AAqs, A, q, _1, _2, _3, _4), 0, num_of_A, 0);
                AAq = AAqs[0];
                for (int k = 1; k < NUM_THREADS; k++)
                    AAq.noalias() += AAqs[k];
            }

            AAq = lambda.asDiagonal() * AAq;

            double alpha = rr_old / (q.transpose() * AAq);
            x += alpha * q;
            r -= alpha * AAq;
            double rr_new = r.transpose() * r;

            if (std::sqrt(rr_new) < tor) {
                std::cout << "iter: " << i << std::endl;
                break;
            }

            q = r + (rr_new / rr_old) * q;
            rr_old = rr_new;
        }
    }
};
