/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <iostream>

#include "FullSystem/HessianBlocks.h"

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{
    //   Top的addPoint和Bot的addPoint
#if 1
#ifndef FRAMES
#define FRAMES (nframes[0])
//#define FRAMES (8)
#endif
    TicToc timer_ACC1;
    double times_ACC1 = 0.0;
    TicToc timer_ACC2;
    double times_ACC2 = 0.0;
    TicToc timer_ACC3;
    double times_ACC3 = 0.0;
    TicToc timer_ACC4;
    double times_ACC4 = 0.0;
    TicToc timer_ACC5;
    double times_ACC5 = 0.0;

    template<int mode>
    void AccumulatedTopHessianSSE::addPoint(MatXXc &H1, VecXc &b1,
                                            EFPoint* p, EnergyFunctional *ef, int tid)	// 0 = active, 1 = linearized, 2=marginalize
    {
        MatXXf Jr1 = MatXXf::Zero(8 * FRAMES, CPARS + 8 * FRAMES);
        VecXf Jr2 = VecXf::Zero(8 * FRAMES);;
        VecXf Jl  = VecXf::Zero(8 * FRAMES);

        assert(mode==0 || mode==2);

        VecCf dc = ef->cDeltaF;

        float bd_acc=0;
        float Hdd_acc=0;
        VecCf  Hcd_acc = VecCf::Zero();

        //! 对该点所有的残差计算相应的矩阵块。Top里的是该残差对应的C, xi部分的偏导，Sch里的是该残差对应的舒尔补
        int k = 0;

        int ngoodres = 0;
        std::vector<int > target_id;

        for(EFResidual* r : p->residualsAll) {
            if(mode==0) {
                if(r->isActive()) ngoodres++;
                assert(!r->isLinearized);
                if(r->isLinearized || !r->isActive()) continue;
            }
            if(mode==2) {
                if(!r->isActive()) continue;
                assert(r->isLinearized);
            }

            RawResidualJacobian* rJ = r->J;
            int htIDX = r->hostIDX + r->targetIDX*nframes[tid];
            target_id.push_back(r->targetIDX);

            Mat18f dp = ef->adHTdeltaF[htIDX];

            VecNRf resApprox;
            if(mode==0)
                resApprox = rJ->resF;
            if(mode==2)
                resApprox = r->res_toZeroF;

            // need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
            Vec2f JI_r(0,0);
            Vec2f Jab_r(0,0);
            float rr=0;
            for(int i=0;i<patternNum;i++) {
                JI_r[0] += resApprox[i] *rJ->JIdx[0][i];
                JI_r[1] += resApprox[i] *rJ->JIdx[1][i];
                Jab_r[0] += resApprox[i] *rJ->JabF[0][i];
                Jab_r[1] += resApprox[i] *rJ->JabF[1][i];
                rr += resApprox[i]*resApprox[i];
            }
            Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;
            bd_acc +=  JI_r[0]*rJ->Jpdd[0] + JI_r[1]*rJ->Jpdd[1];
            Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);
            Hcd_acc += rJ->Jpdc[0]*Ji2_Jpdd[0] + rJ->Jpdc[1]*Ji2_Jpdd[1];
#ifdef USE_MYH
            timer_ACC1.tic();
            //! 上面的是重投影误差的偏导，另外还要有8×2的矩阵JIdx，即8维的residual和x, y的偏导
            Eigen::Matrix<float, 8, 12> J_th = Eigen::Matrix<float, 8, 12>::Zero();
            J_th.block<8, 4>(0, 0) = rJ->JIdx[0] * rJ->Jpdc[0].transpose() +
                                     rJ->JIdx[1] * rJ->Jpdc[1].transpose();
            J_th.block<8, 6>(0, 4) = rJ->JIdx[0] * rJ->Jpdxi[0].transpose() +
                                     rJ->JIdx[1] * rJ->Jpdxi[1].transpose();
            J_th.block<8, 1>(0, 10) = rJ->JabF[0];
            J_th.block<8, 1>(0, 11) = rJ->JabF[1];
            Jr2.segment(8 * k, 8) += resApprox;
#endif

            Jr1.block(8 * k, 0, 8, 4)
                    = J_th.block(0, 0, 8, 4);
            Jr1.block(8 * k, r->hostIDX * 8 + 4, 8, 8)
                    = J_th.block(0, 4, 8, 8) * ef->adHostF[htIDX].transpose();
            Jr1.block(8 * k, r->targetIDX * 8 + 4, 8, 8)
                    = J_th.block(0, 4, 8, 8) * ef->adTargetF[htIDX].transpose();

            Jl.segment(8 * k, 8)
                    = rJ->JIdx[0] * rJ->Jpdd(0, 0) + rJ->JIdx[1] * rJ->Jpdd(1, 0);

            //! 打印host, target, C, xi, ab
            //! 上面的是重投影误差的偏导，另外还要有8×2的矩阵JIdx，即8维的residual和x, y的偏导

            nres[tid]++;
            k++;
            assert(k <= FRAMES);

            times_ACC1 += timer_ACC1.toc();
        }
        if (mode == 0) {
            if (ngoodres == 0) {
                p->HdiF = 0;
                p->bdSumF = 0;
                p->data->idepth_hessian = 0;
                p->data->maxRelBaseline = 0;
                return;
            }
        }
        timer_ACC2.tic();
#if 0
        {  // qr
            int i = 0;
            float a1;
            MatXXf temp1 = MatXXf::Zero(1, CPARS + 8 * FRAMES);
            float temp2 = 0.0;
//            while ((a1 = p->Jl[i]) == 0 && i < 8 * k) {
//                i++;
//            }
            if (i < 8 * k && k > 0) {
                int j = i + 1;
                while (j < 8 * k) {
                    if (Jl[j] == 0) {
                        j++;
                        continue;
                    }
                    float a2 = Jl[j];
                    float r = sqrt(a1 * a1 + a2 * a2);
                    if (r == 0) {
                        std::cout << "r == 0, it's impossible" << std::endl;
                        r = 1e-10;
                        assert(false);
                    }
                    float c = a1 / r;
                    float s = a2 / r;
                    a1 = Jl[i] = r;

                    // 变0的，先到temp
                    temp1 = -s * Jr1.row(i) + c * Jr1.row(j);
                    temp2 = -s * Jr2[i] + c * Jr2[j];
                    // 变大的
                    Jr1.row(i) = c * Jr1.row(i) + s * Jr1.row(j);
                    Jr2[i] = c * Jr2[i] + s * Jr2[j];
                    // 变0的, temp => j
                    Jr1.row(j) = temp1;
                    Jr2[j] = temp2;
                    Jl[j] = 0;
                    ++j;
                }
            } else {
                std::cout << "first.............." << std::endl;
            }
            //! 一个是2 + 4 + 4 + 4 + ... = 2 + 4 * k = o(k)，再乘以列数
            //! 一个是2 + 3 + 4 + 5 + ... + k = o(k^2)， 再乘以列数
            Jr1.row(i).setZero();
            Jr2[i] = 0.0;
        }
#else
        ef->qr3f(Jr1, Jl, Jr2);
        Jr1.row(0).setZero();
        Jr2[0] = 0.0;
#endif
        times_ACC2 += timer_ACC2.toc();

        p->Jr1 = Jr1.middleRows(1, 8 * k - 1).cast<rkf_scalar>();
        p->Jr2 = Jr2.segment(1, 8 * k - 1).cast<rkf_scalar>();

        assert(p->Jr1.rows() == 8 * k - 1);

        H1 += (p->Jr1.transpose() * p->Jr1);
//        H1 += (Jr1.transpose() * Jr1);
        b1 += (p->Jr1.transpose() * p->Jr2);
//        b1 += (Jr1.transpose() * Jr2);
        if (mode == 2) {
            //! 将所有的Jr1, Jr2合并入ef->JM, ef->rM
        }

        p->Hdd_accAF = Hdd_acc;
        p->bd_accAF = bd_acc;
        p->Hcd_accAF = Hcd_acc;

        float Hi = p->Hdd_accAF + p->priorF;
        if(Hi < 1e-10) Hi = 1e-10;

        // 逆深度的信息矩阵，因为逆深度是一维，所以是一个float，逆深度的协方差即1.0 / H
        p->data->idepth_hessian=Hi;

        // 原来HdiF即是协方差
        p->HdiF = 1.0 / Hi;
        p->bdSumF = p->bd_accAF;

//        std::cout << "times_ACC1: " << times_ACC1 << std::endl;
//        std::cout << "times_ACC2: " << times_ACC2 << std::endl;
//        std::cout << "times_ACC3: " << times_ACC3 << std::endl;
//        std::cout << "times_ACC4: " << times_ACC4 << std::endl;
//        std::cout << "times_ACC5: " << times_ACC5 << std::endl;
    }
#endif
#if 0
#ifndef FRAMES
#define FRAMES (nframes[0])
//#define FRAMES (8)
#endif
    TicToc timer_ACC1;
    double times_ACC1 = 0.0;
    TicToc timer_ACC2;
    double times_ACC2 = 0.0;
    TicToc timer_ACC3;
    double times_ACC3 = 0.0;
    TicToc timer_ACC4;
    double times_ACC4 = 0.0;
    TicToc timer_ACC5;
    double times_ACC5 = 0.0;

    template<int mode>
    void AccumulatedTopHessianSSE::addPoint(MatXXf &H1, VecXf &b1,
                                            EFPoint* p, EnergyFunctional const * const ef, int tid)	// 0 = active, 1 = linearized, 2=marginalize
    {
        MatXXf Jr1 = MatXXf::Zero(8 * FRAMES, CPARS + 8 * FRAMES);
        VecXf Jr2 = VecXf::Zero(8 * FRAMES);;
        VecXf Jl  = VecXf::Zero(8 * FRAMES);

        assert(mode==0 || mode==2);

        VecCf dc = ef->cDeltaF;

        float bd_acc=0;
        float Hdd_acc=0;
        VecCf  Hcd_acc = VecCf::Zero();

        //! 对该点所有的残差计算相应的矩阵块。Top里的是该残差对应的C, xi部分的偏导，Sch里的是该残差对应的舒尔补
        int k = 0;

        int ngoodres = 0;
        std::vector<int > target_id;
        int hid;

        for(EFResidual* r : p->residualsAll) {
            if(mode==0) {
                if(r->isActive()) ngoodres++;
                assert(!r->isLinearized);
                if(r->isLinearized || !r->isActive()) continue;
            }
            if(mode==2) {
                if(!r->isActive()) continue;
                assert(r->isLinearized);
            }



            RawResidualJacobian* rJ = r->J;
            int htIDX = r->hostIDX + r->targetIDX*nframes[tid];
            hid = r->hostIDX;
            target_id.push_back(r->targetIDX);

            Mat18f dp = ef->adHTdeltaF[htIDX];

            VecNRf resApprox;
            if(mode==0)
                resApprox = rJ->resF;
            if(mode==2)
                resApprox = r->res_toZeroF;



            // need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
            Vec2f JI_r(0,0);
            Vec2f Jab_r(0,0);
            float rr=0;
            for(int i=0;i<patternNum;i++) {
                JI_r[0] += resApprox[i] *rJ->JIdx[0][i];
                JI_r[1] += resApprox[i] *rJ->JIdx[1][i];
                Jab_r[0] += resApprox[i] *rJ->JabF[0][i];
                Jab_r[1] += resApprox[i] *rJ->JabF[1][i];
                rr += resApprox[i]*resApprox[i];
            }
            Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;
            bd_acc +=  JI_r[0]*rJ->Jpdd[0] + JI_r[1]*rJ->Jpdd[1];
            Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);
            Hcd_acc += rJ->Jpdc[0]*Ji2_Jpdd[0] + rJ->Jpdc[1]*Ji2_Jpdd[1];
//            if (r->data->stereoResidualFlag) {
//                for (int l = 0; l < 8; l++)
//                    Jl[0] += resApprox[l];
//                continue;
//            }
#ifdef USE_MYH
            timer_ACC1.tic();
            //! 上面的是重投影误差的偏导，另外还要有8×2的矩阵JIdx，即8维的residual和x, y的偏导
            Eigen::Matrix<float, 8, 12> J_th = Eigen::Matrix<float, 8, 12>::Zero();
            J_th.block<8, 4>(0, 0) = rJ->JIdx[0] * rJ->Jpdc[0].transpose() +
                                     rJ->JIdx[1] * rJ->Jpdc[1].transpose();
            J_th.block<8, 6>(0, 4) = rJ->JIdx[0] * rJ->Jpdxi[0].transpose() +
                                     rJ->JIdx[1] * rJ->Jpdxi[1].transpose();
            J_th.block<8, 1>(0, 10) = rJ->JabF[0];
            J_th.block<8, 1>(0, 11) = rJ->JabF[1];
//            J_th.block<8, 1>(0, 12) = resApprox;
            Jr2.segment(8 * k, 8) += resApprox;
#endif

            Jr1.block(8 * k, 0, 8, 4)
                    = J_th.block(0, 0, 8, 4);
            Jr1.block(8 * k, r->hostIDX * 8 + 4, 8, 8)
                    = J_th.block(0, 4, 8, 8) * ef->adHostF[htIDX].transpose();
            Jr1.block(8 * k, r->targetIDX * 8 + 4, 8, 8)
                    = J_th.block(0, 4, 8, 8) * ef->adTargetF[htIDX].transpose();

            Jl.segment(8 * k, 8)
                    = rJ->JIdx[0] * rJ->Jpdd(0, 0) + rJ->JIdx[1] * rJ->Jpdd(1, 0);

            //! 打印host, target, C, xi, ab
            //! 上面的是重投影误差的偏导，另外还要有8×2的矩阵JIdx，即8维的residual和x, y的偏导


            nres[tid]++;
            k++;
            assert(k <= FRAMES);

            times_ACC1 += timer_ACC1.toc();
        }
        if (mode == 0) {
            if (ngoodres == 0) {
                p->HdiF = 0;
                p->bdSumF = 0;
                p->data->idepth_hessian = 0;
                p->data->maxRelBaseline = 0;
                return;
            }
        }
        timer_ACC2.tic();
        {
            int i = 0;
            float a1;
            MatXXf temp1 = MatXXf::Zero(1, CPARS + 8 * FRAMES);
            float temp2 = 0.0;
            while ((a1 = Jl[i]) == 0 && i < 8 * k) {
                i++;
            }
            if (i < 8 * k && k > 0) {
                int j = i + 1;
                while (j < 8 * k) {
                    if (Jl[j] == 0) {
                        j++;
                        continue;
                    }
                    float a2 = Jl[j];
                    float r = sqrt(a1 * a1 + a2 * a2);
                    if (r == 0) {
                        std::cout << "r == 0, it's impossible" << std::endl;
                        r = 1e-10;
                        assert(false);
                    }
                    float c = a1 / r;
                    float s = a2 / r;
                    a1 = Jl[i] = r;

                    // 变0的，先到temp
                    temp1 = -s * Jr1.row(i) + c * Jr1.row(j);
                    temp2 = -s * Jr2[i] + c * Jr2[j];
                    // 变大的
                    Jr1.row(i) = c * Jr1.row(i) + s * Jr1.row(j);
                    Jr2[i] = c * Jr2[i] + s * Jr2[j];
                    // 变0的, temp => j
                    Jr1.row(j) = temp1;
                    Jr2[j] = temp2;
                    Jl[j] = 0;
                    ++j;
                }
            } else {
                std::cout << "first.............." << std::endl;
            }
            //! 一个是2 + 4 + 4 + 4 + ... = 2 + 4 * k = o(k)，再乘以列数
            //! 一个是2 + 3 + 4 + 5 + ... + k = o(k^2)， 再乘以列数
            Jr1.row(i).setZero();
            Jr2[i] = 0.0;
        }
        times_ACC2 += timer_ACC2.toc();

//        std::cout << "Jr1:\n" << Jr1 << std::endl;
//        std::cout << "Jr2:\n" << Jr2.transpose() << std::endl;

//        Eigen::SparseMatrix<float> A(8 * k - 1, CPARS + 8 * FRAMES);
        MatXXf A(8 * k - 1, CPARS + 8 * FRAMES);
        VecXf x(CPARS + 8 * FRAMES);
        VecXf b = Jr2.segment(1, 8 * k - 1);
        for (int l = 0; l < 8 * k - 1; l++) {
//            if (Jr1.row(l) == 0)
//                std::cout << 0 << std::endl;
            for (int m = 0; m < CPARS + 8 * FRAMES; m++)
                A(l, m) = Jr1(l + 1, m);
        }

//        std::cout << "A:\n" << A << std::endl;
//        std::cout << "b:\n" << b.transpose() << std::endl;

        timer_ACC5.tic();
        Eigen::LeastSquaresConjugateGradient<MatXXf > lscg;
        lscg.compute(A);
        x = lscg.solve(b);
        times_ACC5 += timer_ACC5.toc();

//        std::cout << "#iterations:     " << lscg.iterations() << std::endl;
//        std::cout << "estimated error: " << lscg.error()      << std::endl;
//        std::cout << "x: " << x.transpose() << std::endl;

        timer_ACC3.tic();
        MatXXf tempH = (Jr1.transpose() * Jr1);
        VecXf tempb  = (Jr1.transpose() * Jr2);
        VecXf x2 = tempH.ldlt().solve(tempb);
        times_ACC3 += timer_ACC3.toc();

//        std::cout << "x2: " << x2.transpose() << std::endl;

        H1 += Jr1.transpose() * Jr1;
        b1 += Jr1.transpose() * Jr2;

        p->Hdd_accAF = Hdd_acc;
        p->bd_accAF = bd_acc;
        p->Hcd_accAF = Hcd_acc;

        float Hi = p->Hdd_accAF + p->priorF;
        if(Hi < 1e-10) Hi = 1e-10;

        // 逆深度的信息矩阵，因为逆深度是一维，所以是一个float，逆深度的协方差即1.0 / H
        p->data->idepth_hessian=Hi;

        // 原来HdiF即是协方差
        p->HdiF = 1.0 / Hi;
        p->bdSumF = p->bd_accAF;

//        std::cout << "times_ACC1: " << times_ACC1 << std::endl;
//        std::cout << "times_ACC2: " << times_ACC2 << std::endl;
//        std::cout << "times_ACC3: " << times_ACC3 << std::endl;
//        std::cout << "times_ACC4: " << times_ACC4 << std::endl;
//        std::cout << "times_ACC5: " << times_ACC5 << std::endl;
    }
#endif

    template void AccumulatedTopHessianSSE::addPoint<0>
            (MatXXc &H1, VecXc &b1,
             EFPoint* p, EnergyFunctional  *ef, int tid);
//template void AccumulatedTopHessianSSE::addPoint<1>(EFPoint* p, EnergyFunctional const * const ef, int tid);
    template void AccumulatedTopHessianSSE::addPoint<2>
            (MatXXc &H1, VecXc &b1,
             EFPoint* p, EnergyFunctional  *ef, int tid);

void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional *EF, bool usePrior, bool useDelta, int tid) {
    H = MatXX::Zero(nframes[tid] * 8 + CPARS, nframes[tid] * 8 + CPARS);
    b = VecX::Zero(nframes[tid] * 8 + CPARS);

    MatXXc J_temp = MatXXc::Zero(CPARS, nframes[tid] * 8 + CPARS);
    VecXc r_temp = VecXc::Zero(CPARS);

    if (usePrior) {
        assert(useDelta);
        H.diagonal().head<CPARS>() += EF->cPrior;
        b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<myscalar>());
#ifdef NEW_METHOD
        J_temp.block(0, 0, CPARS, CPARS)
                = EF->cPrior_new_method.asDiagonal();
        r_temp = EF->cPrior_new_method.cwiseProduct(EF->cDeltaF.cast<rkf_scalar>());
#endif
        for (int h = 0; h < nframes[tid]; h++) {
            if (EF->frames[h]->prior(6) > 0.001) {
                std::cout << "fh->prior == " << EF->frames[h]->prior.transpose() << std::endl;
                H.diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
                b.segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);

#ifdef NEW_METHOD
                if (EF->frames[h]->prior(6) > 0.001) {
                    std::cout << "fh->prior == " << EF->frames[h]->prior.transpose() << std::endl;
                    EF->add_lambda_frame(J_temp, r_temp, h,
                                         EF->frames[h]->prior_new_method,
                                         EF->frames[h]->delta_prior.cast<rkf_scalar>());
                }
#endif
            }

        }
#ifdef NEW_METHOD
        EF->Js.push_back(J_temp);
        EF->rs.push_back(r_temp);
#endif

    }
}

}


