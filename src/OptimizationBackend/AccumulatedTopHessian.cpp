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
#if 0
#ifndef FRAMES
#define FRAMES (nframes[0])
//#define FRAMES (8)
#endif

    template<int mode>
    void AccumulatedTopHessianSSE::addPoint(EFPoint* p, EnergyFunctional *ef, int tid)	// 0 = active, 1 = linearized, 2=marginalize
    {
        MatXXfr Jr1 = MatXXfr::Zero(8 * FRAMES, CPARS + 8 * FRAMES);
//        MatXXf Jr1 = MatXXf::Zero(8 * FRAMES, CPARS + 8 * FRAMES);

        VecXf Jr2 = VecXf::Zero(8 * FRAMES);;
        VecXf Jl  = VecXf::Zero(8 * FRAMES);

        p->Jr1 = MatXXc::Zero(0, 0);
        p->Jr2 = VecXc::Zero(0);

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
//            timer_ACC1.tic();
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
                    = (J_th.block(0, 4, 8, 8) * ef->adHostF[htIDX].transpose());
            Jr1.block(8 * k, r->targetIDX * 8 + 4, 8, 8)
                    = (J_th.block(0, 4, 8, 8) * ef->adTargetF[htIDX].transpose());

            Jl.segment(8 * k, 8)
                    = (rJ->JIdx[0] * rJ->Jpdd(0, 0) + rJ->JIdx[1] * rJ->Jpdd(1, 0));

            //! 打印host, target, C, xi, ab
            //! 上面的是重投影误差的偏导，另外还要有8×2的矩阵JIdx，即8维的residual和x, y的偏导

            nres[tid]++;
            k++;
            assert(k <= FRAMES);
        }
        if (mode == 0) {
            if (ngoodres == 0) {
                p->HdiF = 0;
                p->bdSumF = 0;
                p->data->idepth_hessian = 0;
                p->data->maxRelBaseline = 0;
                p->Jr1 = MatXXc::Zero(0, 0);
                p->Jr2 = VecXc::Zero(0);
                assert(p->Jr1.rows() == 0);
                return;
            }
        }
        ef->qr3f(Jr1, Jl, Jr2);
//        ef->qr3f_householder(Jr1, Jl, Jr2);
        Jr1.row(0).setZero();
        Jr2[0] = 0.0;

//        MatXXc Jr1_temp = Jr1.middleRows(1, 8 * k - 1).cast<rkf_scalar>();
//        VecXc  Jr2_temp = Jr2.segment(1, 8 * k - 1).cast<rkf_scalar>();

//        for (int i = 0; i < Jr1_temp.rows(); i++) {
//            rkf_scalar norm = Jr1_temp.row(i).norm();
//            if (norm == 0)
//                continue;
//            norm = 1 / norm;
//            Jr1_temp.row(i) *= norm;
//            Jr2_temp.row(i) *= (norm);
//        }

//        p->Jr1 = Jr1.middleRows(1, 8 * k - 1).cast<rkf_scalar>();;
        p->Jr1 = Jr1.middleRows(0, 8 * k).cast<rkf_scalar>();;
//        p->Jr2 = Jr2.segment(1, 8 * k - 1).cast<rkf_scalar>();
        p->Jr2 = Jr2.segment(0, 8 * k).cast<rkf_scalar>();
//        ef->compress_Jr(p->Jr1, p->Jr2);

//        assert(p->Jr1.rows() == 8 * k - 1);
        assert(p->Jr1.rows() == 8 * k);

        assert(p->Jr1.rows() > 0);

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
    }
#endif
#ifndef FRAMES
#define FRAMES (nframes[0])
//#define FRAMES (8)
    template<int mode>
    void AccumulatedTopHessianSSE::addPoint(EFPoint* p, EnergyFunctional  * ef, int tid)	// 0 = active, 1 = linearized, 2=marginalize
    {
        int a, b, c;
        std::vector<int> idx;

        p->Jr1 = MatXXfr::Zero(FRAMES, CPARS + 8 * FRAMES);
//        p->Jr1 = MatXXf::Zero(FRAMES, CPARS + 8 * FRAMES);
        p->Jr2 = VecXf::Zero(FRAMES);;
        p->Jl  = VecXf::Zero(FRAMES);

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
            idx.push_back(r->hostIDX);
            idx.push_back(r->targetIDX);

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
//            MatXXf Jp = MatXXf::Zero(8, CPARS + 8 + 1);
            MatXXfr Jp = MatXXfr::Zero(8, CPARS + 8 + 1);

            Jp.leftCols(CPARS) = rJ->JIdx[0] * rJ->Jpdc[0].transpose() +
                                 rJ->JIdx[1] * rJ->Jpdc[1].transpose();

            Jp.middleCols(CPARS, 6) = rJ->JIdx[0] * rJ->Jpdxi[0].transpose() +
                                      rJ->JIdx[1] * rJ->Jpdxi[1].transpose();
            Jp.col(10) = rJ->JabF[0];
            Jp.col(11) = rJ->JabF[1];

            VecXf Jl = rJ->JIdx[0] * rJ->Jpdd(0, 0) + rJ->JIdx[1] * rJ->Jpdd(1, 0);
            Jp.col(12) = resApprox;
//            this->qr2f_householder(Jp, Jl);
            this->qr2f(Jp, Jl);

            times_ACC1 += timer_ACC1.toc();

//            VecXf Jr = resApprox;

//            p->Jr2.segment(8 * k, 8) += resApprox;
#endif

            timer_ACC2.tic();

            //! 要改用稀疏矩阵表示法，CPARS + host + target，总共20个浮点数一行，然后每行有一个表格，
            //! 表格里面两个整数，一个表示host id，一个表示target id
            //! 然后用normx分别乘以每一行，然后对照每一行的表格进行累加

            p->Jr1.block(k, 0, 1, CPARS)
                    = Jp.block(0, 0, 1, CPARS);
            p->Jr1.block(k, r->hostIDX * 8 + 4, 1, 8)
                    = Jp.block(0, CPARS, 1, 8) * ef->adHostF[htIDX].transpose();
            p->Jr1.block(k, r->targetIDX * 8 + 4, 1, 8)
                    = Jp.block(0, 4, 1, 8) * ef->adTargetF[htIDX].transpose();

            p->Jl(k)
                    = Jl(0);
            p->Jr2(k)
                    = Jp(0, 12);

//            Jp.row(0).setZero();
            Jp = Jp.bottomRows(Jp.rows() - 1);

            times_ACC2 += timer_ACC2.toc();

            timer_ACC3.tic();

//            this->myH[htIDX] += Jp.transpose() * Jp;
            add_Jr(this->myJ[htIDX], Jp);

            times_ACC3 += timer_ACC3.toc();


            //! 打印host, target, C, xi, ab
            //! 上面的是重投影误差的偏导，另外还要有8×2的矩阵JIdx，即8维的residual和x, y的偏导


            nres[tid]++;
            k++;
            assert(k <= FRAMES);
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
        timer_ACC4.tic();

        //! 一个是2 + 4 + 4 + 4 + ... = 2 + 4 * k = o(k)，再乘以列数
        //! 一个是2 + 3 + 4 + 5 + ... + k = o(k^2)， 再乘以列数

        std::cout << "Jr1: " << p->Jr1.rows() << " " << p->Jr1.cols() << std::endl;
        std::cout << "p->Jr1:\n" << p->Jr1 << std::endl;
        this->qr3f(p->Jr1, p->Jl, p->Jr2);
//        this->qr3f_householder(p->Jr1, p->Jl, p->Jr2);

        p->Jr1.row(0).setZero();
        p->Jr2(0) = 0.0;

        times_ACC4 += timer_ACC4.toc();

        timer_ACC5.tic();

//        H1 += (p->Jr1.transpose() * p->Jr1).cast<double>();
//        b1 += (p->Jr1.transpose() * p->Jr2).cast<double>();

        times_ACC5 += timer_ACC5.toc();

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

        std::cout << "times_ACC1: " << times_ACC1 << std::endl;
        std::cout << "times_ACC2: " << times_ACC2 << std::endl;
        std::cout << "times_ACC3: " << times_ACC3 << std::endl;
        std::cout << "times_ACC4: " << times_ACC4 << std::endl;
        std::cout << "times_ACC5: " << times_ACC5 << std::endl;
    }
    template void AccumulatedTopHessianSSE::addPoint<0>
            (EFPoint* p, EnergyFunctional  *ef, int tid);
//template void AccumulatedTopHessianSSE::addPoint<1>(EFPoint* p, EnergyFunctional const * const ef, int tid);
    template void AccumulatedTopHessianSSE::addPoint<2>
            (EFPoint* p, EnergyFunctional  *ef, int tid);

void AccumulatedTopHessianSSE::stitchDouble(EnergyFunctional *EF, bool usePrior, bool useDelta, int tid)
{
    MatXXc J_temp = MatXXc::Zero(CPARS, nframes[tid] * 8 + CPARS);
    VecXc r_temp = VecXc::Zero(CPARS);

    for(int h=0;h<nframes[tid];h++)
        for(int t=0;t<nframes[tid];t++) {
            //! h:[0, nframes - 1], t:[0, nframes - 1]
            int hIdx = CPARS + h * 8;
            int tIdx = CPARS + t * 8;
            int aidx = h + nframes[tid] * t;
//            MatPCPC accH = myH[aidx].cast<myscalar>();
            EF->compress_Jr(myJ[aidx]);
            MatXXfr temp1 = MatXXfr::Zero(myJ[aidx].rows(), CPARS + nframes[tid] * 8);
            VecXf temp2 = VecXf::Zero(myJ[aidx].rows());
            temp1.leftCols(CPARS)
                    = myJ[aidx].leftCols(CPARS);
            temp1.middleCols(hIdx, 8)
                    = myJ[aidx].middleCols(CPARS, 8) * EF->adHostF[aidx].transpose();
            temp1.middleCols(tIdx, 8)
                    = myJ[aidx].middleCols(CPARS, 8) * EF->adTargetF[aidx].transpose();
            temp2 = myJ[aidx].rightCols(1);
            J.push_back(temp1);
            r.push_back(temp2);
        }

    if (usePrior) {
        assert(useDelta);
//        H.diagonal().head<CPARS>() += EF->cPrior;
//        b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<myscalar>());
#ifdef NEW_METHOD
        J_temp.block(0, 0, CPARS, CPARS)
                = EF->cPrior_new_method.asDiagonal();
        r_temp = EF->cPrior_new_method.asDiagonal() * (EF->cDeltaF.cast<rkf_scalar>());
#endif
        for (int h = 0; h < nframes[tid]; h++) {
            if (EF->frames[h]->prior(6) > 0.001) {
//                std::cout << "fh->prior == " << EF->frames[h]->prior.transpose() << std::endl;
//                H.diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
//                b.segment<8>(CPARS + h * 8) +=
//                        EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);

#ifdef NEW_METHOD
//                std::cout << "fh->prior == " << EF->frames[h]->prior.transpose() << std::endl;
                EF->add_lambda_frame(J_temp, r_temp, h,
                                         EF->frames[h]->prior_new_method,
//                                         EF->frames[h]->prior,
                                         EF->frames[h]->delta_prior.cast<rkf_scalar>());

//                EF->Js.push_back(J_temp);
//                EF->rs.push_back(r_temp);
#endif
            }
        }
#ifdef NEW_METHOD
        EF->Js.push_back(J_temp);
        EF->rs.push_back(r_temp);
//        std::cout << "J_temp:\n" << J_temp << std::endl;
//        std::cout << "r_temp:\n" << r_temp.transpose() << std::endl;
//        std::cout << (J_temp.transpose() * J_temp).ldlt().solve(J_temp.transpose() * r_temp).transpose()
//                << std::endl;
//        std::cout << H.ldlt().solve(b).transpose() << std::endl;

//        VecXc x_new;
//        std::vector<MatXXc> J_temps;
//        std::vector<VecXc> r_temps;
//        J_temps.push_back(J_temp);
//        r_temps.push_back(r_temp);
//        EF->pcgMT(EF->red, &J_temps, &r_temps, EF, x_new, 1e-8, 100, false);
//        std::cout << x_new.transpose() << std::endl;
//        std::cout << "........" << std::endl;
#endif

    }
}
#ifdef ROOTBA
    void AccumulatedTopHessianSSE::qr2f(MatXXfr &Jp, VecXf &Jl)
    {
        MatXXfr temp1;
        VecXf temp2, temp3;
        int nres = Jl.rows();
        int cols = Jl.cols();

//        assert(nres > 3);

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
//                temp3 = -s * Jr.row(j) + c * Jr.row(i);
// 变大的.  j是pivot，在上面，i在下面
                Jl.row(j) = c * Jl.row(j) + s * Jl.row(i);
                Jp.row(j) = c * Jp.row(j) + s * Jp.row(i);
//                Jr.row(j) = c * Jr.row(j) + s * Jr.row(i);
// 变0的, temp => i
                Jl.row(i) = temp2;
                Jp.row(i) = temp1;
//                Jr.row(i) = temp3;

                Jl(j, j) = pivot = r;
                Jl(i, j) = 0;
            }
        }
    }
    void AccumulatedTopHessianSSE::qr3f(MatXXfr &Jp, VecXf &Jl, VecXf &Jr)
    {
        MatXXfr temp1;
        VecXf temp2, temp3;
        int nres = Jl.rows();
        int cols = Jl.cols();

//        assert(nres > 3);

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
    void AccumulatedTopHessianSSE::qr2f_householder(MatXXf &Jp, VecXf &Jl)
    {
        int m = Jl.rows();
        int n = Jl.cols();
//    MatXX Q = MatXX::Identity(m, m);

//        assert(n == 1);
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
            {
                Jl.bottomRows(m - j) = Jl.bottomRows(m - j) -
                                       (tau * w) * (w.transpose() * Jl.bottomRows(m - j));
//                Jl(j) = normx;
//            Jl.bottomRows(m - j - 1).setZero();
            }

            Jp.bottomRows(m - j) = Jp.bottomRows(m - j) -
                                   (tau * w) * (w.transpose() * Jp.bottomRows(m - j));
        }
//        for (int i = 0; i < m; i++)
//            for (int j = 0; j < n; j++)
//                if (std::abs(Jl(i, j)) < 1e-10)
//                    Jl(i, j) = 0.0;
    }
    void AccumulatedTopHessianSSE::qr3f_householder(MatXXf &Jp, VecXf &Jl, VecXf &r)
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
            {
                Jl.bottomRows(m - j) = Jl.bottomRows(m - j) -
                                       (tau * w) * (w.transpose() * Jl.bottomRows(m - j));
//                Jl(j) = normx;
            }
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
#endif

}


