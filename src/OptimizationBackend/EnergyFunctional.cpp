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


#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "util/NumType.h"

#include <execution>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{
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
    TicToc timer_ACC6;
    double times_ACC6 = 0.0;
    TicToc timer_ACC7;
    double times_ACC7 = 0.0;
    TicToc timer_ACC8;
    double times_ACC8 = 0.0;

bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;


void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib)
{

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;
	adHost = new Mat88[nFrames*nFrames];
	adTarget = new Mat88[nFrames*nFrames];

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++) {
			FrameHessian* host = frames[h]->data;
			FrameHessian* target = frames[t]->data;

			SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

			Mat88 AH = Mat88::Identity();
			Mat88 AT = Mat88::Identity();

			AH.topLeftCorner<6,6>() = -hostToTarget.Adj().transpose();
			AT.topLeftCorner<6,6>() = Mat66::Identity();


			Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
			AT(6,6) = -affLL[0];
			AH(6,6) = affLL[0];
			AT(7,7) = -1;
			AH(7,7) = affLL[0];

			AH.block<3,8>(0,0) *= SCALE_XI_TRANS;
			AH.block<3,8>(3,0) *= SCALE_XI_ROT;
			AH.block<1,8>(6,0) *= SCALE_A;
			AH.block<1,8>(7,0) *= SCALE_B;
			AT.block<3,8>(0,0) *= SCALE_XI_TRANS;
			AT.block<3,8>(3,0) *= SCALE_XI_ROT;
			AT.block<1,8>(6,0) *= SCALE_A;
			AT.block<1,8>(7,0) *= SCALE_B;

			adHost[h+t*nFrames] = AH;
			adTarget[h+t*nFrames] = AT;
		}
	cPrior = VecC::Constant(setting_initialCalibHessian);
#ifdef NEW_METHOD
    cPrior_new_method = VecCc::Constant(std::sqrt((rkf_scalar)setting_initialCalibHessian));
#endif


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	adHostF = new Mat88f[nFrames*nFrames];
	adTargetF = new Mat88f[nFrames*nFrames];

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++) {
			adHostF[h+t*nFrames] = adHost[h+t*nFrames].cast<float>();
			adTargetF[h+t*nFrames] = adTarget[h+t*nFrames].cast<float>();
		}

//	cPriorF = cPrior.cast<float>();

	EFAdjointsValid = true;
}



EnergyFunctional::EnergyFunctional()
{
	adHost=0;
	adTarget=0;

	red=0;

	adHostF=0;
	adTargetF=0;
	adHTdeltaF=0;

	nFrames = nResiduals = nPoints = 0;

	HM = MatXX::Zero(CPARS,CPARS);
	bM = VecX::Zero(CPARS);
#ifdef NEW_METHOD
    JM = MatXXc::Zero(0, CPARS);
    rM = VecXc::Zero(0);
    num_of_iter = 0;
#endif

	accSSE_top_A = new AccumulatedTopHessianSSE();
	accSSE_bot = new AccumulatedSCHessianSSE();

	resInA = resInL = resInM = 0;
}
EnergyFunctional::~EnergyFunctional()
{
	for(EFFrame* f : frames) {
		for(EFPoint* p : f->points) {
			for(EFResidual* r : p->residualsAll) {
				r->data->efResidual=0;
				delete r;
			}
			p->data->efPoint=0;
			delete p;
		}
        // f -> data -> efFrame即f自己
		f->data->efFrame=0;
		delete f;
	}

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	if(adHTdeltaF != 0) delete[] adHTdeltaF;


	delete accSSE_top_A;
	delete accSSE_bot;
}

// 从前端读取数据，设置f -> delta, f -> delta_prior, p -> deltaF
void EnergyFunctional::setDeltaF(CalibHessian* HCalib)
{
	if(adHTdeltaF != 0) delete[] adHTdeltaF;
	adHTdeltaF = new Mat18f[nFrames*nFrames];
	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++) {
			int idx = h+t*nFrames;
			adHTdeltaF[idx] =
                    frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() *
                            adHostF[idx] +
                            frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() *
                                    adTargetF[idx];
		}

	cDeltaF = HCalib->value_minus_value_zero.cast<float>();
	for(EFFrame* f : frames) {
		f->delta = f->data->get_state_minus_stateZero().head<8>();
		f->delta_prior = f->data->get_state().head<8>();

		/*for(EFPoint* p : f->points) {
            p->deltaF = p->data->idepth - p->data->idepth_zero;
            // 逆深度其实并没有用FEJ，所以delta === 0.0
            assert(p -> deltaF < 0.00001);
        }*/
	}

	EFDeltaValid = true;
}

// accumulates & shifts L.
    void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT)
    {
        if (MT) {
            std::cout << "MT" << std::endl;
            Js.clear();
            rs.clear();
            red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames, _1, _2, _3, _4), 0, 0,
                        0);
            red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
                                    accSSE_top_A, &allPoints, this, _1, _2, _3, _4),
                        0, allPoints.size(), 0);
//            accSSE_top_A->stitchDoubleMT(red, H, b, this, false, true);

            int total_rows = 0;
            for (EFFrame *f: frames) {
                for (EFPoint *p: f->points) {
//                    accSSE_top_A->addPoint<0>(p, this, 0);
                    total_rows += p->Jr1.rows();
                    if (p->Jr1.rows() > 0) {
                        Js.push_back(p->Jr1);
                        rs.push_back(p->Jr2);
                    }
                }
            }
            accSSE_top_A->stitchDouble(this, true, true);

            resInA = accSSE_top_A->nres[0];
        } else {
            accSSE_top_A->setZero(nFrames);
            Js.clear();
            rs.clear();
            int total_rows = 0;
            for (EFFrame *f: frames) {
                for (EFPoint *p: f->points) {
                    accSSE_top_A->addPoint<0>(p, this, 0);
                    total_rows += p->Jr1.rows();
                    if (p->Jr1.rows() > 0) {
                        Js.push_back(p->Jr1);
                        rs.push_back(p->Jr2);
                    }
                }
            }
            accSSE_top_A->stitchDouble(this, true, true);

            resInA = accSSE_top_A->nres[0];
        }
    }
//! resubstitute是输出，把后端的解输出到前端
void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
{
	assert(x.size() == CPARS+nFrames*8);

	VecXf xF = x.cast<float>();
	HCalib->step = - x.head<CPARS>();

	Mat18f* xAd = new Mat18f[nFrames*nFrames];
	VecCf cstep = xF.head<CPARS>();

//    std::cout << "....cstep...." << cstep.matrix() << std::endl;

	for(EFFrame* h : frames) {
        // 后端数据生成，推送到前端( ->data )
		h->data->step.head<8>() = - x.segment<8>(CPARS+8*h->idx);
		h->data->step.tail<2>().setZero();

		for(EFFrame* t : frames)
            // xAd[h][t] = x[h][0].t() * adHostF[t][h] + x[t][0].t() * adTargetF[t][h]
			xAd[nFrames*h->idx + t->idx] = xF.segment<8>(CPARS+8*h->idx).transpose()
                                           * adHostF[h->idx+nFrames*t->idx]
                                           + xF.segment<8>(CPARS+8*t->idx).transpose()
                                             * adTargetF[h->idx+nFrames*t->idx];
	}
    //! 通过逆深度对于光度内参的偏导，乘以光度参数的delta，可以得到优化更新的逆深度
    resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0,0);

	delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(
        const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++) {
        // 后端的points
		EFPoint* p = allPoints[k];

		int ngoodres = 0;
		for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
		if(ngoodres==0) {
			p->data->step = 0;
			continue;
		}
		float b = p->bdSumF;
//        std::cout << " 1 b: " << b << std::endl;
//        std::cout << " xc: " << xc.matrix() << std::endl;
		b -= xc.dot(p->Hcd_accAF /*+ p->Hcd_accLF*/);
//        std::cout << " 2 b: " << b << std::endl;

		for(EFResidual* r : p->residualsAll) {
			if(!r->isActive()) continue;
			b -= xAd[r->hostIDX*nFrames + r->targetIDX] * r->JpJdF;

//            std::cout << " 3 b: " << b << std::endl;
		}

//        assert(std::isfinite(b));
//        assert(std::isfinite(p->HdiF));
		p->data->step = - b*p->HdiF;
		assert(std::isfinite(p->data->step));
	}
}

double EnergyFunctional::calcMEnergyF()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	VecX delta = getStitchedDeltaF();
	return delta.dot(2*bM + HM*delta);
}

EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
{
//  存雅克比
	EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
	efr->idxInAll = r->point->efPoint->residualsAll.size();
	r->point->efPoint->residualsAll.push_back(efr);
	if(efr->data->stereoResidualFlag == false)
	    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;
//     connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;
	nResiduals++;
	r->efResidual = efr;
	
	return efr;
}
EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib)
{
	EFFrame* eff = new EFFrame(fh);
	eff->idx = frames.size();
	frames.push_back(eff);

	nFrames++;
	fh->efFrame = eff;
	//stereo
    // 这里增加右图
	EFFrame* eff_right = new EFFrame(fh->frame_right);
	eff_right->idx = frames.size()+1000000;
	fh->frame_right->efFrame = eff_right;

#ifndef NEW_METHOD
	assert(HM.cols() == 8*nFrames+CPARS-8);
	bM.conservativeResize(8*nFrames+CPARS);
	HM.conservativeResize(8*nFrames+CPARS,8*nFrames+CPARS);
	bM.tail<8>().setZero();
	HM.rightCols<8>().setZero();
	HM.bottomRows<8>().setZero();
#endif
#ifdef NEW_METHOD
    //! 新加入一帧，会给列扩容
    JM.conservativeResize(JM.rows(), 8 * nFrames + CPARS);
    JM.rightCols<8>().setZero();
#endif

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	setAdjointsF(Hcalib);
	makeIDX();

	for(EFFrame* fh2 : frames) {
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0,0);
		if(fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0,0);
	}

	return eff;
}
EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
{
	EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
	efp->idxInPoints = ph->host->efFrame->points.size();
	ph->host->efFrame->points.push_back(efp);

	nPoints++;
	ph->efPoint = efp;

	EFIndicesValid = false;

	return efp;
}

void EnergyFunctional::dropResidual(EFResidual* r)
{
	EFPoint* p = r->point;
	assert(r == p->residualsAll[r->idxInAll]);

	p->residualsAll[r->idxInAll] = p->residualsAll.back();
	p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
	p->residualsAll.pop_back();

	if(r->isActive())
		r->host->data->shell->statistics_goodResOnThis++;
	else
		r->host->data->shell->statistics_outlierResOnThis++;

	if(r->data->stereoResidualFlag == false)
		connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
//     connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
	nResiduals--;
	r->data->efResidual=0;
	delete r;
}
void EnergyFunctional::marginalizeFrame(EFFrame* fh)
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	assert((int)fh->points.size()==0);

#ifdef NEW_METHOD
    //! 在JM, rM的idx所在的8列下增加Lambda  Lambda * alpha如下
    //! JM              rM                   ==        JM'       rM'
    //! Lambda          Lambda * alpha
    //!  JM'.transpose * JM' = JM.transpose * JM + Lambda^2
    //!  JM'.transpose * rM' = JM.transpose * rM + Lambda * (Lambda * alpha)
    //! Lambda^2即prior.asDiagonal()， alpha即delta_prior
    if (fh->prior(6) > 0.001) {
//        std::cout << "JM, rM:\n"
//                  << (JM.transpose() * JM).ldlt().solve(JM.transpose() * rM).transpose() << std::endl;
//        std::cout << "HM, bM:\n"
//                  << HM.ldlt().solve(bM).transpose() << std::endl;
        //!void add_lambda_frame(MatXXc &J, VecXc &r, int idx, Vec8c Lambda, Vec8c alpha);
        add_lambda_frame(JM, rM, fh->idx,
                         fh->prior_new_method,
                         fh->delta_prior.cast<rkf_scalar>());
//        std::cout << "marg frame: " << fh->idx << std::endl;
//        std::cout << "fh->prior fh->delta_prior:\n" << fh->prior.transpose() << "\n"
//                  << fh->delta_prior.transpose() << std::endl;

//        std::cout << "JM:\n" << JM << std::endl;
//        std::cout << "rM:\n" << rM.transpose() << std::endl;
    }
#endif
    std::cout << "marg frame: " << fh->idx << std::endl;
#ifndef NEW_METHOD
	int ndim = nFrames*8+CPARS-8;// new dimension
	int odim = nFrames*8+CPARS;// old dimension

//	VecX eigenvaluesPre = HM.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//

    //! 如果fh的index不是frames的最后一个
	if((int)fh->idx != (int)frames.size()-1) {
		int io = fh->idx*8+CPARS;	// index of frame to move to end
		int ntail = 8*(nFrames-fh->idx-1);
		assert((io+8+ntail) == nFrames*8+CPARS);

        //! 用eigen的矩阵操作将bM和HM里的目标内容移到右下角和末尾
		Vec8 bTmp = bM.segment<8>(io);
		VecX tailTMP = bM.tail(ntail);
		bM.segment(io,ntail) = tailTMP;
		bM.tail<8>() = bTmp;

		MatXX HtmpCol = HM.block(0,io,odim,8);
		MatXX rightColsTmp = HM.rightCols(ntail);
		HM.block(0,io,odim,ntail) = rightColsTmp;
		HM.rightCols(8) = HtmpCol;

		MatXX HtmpRow = HM.block(io,0,8,odim);
		MatXX botRowsTmp = HM.bottomRows(ntail);
		HM.block(io,0,ntail,odim) = botRowsTmp;
		HM.bottomRows(8) = HtmpRow;
	}


//	// marginalize. First add prior here, instead of to active.
    HM.bottomRightCorner<8,8>().diagonal() += fh->prior;
    //! cwiseProduct 点乘
    //! 逐点相乘，如果是两个向量，结果也为一个等维度向量
    //! 因为这个fh->prior是在对角线上，所以会有这么一个操作
    bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);
//    assert(fh->prior.asDiagonal() * fh->delta_prior
//           == fh->prior.cwiseProduct(fh->delta_prior));

#endif
//    ! 只有DSO碰到的第一帧会有fh->prior，需要特别加成
//    std::cout << "fh->prior fh->delta_prior:\n" << fh->prior.transpose() << "\n"
//            << fh->delta_prior.transpose() << std::endl;

//!  边缘化目标帧
#ifdef NEW_METHOD
    marg_frame(JM, rM, fh->idx);
#else
//	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";

	VecX SVec = (HM.diagonal().cwiseAbs()+VecX::Constant(HM.cols(), 10)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();

//	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() << "\n\n";
//	std::cout << std::setprecision(16) << "SVecI: " << SVecI.transpose() << "\n\n";

	// scale!
	MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
	VecX bMScaled =  SVecI.asDiagonal() * bM;

	// invert bottom part!
	Mat88 hpi = HMScaled.bottomRightCorner<8,8>();
	hpi = 0.5f*(hpi+hpi);
	hpi = hpi.inverse();
	hpi = 0.5f*(hpi+hpi);

	// schur-complement!
    //! 把右下角舒尔补掉了
	MatXX bli = HMScaled.bottomLeftCorner(8,ndim).transpose() * hpi;
	HMScaled.topLeftCorner(ndim,ndim).noalias() -=
            bli * HMScaled.bottomLeftCorner(8,ndim);
	bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<8>();

	//unscale!
	HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	bMScaled = SVec.asDiagonal() * bMScaled;

	// set.
    //! 把右下角丢弃
	HM = 0.5*(HMScaled.topLeftCorner(ndim,ndim) +
            HMScaled.topLeftCorner(ndim,ndim).transpose());
	bM = bMScaled.head(ndim);

    if (fh->prior(6) > 0.001) {
        std::cout << "marg JM, rM:\n"
                  << (JM.transpose() * JM).ldlt().solve(JM.transpose() * rM).transpose() << std::endl;
        std::cout << "marg HM, bM:\n"
                  << HM.ldlt().solve(bM).transpose() << std::endl;
    }

//    std::cout << "bM: " << bM.size() << std::endl;
//    std::cout << "bMScaled: " << bMScaled.size() << std::endl;

#endif

	// remove from vector, without changing the order!
	for(unsigned int i=fh->idx; i+1<frames.size();i++) {
		frames[i] = frames[i+1];
		frames[i]->idx = i;
	}
	frames.pop_back();
	nFrames--;
    //! 这里是让efFrame从上级的FrameHessian中脱钩，然后在本函数的末尾会delete fh
    fh->data->efFrame = 0;

#ifndef NEW_METHOD
	assert((int)frames.size()*8+CPARS == (int)HM.rows());
	assert((int)frames.size()*8+CPARS == (int)HM.cols());
	assert((int)frames.size()*8+CPARS == (int)bM.size());
	assert((int)frames.size() == (int)nFrames);
#endif

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	makeIDX();
	delete fh;
}

void EnergyFunctional::marginalizePointsF()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	allPointsToMarg.clear();
	for(EFFrame* f : frames) {
		for(int i=0;i<(int)f->points.size();i++) {
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_MARGINALIZE) {
				p->priorF *= setting_idepthFixPriorMargFac;
				for(EFResidual* r : p->residualsAll)
					if(r->isActive())
						if(r->data->stereoResidualFlag == false)
							 connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
				allPointsToMarg.push_back(p);
			}
		}
	}
//    std::cout << "allPointsToMarg.size(): " << allPointsToMarg.size() << std::endl;

//    MatXXc M = MatXXc::Zero(accSSE_top_A->nframes[0]*8+CPARS, accSSE_top_A->nframes[0]*8+CPARS);
//    VecXc Mb = VecXc::Zero(accSSE_top_A->nframes[0] * 8+CPARS);
#ifdef NEW_METHOD
#endif

    //! HM, bM是祖传的。这里M - Msc, Mb - Mbsc是本次新的增量，再加到祖传的HM，bM上。

	accSSE_bot->setZero(nFrames);
	accSSE_top_A->setZero(nFrames);

    int total_rows = 0;
	for(EFPoint* p : allPointsToMarg) {
        accSSE_top_A->addPoint<2>( p,this);
        //! 这之前要把marg掉的JM, rM提取出来，否则点删掉了就没了
//		removePoint(p);
        total_rows += p->Jr1.rows();
	}
//    std::cout << "totol_rows: " << total_rows << std::endl;
//    if (rM.rows() > 10)
//        std::cout << "rM      1:\n" << rM.topRows(10).transpose() << std::endl;
    int m = JM.rows();
    JM.conservativeResize(m + total_rows, JM.cols());
    rM.conservativeResize(m + total_rows);
//    JM.bottomRows(total_rows).setZero();
//    rM.bottomRows(total_rows).setZero();
    for (EFPoint *p : allPointsToMarg) {
        JM.middleRows(m, p->Jr1.rows()) = 0.5 * p->Jr1;
        rM.middleRows(m, p->Jr2.rows()) = 0.5 * p->Jr2;
        m += p->Jr2.rows();
//        removePoint(p);
    }
    for (EFPoint *p : allPointsToMarg) {
        removePoint(p);
    }
//    std::cout << "before compress JM, rM:\n"
//              << (JM.transpose() * JM).ldlt().solve(JM.transpose() * rM).transpose() << std::endl;
#if 0
    timer_ACC4.tic();
    std::vector<MatXXc> JMs;
    std::vector<VecXc> rMs;
    int JMs_step = JM.rows() / NUM_THREADS + 1;
    if (JMs_step < 1000)
        JMs_step = 1000;
    m = 0;
    std::cout << "JM rows: " << JM.rows() << std::endl;
    while (m + JMs_step < JM.rows()) {
        MatXXc temp1 = JM.middleRows(m, JMs_step);
        VecXc temp2  = rM.middleRows(m, JMs_step);
        JMs.push_back(temp1);
        rMs.push_back(temp2);
        m += JMs_step;
    }
    JMs.push_back(JM.bottomRows(JM.rows() - m));
    rMs.push_back(rM.bottomRows(rM.rows() - m));
    std::cout << "JMs.size()" << JMs.size() << std::endl;
    assert(JMs.size() <= NUM_THREADS);

    timer_ACC3.tic();
    compress_JrMT(red, &JMs, &rMs, this, true);
//    compress_Jr(JM, rM);
    times_ACC3 += timer_ACC3.toc();


    int rows_of_JM_compressed = 0;
    for (int i = 0; i < JMs.size(); i++)
        rows_of_JM_compressed += JMs[i].rows();
    int cols_of_JM_compressed = JM.cols();

    JM = MatXXc::Zero(rows_of_JM_compressed, cols_of_JM_compressed);
    rM = VecXc::Zero(rows_of_JM_compressed);

    m = 0;
    for (int i = 0; i < JMs.size(); i++) {
        JM.middleRows(m, JMs[i].rows()) = JMs[i];
        rM.middleRows(m, rMs[i].rows()) = rMs[i];
        m += JMs[i].rows();
    }
    times_ACC4 += timer_ACC4.toc();

    std::cout << "marg p multi: " << times_ACC3 << std::endl;
    std::cout << "marg p total: " << times_ACC4 << std::endl;
#else
    timer_ACC4.tic();
    compress_Jr(JM, rM);
    times_ACC4 += timer_ACC4.toc();
#endif
    std::cout << "marg p single: " << times_ACC4 << std::endl;

    std::cout << "JM size: " << JM.rows() << " " << JM.cols() << std::endl;
//    std::cout << "JM^T * rM:\n" << (JM.transpose() * rM).transpose() << std::endl;
//    std::cout << "JM       :\n" << JM_new << std::endl;
//    if (rM.rows() > 10)
//        std::cout << "rM       :\n" << rM.topRows(10).transpose() << std::endl;

//    std::cout << "after compress JM, rM:\n"
//            << (JM.transpose() * JM).ldlt().solve(JM.transpose() * rM).transpose() << std::endl;

//    VecXc x_new;
//    std::vector<MatXXc > JMs;
//    std::vector<VecXc > rMs;
//    JMs.push_back(JM);
//    rMs.push_back(rM);
//    pcgMT(red, &JMs, &rMs, this, x_new, 1e-8, 100, false);
//    if (JM.rows() > 0) {
//        cg(JM, rM, x_new, 1e-8, JM.cols());
//        std::cout << "cg:\n" << x_new.transpose() << std::endl;
//    }

	resInM+= accSSE_top_A->nres[0];

#ifndef NEW_METHOD
	MatXX H = M.cast<myscalar>();
    VecX b = Mb.cast<myscalar>();
    // 每一个点都对应一个完整的H, b。或者说，Marg: point -> (H_nxn, b_n)
    //! 点删掉了，但是相应的约束信息整合到HM，bM里了。
	HM += setting_margWeightFac*H;
	bM += setting_margWeightFac*b;
#endif

//    if (bM.rows() > 10)
//        std::cout << "bM       :\n" << bM.topRows(10).transpose() << std::endl;
//    std::cout << "bM size in marg points: " << bM.size() << std::endl;

//    std::cout << "reference HM, bM:\n"
//            << HM.ldlt().solve(bM).transpose() << std::endl;
//    std::cout << std::endl;

	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::dropPointsF()
{
	for(EFFrame* f : frames) {
		for(int i=0;i<(int)f->points.size();i++) {
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_DROP) {
				removePoint(p);
				i--;
			}
		}
	}

	EFIndicesValid = false;
	makeIDX();
}


void EnergyFunctional::removePoint(EFPoint* p)
{
	for(EFResidual* r : p->residualsAll)
		dropResidual(r);

	EFFrame* h = p->host;
	h->points[p->idxInPoints] = h->points.back();
	h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
	h->points.pop_back();

	nPoints--;
	p->data->efPoint = 0;

	EFIndicesValid = false;

	delete p;
}

void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
{
//	VecX eigenvaluesPre = H.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";

	// decide to which nullspaces to orthogonalize.
	std::vector<VecX> ns;
	ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
	ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
//	if(setting_affineOptModeA <= 0)
//		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
//	if(setting_affineOptModeB <= 0)
//		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());

	// make Nullspaces matrix
	MatXX N(ns[0].rows(), ns.size());
	for(unsigned int i=0;i<ns.size();i++)
		N.col(i) = ns[i].normalized();

	// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
	Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

	VecX SNN = svdNN.singularValues();
	myscalar minSv = 1e10, maxSv = 0;
	for(int i=0;i<SNN.size();i++) {
		if(SNN[i] < minSv) minSv = SNN[i];
		if(SNN[i] > maxSv) maxSv = SNN[i];
	}
	for(int i=0;i<SNN.size();i++)
		{ if(SNN[i] > setting_solverModeDelta*maxSv) SNN[i] = 1.0 / SNN[i]; else SNN[i] = 0; }

	MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
	MatXX NNpiT = N*Npi.transpose(); 	// [dim] x [dim].
	MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

	if(b!=0) *b -= NNpiTS * *b;
	if(H!=0) *H -= NNpiTS * *H * NNpiTS;

//	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

//	VecX eigenvaluesPost = H.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
//	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";
}

void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib)
{
    timer_ACC5.tic();

#ifdef NEW_METHOD
//    test_qr();
//    exit(0);
#endif
    TicToc timer_solveSystemF;

	if(setting_solverMode & SOLVER_USE_GN) lambda=0;
	if(setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);


	MatXX  HA_top;
	VecX   bA_top, bM_top;

    timer_ACC1.tic();
    accumulateAF_MT(HA_top, bA_top, multiThreading);
    times_ACC1 += timer_ACC1.toc();

    //! 这里的关键是zero点，也就是固定线性化点，之前制作HM, bM的时候，在固定线性化点做了一次优化，然后再回退，
    //! 即回退到了zero点，制作了HM，bM;
    //! 之后应该是在前端一直保存了zero点，所以每次要使用这个祖传HM, bM的时候，都需要现求一次state - state_zero
    //! 然后将这个bM调整如下
#ifndef NEW_METHOD
	bM_top = (bM+ HM * getStitchedDeltaF());
#endif
#ifdef NEW_METHOD
    VecXc rM_top;
    if (JM.rows() > 0) {
        rM_top = rM + JM * getStitchedDeltaF().cast<rkf_scalar>();
        Js.push_back(JM);
        rs.push_back(rM_top);
    } else {
        std::cout << "JM is empty" << std::endl;
        std::cout << "HM:\n" << HM << std::endl;
        std::cout << "bM:\n" << bM.transpose() << std::endl;
//        assert(false);
    }
#endif
//    std::cout << "getStitchedDeltaF(): " << getStitchedDeltaF() << std::endl;

#ifdef NEW_METHOD
//! 加Damping
    MatXXc Damping = 1e-3 * MatXXc::Identity(Js[0].cols(), Js[0].cols());
    Js.push_back(Damping);
    rs.push_back(VecXc::Zero(Js[0].cols()));
#endif

#if 1
    int total = 0;
    for (int i = 0; i < Js.size(); i++) {
        total += Js[i].rows();
    }
    std::cout << "total " << total << std::endl;

    MatXXc JJ = MatXXc::Zero(total, Js[0].cols());
    VecXc rr = VecXc::Zero(total);
    int m = 0;
    for (int i = 0; i < Js.size(); i++) {
        JJ.middleRows(m, Js[i].rows()) = Js[i];
        rr.middleRows(m, rs[i].rows()) = rs[i];
        m += rs[i].rows();
    }
//    compress_Jr(JJ, rr);

    //! normalize full cols
//    VecXc JJ_norms = VecXc::Zero(JJ.cols());
//    for (int i = 0; i < JJ.cols(); i++) {
//        JJ_norms(i) = JJ.col(i).norm();
//        JJ.col(i) = JJ.col(i) / JJ_norms(i);
//    }
//    for (int i = 0; i < JJ.cols(); i++)
//        JJ_norms(i) = 1.0 / JJ_norms(i);

//    std::vector<MatXXc> JJs;
//    std::vector<VecXc> rrs;
//    JJs.push_back(JJ);
//    rrs.push_back(rr);
//    int JJs_step = JJ.rows() / NUM_THREADS + 8;
//    m = 0;
//    while (m + JJs_step < JJ.rows()) {
//        JJs.push_back(JJ.middleRows(m, JJs_step));
//        rrs.push_back(rr.middleRows(m, JJs_step));
//        m += JJs_step;
//    }
//    JJs.push_back(JJ.middleRows(m, JJ.rows() - m));
//    rrs.push_back(rr.middleRows(m, rr.rows() - m));
//    assert(JJs.size() <= NUM_THREADS);
//    std::cout << "JJs.size: " << JJs.size() << std::endl;

//    timer_ACC4.tic();
//    compress_JrMT(red, &JJs, &rrs, this, true);
//    times_ACC4 += timer_ACC4.toc();
#endif

//    std::vector<int> indices;
//    for (int i = 0; i < JJs.size(); i++)
//        indices.push_back(i);

//    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
//                  [&](auto &i) {
//    for (int i = 0; i < JJs.size(); i++)
//                      compress_Jr(JJs[i], rrs[i]);
//                  });

//    std::cout << "JJs size " << JJs.size() << std::endl;

//    timer_ACC4.tic();
//    compress_JrMT(red, &JJs, &rrs, this, true);
//    times_ACC4 += timer_ACC4.toc();

//    std::cout << "rows:.... " << JJs[23].rows() << std::endl;



//    compress_Jr(JJ, rr);
//    std::cout << "Js:\n" << Js[Js.size() - 1] << std::endl;
//    std::cout << "rr:\n" << rs[rs.size() - 1].transpose() << std::endl;
//    VecXc x_new;
//    pcgMT(red, &Js, &rs, this, x_new, 1e-10, 100, false);

//    compress_Jr(JJ, rr);

#if 1
    timer_ACC2.tic();
    Eigen::LeastSquaresConjugateGradient<MatXXc > lscg;
    lscg.setMaxIterations(1000);
    lscg.setTolerance(1e-2);
    lscg.compute(JJ);
    VecXc y = lscg.solve(rr);
    times_ACC2 += timer_ACC2.toc();
//    std::cout << "lscg  x:\n" << y.transpose() << std::endl;
    std::cout << "lscg iter: " << lscg.iterations() << std::endl;
    std::cout << "lscg error: " << lscg.error() << std::endl;

//    std::cout << "JJ cols:   " << JJ.cols() << std::endl;

//    VecXc y;

//    cg(JJ, rr, y, 1e-6, 10);
//    timer_ACC4.tic();
//    compress_Jr(JJ, rr);
//    times_ACC4 += timer_ACC4.toc();

//    timer_ACC4.tic();
//    MatXXc JJJJ = JJ.transpose() * JJ;
//    VecXc rrrr = JJ.transpose() * rr;
//    times_ACC4 += timer_ACC4.toc();
//    y = JJ.householderQr().solve(rr);
//    y = JJJJ.ldlt().solve(rrrr);

//    timer_ACC2.tic();
//    pcg_orig(JJJJ, rrrr, y, 1e-6, 1000);
//    leastsquare_pcg_orig(JJ, rr, y, 1e-6, 1000);
//    times_ACC2 += timer_ACC2.toc();
//    y = JJ_norms.asDiagonal() * y;

//    timer_ACC2.tic();
//    y = JJ.householderQr().solve(rr);
//    leastsquare_pcg_origMT(red, &Js, &rs, this, y, 1e-4, 20000, false);
//    times_ACC2 += timer_ACC2.toc();

//    y = JJ_norms.asDiagonal() * y;

    times_ACC5 += timer_ACC5.toc();


//    VecXc yy;
//    timer_ACC4.tic();
//    leastsquare_pcg_origMT(red, &Js, &rs, this,  yy, 1e-6, 1000, false);
//    times_ACC4 += timer_ACC4.toc();


    std::cout << "............" << std::endl;
    std::cout << "qr  " << times_ACC1 << std::endl;
//    std::cout << "optimize" << times_ACC1 << std::endl;
    std::cout << "pcg " << times_ACC2 << std::endl;
//    std::cout <<"marginalize " << times_ACC4 << std::endl;
    std::cout <<"solveS " << times_ACC5 << std::endl;
//    std::cout << num_of_iter << std::endl;
//    y = JJJJ.ldlt().solve(rrrr);
//! 可能是没有保证正定的缘故
//! 可以在sandbox制作一个半正定的矩阵试试
//    pcgMT(red, &Js, &rs, this, yy, 1e-6, 100, true);
//    std::cout << "pcgMT x:\n" << yy.transpose() << std::endl;
//    cg(JJ, rr, yy2, 1e-5, 100);
//    exit(0);

#else
    VecXc y;
//    y = (JJ.transpose() * JJ).ldlt().solve(JJ.transpose() * rr);
//    pcgMT(red, &Js, &rs, this, y, 1e-6, 10, false);
    assert(JJ.rows() > 0);
    std::cout << "JJ:\n" << JJ << std::endl;
    std::cout << "rr:\n" << rr.transpose() << std::endl;
    pcg(JJ, rr, y, 1e-6, 10);
#endif
//    y.setZero();
//    pcg(JJ, rr, y, 1e-6, 200);
//    std::cout << "pcg   x:\n" << y.transpose() << std::endl;

#ifndef NEW_METHOD
	MatXX HFinal_top;
	VecX bFinal_top;

    {  // 就是 M + A + SC，只是因爲lambda稍有改動
		HFinal_top =  HM + HA_top;
		bFinal_top =  bM_top + bA_top /*- b_sc*/;

        // lastHS和lastbS只有打印時會調用，無實際作用
		lastHS = HFinal_top /*- H_sc*/;
		lastbS = bFinal_top;

//		for(int i=0;i<8*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);

//        HFinal_top -= H_sc * (1.0f/(1+lambda));

        // 上面的把HM, HA_top略爲放大，把H_sc略爲縮小，整體的HFinal_top略爲放大
//        std::cout << "H:\n" << HFinal_top << std::endl;
//        std::cout << "b:\n" << bFinal_top << std::endl;
	}

    auto times_solveSystemF = timer_solveSystemF.toc();
//    std::cout << "solveSystemF cost time " << times_solveSystemF << std::endl;
//    exit(1);

	VecX x;

//    printf("setting solverMode: %x\n", setting_solverMode);
	if(setting_solverMode & SOLVER_SVD) {
		VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		VecX bFinalScaled  = SVecI.asDiagonal() * bFinal_top;
		Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX S = svd.singularValues();
		myscalar minSv = 1e10, maxSv = 0;
		for(int i=0;i<S.size();i++) {
			if(S[i] < minSv) minSv = S[i];
			if(S[i] > maxSv) maxSv = S[i];
		}

		VecX Ub = svd.matrixU().transpose()*bFinalScaled;
		int setZero=0;
		for(int i=0;i<Ub.size();i++) {
			if(S[i] < setting_solverModeDelta*maxSv)
			{ Ub[i] = 0; setZero++; }

			if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7))
			{ Ub[i] = 0; setZero++; }
			else Ub[i] /= S[i];
		}
		x = SVecI.asDiagonal() * svd.matrixV() * Ub;
	} else {
//        printf("........hello\n");
		VecX SVecI = (HFinal_top.diagonal()+VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
	}

//    std::cout << "x_new: " << x_new.transpose() << std::endl;
//    std::cout << "x_old: " << HFinal_top.ldlt().solve(bFinal_top).transpose() << std::endl;
    std::cout <<   "x    : " << x.transpose() << std::endl;

//    MatXXc H_rkf2 = MatXXc::Zero(CPARS + nFrames * 8, CPARS + nFrames * 8);
//    VecX b_rkf2 = VecX::Zero(CPARS + nFrames * 8);
//
//    for (int i = 0; i < Js.size(); i++) {
//        H_rkf2 += Js[i].transpose() * Js[i];
//        b_rkf2 += Js[i].transpose() * rs[i];
//    }
//    std::cout << "x_3rd: " << H_rkf2.ldlt().solve(b_rkf2).transpose() << std::endl;
//    std::cout << std::endl;
#endif

#ifdef NEW_METHOD
    VecX x = y.cast<double>();
#endif
	if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) ||
            (iteration >= 2 &&
                    (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER))) {
		VecX xOld = x;
		orthogonalize(&x, 0);
	}

	lastX = x;

	//resubstituteF(x, HCalib);
	resubstituteF_MT(x, HCalib,multiThreading);
}
void EnergyFunctional::makeIDX()
{
	for(unsigned int idx=0;idx<frames.size();idx++)
		frames[idx]->idx = idx;

	allPoints.clear();

	for(EFFrame* f : frames)
		for(EFPoint* p : f->points) {
			allPoints.push_back(p);
			for(EFResidual* r : p->residualsAll) {
				r->hostIDX = r->host->idx;
				r->targetIDX = r->target->idx;
				if(r->data->stereoResidualFlag == true) {
				  r->targetIDX = frames[frames.size()-1]->idx;
				}
			}
		}

	EFIndicesValid=true;
}

VecX EnergyFunctional::getStitchedDeltaF() const
{
	VecX d = VecX(CPARS+nFrames*8).setZero();
    d.head<CPARS>() = cDeltaF.cast<myscalar>();
    // rkf
	for(int h=0;h<nFrames;h++)
        d.segment<8>(CPARS+8*h) = frames[h]->delta;
	return d;
}

}
