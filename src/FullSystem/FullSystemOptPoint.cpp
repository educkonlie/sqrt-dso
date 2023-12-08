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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ImmaturePoint.h"
#include "math.h"

namespace dso
{

#ifndef DSO_LITE
PointHessian* FullSystem::optimizeImmaturePoint(
		ImmaturePoint* point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
{
	int nres = 0;
	for(FrameHessian* fh : frameHessians) {
        // 为该点建立到其他帧的残差项
		if(fh != point->host) {
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh;
			nres++;
		}
	}
	assert(nres == ((int)frameHessians.size())-1);

	bool print = false;//rand()%50==0;

    static int k = 0;
    if (k++ < 4)
        print = true;
    else
        print = false;

	float lastEnergy = 0;
	float lastHdd=0;
	float lastbd=0;
	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f;

	for(int i=0;i<nres;i++) {
        // 返回的除了lastEnergy还有lastHdd, lastbd
		lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i,
                                               lastHdd, lastbd, currentIdepth);
		residuals[i].state_state = residuals[i].state_NewState;
		residuals[i].state_energy = residuals[i].state_NewEnergy;
	}

    // Hdd矩阵其实是一个实数，因为逆深度是一维的
    // 所以是單點直接高斯牛頓的
	if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) {
		if(print)
			printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
				nres, lastHdd, lastEnergy);
		return 0;
	}

	if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
			nres, lastHdd,lastEnergy,currentIdepth);

	float lambda = 0.1;
	for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++) {
		float H = lastHdd;
		H *= 1+lambda;
        // step就是逆深度的改变量
		float step = (1.0/H) * lastbd;
		float newIdepth = currentIdepth - step;

		float newHdd=0; float newbd=0; float newEnergy=0;
		for(int i=0;i<nres;i++)
			newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i,
                                                  newHdd, newbd, newIdepth);
        // Hx = b
		if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act) {
			if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres,
					newHdd,
					lastEnergy);
			return 0;
		}

		if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				"",
				lastEnergy, newEnergy, newIdepth);

		if(newEnergy < lastEnergy) {
			currentIdepth = newIdepth;
			lastHdd = newHdd;
			lastbd = newbd;
			lastEnergy = newEnergy;
			for(int i=0;i<nres;i++) {
				residuals[i].state_state = residuals[i].state_NewState;
				residuals[i].state_energy = residuals[i].state_NewEnergy;
			}

			lambda *= 0.5;
		} else {
			lambda *= 5;
		}

		if(fabsf(step) < 0.0001*currentIdepth)
			break;
	}

	if(!std::isfinite(currentIdepth)) {
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

	int numGoodRes=0;
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) numGoodRes++;

	if(numGoodRes < minObs) {
		if(print) printf("OptPoint: OUTLIER!\n");
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

    // 如果p成功成为活跃点，则加入PBA
	PointHessian* p = new PointHessian(point, &Hcalib);
	if(!std::isfinite(p->energyTH)) {delete p; return (PointHessian*)((long)(-1));}

	p->lastResiduals[0].first = 0;
	p->lastResiduals[0].second = ResState::OOB;
	p->lastResiduals[1].first = 0;
	p->lastResiduals[1].second = ResState::OOB;
	p->setIdepthZero(currentIdepth);
	p->setIdepth(currentIdepth);
	p->setPointStatus(PointHessian::ACTIVE);

    // 给该点增加右目优化
    for (int i = 0; i < 1/*coupling_factor*/; i++) {
        PointFrameResidual *r = new PointFrameResidual(p, p->host, p->host->frame_right);
        r->state_NewEnergy = r->state_energy = 0;
        r->state_NewState = ResState::OUTLIER;
        r->setState(ResState::IN);
        // 属于stereo优化，即右目优化
        r->stereoResidualFlag = true;
        p->residuals.push_back(r);
    }
// temporal的能量项
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) {
			PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
			r->state_NewEnergy = r->state_energy = 0;
			r->state_NewState = ResState::OUTLIER;
			r->setState(ResState::IN);
			p->residuals.push_back(r);

            // p -> lastResiduals里特别记录到最后两帧的残差
			if(r->target == frameHessians.back()) {
				p->lastResiduals[0].first = r;
				p->lastResiduals[0].second = ResState::IN;
			} else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2])) {
				p->lastResiduals[1].first = r;
				p->lastResiduals[1].second = ResState::IN;
			}
		}

	if(print) printf("point activated!\n");

	statistics_numActivatedPoints++;
	return p;
}
#else
PointHessian* FullSystem::optimizeImmaturePoint(
		ImmaturePoint* point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
{
	int nres = 0;
	for(FrameHessian* fh : frameHessians) {
        // 为该点建立到其他帧的残差项
		if(fh != point->host) {
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh;
			nres++;
		}
	}
	assert(nres == ((int)frameHessians.size())-1);

	bool print = false;//rand()%50==0;

    static int k = 0;
    if (k++ < 4)
        print = true;
    else
        print = false;

	float lastEnergy = 0;
	float lastHdd=0;
	float lastbd=0;
	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f;

	for(int i=0;i<nres;i++) {
        // 返回的除了lastEnergy还有lastHdd, lastbd
		lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i,
                                               lastHdd, lastbd, currentIdepth);
		residuals[i].state_state = residuals[i].state_NewState;
		residuals[i].state_energy = residuals[i].state_NewEnergy;
	}

    // Hdd矩阵其实是一个实数，因为逆深度是一维的
    // 所以是單點直接高斯牛頓的
	if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) {
		if(print)
			printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
				nres, lastHdd, lastEnergy);
		return 0;
	}

	if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
			nres, lastHdd,lastEnergy,currentIdepth);

	float lambda = 0.1;
	for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++) {
		float H = lastHdd;
		H *= 1+lambda;
        // step就是逆深度的改变量
		float step = (1.0/H) * lastbd;
		float newIdepth = currentIdepth - step;

		float newHdd=0; float newbd=0; float newEnergy=0;
		for(int i=0;i<nres;i++)
			newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i,
                                                  newHdd, newbd, newIdepth);
        // Hx = b
		if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act) {
			if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres,
					newHdd,
					lastEnergy);
			return 0;
		}

		if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				"",
				lastEnergy, newEnergy, newIdepth);

		if(newEnergy < lastEnergy) {
			currentIdepth = newIdepth;
			lastHdd = newHdd;
			lastbd = newbd;
			lastEnergy = newEnergy;
			for(int i=0;i<nres;i++) {
				residuals[i].state_state = residuals[i].state_NewState;
				residuals[i].state_energy = residuals[i].state_NewEnergy;
			}

			lambda *= 0.5;
		} else {
			lambda *= 5;
		}

		if(fabsf(step) < 0.0001*currentIdepth)
			break;
	}

	if(!std::isfinite(currentIdepth)) {
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

	int numGoodRes=0;
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) numGoodRes++;

	if(numGoodRes < minObs) {
		if(print) printf("OptPoint: OUTLIER!\n");
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

    // 如果p成功成为活跃点，则加入PBA
	PointHessian* p = new PointHessian(point, &Hcalib);
	if(!std::isfinite(p->energyTH)) {delete p; return (PointHessian*)((long)(-1));}

	p->lastResiduals[0].first = 0;
	p->lastResiduals[0].second = ResState::OOB;
	p->lastResiduals[1].first = 0;
	p->lastResiduals[1].second = ResState::OOB;
	p->setIdepthZero(currentIdepth);
	p->setIdepth(currentIdepth);
	p->setPointStatus(PointHessian::ACTIVE);

    /*// 给该点增加右目优化
    for (int i = 0; i < 1*//*coupling_factor*//*; i++) {
        PointFrameResidual *r = new PointFrameResidual(p, p->host, p->host->frame_right);
        r->state_NewEnergy = r->state_energy = 0;
        r->state_NewState = ResState::OUTLIER;
        r->setState(ResState::IN);
        // 属于stereo优化，即右目优化
        r->stereoResidualFlag = true;
        p->residuals.push_back(r);
    }*/
// temporal的能量项
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) {
			PointFrameResidual* r1 = new PointFrameResidual(p, p->host,
                                                            residuals[i].target);
            PointFrameResidual* r2 = new PointFrameResidual(p, p->host,
                                                            (residuals[i].target));
			r1->state_NewEnergy = r1->state_energy = 0;
            r2->state_NewEnergy = r2->state_energy = 0;
			r1->state_NewState = ResState::OUTLIER;
            r2->state_NewState = ResState::OUTLIER;
			r1->setState(ResState::IN);
            r2->setState(ResState::IN);
			p->residuals.push_back(r1);
            p->residuals.push_back(r2);

            r2->stereoResidualFlag = true;

            // p -> lastResiduals里特别记录到最后两帧的残差
			if(r1->target == frameHessians.back()) {
				p->lastResiduals[0].first = r1;
				p->lastResiduals[0].second = ResState::IN;
			} else if(r1->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2])) {
				p->lastResiduals[1].first = r1;
				p->lastResiduals[1].second = ResState::IN;
			}
		}

	if(print) printf("point activated!\n");

	statistics_numActivatedPoints++;
	return p;
}
#endif



}
