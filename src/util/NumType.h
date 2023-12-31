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


#pragma once

#include "Eigen/Core"
#include "sophus/sim3.hpp"
#include "sophus/se3.hpp"
#include <glog/logging.h>
#include <ctime>
#include <cstdlib>
#include <chrono>


namespace dso
{

// CAMERA MODEL TO USE


#define SSEE(val,idx) (*(((float*)&val)+idx))


#define MAX_RES_PER_POINT 8
#define NUM_THREADS 15


#define todouble(x) (x).cast<double>()

//#define Quaterniond Sophus::Quaternionf
//#define myscalar float
//typedef Sophus::SE3f SE3;
//typedef Sophus::Sim3f Sim3;
//typedef Sophus::SO3f SO3;

#define Quaterniond Sophus::Quaterniond
#define myscalar double
typedef Sophus::SE3d SE3;
typedef Sophus::Sim3d Sim3;
typedef Sophus::SO3d SO3;

//#define RKF_BASELINE
#ifdef RKF_BASELINE
#define CPARS 5    // 似乎是相机内参
#else
#define CPARS 4    // 似乎是相机内参
#endif

#define NEW_METHOD // 我们的rootba算法将使用这个宏来表明 2023.12.04
#ifdef NEW_METHOD
#define rkf_scalar float
typedef Eigen::Matrix<rkf_scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXXcr;
typedef Eigen::Matrix<rkf_scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatXXc;
//typedef Eigen::Matrix<rkf_scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXXc;
typedef Eigen::Matrix<rkf_scalar, Eigen::Dynamic, 1> VecXc;
typedef Eigen::Matrix<rkf_scalar, 8, 1> Vec8c;
//! 即对所有残差都进行compress_Jr压缩
//#define QR_QR
#endif

#define ROOTBA
#define ROOTBA_PREPARE

#define USE_MYH
//#define USE_ACC_INSTEAD_OF_myH
//#define MatXXg MatXX
//#define VecXg VecX

typedef Eigen::Matrix<myscalar,Eigen::Dynamic,Eigen::Dynamic> MatXX;
//typedef Eigen::Matrix<double,CPARS,CPARS> MatCC;
//#define MatToDynamic(x) MatXX(x)

//typedef Eigen::Matrix<double,CPARS,10> MatC10;
//typedef Eigen::Matrix<myscalar,10,10> Mat1010;
//typedef Eigen::Matrix<myscalar,13,13> Mat1313;

//typedef Eigen::Matrix<myscalar,8,10> Mat810;
//typedef Eigen::Matrix<myscalar,8,3> Mat83;
typedef Eigen::Matrix<myscalar,6,6> Mat66;
//typedef Eigen::Matrix<myscalar,5,3> Mat53;
//typedef Eigen::Matrix<myscalar,4,3> Mat43;
typedef Eigen::Matrix<myscalar,4,2> Mat42;
typedef Eigen::Matrix<myscalar,3,3> Mat33;
//typedef Eigen::Matrix<myscalar,2,2> Mat22;
typedef Eigen::Matrix<myscalar,8,CPARS> Mat8C;
//typedef Eigen::Matrix<double,CPARS,8> MatC8;
//typedef Eigen::Matrix<float,8,CPARS> Mat8Cf;
//typedef Eigen::Matrix<float,CPARS,8> MatC8f;

typedef Eigen::Matrix<myscalar,8,8> Mat88;
//typedef Eigen::Matrix<myscalar,7,7> Mat77;

typedef Eigen::Matrix<myscalar,CPARS,1> VecC;
typedef Eigen::Matrix<rkf_scalar, CPARS, 1> VecCc;
typedef Eigen::Matrix<float,CPARS,1> VecCf;
//typedef Eigen::Matrix<myscalar,13,1> Vec13;
typedef Eigen::Matrix<myscalar,10,1> Vec10;
//typedef Eigen::Matrix<myscalar,16,1> Vec16;
//typedef Eigen::Matrix<myscalar,9,1> Vec9;
typedef Eigen::Matrix<myscalar,8,1> Vec8;
typedef Eigen::Matrix<myscalar,7,1> Vec7;
typedef Eigen::Matrix<myscalar,6,1> Vec6;
typedef Eigen::Matrix<myscalar,5,1> Vec5;
typedef Eigen::Matrix<myscalar,4,1> Vec4;
typedef Eigen::Matrix<myscalar,3,1> Vec3;
typedef Eigen::Matrix<myscalar,2,1> Vec2;
typedef Eigen::Matrix<myscalar,Eigen::Dynamic,1> VecX;
typedef Eigen::Matrix<myscalar,Eigen::Dynamic,1> VectorXd;

typedef Eigen::Matrix<float,3,3> Mat33f;
//typedef Eigen::Matrix<float,10,3> Mat103f;
typedef Eigen::Matrix<float,2,2> Mat22f;
typedef Eigen::Matrix<float,3,1> Vec3f;
typedef Eigen::Matrix<float,2,1> Vec2f;
typedef Eigen::Matrix<float,6,1> Vec6f;
typedef Eigen::Matrix<float,5,1> Vec5f;



//typedef Eigen::Matrix<double,4,9> Mat49;
//typedef Eigen::Matrix<double,8,9> Mat89;
//
//typedef Eigen::Matrix<double,9,4> Mat94;
//typedef Eigen::Matrix<double,9,8> Mat98;
//
//typedef Eigen::Matrix<double,8,1> Mat81;
//typedef Eigen::Matrix<double,1,8> Mat18;
//typedef Eigen::Matrix<double,9,1> Mat91;
//typedef Eigen::Matrix<double,1,9> Mat19;
//
//
//typedef Eigen::Matrix<double,8,4> Mat84;
//typedef Eigen::Matrix<double,4,8> Mat48;
//typedef Eigen::Matrix<double,4,4> Mat44;


typedef Eigen::Matrix<float,MAX_RES_PER_POINT,1> VecNRf;
typedef Eigen::Matrix<float,12,1> Vec12f;
typedef Eigen::Matrix<float,1,8> Mat18f;
typedef Eigen::Matrix<float,6,6> Mat66f;
typedef Eigen::Matrix<float,8,8> Mat88f;
typedef Eigen::Matrix<float,8,4> Mat84f;
typedef Eigen::Matrix<float,8,1> Vec8f;
typedef Eigen::Matrix<float,10,1> Vec10f;
typedef Eigen::Matrix<float,6,6> Mat66f;
typedef Eigen::Matrix<float,4,1> Vec4f;
typedef Eigen::Matrix<float,4,4> Mat44f;
typedef Eigen::Matrix<float,12,12> Mat1212f;
typedef Eigen::Matrix<float,12,1> Vec12f;
typedef Eigen::Matrix<float,13,13> Mat1313f;
typedef Eigen::Matrix<float,10,10> Mat1010f;
typedef Eigen::Matrix<float,13,1> Vec13f;
typedef Eigen::Matrix<float,9,9> Mat99f;
typedef Eigen::Matrix<float,9,1> Vec9f;

typedef Eigen::Matrix<float,4,2> Mat42f;
typedef Eigen::Matrix<float,6,2> Mat62f;
typedef Eigen::Matrix<float,1,2> Mat12f;

typedef Eigen::Matrix<float,Eigen::Dynamic,1> VecXf;
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> MatXXf;
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> MatXXfr;



//typedef Eigen::Matrix<double,8+CPARS+1,8+CPARS+1> MatPCPC;
//typedef Eigen::Matrix<float,8+CPARS+1,8+CPARS+1> MatPCPCf;
typedef Eigen::Matrix<myscalar,8+CPARS+1,8+CPARS+1> MatPCPC;
//typedef Eigen::Matrix<double,8+CPARS+1,1> VecPC;
//typedef Eigen::Matrix<float,8+CPARS+1,1> VecPCf;

//typedef Eigen::Matrix<float,14,14> Mat1414f;
//typedef Eigen::Matrix<float,14,1> Vec14f;
//typedef Eigen::Matrix<double,14,14> Mat1414;
//typedef Eigen::Matrix<double,14,1> Vec14;



// transforms points from one frame to another.
struct AffLight {
	AffLight(double a_, double b_) : a(a_), b(b_) {};
	AffLight() : a(0), b(0) {};

	// Affine Parameters:
	double a,b;	// I_frame = exp(a)*I_global + b. // I_global = exp(-a)*(I_frame - b).

	static Vec2 fromToVecExposure(float exposureF, float exposureT, AffLight g2F, AffLight g2T)
	{
		if(exposureF==0 || exposureT==0) {
			exposureT = exposureF = 1;
			//printf("got exposure value of 0! please choose the correct model.\n");
			//assert(setting_brightnessTransferFunc < 2);
		}

		double a = exp(g2T.a-g2F.a) * exposureT / exposureF;
		double b = g2T.b - a*g2F.b;
		return Vec2(a,b);
	}

	Vec2 vec() {
		return Vec2(a,b);
	}
};

class TicToc {
    public:
        TicToc() {tic();}

        void tic() {start = std::chrono::system_clock::now();}

        double toc()
        {
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            return elapsed_seconds.count() /** 1000*/;
        }

    private:
        std::chrono::time_point<std::chrono::system_clock> start, end;
};

}

