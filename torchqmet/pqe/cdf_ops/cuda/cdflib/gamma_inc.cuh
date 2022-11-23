#pragma once

#include "error_fc.cuh"
#include "fifidint.cuh"
#include "gam1.cuh"
#include "rlog.cuh"
#include "rexp.cuh"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <limits>
#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>


namespace cdflib {
namespace cuda {

template<typename scalar_t, int prec_ind>
__host__ __device__ __forceinline__
void gamma_inc ( scalar_t a, scalar_t x, scalar_t *ans, scalar_t *qans )

//****************************************************************************80
//
//  Purpose:
//
//    GAMMA_INC evaluates the incomplete gamma ratio functions P(A,X) and Q(A,X).
//
//  Discussion:
//
//    This is certified spaghetti code.
//
//  Author:
//
//    Alfred H Morris, Jr,
//    Naval Surface Weapons Center,
//    Dahlgren, Virginia.
//
//  Parameters:
//
//    Input, double *A, *X, the arguments of the incomplete
//    gamma ratio.  A and X must be nonnegative.  A and X cannot
//    both be zero.
//
//    Output, double *ANS, *QANS.  On normal output,
//    ANS = P(A,X) and QANS = Q(A,X).  However, ANS is set to 2 if
//    A or X is negative, or both are 0, or when the answer is
//    computationally indeterminate because A is extremely large
//    and X is very close to A.
//
//    Input, bool PREC_IND, indicates the accuracy request:
//    0, as much accuracy as possible.
//    1, to within 1 unit of the 6-th significant digit,
//    otherwise, to within 1 unit of the 3rd significant digit.
//
{
  static const scalar_t alog10 = 2.30258509299405e0;
  static const scalar_t d10 = -.185185185185185e-02;
  static const scalar_t d20 = .413359788359788e-02;
  static const scalar_t d30 = .649434156378601e-03;
  static const scalar_t d40 = -.861888290916712e-03;
  static const scalar_t d50 = -.336798553366358e-03;
  static const scalar_t d60 = .531307936463992e-03;
  static const scalar_t d70 = .344367606892378e-03;
  static const scalar_t rt2pin = .398942280401433e0;
  static const scalar_t rtpi = 1.77245385090552e0;
  static const scalar_t third = .333333333333333e0;
  static const scalar_t acc0[3] = {
    5.e-15,5.e-7,5.e-4
  };
  static const scalar_t big[3] = {
    20.0e0,14.0e0,10.0e0
  };
  static const scalar_t d0[13] = {
    .833333333333333e-01,-.148148148148148e-01,.115740740740741e-02,
    .352733686067019e-03,-.178755144032922e-03,.391926317852244e-04,
    -.218544851067999e-05,-.185406221071516e-05,.829671134095309e-06,
    -.176659527368261e-06,.670785354340150e-08,.102618097842403e-07,
    -.438203601845335e-08
  };
  static const scalar_t d1[12] = {
    -.347222222222222e-02,.264550264550265e-02,-.990226337448560e-03,
    .205761316872428e-03,-.401877572016461e-06,-.180985503344900e-04,
    .764916091608111e-05,-.161209008945634e-05,.464712780280743e-08,
    .137863344691572e-06,-.575254560351770e-07,.119516285997781e-07
  };
  static const scalar_t d2[10] = {
    -.268132716049383e-02,.771604938271605e-03,.200938786008230e-05,
    -.107366532263652e-03,.529234488291201e-04,-.127606351886187e-04,
    .342357873409614e-07,.137219573090629e-05,-.629899213838006e-06,
    .142806142060642e-06
  };
  static const scalar_t d3[8] = {
    .229472093621399e-03,-.469189494395256e-03,.267720632062839e-03,
    -.756180167188398e-04,-.239650511386730e-06,.110826541153473e-04,
    -.567495282699160e-05,.142309007324359e-05
  };
  static const scalar_t d4[6] = {
    .784039221720067e-03,-.299072480303190e-03,-.146384525788434e-05,
    .664149821546512e-04,-.396836504717943e-04,.113757269706784e-04
  };
  static const scalar_t d5[4] = {
    -.697281375836586e-04,.277275324495939e-03,-.199325705161888e-03,
    .679778047793721e-04
  };
  static const scalar_t d6[2] = {
    -.592166437353694e-03,.270878209671804e-03
  };
  static const scalar_t e00[3] = {
    .25e-3,.25e-1,.14e0
  };
  static const scalar_t x00[3] = {
    31.0e0,17.0e0,9.7e0
  };
  scalar_t a2n,a2nm1,acc,am0,amn,an,an0,apn,b2n,b2nm1,c,c0,c1,c2,c3,c4,c5,c6,
    cma,e0,g,h,j,l,r,rta,rtx,s,sum,t,t1,tol,twoa,u,w,x0,y,z;
  int i,iop,m,max,n;
  scalar_t wk[20],T3;
  int T4,T5;
  scalar_t T6,T7;

//
//  E IS A MACHINE DEPENDENT CONSTANT. E IS THE SMALLEST
//  NUMBER FOR WHICH 1.0 + E .GT. 1.0 .
//
    constexpr scalar_t e = std::numeric_limits<scalar_t>::epsilon();
    if(a < 0.0e0 || x < 0.0e0) goto S430;
    if(a == 0.0e0 && x == 0.0e0) goto S430;
    if(a*x == 0.0e0) goto S420;
    iop = prec_ind+1;
    if(iop != 1 && iop != 2) iop = 3;
    acc = std::max(acc0[iop-1],e);
    e0 = e00[iop-1];
    x0 = x00[iop-1];
//
//  SELECT THE APPROPRIATE ALGORITHM
//
    if(a >= 1.0e0) goto S10;
    if(a == 0.5e0) goto S390;
    if(x < 1.1e0) goto S160;
    t1 = a*std::log(x)-x;
    u = a*std::exp(t1);
    if(u == 0.0e0) goto S380;
    r = u*(1.0e0+gam1<scalar_t>(a));
    goto S250;
S10:
    if(a >= big[iop-1]) goto S30;
    if(a > x || x >= x0) goto S20;
    twoa = a+a;
    m = static_cast<int>(fifidint<scalar_t>(twoa));
    if(twoa != static_cast<scalar_t>(m)) goto S20;
    i = m/2;
    if(a == static_cast<scalar_t>(i)) goto S210;
    goto S220;
S20:
    t1 = a*std::log(x)-x;
    r = std::exp(t1)/ std::tgamma(a);
    goto S40;
S30:
    l = x/ a;
    if(l == 0.0e0) goto S370;
    s = 0.5e0+(0.5e0-l);
    z = rlog<scalar_t>(l);
    if(z >= 700.0e0/ a) goto S410;
    y = a*z;
    rta = std::sqrt(a);
    if(std::abs(s) <= e0/rta) goto S330;
    if(std::abs(s) <= 0.4e0) goto S270;
    t = std::pow(1.0e0/ a,2.0);
    t1 = (((0.75e0*t-1.0e0)*t+3.5e0)*t-105.0e0)/(a*1260.0e0);
    t1 -= y;
    r = rt2pin*rta*std::exp(t1);
S40:
    if(r == 0.0e0) goto S420;
    if(x <= std::max(a,alog10)) goto S50;
    if(x < x0) goto S250;
    goto S100;
S50:
//
//  TAYLOR SERIES FOR P/R
//
    apn = a+1.0e0;
    t = x/apn;
    wk[0] = t;
    for ( n = 2; n <= 20; n++ )
    {
        apn += 1.0e0;
        t *= (x/apn);
        if(t <= 1.e-3) goto S70;
        wk[n-1] = t;
    }
    n = 20;
S70:
    sum = t;
    tol = 0.5e0*acc;
S80:
    apn += 1.0e0;
    t *= (x/apn);
    sum += t;
    if(t > tol) goto S80;
    max = n-1;
    for ( m = 1; m <= max; m++ )
    {
        n -= 1;
        sum += wk[n-1];
    }
    *ans = r/ a*(1.0e0+sum);
    *qans = 0.5e0+(0.5e0-*ans);
    return;
S100:
//
//  ASYMPTOTIC EXPANSION
//
    amn = a-1.0e0;
    t = amn/ x;
    wk[0] = t;
    for ( n = 2; n <= 20; n++ )
    {
        amn -= 1.0e0;
        t *= (amn/ x);
        if(std::abs(t) <= 1.e-3) goto S120;
        wk[n-1] = t;
    }
    n = 20;
S120:
    sum = t;
S130:
    if(std::abs(t) <= acc) goto S140;
    amn -= 1.0e0;
    t *= (amn/ x);
    sum += t;
    goto S130;
S140:
    max = n-1;
    for ( m = 1; m <= max; m++ )
    {
        n -= 1;
        sum += wk[n-1];
    }
    *qans = r/ x*(1.0e0+sum);
    *ans = 0.5e0+(0.5e0-*qans);
    return;
S160:
//
//  TAYLOR SERIES FOR P(A,X)/X**A
//
    an = 3.0e0;
    c = x;
    sum = x/(a+3.0e0);
    tol = 3.0e0*acc/(a+1.0e0);
S170:
    an += 1.0e0;
    c = -(c*(x/an));
    t = c/(a+an);
    sum += t;
    if(std::abs(t) > tol) goto S170;
    j = a*x*((sum/6.0e0-0.5e0/(a+2.0e0))*x+1.0e0/(a+1.0e0));
    z = a*std::log(x);
    h = gam1<scalar_t>(a);
    g = 1.0e0+h;
    if(x < 0.25e0) goto S180;
    if(a < x/2.59e0) goto S200;
    goto S190;
S180:
    if(z > -.13394e0) goto S200;
S190:
    w = std::exp(z);
    *ans = w*g*(0.5e0+(0.5e0-j));
    *qans = 0.5e0+(0.5e0-*ans);
    return;
S200:
    l = rexp<scalar_t>(z);
    w = 0.5e0+(0.5e0+l);
    *qans = (w*j-l)*g-h;
    if(*qans < 0.0e0) goto S380;
    *ans = 0.5e0+(0.5e0-*qans);
    return;
S210:
//
//  FINITE SUMS FOR Q WHEN A .GE. 1 AND 2*A IS AN INTEGER
//
    sum = std::exp(-x);
    t = sum;
    n = 1;
    c = 0.0e0;
    goto S230;
S220:
    rtx = std::sqrt(x);
    sum = std::erfc ( rtx );
    t = std::exp(-x)/(rtpi*rtx);
    n = 0;
    c = -0.5e0;
S230:
    if(n == i) goto S240;
    n += 1;
    c += 1.0e0;
    t = x*t/c;
    sum += t;
    goto S230;
S240:
    *qans = sum;
    *ans = 0.5e0+(0.5e0-*qans);
    return;
S250:
//
//  CONTINUED FRACTION EXPANSION
//
    tol = std::max( static_cast<scalar_t>(5.0e0) * e, acc);
    a2nm1 = a2n = 1.0e0;
    b2nm1 = x;
    b2n = x+(1.0e0-a);
    c = 1.0e0;
S260:
    a2nm1 = x*a2n+c*a2nm1;
    b2nm1 = x*b2n+c*b2nm1;
    am0 = a2nm1/b2nm1;
    c += 1.0e0;
    cma = c-a;
    a2n = a2nm1+cma*a2n;
    b2n = b2nm1+cma*b2n;
    an0 = a2n/b2n;
    if(std::abs(an0-am0) >= tol*an0) goto S260;
    *qans = r*an0;
    *ans = 0.5e0+(0.5e0-*qans);
    return;
S270:
//
//  GENERAL TEMME EXPANSION
//
    if(std::abs(s) <= 2.0e0*e && a*e*e > 3.28e-3) goto S430;
    c = std::exp(-y);
    T3 = std::sqrt(y);
    w = 0.5e0 * error_fc<scalar_t, true> ( T3 );
    u = 1.0e0/ a;
    z = std::sqrt(z+z);
    if(l < 1.0e0) z = -z;
    T4 = iop-2;
    if(T4 < 0) goto S280;
    else if(T4 == 0) goto S290;
    else  goto S300;
S280:
    if(std::abs(s) <= 1.e-3) goto S340;
    c0 = ((((((((((((d0[12]*z+d0[11])*z+d0[10])*z+d0[9])*z+d0[8])*z+d0[7])*z+d0[
      6])*z+d0[5])*z+d0[4])*z+d0[3])*z+d0[2])*z+d0[1])*z+d0[0])*z-third;
    c1 = (((((((((((d1[11]*z+d1[10])*z+d1[9])*z+d1[8])*z+d1[7])*z+d1[6])*z+d1[5]
      )*z+d1[4])*z+d1[3])*z+d1[2])*z+d1[1])*z+d1[0])*z+d10;
    c2 = (((((((((d2[9]*z+d2[8])*z+d2[7])*z+d2[6])*z+d2[5])*z+d2[4])*z+d2[3])*z+
      d2[2])*z+d2[1])*z+d2[0])*z+d20;
    c3 = (((((((d3[7]*z+d3[6])*z+d3[5])*z+d3[4])*z+d3[3])*z+d3[2])*z+d3[1])*z+
      d3[0])*z+d30;
    c4 = (((((d4[5]*z+d4[4])*z+d4[3])*z+d4[2])*z+d4[1])*z+d4[0])*z+d40;
    c5 = (((d5[3]*z+d5[2])*z+d5[1])*z+d5[0])*z+d50;
    c6 = (d6[1]*z+d6[0])*z+d60;
    t = ((((((d70*u+c6)*u+c5)*u+c4)*u+c3)*u+c2)*u+c1)*u+c0;
    goto S310;
S290:
    c0 = (((((d0[5]*z+d0[4])*z+d0[3])*z+d0[2])*z+d0[1])*z+d0[0])*z-third;
    c1 = (((d1[3]*z+d1[2])*z+d1[1])*z+d1[0])*z+d10;
    c2 = d2[0]*z+d20;
    t = (c2*u+c1)*u+c0;
    goto S310;
S300:
    t = ((d0[2]*z+d0[1])*z+d0[0])*z-third;
S310:
    if(l < 1.0e0) goto S320;
    *qans = c*(w+rt2pin*t/rta);
    *ans = 0.5e0+(0.5e0-*qans);
    return;
S320:
    *ans = c*(w-rt2pin*t/rta);
    *qans = 0.5e0+(0.5e0-*ans);
    return;
S330:
//
//  TEMME EXPANSION FOR L = 1
//
    if(a*e*e > 3.28e-3) goto S430;
    c = 0.5e0+(0.5e0-y);
    w = (0.5e0-std::sqrt(y)*(0.5e0+(0.5e0-y/3.0e0))/rtpi)/c;
    u = 1.0e0/ a;
    z = std::sqrt(z+z);
    if(l < 1.0e0) z = -z;
    T5 = iop-2;
    if(T5 < 0) goto S340;
    else if(T5 == 0) goto S350;
    else  goto S360;
S340:
    c0 = ((((((d0[6]*z+d0[5])*z+d0[4])*z+d0[3])*z+d0[2])*z+d0[1])*z+d0[0])*z-
      third;
    c1 = (((((d1[5]*z+d1[4])*z+d1[3])*z+d1[2])*z+d1[1])*z+d1[0])*z+d10;
    c2 = ((((d2[4]*z+d2[3])*z+d2[2])*z+d2[1])*z+d2[0])*z+d20;
    c3 = (((d3[3]*z+d3[2])*z+d3[1])*z+d3[0])*z+d30;
    c4 = (d4[1]*z+d4[0])*z+d40;
    c5 = (d5[1]*z+d5[0])*z+d50;
    c6 = d6[0]*z+d60;
    t = ((((((d70*u+c6)*u+c5)*u+c4)*u+c3)*u+c2)*u+c1)*u+c0;
    goto S310;
S350:
    c0 = (d0[1]*z+d0[0])*z-third;
    c1 = d1[0]*z+d10;
    t = (d20*u+c1)*u+c0;
    goto S310;
S360:
    t = d0[0]*z-third;
    goto S310;
S370:
//
//  SPECIAL CASES
//
    *ans = 0.0e0;
    *qans = 1.0e0;
    return;
S380:
    *ans = 1.0e0;
    *qans = 0.0e0;
    return;
S390:
    if(x >= 0.25e0) goto S400;
    T6 = std::sqrt(x);
    *ans = std::erf ( T6 );
    *qans = 0.5e0+(0.5e0-*ans);
    return;
S400:
    T7 = std::sqrt(x);
    *qans = std::erfc ( T7 );
    *ans = 0.5e0+(0.5e0-*qans);
    return;
S410:
    if(std::abs(s) <= 2.0e0*e) goto S430;
S420:
    if(x <= a) goto S370;
    goto S380;
S430:
//
//  ERROR RETURN
//
    *ans = 2.0e0;
    return;
}

} // namespace cuda
} // namespace cdflib
