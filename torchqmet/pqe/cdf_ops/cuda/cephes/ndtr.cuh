#pragma once

/*
 * From
 * https://github.com/scipy/scipy/blob/7c7a5f8393e7b16e5bc81c739c84fe2e639c367f/scipy/special/cephes/ndtr.c
 */

/*                                                     ndtr.c
 *
 *     Normal distribution function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, ndtr();
 *
 * y = ndtr( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the area under the Gaussian probability density
 * function, integrated from minus infinity to x:
 *
 *                            x
 *                             -
 *                   1        | |          2
 *    ndtr(x)  = ---------    |    exp( - t /2 ) dt
 *               sqrt(2pi)  | |
 *                           -
 *                          -inf.
 *
 *             =  ( 1 + erf(z) ) / 2
 *             =  erfc(z) / 2
 *
 * where z = x/sqrt(2). Computation is via the functions
 * erf and erfc.
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE     -13,0        30000       3.4e-14     6.7e-15
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition         value returned
 * erfc underflow    x > 37.519379347       0.0
 *
 */
/*							erf.c
 *
 *	Error function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, erf();
 *
 * y = erf( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * The integral is
 *
 *                           x
 *                            -
 *                 2         | |          2
 *   erf(x)  =  --------     |    exp( - t  ) dt.
 *              sqrt(pi)   | |
 *                          -
 *                           0
 *
 * For 0 <= |x| < 1, erf(x) = x * P4(x**2)/Q5(x**2); otherwise
 * erf(x) = 1 - erfc(x).
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,1         30000       3.7e-16     1.0e-16
 *
 */
/*							erfc.c
 *
 *	Complementary error function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, erfc();
 *
 * y = erfc( x );
 *
 *
 *
 * DESCRIPTION:
 *
 *
 *  1 - erf(x) =
 *
 *                           inf.
 *                             -
 *                  2         | |          2
 *   erfc(x)  =  --------     |    exp( - t  ) dt
 *               sqrt(pi)   | |
 *                           -
 *                            x
 *
 *
 * For small x, erfc(x) = 1 - erf(x); otherwise rational
 * approximations are computed.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,26.6417   30000       5.7e-14     1.5e-14
 */


/*
 * Cephes Math Library Release 2.2:  June, 1992
 * Copyright 1984, 1987, 1988, 1992 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

// #include "polevl.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <limits>
#include <cmath>

namespace cephes { namespace cuda {



template<typename scalar_t>
static __host__ __device__ __forceinline__
scalar_t ndtr(scalar_t a)
{
    scalar_t x, y, z;

    if (std::isnan(a)) {
        // throw std::runtime_error("ndtr sees NaN")
        return NAN;
    }

    // std::sqrt(0.5);  std::sqrt is not constexpr
    constexpr scalar_t SQRT1_2 = 0.707106781186547524400844362104849039284835937688474036588339868995366239231053519425193767163820786367507;

    x = a * SQRT1_2;
    z = std::fabs(x);

    if (z < SQRT1_2) {
	    y = 0.5 + 0.5 * std::erf(x);

    } else {
	    y = 0.5 * std::erfc(z);

        if (x > 0) {
            y = 1.0 - y;
        }
    }

    return (y);
}


// template<typename scalar_t>
// static inline scalar_t erfc(scalar_t a)
// {
//     scalar_t p, q, x, y, z;

//     if (std::isnan(a)) {
//         // throw std::runtime_error("erfc sees NaN")
// 	    return NAN;
//     }

//     if (a < 0.0)
// 	    x = -a;
//     else
// 	    x = a;

//     if (x < 1.0)
// 	    return (1.0 - erf<scalar_t>(a));

//     z = -a * a;

//     if (z < -MAXLOG) {
//         under:
//         // sf_error("erfc", SF_ERROR_UNDERFLOW, NULL);
//         if (a < 0)
//             return (2.0);
//         else
//             return (0.0);
//     }

//     z = std::exp(z);

//     if (x < 8.0) {

//         static const scalar_t P[] = {
//             2.46196981473530512524E-10,
//             5.64189564831068821977E-1,
//             7.46321056442269912687E0,
//             4.86371970985681366614E1,
//             1.96520832956077098242E2,
//             5.26445194995477358631E2,
//             9.34528527171957607540E2,
//             1.02755188689515710272E3,
//             5.57535335369399327526E2
//         };

//         static const scalar_t Q[] = {
//             /* 1.00000000000000000000E0, */
//             1.32281951154744992508E1,
//             8.67072140885989742329E1,
//             3.54937778887819891062E2,
//             9.75708501743205489753E2,
//             1.82390916687909736289E3,
//             2.24633760818710981792E3,
//             1.65666309194161350182E3,
//             5.57535340817727675546E2
//         };

//         p = polevl<scalar_t>(x, P, 8);
//         q = p1evl<scalar_t>(x, Q, 8);
//     }
//     else {

//         static const scalar_t R[] = {
//             5.64189583547755073984E-1,
//             1.27536670759978104416E0,
//             5.01905042251180477414E0,
//             6.16021097993053585195E0,
//             7.40974269950448939160E0,
//             2.97886665372100240670E0
//         };

//         static const scalar_t S[] = {
//             /* 1.00000000000000000000E0, */
//             2.26052863220117276590E0,
//             9.39603524938001434673E0,
//             1.20489539808096656605E1,
//             1.70814450747565897222E1,
//             9.60896809063285878198E0,
//             3.36907645100081516050E0
//         };

//         p = polevl<scalar_t>(x, R, 5);
//         q = p1evl<scalar_t>(x, S, 6);
//     }
//     y = (z * p) / q;

//     if (a < 0)
// 	    y = 2.0 - y;

//     if (y == 0.0)
// 	    goto under;

//     return (y);
// }


// template<typename scalar_t>
// static inline scalar_t erf(scalar_t x)
// {
//     scalar_t y, z;

//     if (std::isnan(x)) {
//         // throw std::runtime_error("erf sees NaN")
// 	    return NAN;
//     }

//     if (x < 0.0) {
// 	    return -erf<scalar_t>(-x);
//     }

//     if (std::fabs(x) > 1.0)
// 	    return (1.0 - erfc<scalar_t>(x));

//     z = x * x;

//     static const scalar_t T[] = {
//         9.60497373987051638749E0,
//         9.00260197203842689217E1,
//         2.23200534594684319226E3,
//         7.00332514112805075473E3,
//         5.55923013010394962768E4
//     };

//     static const scalar_t U[] = {
//         /* 1.00000000000000000000E0, */
//         3.35617141647503099647E1,
//         5.21357949780152679795E2,
//         4.59432382970980127987E3,
//         2.26290000613890934246E4,
//         4.92673942608635921086E4
//     };

//     y = x * polevl<scalar_t>(z, T, 4) / p1evl<scalar_t>(z, U, 5);
//     return (y);

// }

/*
 * double log_ndtr(double a)
 *
 * For a > -20, use the existing ndtr technique and take a log.
 * for a <= -20, we use the Taylor series approximation of erf to compute
 * the log CDF directly. The Taylor series consists of two parts which we will name "left"
 * and "right" accordingly.  The right part involves a summation which we compute until the
 * difference in terms falls below the machine-specific EPSILON.
 *
 * \Phi(z) &=&
 *   \frac{e^{-z^2/2}}{-z\sqrt{2\pi}}  * [1 +  \sum_{n=1}^{N-1}  (-1)^n \frac{(2n-1)!!}{(z^2)^n}]
 *   + O(z^{-2N+2})
 *   = [\mbox{LHS}] * [\mbox{RHS}] + \mbox{error}.
 *
 */

template<typename scalar_t>
static __host__ __device__ __forceinline__
scalar_t log_ndtr(scalar_t a)
{

    if (a > 6) {
	    return -ndtr<scalar_t>(-a);     /* log(1+x) \approx x */
    }
    if (a > -20) {
	    return std::log(ndtr<scalar_t>(a));
    }

    scalar_t log_LHS,		/* we compute the left hand side of the approx (LHS) in one shot */
     last_total = 0,		/* variable used to check for convergence */
	 right_hand_side = 1,	/* includes first term from the RHS summation */
	 numerator = 1,		/* numerator for RHS summand */
	 denom_factor = 1,	/* use reciprocal for denominator to avoid division */
	 denom_cons = 1.0 / (a * a);	/* the precomputed division we use to adjust the denominator */
    long sign = 1, i = 0;

    log_LHS = -0.5 * a * a - std::log(-a) - 0.5 * std::log(2 * M_PI);

    while (std::fabs(last_total - right_hand_side) > std::numeric_limits<scalar_t>::epsilon()) {
        i += 1;
        last_total = right_hand_side;
        sign = -sign;
        denom_factor *= denom_cons;
        numerator *= 2 * i - 1;
        right_hand_side += sign * numerator * denom_factor;
    }
    return log_LHS + std::log(right_hand_side);
}


template<typename scalar_t>
static __host__ __device__ __forceinline__
std::pair<scalar_t, scalar_t> ndtr_log_ndtr(scalar_t a)
{

    if (a > 6) {
        /* log(1+x) \approx x */
	    scalar_t x = -ndtr<scalar_t>(-a);  /* x = -ndtr_remain_val */
	    return std::make_pair(1 + x, x);
    }
    if (a > -20) {
        scalar_t ndtr_val = ndtr<scalar_t>(a);
	    return std::make_pair(ndtr_val, std::log(ndtr_val));
    }

    scalar_t log_LHS,		/* we compute the left hand side of the approx (LHS) in one shot */
     last_total = 0,		/* variable used to check for convergence */
	 right_hand_side = 1,	/* includes first term from the RHS summation */
	 numerator = 1,		/* numerator for RHS summand */
	 denom_factor = 1,	/* use reciprocal for denominator to avoid division */
	 denom_cons = 1.0 / (a * a);	/* the precomputed division we use to adjust the denominator */
    long sign = 1, i = 0;

    log_LHS = -0.5 * a * a - std::log(-a) - 0.5 * std::log(2 * M_PI);

    while (std::fabs(last_total - right_hand_side) > std::numeric_limits<scalar_t>::epsilon()) {
        i += 1;
        last_total = right_hand_side;
        sign = -sign;
        denom_factor *= denom_cons;
        numerator *= 2 * i - 1;
        right_hand_side += sign * numerator * denom_factor;
    }
    return std::make_pair(ndtr<scalar_t>(a), log_LHS + std::log(right_hand_side));
}

} // namespace cuda
} // namespace cephes
