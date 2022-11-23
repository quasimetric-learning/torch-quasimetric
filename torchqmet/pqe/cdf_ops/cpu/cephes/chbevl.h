#pragma once

/*
 * From
 * https://github.com/scipy/scipy/blob/5caee5d4ad564cfae4596f8dfa8b45997767035b/scipy/special/cephes/chbevl.c
 */

/*                                                     chbevl.c
 *
 *     Evaluate Chebyshev series
 *
 *
 *
 * SYNOPSIS:
 *
 * int N;
 * double x, y, coef[N], chebevl();
 *
 * y = chbevl( x, coef, N );
 *
 *
 *
 * DESCRIPTION:
 *
 * Evaluates the series
 *
 *        N-1
 *         - '
 *  y  =   >   coef[i] T (x/2)
 *         -            i
 *        i=0
 *
 * of Chebyshev polynomials Ti at argument x/2.
 *
 * Coefficients are stored in reverse order, i.e. the zero
 * order term is last in the array.  Note N is the number of
 * coefficients, not the order.
 *
 * If coefficients are for the interval a to b, x must
 * have been transformed to x -> 2(2x - b - a)/(b-a) before
 * entering the routine.  This maps x from (a, b) to (-1, 1),
 * over which the Chebyshev polynomials are defined.
 *
 * If the coefficients are for the inverted interval, in
 * which (a, b) is mapped to (1/b, 1/a), the transformation
 * required is x -> 2(2ab/x - b - a)/(b-a).  If b is infinity,
 * this becomes x -> 4a/x - 1.
 *
 *
 *
 * SPEED:
 *
 * Taking advantage of the recurrence properties of the
 * Chebyshev polynomials, the routine requires one more
 * addition per loop than evaluating a nested polynomial of
 * the same degree.
 *
 */
/*							chbevl.c	*/

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1985, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */


namespace cephes { namespace cpu {

template<typename scalar_t>
static inline typename std::enable_if<std::is_floating_point<scalar_t>::value, scalar_t>::type
chbevl(const scalar_t x, const scalar_t array[], const size_t len)
{
    scalar_t b0, b1, b2;

    b0 = array[0];
    b1 = static_cast<scalar_t>(0.0);

    for (size_t i = 1; i < len; ++i)  {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + array[i];
    }

    return (0.5 * (b0 - b2));
}

} // namespace cpu
} // namespace cephes
