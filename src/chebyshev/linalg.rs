//! # Linalg functions for chebyshev space
use crate::Scalar;
use ndarray::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

/// Tridiagonal matrix solver
///     Ax = d
/// where A is banded with diagonals in offsets -2, 0, 2
///
/// a: sub-diagonal (-2)
/// b: main-diagonal
/// c: sub-diagonal (+2)
#[allow(clippy::many_single_char_names)]
pub fn tdma<S1, S2, T1, T2>(
    a: &ArrayBase<S1, Ix1>,
    b: &ArrayBase<S1, Ix1>,
    c: &ArrayBase<S1, Ix1>,
    d: &mut ArrayBase<S2, Ix1>,
) where
    S1: ndarray::Data<Elem = T1>,
    S2: ndarray::Data<Elem = T2> + ndarray::DataMut,
    T1: Scalar,
    T2: Scalar
        + Add<T1, Output = T2>
        + Mul<T1, Output = T2>
        + Div<T1, Output = T2>
        + Sub<T1, Output = T2>,
{
    let n = d.len();
    let mut x = Array1::<T2>::zeros(n);
    let mut w = Array1::<T1>::zeros(n - 2);
    let mut g = Array1::<T2>::zeros(n);

    // Forward sweep
    w[0] = c[0] / b[0];
    g[0] = d[0] / b[0];
    if c.len() > 1 {
        w[1] = c[1] / b[1];
    }
    g[1] = d[1] / b[1];

    for i in 2..n - 2 {
        w[i] = c[i] / (b[i] - a[i - 2] * w[i - 2]);
    }
    for i in 2..n {
        g[i] = (d[i] - g[i - 2] * a[i - 2]) / (b[i] - a[i - 2] * w[i - 2]);
    }

    // Back substitution
    x[n - 1] = g[n - 1];
    x[n - 2] = g[n - 2];
    for i in (1..n - 1).rev() {
        x[i - 1] = g[i - 1] - x[i + 1] * w[i - 1];
    }

    d.assign(&x);
}

/// Pentadiagonal matrix solver
///     Ax = rhs
/// where A is banded with diagonals in offsets -2,-1,0,1,2
///
/// l2: sub-diagonal (-2)
/// l1: sub-diagonal (-1)
/// d0: main-diagonal
/// u1: sub-diagonal (+1)
/// u2: sub-diagonal (+2)
/// rhs: rhs (input), returns solution
///
/// ## Reference
/// `https://www.hindawi.com/journals/mpe/2015/232456/`
#[allow(clippy::many_single_char_names)]
pub fn pdma<S1, S2, T1, T2>(
    l2: &ArrayBase<S1, Ix1>,
    l1: &ArrayBase<S1, Ix1>,
    d0: &ArrayBase<S1, Ix1>,
    u1: &ArrayBase<S1, Ix1>,
    u2: &ArrayBase<S1, Ix1>,
    rhs: &mut ArrayBase<S2, Ix1>,
) where
    S1: ndarray::Data<Elem = T1>,
    S2: ndarray::Data<Elem = T2> + ndarray::DataMut,
    T1: Scalar,
    T2: Scalar
        + Add<T1, Output = T2>
        + Mul<T1, Output = T2>
        + Div<T1, Output = T2>
        + Sub<T1, Output = T2>,
{
    let n = rhs.len();

    let mut al = Array1::<T1>::zeros(n);
    let mut be = Array1::<T1>::zeros(n);
    let mut ze = Array1::<T2>::zeros(n);
    let mut ga = Array1::<T1>::zeros(n);
    let mut mu = Array1::<T1>::zeros(n);

    mu[0] = d0[0];
    al[0] = u1[0] / mu[0];
    be[0] = u2[0] / mu[0];
    ze[0] = rhs[0] / mu[0];

    ga[1] = l1[0];
    mu[1] = d0[1] - al[0] * ga[1];
    al[1] = (u1[1] - be[0] * ga[1]) / mu[1];
    be[1] = u2[1] / mu[1];
    ze[1] = (rhs[1] - ze[0] * ga[1]) / mu[1];

    for i in 2..n - 2 {
        ga[i] = l1[i - 1] - al[i - 2] * l2[i - 2];
        mu[i] = d0[i] - be[i - 2] * l2[i - 2] - al[i - 1] * ga[i];
        al[i] = (u1[i] - be[i - 1] * ga[i]) / mu[i];
        be[i] = u2[i] / mu[i];
        ze[i] = (rhs[i] - ze[i - 2] * l2[i - 2] - ze[i - 1] * ga[i]) / mu[i];
    }

    ga[n - 2] = l1[n - 3] - al[n - 4] * l2[n - 4];
    mu[n - 2] = d0[n - 2] - be[n - 4] * l2[n - 4] - al[n - 3] * ga[n - 2];
    al[n - 2] = (u1[n - 2] - be[n - 3] * ga[n - 2]) / mu[n - 2];

    ga[n - 1] = l1[n - 2] - al[n - 3] * l2[n - 3];
    mu[n - 1] = d0[n - 1] - be[n - 3] * l2[n - 3] - al[n - 2] * ga[n - 1];

    ze[n - 2] = (rhs[n - 2] - ze[n - 4] * l2[n - 4] - ze[n - 3] * ga[n - 2]) / mu[n - 2];
    ze[n - 1] = (rhs[n - 1] - ze[n - 3] * l2[n - 3] - ze[n - 2] * ga[n - 1]) / mu[n - 1];

    // Backward substitution
    rhs[n - 1] = ze[n - 1];
    rhs[n - 2] = ze[n - 2] - rhs[n - 1] * al[n - 2];

    for i in (0..n - 2).rev() {
        rhs[i] = ze[i] - rhs[i + 1] * al[i] - rhs[i + 2] * be[i];
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::approx_eq;

    #[test]
    /// Test Pentadiagonal matrix solver
    fn test_pdma() {
        use ndarray::array;
        let n = 6;
        // Test matrix diagonals (symmetrix about diagonal, hence only 3)
        let d0 = Array1::range(0., n as f64, 1.) + 1.;
        let d1 = 2.5 * &d0.slice(ndarray::s![..d0.len() - 1]);
        let d2 = -1.5 * &d0.slice(ndarray::s![..d0.len() - 2]);
        // Test rhs
        let mut rhs = Array1::range(0., n as f64, 1.) - 0.5;
        // Solve
        pdma(&d2, &d1, &d0, &d1, &d2, &mut rhs);

        approx_eq(
            &rhs,
            &array![
                -0.48507635,
                0.16677649,
                0.28790992,
                0.02013722,
                0.23916798,
                0.2718706
            ],
        );
    }
}
