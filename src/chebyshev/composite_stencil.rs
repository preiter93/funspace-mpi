//! Transformation stencils from orthogonal chebyshev space to composite space
//! $$
//! p = S c
//! $$
//! where $S$ is a two-dimensional transform matrix.
#![allow(clippy::used_underscore_binding)]
use crate::{FloatNum, Scalar};
use ndarray::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

/// Elementary methods for stencils
#[enum_dispatch]
pub trait Stencil<A> {
    /// Multiply stencil with a 1d array
    fn multiply_vec<S, T>(&self, composite_coeff: &ArrayBase<S, Ix1>) -> Array1<T>
    where
        S: ndarray::Data<Elem = T>,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>;

    /// Multiply stencil with a 1d array (output must be supplied)
    fn multiply_vec_inplace<S1, S2, T>(
        &self,
        composite_coeff: &ArrayBase<S1, Ix1>,
        orthonorm_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>;

    /// Solve linear system $A c = p$, where stencil is matrix $A$.
    fn solve_vec<S, T>(&self, orthonorm_coeff: &ArrayBase<S, Ix1>) -> Array1<T>
    where
        S: ndarray::Data<Elem = T>,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>;

    /// Solve linear system $A c = p$, where stencil is matrix $A$ (output must be supplied)
    fn solve_vec_inplace<S1, S2, T>(
        &self,
        orthonorm_coeff: &ArrayBase<S1, Ix1>,
        composite_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>;

    /// Return stencil as 2d array
    fn to_array(&self) -> Array2<A>;
}

#[allow(clippy::pub_enum_variant_names)]
#[enum_dispatch(Stencil<A>)]
#[derive(Clone)]
pub enum ChebyshevStencil<A: FloatNum> {
    StencilChebyshev(StencilChebyshev<A>),
    StencilChebyshevDirchletNeumann(StencilChebyshevDirchletNeumann<A>),
    StencilChebyshevBoundary(StencilChebyshevBoundary<A>),
}

/// Container for Chebyshev Stencil (internally used)
#[derive(Clone)]
pub struct StencilChebyshev<A> {
    /// Number of coefficients in orthonormal space
    n: usize,
    /// Number of coefficients in composite space
    m: usize,
    /// Main diagonal
    diag: Array1<A>,
    /// Subdiagonal offset -2
    low2: Array1<A>,
    /// For tdma (diagonal)
    main: Array1<A>,
    /// For tdma (off-diagonal)
    off: Array1<A>,
}

/// Container for Boundary Condition Stencil
///
/// This stencil is fully defined by the
/// the 2 coefficients that act on $T_0$ and the
/// 2 coefficients that act on $T_1$, where $T$
/// are the basis function of the orthonormal
/// chebyshev basis.
#[derive(Clone)]
pub struct StencilChebyshevBoundary<A> {
    /// Number of coefficients in orthonormal space
    n: usize,
    /// Number of coefficients in composite space
    m: usize,
    /// T0
    t0: Array1<A>,
    /// T1
    t1: Array1<A>,
}

impl<A: FloatNum> StencilChebyshev<A> {
    /// Return stencil of chebyshev dirichlet space
    /// $$
    ///  \phi_k = T_k - T_{k+2}
    /// $$
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    pub fn dirichlet(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = Array::from_vec(vec![A::one(); m]);
        let low2 = Array::from_vec(vec![-A::one(); m]);
        let (main, off) = Self::_get_main_off(&diag.view(), &low2.view());
        Self {
            n,
            m,
            diag,
            low2,
            main,
            off,
        }
    }

    /// Return stencil of chebyshev neumann space
    /// $$
    /// \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
    /// $$
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    pub fn neumann(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = Array::from_vec(vec![A::one(); m]);
        let mut low2 = Array::from_vec(vec![A::zero(); m]);
        for (k, v) in low2.iter_mut().enumerate() {
            let k_ = A::from_f64(k.pow(2) as f64).unwrap();
            let k2_ = A::from_f64((k + 2).pow(2) as f64).unwrap();
            *v = -A::one() * k_ / k2_;
        }
        let (main, off) = Self::_get_main_off(&diag.view(), &low2.view());
        Self {
            n,
            m,
            diag,
            low2,
            main,
            off,
        }
    }

    /// Get main diagonal and off diagonal, used in [`StencilChebyshev::solve_vec_inplace`]
    fn _get_main_off(diag: &ArrayView1<A>, low2: &ArrayView1<A>) -> (Array1<A>, Array1<A>) {
        let m = diag.len();
        let mut main = Array::from_vec(vec![A::zero(); m]);
        let mut off = Array::from_vec(vec![A::zero(); m - 2]);
        for (i, v) in main.iter_mut().enumerate() {
            *v = diag[i] * diag[i] + low2[i] * low2[i];
        }
        for (i, v) in off.iter_mut().enumerate() {
            *v = diag[i + 2] * low2[i];
        }
        (main, off)
    }

    /// Composite spaces can be smaller than its orthonormal counterpart
    pub fn get_m(n: usize) -> usize {
        n - 2
    }
}

impl<A: FloatNum> Stencil<A> for StencilChebyshev<A> {
    /// Returns transform stencil to a 2d ndarray
    fn to_array(&self) -> Array2<A> {
        let mut mat = Array2::<A>::zeros((self.n, self.m).f());
        for (i, (d, l)) in self.diag.iter().zip(self.low2.iter()).enumerate() {
            mat[[i, i]] = *d;
            mat[[i + 2, i]] = *l;
        }
        mat
    }

    /// Multiply stencil with a 1d array (transforms to orthonorm coefficents)
    /// input and output array do usually differ in size.
    fn multiply_vec<S, T>(&self, composite_coeff: &ArrayBase<S, Ix1>) -> Array1<T>
    where
        S: ndarray::Data<Elem = T>,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        let mut orthonorm_coeff = Array1::<T>::zeros(self.n);
        self.multiply_vec_inplace(composite_coeff, &mut orthonorm_coeff);
        orthonorm_coeff
    }

    /// See [`StencilChebyshev::multiply_vec`]
    fn multiply_vec_inplace<S1, S2, T>(
        &self,
        composite_coeff: &ArrayBase<S1, Ix1>,
        orthonorm_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        orthonorm_coeff.mapv_inplace(|x| x * T::zero());
        orthonorm_coeff[0] = composite_coeff[0] * self.diag[0];
        orthonorm_coeff[1] = composite_coeff[1] * self.diag[1];
        for i in 2..self.n - 2 {
            orthonorm_coeff[i] =
                composite_coeff[i] * self.diag[i] + composite_coeff[i - 2] * self.low2[i - 2];
        }
        orthonorm_coeff[self.n - 2] = composite_coeff[self.n - 4] * self.low2[self.n - 4];
        orthonorm_coeff[self.n - 1] = composite_coeff[self.n - 3] * self.low2[self.n - 3];
    }

    /// Solve linear algebraic system $p = S c$ for $p$ with given composite
    /// coefficents $c$.
    ///
    /// Input and output array do usually differ in size.
    fn solve_vec<S, T>(&self, orthonorm_coeff: &ArrayBase<S, Ix1>) -> Array1<T>
    where
        S: ndarray::Data<Elem = T>,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        let mut composite_coeff = Array1::<T>::zeros(self.m);
        self.solve_vec_inplace(orthonorm_coeff, &mut composite_coeff);
        composite_coeff
    }

    /// See [`StencilChebyshev::solve_vec`]
    fn solve_vec_inplace<S1, S2, T>(
        &self,
        orthonorm_coeff: &ArrayBase<S1, Ix1>,
        composite_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        use super::linalg::tdma;
        // Multiply right hand side
        for i in 0..self.m {
            composite_coeff[i] =
                orthonorm_coeff[i] * self.diag[i] + orthonorm_coeff[i + 2] * self.low2[i];
        }
        // Solve tridiagonal system
        tdma(
            &self.off.view(),
            &self.main.view(),
            &self.off.view(),
            composite_coeff,
        );
    }
}

impl<A: FloatNum> StencilChebyshevBoundary<A> {
    /// dirichlet_bc basis
    /// $$
    ///     \phi_0 = 0.5 T_0 - 0.5 T_1
    /// $$
    /// $$
    ///     \phi_1 = 0.5 T_0 + 0.5 T_1
    /// $$
    pub fn dirichlet(n: usize) -> Self {
        let m = Self::get_m(n);
        let _05 = A::from_f64(0.5).unwrap();
        let t0 = Array::from_vec(vec![_05, _05]);
        let t1 = Array::from_vec(vec![-(_05), _05]);
        StencilChebyshevBoundary { n, m, t0, t1 }
    }

    /// neumann_bc basis
    /// $$
    ///     \phi_0 = 0.5T_0 - 1/8T_1
    /// $$
    /// $$
    ///     \phi_1 = 0.5T_0 + 1/8T_1
    /// $$
    pub fn neumann(n: usize) -> Self {
        let m = Self::get_m(n);
        let _05 = A::from_f64(0.5).unwrap();
        let _18 = A::from_f64(1. / 8.).unwrap();
        let t0 = Array::from_vec(vec![_05, _05]);
        let t1 = Array::from_vec(vec![-(_18), _18]);
        StencilChebyshevBoundary { n, m, t0, t1 }
    }

    /// Return size of spectral space (number of coefficients) from size in physical space
    pub fn get_m(_n: usize) -> usize {
        2
    }
}

impl<A: FloatNum> Stencil<A> for StencilChebyshevBoundary<A> {
    /// Returns transform stencil as 2d ndarray
    fn to_array(&self) -> Array2<A> {
        let mut mat = Array2::<A>::zeros((self.n, self.m).f());
        mat[[0, 0]] = self.t0[0];
        mat[[0, 1]] = self.t0[1];
        mat[[1, 0]] = self.t1[0];
        mat[[1, 1]] = self.t1[1];
        mat
    }

    /// Multiply stencil with a 1d array (transforms to orthonorm coefficents)
    /// input and output array do usually differ in size.
    fn multiply_vec<S, T>(&self, composite_coeff: &ArrayBase<S, Ix1>) -> Array1<T>
    where
        S: ndarray::Data<Elem = T>,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        let mut orthonorm_coeff = Array1::<T>::zeros(self.n);
        self.multiply_vec_inplace(composite_coeff, &mut orthonorm_coeff);
        orthonorm_coeff
    }

    /// See [`StencilChebyshevBoundary::multiply_vec`]
    fn multiply_vec_inplace<S1, S2, T>(
        &self,
        composite_coeff: &ArrayBase<S1, Ix1>,
        orthonorm_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        orthonorm_coeff.mapv_inplace(|x| x * T::zero());
        orthonorm_coeff[0] = composite_coeff[0] * self.t0[0] + composite_coeff[1] * self.t0[1];
        orthonorm_coeff[1] = composite_coeff[0] * self.t1[0] + composite_coeff[1] * self.t1[1];
    }

    /// Solve linear algebraic system $p = S c$ for $p$ with given composite
    /// coefficents $c$.
    ///
    /// Input and output array do usually differ in size.
    fn solve_vec<S, T>(&self, orthonorm_coeff: &ArrayBase<S, Ix1>) -> Array1<T>
    where
        S: ndarray::Data<Elem = T>,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        let mut composite_coeff = Array1::<T>::zeros(self.m);
        self.solve_vec_inplace(orthonorm_coeff, &mut composite_coeff);
        composite_coeff
    }

    /// See [`StencilChebyshevBoundary::solve_vec`]
    fn solve_vec_inplace<S1, S2, T>(
        &self,
        orthonorm_coeff: &ArrayBase<S1, Ix1>,
        composite_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        let c0 = orthonorm_coeff[0] * self.t0[0] + orthonorm_coeff[1] * self.t1[0];
        let c1 = orthonorm_coeff[0] * self.t0[1] + orthonorm_coeff[1] * self.t1[1];
        // Determinante
        let a = self.t0[0] * self.t0[0] + self.t1[0] * self.t1[0];
        let b = self.t0[0] * self.t0[1] + self.t1[0] * self.t1[1];
        let c = self.t0[1] * self.t0[0] + self.t1[1] * self.t1[0];
        let d = self.t0[1] * self.t0[1] + self.t1[1] * self.t1[1];

        let det = A::one() / (a * d - b * c);
        composite_coeff[0] = (c0 * d - c1 * b) * det;
        composite_coeff[1] = (c1 * a - c0 * c) * det;
    }
}

/// Stencil with Dirichlet boundary conditions at x=-1
/// and Neumann boundary conditions at x=1
#[derive(Clone)]
pub struct StencilChebyshevDirchletNeumann<A> {
    /// Number of coefficients in orthonormal space
    n: usize,
    /// Number of coefficients in composite space
    m: usize,
    /// Main diagonal
    diag: Array1<A>,
    /// Subdiagonal offset -1
    low1: Array1<A>,
    /// Subdiagonal offset -2
    low2: Array1<A>,
    /// For fdma (diagonal)
    _fdma_diag: Array1<A>,
    /// For tdma (off-diagonal 1)
    _fdma_off1: Array1<A>,
    /// For tdma (off-diagonal 2)
    _fdma_off2: Array1<A>,
}

impl<A: FloatNum> StencilChebyshevDirchletNeumann<A> {
    /// Initialize stencil
    ///
    /// # Panics
    /// If casting from f64 to generic float fails
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn new(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = Array::from_vec(vec![A::one(); m]);
        let mut low1 = Array::from_vec(vec![A::zero(); m]);
        let mut low2 = Array::from_vec(vec![A::zero(); m]);
        for (k, (v1, v2)) in low1.iter_mut().zip(low2.iter_mut()).enumerate() {
            let kf64 = k as f64;
            *v1 = A::from_f64(
                (-1. * kf64.powi(2) + (kf64 + 2.).powi(2))
                    / ((kf64 + 1.).powi(2) + (kf64 + 2.).powi(2)),
            )
            .unwrap();
            *v2 = A::from_f64(
                (-1. * kf64.powi(2) - (kf64 + 1.).powi(2))
                    / ((kf64 + 1.).powi(2) + (kf64 + 2.).powi(2)),
            )
            .unwrap();
        }
        let (_fdma_diag, _fdma_off1, _fdma_off2) =
            Self::_get_main_off(&diag.view(), &low1.view(), &low2.view());

        Self {
            n,
            m,
            diag,
            low1,
            low2,
            _fdma_diag,
            _fdma_off1,
            _fdma_off2,
        }
    }

    /// Get main diagonal and two off diagonals of S.T@S, where S is stencil matrix
    /// S.T@S is five-diagonal, but symmetric about the diagonal
    fn _get_main_off(
        diag: &ArrayView1<A>,
        low1: &ArrayView1<A>,
        low2: &ArrayView1<A>,
    ) -> (Array1<A>, Array1<A>, Array1<A>) {
        let m = diag.len();
        let mut main = Array::from_vec(vec![A::zero(); m]);
        let mut off1 = Array::from_vec(vec![A::zero(); m - 1]);
        let mut off2 = Array::from_vec(vec![A::zero(); m - 2]);
        for (i, v) in main.iter_mut().enumerate() {
            *v = diag[i] * diag[i] + low1[i] * low1[i] + low2[i] * low2[i];
        }
        for (i, v) in off1.iter_mut().enumerate() {
            *v = diag[i + 1] * low1[i] + low1[i + 1] * low2[i];
        }
        for (i, v) in off2.iter_mut().enumerate() {
            *v = diag[i + 2] * low2[i];
        }
        (main, off1, off2)
    }

    /// Composite spaces can be smaller than its orthonormal counterpart
    pub fn get_m(n: usize) -> usize {
        n - 2
    }
}

impl<A: FloatNum> Stencil<A> for StencilChebyshevDirchletNeumann<A> {
    /// Returns transform stencil to a 2d ndarray
    fn to_array(&self) -> Array2<A> {
        let mut mat = Array2::<A>::zeros((self.n, self.m).f());
        for (i, ((d, l1), l2)) in self
            .diag
            .iter()
            .zip(self.low1.iter())
            .zip(self.low2.iter())
            .enumerate()
        {
            mat[[i, i]] = *d;
            mat[[i + 1, i]] = *l1;
            mat[[i + 2, i]] = *l2;
        }
        mat
    }

    /// Multiply stencil with a 1d array (transforms to parent coefficents)
    /// input and output array do usually differ in size.
    fn multiply_vec<S, T>(&self, composite_coeff: &ArrayBase<S, Ix1>) -> Array1<T>
    where
        S: ndarray::Data<Elem = T>,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        let mut orthonorm_coeff = Array1::<T>::zeros(self.n);
        self.multiply_vec_inplace(composite_coeff, &mut orthonorm_coeff);
        orthonorm_coeff
    }

    /// See [`StencilChebyshevDirchletNeumann::multiply_vec`]
    fn multiply_vec_inplace<S1, S2, T>(
        &self,
        composite_coeff: &ArrayBase<S1, Ix1>,
        orthonorm_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        orthonorm_coeff.mapv_inplace(|x| x * T::zero());
        orthonorm_coeff[0] = composite_coeff[0] * self.diag[0];
        orthonorm_coeff[1] = composite_coeff[1] * self.diag[1] + composite_coeff[0] * self.low1[0];
        for i in 2..self.n - 2 {
            orthonorm_coeff[i] = composite_coeff[i] * self.diag[i]
                + composite_coeff[i - 1] * self.low1[i - 1]
                + composite_coeff[i - 2] * self.low2[i - 2];
        }
        orthonorm_coeff[self.n - 2] = composite_coeff[self.n - 3] * self.low1[self.n - 3]
            + composite_coeff[self.n - 4] * self.low2[self.n - 4];
        orthonorm_coeff[self.n - 1] = composite_coeff[self.n - 3] * self.low2[self.n - 3];
    }

    /// Solve linear algebraic system $p = S c$ for $p$ with given composite
    /// coefficents $c$.
    ///
    /// Input and output array do usually differ in size.
    fn solve_vec<S, T>(&self, orthonorm_coeff: &ArrayBase<S, Ix1>) -> Array1<T>
    where
        S: ndarray::Data<Elem = T>,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        let mut composite_coeff = Array1::<T>::zeros(self.m);
        self.solve_vec_inplace(orthonorm_coeff, &mut composite_coeff);
        composite_coeff
    }

    /// See [`StencilChebyshevDirchletNeumann::solve_vec`]
    fn solve_vec_inplace<S1, S2, T>(
        &self,
        orthonorm_coeff: &ArrayBase<S1, Ix1>,
        composite_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        T: Scalar
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        use super::linalg::pdma;
        // Multiply right hand side
        for i in 0..self.m {
            composite_coeff[i] = orthonorm_coeff[i] * self.diag[i]
                + orthonorm_coeff[i + 1] * self.low1[i]
                + orthonorm_coeff[i + 2] * self.low2[i];
        }
        // Solve tridiagonal system
        pdma(
            &self._fdma_off2.view(),
            &self._fdma_off1.view(),
            &self._fdma_diag.view(),
            &self._fdma_off1.view(),
            &self._fdma_off2.view(),
            composite_coeff,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::{approx_eq, approx_eq_complex};
    use num_complex::Complex;

    #[test]
    fn test_stench_cheb() {
        let stencil = StencilChebyshev::<f64>::dirichlet(5);
        let orthonorm = Array::from_vec(vec![2., 0.7071, -1., -0.7071, -1.]);
        let composite = stencil.solve_vec(&orthonorm);
        approx_eq(&composite, &array![2., 0.70710678, 1.]);

        let stencil = StencilChebyshev::<f64>::dirichlet(5);
        let composite = Array::from_vec(vec![2., 0.70710678, 1.]);
        let orthonorm = stencil.multiply_vec(&composite);
        approx_eq(&orthonorm, &array![2., 0.7071, -1., -0.7071, -1.]);
    }

    #[test]
    fn test_stench_cheb_complex() {
        let stencil = StencilChebyshev::<f64>::dirichlet(5);
        let orthonorm = array![2., 0.7071, -1., -0.7071, -1.].mapv(|x| Complex::new(x, x));
        let expected = array![2., 0.70710678, 1.].mapv(|x| Complex::new(x, x));
        let composite = stencil.solve_vec(&orthonorm);
        approx_eq_complex(&composite, &expected);

        let stencil = StencilChebyshev::<f64>::dirichlet(5);
        let composite = array![2., 0.70710678, 1.].mapv(|x| Complex::new(x, x));
        let expected = array![2., 0.7071, -1., -0.7071, -1.].mapv(|x| Complex::new(x, x));
        let orthonorm = stencil.multiply_vec(&composite);
        approx_eq_complex(&orthonorm, &expected);
    }

    #[test]
    fn test_stench_cheb_boundary() {
        let stencil = StencilChebyshevBoundary::<f64>::dirichlet(4);
        let orthonorm = Array::from_vec(vec![1., 2., 3., 4.]);
        let composite = stencil.solve_vec(&orthonorm);
        approx_eq(&composite, &array![-1., 3.]);

        let stencil = StencilChebyshevBoundary::<f64>::dirichlet(4);
        let composite = Array::from_vec(vec![1., 2.]);
        let orthonorm = stencil.multiply_vec(&composite);
        approx_eq(&orthonorm, &array![1.5, 0.5, 0., 0.]);
    }

    #[test]
    fn test_stench_cheb_boundary_neumann() {
        let stencil = StencilChebyshevBoundary::<f64>::neumann(4);
        let orthonorm = Array::from_vec(vec![1., 2., 3., 4.]);
        let composite = stencil.solve_vec(&orthonorm);
        approx_eq(&composite, &array![-7., 9.]);

        let stencil = StencilChebyshevBoundary::<f64>::neumann(4);
        let composite = Array::from_vec(vec![-7., 9.]);
        let orthonorm = stencil.multiply_vec(&composite);
        approx_eq(&orthonorm, &array![1., 2., 0., 0.]);
    }

    #[test]
    fn test_stencil_cheb_dirichlet_neumann() {
        let stencil = StencilChebyshevDirchletNeumann::<f64>::new(8);
        // Test multiplication
        let composite = Array::from_vec(vec![1., 2., 3., 4., 5., 6.]);
        let orthonorm = stencil.multiply_vec(&composite);
        approx_eq(
            &orthonorm,
            &array![
                1.,
                2.8,
                4.03076923,
                4.67076923,
                5.00097561,
                5.20031987,
                -1.66653809,
                -4.30588235
            ],
        );
        // Test solution S.T @ S x = b (solve for x)
        let orthonorm = Array::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let composite = stencil.solve_vec(&orthonorm);
        approx_eq(
            &composite,
            &array![0.80428135, 1.35351682, 1.33709715, 2.28550459, 1.4272395, 2.15195267],
        );
    }
}
