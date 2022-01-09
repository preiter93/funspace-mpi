//! Chebyshev base with Dirichlet boundary conditions
use super::stencil::Stencil;
use crate::types::{FloatNum, Scalar};
use ndarray::{Array, Array1, Array2, ArrayBase, ArrayView1, Ix1, ShapeBuilder};
use std::ops::{Add, Div, Mul, Sub};

/// Container for Chebyshev Stencil with Dirichlet boundary conditions
#[derive(Clone)]
pub struct StencilChebDirichlet<A> {
    /// Number of coefficients in orthonormal space
    n: usize,
    /// Number of coefficients in composite space
    m: usize,
    /// Main diagonal
    diag: Array1<A>,
    /// Subdiagonal offset -2
    low2: Array1<A>,
    /// For tdma (diagonal)
    tdma_diag: Array1<A>,
    /// For tdma (off-diagonal)
    tdma_off2: Array1<A>,
}

impl<A: FloatNum> StencilChebDirichlet<A> {
    /// Return stencil of chebyshev dirichlet space
    /// $$
    ///  \phi_k = T_k - T_{k+2}
    /// $$
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    pub fn new(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = Array::from_vec(vec![A::one(); m]);
        let low2 = Array::from_vec(vec![-A::one(); m]);
        let (tdma_diag, tdma_off2) = Self::get_tdma_diags(&diag.view(), &low2.view());
        Self {
            n,
            m,
            diag,
            low2,
            tdma_diag,
            tdma_off2,
        }
    }

    /// Get main diagonal and off diagonal, used in [`StencilChebyshev::solve_vec_inplace`]
    fn get_tdma_diags(diag: &ArrayView1<A>, low2: &ArrayView1<A>) -> (Array1<A>, Array1<A>) {
        let m = diag.len();
        let mut tdma_diag = Array::from_vec(vec![A::zero(); m]);
        let mut tdma_off2 = Array::from_vec(vec![A::zero(); m - 2]);
        for (i, v) in tdma_diag.iter_mut().enumerate() {
            *v = diag[i] * diag[i] + low2[i] * low2[i];
        }
        for (i, v) in tdma_off2.iter_mut().enumerate() {
            *v = diag[i + 2] * low2[i];
        }
        (tdma_diag, tdma_off2)
    }

    /// Composite spaces can be smaller than its orthonormal counterpart
    pub fn get_m(n: usize) -> usize {
        n - 2
    }
}

impl<A: FloatNum> Stencil<A> for StencilChebDirichlet<A> {
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
        use crate::chebyshev::linalg::tdma;
        // Multiply right hand side
        for i in 0..self.m {
            composite_coeff[i] =
                orthonorm_coeff[i] * self.diag[i] + orthonorm_coeff[i + 2] * self.low2[i];
        }
        // Solve tridiagonal system
        tdma(
            &self.tdma_off2.view(),
            &self.tdma_diag.view(),
            &self.tdma_off2.view(),
            composite_coeff,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::{approx_eq, approx_eq_complex};
    use ndarray::array;
    use num_complex::Complex;

    #[test]
    fn test_stench_cheb() {
        let stencil = StencilChebDirichlet::<f64>::new(5);
        let orthonorm = Array::from_vec(vec![2., 0.7071, -1., -0.7071, -1.]);
        let composite = stencil.solve_vec(&orthonorm);
        approx_eq(&composite, &array![2., 0.70710678, 1.]);

        let stencil = StencilChebDirichlet::<f64>::new(5);
        let composite = Array::from_vec(vec![2., 0.70710678, 1.]);
        let orthonorm = stencil.multiply_vec(&composite);
        approx_eq(&orthonorm, &array![2., 0.7071, -1., -0.7071, -1.]);
    }

    #[test]
    fn test_stench_cheb_complex() {
        let stencil = StencilChebDirichlet::<f64>::new(5);
        let orthonorm = array![2., 0.7071, -1., -0.7071, -1.].mapv(|x| Complex::new(x, x));
        let expected = array![2., 0.70710678, 1.].mapv(|x| Complex::new(x, x));
        let composite = stencil.solve_vec(&orthonorm);
        approx_eq_complex(&composite, &expected);

        let stencil = StencilChebDirichlet::<f64>::new(5);
        let composite = array![2., 0.70710678, 1.].mapv(|x| Complex::new(x, x));
        let expected = array![2., 0.7071, -1., -0.7071, -1.].mapv(|x| Complex::new(x, x));
        let orthonorm = stencil.multiply_vec(&composite);
        approx_eq_complex(&orthonorm, &expected);
    }
}
