//! Mixed Dirichlet - Neumann base
use super::stencil::Stencil;
use crate::types::{FloatNum, Scalar};
use ndarray::{Array, Array1, Array2, ArrayBase, ArrayView1, Ix1, ShapeBuilder};
use std::ops::{Add, Div, Mul, Sub};

/// Stencil with Dirichlet boundary conditions at x=-1
/// and Neumann boundary conditions at x=1
#[derive(Clone)]
pub struct StencilChebDirichletNeumann<A> {
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
    fdma_diag: Array1<A>,
    /// For tdma (off-diagonal 1)
    fdma_off1: Array1<A>,
    /// For tdma (off-diagonal 2)
    fdma_off2: Array1<A>,
}

impl<A: FloatNum> StencilChebDirichletNeumann<A> {
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
        let (fdma_diag, fdma_off1, fdma_off2) =
            Self::_get_main_off(&diag.view(), &low1.view(), &low2.view());

        Self {
            n,
            m,
            diag,
            low1,
            low2,
            fdma_diag,
            fdma_off1,
            fdma_off2,
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

impl<A: FloatNum> Stencil<A> for StencilChebDirichletNeumann<A> {
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
        use crate::chebyshev::linalg::pdma;
        // Multiply right hand side
        for i in 0..self.m {
            composite_coeff[i] = orthonorm_coeff[i] * self.diag[i]
                + orthonorm_coeff[i + 1] * self.low1[i]
                + orthonorm_coeff[i + 2] * self.low2[i];
        }
        // Solve tridiagonal system
        pdma(
            &self.fdma_off2.view(),
            &self.fdma_off1.view(),
            &self.fdma_diag.view(),
            &self.fdma_off1.view(),
            &self.fdma_off2.view(),
            composite_coeff,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::approx_eq;
    use ndarray::array;

    #[test]
    fn test_stencil_cheb_dirichlet_neumann() {
        let stencil = StencilChebDirichletNeumann::<f64>::new(8);
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
