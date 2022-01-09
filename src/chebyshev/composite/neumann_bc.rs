//! Chebyshev base for Neumann boundary conditions
#![allow(clippy::used_underscore_binding)]
use super::stencil::Stencil;
use crate::types::{FloatNum, Scalar};
use ndarray::{Array, Array1, Array2, ArrayBase, Ix1, ShapeBuilder};
use std::ops::{Add, Div, Mul, Sub};

/// Container for Boundary Condition Stencil
///
/// This stencil is fully defined by the
/// the 2 coefficients that act on $T_1$ and the
/// 2 coefficients that act on $T_2$, where $T$
/// are the basis function of the orthonormal
/// chebyshev basis.
#[derive(Clone)]
pub struct StencilChebNeumannBc<A> {
    /// Number of coefficients in orthonormal space
    n: usize,
    /// Number of coefficients in composite space
    m: usize,
    /// T1
    t1: Array1<A>,
    /// T2
    t2: Array1<A>,
}

impl<A: FloatNum> StencilChebNeumannBc<A> {
    /// neumann bc basis
    /// $$
    ///     \phi_0 = 0.5T_1 - 1/8T_2
    /// $$
    /// $$
    ///     \phi_1 = 0.5T_1 + 1/8T_2
    /// $$
    pub fn new(n: usize) -> Self {
        let m = Self::get_m(n);
        let _05 = A::from_f64(0.5).unwrap();
        let _18 = A::from_f64(1. / 8.).unwrap();
        let t1 = Array::from_vec(vec![_05, _05]);
        let t2 = Array::from_vec(vec![-(_18), _18]);
        Self { n, m, t1, t2 }
    }
    /// Return size of spectral space (number of coefficients) from size in physical space
    pub fn get_m(_n: usize) -> usize {
        2
    }
}

impl<A: FloatNum> Stencil<A> for StencilChebNeumannBc<A> {
    /// Returns transform stencil as 2d ndarray
    fn to_array(&self) -> Array2<A> {
        let mut mat = Array2::<A>::zeros((self.n, self.m).f());
        mat[[1, 0]] = self.t1[0];
        mat[[1, 1]] = self.t1[1];
        mat[[2, 0]] = self.t2[0];
        mat[[2, 1]] = self.t2[1];
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
        orthonorm_coeff[1] = composite_coeff[0] * self.t1[0] + composite_coeff[1] * self.t1[1];
        orthonorm_coeff[2] = composite_coeff[0] * self.t2[0] + composite_coeff[1] * self.t2[1];
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
        let c0 = orthonorm_coeff[1] * self.t1[0] + orthonorm_coeff[2] * self.t2[0];
        let c1 = orthonorm_coeff[1] * self.t1[1] + orthonorm_coeff[2] * self.t2[1];
        // Determinante
        let a = self.t1[0] * self.t1[0] + self.t2[0] * self.t2[0];
        let b = self.t1[0] * self.t1[1] + self.t2[0] * self.t2[1];
        let c = self.t1[1] * self.t1[0] + self.t2[1] * self.t2[0];
        let d = self.t1[1] * self.t1[1] + self.t2[1] * self.t2[1];

        let det = A::one() / (a * d - b * c);
        composite_coeff[0] = (c0 * d - c1 * b) * det;
        composite_coeff[1] = (c1 * a - c0 * c) * det;
    }
}
