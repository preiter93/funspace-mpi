//! Transformation stencils from orthogonal chebyshev space to composite space
//! $$
//! p = S c
//! $$
//! where $S$ is a two-dimensional transform matrix.
use super::dirichlet::StencilChebDirichlet;
use super::dirichlet_bc::StencilChebDirichletBc;
use super::dirichlet_neumann::StencilChebDirichletNeumann;
use super::neumann::StencilChebNeumann;
use super::neumann_bc::StencilChebNeumannBc;
use crate::{FloatNum, Scalar};
use ndarray::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

#[allow(clippy::pub_enum_variant_names)]
#[enum_dispatch(Stencil<A>)]
#[derive(Clone)]
pub enum StencilChebyshev<A: FloatNum> {
    StencilChebDirichlet(StencilChebDirichlet<A>),
    StencilChebNeumann(StencilChebNeumann<A>),
    StencilChebDirichletNeumann(StencilChebDirichletNeumann<A>),
    StencilChebDirichletBc(StencilChebDirichletBc<A>),
    StencilChebNeumannBc(StencilChebNeumannBc<A>),
}

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
