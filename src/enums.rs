//! Inidividual enums for all bases who transform either
//! real-to-real, real-to-complex or complex-to-complex plus
//! a collective enum of enums, which combines these three types
//! and which can be used to put all Bases in a single array.
//!
//! Real-to-complex transforms implement the differentiate and
//! from ortho trait only for complex numbers
use crate::chebyshev::Chebyshev;
use crate::chebyshev::CompositeChebyshev;
use crate::fourier::FourierC2c;
use crate::fourier::FourierR2c;
use crate::traits::BaseOperators;
use crate::traits::Differentiate;
use crate::traits::DifferentiatePar;
use crate::traits::FromOrtho;
use crate::traits::FromOrthoPar;
use crate::traits::Transform;
use crate::traits::TransformPar;
use crate::types::FloatNum;
use ndarray::prelude::*;
use num_complex::Complex;

// /// Define which number format the
// /// arrays have before and after
// /// a transform (Type in physical space
// /// and type in spectral space)
// #[derive(Clone)]
// pub enum TransformKind {
//     /// Real to real transform
//     RealToReal,
//     /// Complex to complex transform
//     ComplexToComplex,
//     /// Real to complex transform
//     RealToComplex,
// }

//#[enum_dispatch(BaseSize, Basics<T>, LaplacianInverse<T>)]
/// Enum with all available bases
#[derive(Debug, Clone, Copy)]
pub enum BaseKind {
    /// Chebyshev orthogonal base
    Chebyshev,
    /// Chebyshev dirichlet base
    ChebDirichlet,
    /// Chebyshev neumann base
    ChebNeumann,
    /// Chebyshev dirichlet - neumann base
    ChebDirichletNeumann,
    /// Chebyshev dirichlet base (for boundary conditions)
    ChebDirichletBc,
    /// Chebyshev neumann base (for boundary conditions)
    ChebNeumannBc,
    /// Fourier real to complex
    FourierR2c,
    /// Fourier complex to complex
    FourierC2c,
}

impl std::fmt::Display for BaseKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            BaseKind::Chebyshev => write!(f, "Chebyhev"),
            BaseKind::ChebDirichlet => write!(f, "ChebDirichlet"),
            BaseKind::ChebNeumann => write!(f, "ChebNeumann"),
            BaseKind::ChebDirichletNeumann => write!(f, "ChebDirichletNeumann"),
            BaseKind::ChebDirichletBc => write!(f, "ChebDirichletBc"),
            BaseKind::ChebNeumannBc => write!(f, "ChebNeumannBc"),
            BaseKind::FourierR2c => write!(f, "FourierR2c"),
            BaseKind::FourierC2c => write!(f, "FourierC2c"),
        }
    }
}

// #[allow(clippy::pub_enum_variant_names)]
// #[enum_dispatch(BaseSize, Basics<T>, LaplacianInverse<T>)]
// #[derive(Clone)]
// /// Enum of enums which binds all bases
// pub enum BaseAll<T: FloatNum> {
//     /// Real-to-real bases
//     BaseR2r(BaseR2r<T>),
//     /// Real-to-complex bases
//     BaseR2c(BaseR2c<T>),
//     /// Complex-to-complex bases
//     BaseC2c(BaseC2c<T>),
// }

#[allow(clippy::large_enum_variant)]
#[enum_dispatch(BaseSize, Basics<T>, LaplacianInverse<T>)]
#[derive(Clone)]
/// All bases who transform: real-to-real
pub enum BaseR2r<T: FloatNum> {
    /// Chebyshev polynomials (orthogonal)
    Chebyshev(Chebyshev<T>),
    /// Chebyshev polynomials (composite)
    CompositeChebyshev(CompositeChebyshev<T>),
}

#[enum_dispatch(BaseSize, Basics<T>, LaplacianInverse<T>)]
#[derive(Clone)]
/// All bases who transform: real-to-complex
pub enum BaseR2c<T: FloatNum> {
    /// Fourier polynomials
    FourierR2c(FourierR2c<T>),
}

#[enum_dispatch(BaseSize, Basics<T>, LaplacianInverse<T>)]
#[derive(Clone)]
/// All bases who transform: complex-to-complex
pub enum BaseC2c<T: FloatNum> {
    /// Fourier polynomials
    FourierC2c(FourierC2c<T>),
}

// Implement traits on real-to-real
impl_transform_trait_for_base!(BaseR2r, A, A, Chebyshev, CompositeChebyshev);
impl_differentiate_trait_for_base!(BaseR2r, A, Chebyshev, CompositeChebyshev);
impl_differentiate_trait_for_base!(BaseR2r, Complex<A>, Chebyshev, CompositeChebyshev);
impl_from_ortho_trait_for_base!(BaseR2r, A, Chebyshev, CompositeChebyshev);
impl_from_ortho_trait_for_base!(BaseR2r, Complex<A>, Chebyshev, CompositeChebyshev);
impl_baseoperator_trait_for_base!(BaseR2r, A, A, Chebyshev, CompositeChebyshev);

// Implement traits on real-to-complex
impl_transform_trait_for_base!(BaseR2c, A, Complex<A>, FourierR2c);
impl_differentiate_trait_for_base!(BaseR2c, Complex<A>, FourierR2c);
impl_from_ortho_trait_for_base!(BaseR2c, Complex<A>, FourierR2c);
impl_baseoperator_trait_for_base!(BaseR2c, A, Complex<A>, FourierR2c);

// Implement traits on complex-to-complex
impl_transform_trait_for_base!(BaseC2c, Complex<A>, Complex<A>, FourierC2c);
impl_differentiate_trait_for_base!(BaseC2c, Complex<A>, FourierC2c);
impl_from_ortho_trait_for_base!(BaseC2c, Complex<A>, FourierC2c);
impl_baseoperator_trait_for_base!(BaseC2c, Complex<A>, Complex<A>, FourierC2c);
