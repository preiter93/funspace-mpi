//! # Real - to - complex fourier space
//! # Example
//! Initialize new fourier basis
//! ```
//! use funspace::fourier::FourierR2c;
//! let fo = FourierR2c::<f64>::new(4);
//! ```
#![allow(clippy::module_name_repetitions)]
use super::FourierC2c;
use crate::enums::BaseKind;
use crate::traits::BaseOperators;
use crate::traits::BaseSize;
use crate::traits::Basics;
use crate::traits::Differentiate;
use crate::traits::DifferentiatePar;
use crate::traits::FromOrtho;
use crate::traits::FromOrthoPar;
use crate::traits::LaplacianInverse;
use crate::traits::Transform;
use crate::traits::TransformPar;
use crate::types::FloatNum;
use crate::types::Scalar;
use ndarray::prelude::*;
use ndrustfft::R2cFftHandler;
use num_complex::Complex;

/// # Container for fourier space (Real-to-complex)
#[derive(Clone)]
pub struct FourierR2c<A> {
    /// Number of coefficients in physical space
    pub n: usize,
    /// Number of coefficients in spectral space
    pub m: usize,
    /// Grid coordinates of fourier nodes
    pub x: Array1<A>,
    /// Complex wavenumber vector
    pub k: Array1<Complex<A>>,
    /// Handles discrete cosine transform
    pub fft_handler: R2cFftHandler<A>,
}

impl<A: FloatNum> FourierR2c<A> {
    /// Returns a new Fourier Basis for real-to-complex transforms
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            m: n / 2 + 1,
            x: FourierC2c::nodes(n),
            k: Self::wavenumber(n),
            fft_handler: R2cFftHandler::new(n),
        }
    }

    /// Return complex wavenumber vector for r2c transform (0, 1, 2, 3)
    #[allow(clippy::missing_panics_doc)]
    fn wavenumber(n: usize) -> Array1<Complex<A>> {
        let mut k: Array1<Complex<A>> = Array1::zeros(n / 2 + 1);
        for (i, ki) in k.iter_mut().enumerate() {
            ki.im = A::from(i).unwrap();
        }
        k
    }

    /// Differentiate 1d Array *n_times*
    /// # Example
    /// Differentiate along lane
    /// ```
    /// use funspace::fourier::FourierR2c;
    /// use funspace::utils::approx_eq_complex;
    /// use ndarray::prelude::*;
    /// let fo = FourierR2c::<f64>::new(5);
    /// let mut k = fo.k.clone();
    /// let expected = k.mapv(|x| x.powf(2.));
    /// fo.differentiate_lane(&mut k, 1);
    /// approx_eq_complex(&k, &expected);
    /// ```
    ///
    /// # Panics
    /// When type conversion fails ( safe )
    pub fn differentiate_lane<S, T2>(&self, data: &mut ArrayBase<S, Ix1>, n_times: usize)
    where
        S: ndarray::Data<Elem = T2> + ndarray::DataMut,
        T2: Scalar + From<Complex<A>>,
    {
        let k = self.k.mapv(T2::from);
        for _ in 0..n_times {
            for (d, ki) in data.iter_mut().zip(k.iter()) {
                *d = *d * *ki;
            }
        }
    }
}

impl<A: FloatNum> BaseSize for FourierR2c<A> {
    /// Size in physical space
    fn len_phys(&self) -> usize {
        self.n
    }
    /// Size in spectral space
    fn len_spec(&self) -> usize {
        self.m
    }
    /// Size of orthogonal space
    fn len_orth(&self) -> usize {
        self.m
    }
}

impl<A: FloatNum> Basics<A> for FourierR2c<A> {
    /// Coordinates in physical space
    fn coords(&self) -> &Array1<A> {
        &self.x
    }
    /// Return mass matrix (= eye)
    fn mass(&self) -> Array2<A> {
        Array2::<A>::eye(self.m)
    }
    /// Return transform kind
    fn base_kind(&self) -> BaseKind {
        BaseKind::FourierR2c
    }
}

/// Copied from c2c
impl<A: FloatNum> Transform for FourierR2c<A> {
    type Physical = A;
    type Spectral = Complex<A>;

    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::fourier::FourierR2c;
    /// use funspace::Transform;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = FourierR2c::new(4);
    /// let input = array![1., 2., 3., 4.];
    /// let expected = array![
    ///     Complex::new(10., 0.),
    ///     Complex::new(-2., 2.),
    ///     Complex::new(-2., 0.)
    /// ];
    /// let output = fo.forward(&input, 0);
    /// approx_eq_complex(&output, &expected);
    /// ```
    fn forward<S, D>(&mut self, input: &ArrayBase<S, D>, axis: usize) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.forward_inplace(input, &mut output, axis);
        output
    }

    /// See [`FourierR2c::forward`]
    fn forward_inplace<S1, S2, D>(
        &mut self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndfft_r2c;
        check_array_axis(input, self.n, axis, Some("fourier forward"));
        check_array_axis(output, self.m, axis, Some("fourier forward"));
        ndfft_r2c(input, output, &mut self.fft_handler, axis);
    }

    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use funspace::fourier::FourierR2c;
    /// use funspace::Transform;
    /// use funspace::utils::approx_eq;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = FourierR2c::new(4);
    /// let input = array![
    ///     Complex::new(10., 0.),
    ///     Complex::new(-2., 2.),
    ///     Complex::new(-2., 0.)
    /// ];
    /// let expected = array![1., 2., 3., 4.];
    /// let output = fo.backward(&input, 0);
    /// approx_eq(&output, &expected);
    /// ```
    fn backward<S, D>(&mut self, input: &ArrayBase<S, D>, axis: usize) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.n, axis);
        self.backward_inplace(input, &mut output, axis);
        output
    }

    /// See [`FourierR2c::backward`]
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    #[allow(clippy::used_underscore_binding)]
    fn backward_inplace<S1, S2, D>(
        &mut self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndifft_r2c;
        check_array_axis(input, self.m, axis, Some("fourier backward"));
        check_array_axis(output, self.n, axis, Some("fourier backward"));
        ndifft_r2c(input, output, &mut self.fft_handler, axis);
    }
}

impl<A: FloatNum> TransformPar for FourierR2c<A> {
    type Physical = A;
    type Spectral = Complex<A>;

    /// Parallel version. See [`FourierR2c::forward`]
    fn forward_par<S, D>(
        &mut self,
        input: &ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.forward_inplace_par(input, &mut output, axis);
        output
    }

    /// Parallel version. See [`FourierR2c::forward_inplace`]
    fn forward_inplace_par<S1, S2, D>(
        &mut self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndfft_r2c_par;
        check_array_axis(input, self.n, axis, Some("fourier forward"));
        check_array_axis(output, self.m, axis, Some("fourier forward"));
        ndfft_r2c_par(input, output, &mut self.fft_handler, axis);
    }

    /// Parallel version. See [`FourierR2c::backward`]
    fn backward_par<S, D>(
        &mut self,
        input: &ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.n, axis);
        self.backward_inplace_par(input, &mut output, axis);
        output
    }

    /// Parallel version. See [`FourierR2c::backward_inplace`]
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    #[allow(clippy::used_underscore_binding)]
    fn backward_inplace_par<S1, S2, D>(
        &mut self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndifft_r2c_par;
        check_array_axis(input, self.m, axis, Some("fourier backward"));
        check_array_axis(output, self.n, axis, Some("fourier backward"));
        ndifft_r2c_par(input, output, &mut self.fft_handler, axis);
    }
}

/// Perform differentiation in spectral space
impl<A: FloatNum> Differentiate<Complex<A>> for FourierR2c<A> {
    fn differentiate<S, D>(
        &self,
        data: &ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        let mut output = data.to_owned();
        self.differentiate_inplace(&mut output, n_times, axis);
        output
    }

    fn differentiate_inplace<S, D>(&self, data: &mut ArrayBase<S, D>, n_times: usize, axis: usize)
    where
        S: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        check_array_axis(data, self.m, axis, Some("fourier differentiate"));
        ndarray::Zip::from(data.lanes_mut(Axis(axis))).for_each(|mut lane| {
            self.differentiate_lane(&mut lane, n_times);
        });
    }
}

/// Perform differentiation in spectral space
impl<A: FloatNum> DifferentiatePar<Complex<A>> for FourierR2c<A> {
    fn differentiate_par<S, D>(
        &self,
        data: &ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        let mut output = data.to_owned();
        self.differentiate_inplace_par(&mut output, n_times, axis);
        output
    }

    fn differentiate_inplace_par<S, D>(
        &self,
        data: &mut ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) where
        S: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        check_array_axis(data, self.m, axis, Some("fourier differentiate"));
        ndarray::Zip::from(data.lanes_mut(Axis(axis))).par_for_each(|mut lane| {
            self.differentiate_lane(&mut lane, n_times);
        });
    }
}

impl<A: FloatNum> BaseOperators for FourierR2c<A> {
    type Spectral = Complex<A>;
    /// Return explicit differentiation operator
    ///
    /// This function might be useful to build the operator
    /// matrices explicitly.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    fn diff_op(&self, deriv: usize) -> Array2<Self::Spectral> {
        let mut diff_mat = Array2::<Complex<A>>::zeros((self.m, self.m));
        for (l, k) in diff_mat.diag_mut().iter_mut().zip(self.k.iter()) {
            *l = Complex::new(A::zero(), k.im).powi(deriv as i32);
        }
        diff_mat
    }

    fn stencil(&self) -> Option<Array2<Self::Spectral>> {
        None
    }
}

impl<A: FloatNum> LaplacianInverse<A> for FourierR2c<A> {
    /// Laplacian ( = |k^2| ) diagonal matrix
    fn laplace(&self) -> Array2<A> {
        let mut lap = Array2::<A>::zeros((self.m, self.m));
        for (l, k) in lap.diag_mut().iter_mut().zip(self.k.iter()) {
            *l = -k.im * k.im;
        }
        lap
    }

    /// Pseudoinverse Laplacian for `FourierR2c` basis
    fn laplace_inv(&self) -> Array2<A> {
        let mut pinv = self.laplace();
        for p in pinv.slice_mut(s![1.., 1..]).diag_mut().iter_mut() {
            *p = A::one() / *p;
        }
        pinv
    }

    /// Pseudoidentity matrix (= eye matrix with removed
    /// first row for `FourierR2c`)
    fn laplace_inv_eye(&self) -> Array2<A> {
        let eye = Array2::<A>::eye(self.m);
        eye.slice(s![1.., ..]).to_owned()
    }
}

impl<A: FloatNum> FromOrtho<Complex<A>> for FourierR2c<A> {
    /// Return itself
    fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, _axis: usize) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        input.to_owned()
    }

    /// Return itself
    fn to_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        _axis: usize,
    ) where
        S1: ndarray::Data<Elem = Complex<A>>,
        S2: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        output.assign(input);
    }

    /// Return itself
    fn from_ortho<S, D>(&self, input: &ArrayBase<S, D>, _axis: usize) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        input.to_owned()
    }

    /// Return itself
    fn from_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        _axis: usize,
    ) where
        S1: ndarray::Data<Elem = Complex<A>>,
        S2: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        output.assign(input);
    }
}

impl<A: FloatNum> FromOrthoPar<Complex<A>> for FourierR2c<A> {
    /// Return itself
    fn to_ortho_par<S, D>(&self, input: &ArrayBase<S, D>, _axis: usize) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        input.to_owned()
    }

    /// Return itself
    fn to_ortho_inplace_par<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        _axis: usize,
    ) where
        S1: ndarray::Data<Elem = Complex<A>>,
        S2: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        output.assign(input);
    }

    /// Return itself
    fn from_ortho_par<S, D>(&self, input: &ArrayBase<S, D>, _axis: usize) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        input.to_owned()
    }

    /// Return itself
    fn from_ortho_inplace_par<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        _axis: usize,
    ) where
        S1: ndarray::Data<Elem = Complex<A>>,
        S2: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        output.assign(input);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::approx_eq_complex;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_fourier_r2c_differentiation_operator() {
        let fourier = FourierR2c::new(5);
        let diff_op: Array2<Complex<f64>> = fourier.diff_op(1);

        // Higher order derivative
        let diff_op2: Array2<Complex<f64>> = fourier.diff_op(2);
        approx_eq_complex(&diff_op2, &diff_op.dot(&diff_op));

        // Compare with implicit differentiation
        let mut data = Array1::<Complex<f64>>::zeros(3);
        for (i, v) in data.iter_mut().enumerate() {
            *v = Complex::new(i as f64, i as f64);
        }
        let data_deriv = fourier.differentiate(&data, 1, 0);
        approx_eq_complex(&data_deriv, &diff_op.dot(&data));
    }
}
