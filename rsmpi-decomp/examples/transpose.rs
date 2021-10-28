//! Run with
//!
//! cargo mpirun --np 2 --example transpose
use ndarray::Array2;
use rsmpi_decomp::mpi::initialize;
use rsmpi_decomp::Decomp2d;

fn main() {
    let (nx, ny) = (7, 4);
    let universe = initialize().unwrap();
    let decomp = Decomp2d::new(&universe, [nx, ny]);
    let mut global: Array2<f64> = Array2::zeros([nx, ny]);
    for (i, v) in global.iter_mut().enumerate() {
        *v = i as f64;
    }
    let xpen: Array2<f64> = decomp.split_array_x_pencil(&global);
    let ypen: Array2<f64> = decomp.split_array_y_pencil(&global);

    // test x to y
    let mut work: Array2<f64> = Array2::zeros(decomp.y_pencil.sz);
    decomp.transpose_x_to_y(&xpen, &mut work);
    assert_eq!(work, ypen);

    // test y to x
    let mut work: Array2<f64> = Array2::zeros(decomp.x_pencil.sz);
    decomp.transpose_y_to_x(&ypen, &mut work);
    assert_eq!(work, xpen);
}
