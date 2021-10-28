//! Run with
//!
//! cargo mpirun --np 2 --example gather_root
use ndarray::Array2;
use rsmpi_decomp::mpi::initialize;
use rsmpi_decomp::Decomp2d;

fn main() {
    let (nx, ny) = (7, 3);
    let universe = initialize().unwrap();
    let decomp = Decomp2d::new(&universe, [nx, ny]);

    let mut global: Array2<f64> = Array2::zeros([nx, ny]);
    for (i, v) in global.iter_mut().enumerate() {
        *v = i as f64;
    }
    let xpen: Array2<f64> = decomp.split_array_x_pencil(&global);
    let ypen: Array2<f64> = decomp.split_array_y_pencil(&global);

    // x-pencil
    let mut rcv: Array2<f64> = Array2::zeros(decomp.n_global);
    decomp.gather_x(&xpen, &mut rcv);
    if decomp.nrank == 0 {
        assert_eq!(global, rcv);
    }

    // y-pencil
    let mut rcv: Array2<f64> = Array2::zeros(decomp.n_global);
    decomp.gather_y(&ypen, &mut rcv);
    if decomp.nrank == 0 {
        assert_eq!(global, rcv);
    }
}
