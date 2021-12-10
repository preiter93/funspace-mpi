//! Run with
//!
//! cargo mpirun --np 2 --example gather_sum
use rsmpi_decomp::functions::all_gather_sum;
use rsmpi_decomp::functions::gather_sum;
use rsmpi_decomp::mpi::initialize;
use rsmpi_decomp::mpi::traits::Communicator;

fn main() {
    let universe = initialize().unwrap();
    let world = universe.world();
    let x = world.rank() as f64;
    let mut y = 0.;

    // gather
    gather_sum(&universe, &x, &mut y);
    if world.rank() == 0 {
        let mut y2 = 0.;
        for i in 0..world.size() {
            y2 += i as f64;
        }
        assert_eq!(y, y2);
    }

    // all gather
    all_gather_sum(&universe, &x, &mut y);
    let mut y2 = 0.;
    for i in 0..world.size() {
        y2 += i as f64;
    }
    assert_eq!(y, y2);
}
