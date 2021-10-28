//! Run with
//!
//! cargo mpirun --np 2 --example broadcast_root
use rsmpi_decomp::functions::broadcast_scalar;
use rsmpi_decomp::mpi::initialize;
use rsmpi_decomp::mpi::traits::Communicator;

fn main() {
    let universe = initialize().unwrap();
    let world = universe.world();
    let mut x = if world.rank() == 0 { 1000.4 } else { 0. };
    broadcast_scalar(&universe, &mut x);
    assert_eq!(x, 1000.4);
}
