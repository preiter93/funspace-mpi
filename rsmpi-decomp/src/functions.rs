//! Collection of simplified mpi routines
use mpi::collective::Root;
use mpi::environment::Universe;
use mpi::topology::Communicator;
use mpi::traits::Equivalence;
use num_traits::Zero;

/// Broadcast scalar value from root to all processes
pub fn broadcast_scalar<T: Zero + Equivalence>(universe: &Universe, data: &mut T) {
    let world = universe.world();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);
    // let mut x = if world.rank() == root_rank {
    //     data
    // } else {
    //     T::zero()
    // };
    root_process.broadcast_into(data);
}
