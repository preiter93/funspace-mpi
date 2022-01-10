//! Feature: Mpi parallel spaces in two-dimensions
#![cfg(feature = "mpi")]
pub mod decomp_handler;
pub mod space2;
pub mod space_traits;
pub use decomp_handler::DecompHandler;
pub use mpi_decomp::functions::all_gather_sum;
pub use mpi_decomp::functions::broadcast_scalar;
pub use mpi_decomp::functions::gather_sum;
pub use mpi_decomp::mpi::environment::Universe;
pub use mpi_decomp::mpi::initialize;
pub use mpi_decomp::mpi::topology::Communicator;
pub use mpi_decomp::mpi::traits::Equivalence;
pub use mpi_decomp::Decomp2d;
pub use space2::Space2;
pub use space_traits::BaseSpaceMpi;
