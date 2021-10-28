# rsmpi-decomp

## `rsmpi-decomp`: Domain decomposition for mpi data distribution

This library is work in progress...

## Examples
See `examples/`

## Supported:
- *2D domain*: Split 1 dimension.


### x-pencil domain:
```rust
--------------
|     P1     |
--------------
|     P0     |
--------------
```
### y-pencil domain:
```rust
---------------
|      |      |
|  P0  |  P1  |
|      |      |
---------------
```

## `cargo-mpirun`
Install:
```rust
cargo install cargo-mpirun
```
Run:
```rust
cargo mpirun --np 2 --example gather_root
```

License: MIT
