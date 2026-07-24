# Build

## Prerequisites

The MLX backend requires:

- an Apple Silicon Mac running macOS 14 or newer;
- a native arm64 Python 3.11 or newer; and
- Xcode Command Line Tools (`xcode-select --install`); and
- an up-to-date Rust toolchain.

Rosetta/x86 Python cannot use MLX. Mortal uses the Metal GPU by default and
shares memory between the CPU and GPU, so there is no CUDA or PyTorch
installation step.

## Clone and create the environment

```shell
$ git clone https://github.com/Equim-chan/Mortal.git
$ cd Mortal
$ conda env create -f environment.yml
$ conda activate mortal
```

Confirm that Python and MLX see the native GPU:

```shell
$ python -c 'import platform, mlx.core as mx; print(platform.machine(), mx.default_device())'
arm64 Device(gpu, 0)
```

## Build and install libriichi

> Working directory: the Mortal repository root.

```shell
$ cargo build -p libriichi --lib --release
$ cp target/release/libriichi.dylib mortal/libriichi.so
```

Test the Python extension and Metal execution:

```shell
$ PYTHONPATH=mortal python -c 'import libriichi, mlx.core as mx; x = mx.ones((2, 2)); mx.eval(x @ x); print("ready")'
ready
```

## Configure Mortal

> Working directory: the Mortal repository root.

```shell
$ cp mortal/config.example.toml mortal/config.toml
```

At minimum, set `control.state_file` to a native MLX `.safetensors`
checkpoint. `device = "gpu"` selects Metal; `"auto"` falls back to the CPU
when Metal is unavailable. Checkpoints contain model arrays, optimizer state,
and JSON metadata in one safe, portable Safetensors file.

## Convert an existing PyTorch checkpoint

The runtime has no PyTorch dependency. To reuse an old `.pth` model, perform a
one-time conversion in a temporary environment that has both PyTorch and MLX:

```shell
$ python mortal/convert_torch_checkpoint.py \
    old-model.pth mortal.safetensors
```

Convert the separate game-result predictor with:

```shell
$ python mortal/convert_torch_checkpoint.py \
    --grp old-grp.pth grp.safetensors
```

Conv1d kernels are transposed into MLX layout, and stacked GRU parameters are
mapped to native `mlx.nn.GRU` layers. GRP is converted from float64 to float32
because Metal does not support float64. Optimizer state is intentionally reset;
model weights, configuration, counters, timestamps, and tags are retained.

Dataset file indexes also changed from Torch serialization to JSON. Delete old
`.pth` indexes and let Mortal rebuild the configured `.json` indexes.
When reusing an old config, replace every `cuda:*` device with `"gpu"` or
`"auto"` and remove the obsolete CUDA/cuDNN/AMP options. Also remove
`dataset.num_workers`: native MLX batching currently performs dataset and
Rust/GRP preprocessing in the training process. If Metal utilization is low,
increase `dataset.file_batch_size` within the available unified memory.

## Online training compatibility

The MLX migration replaces Torch's online parameter serialization with a
versioned MessagePack protocol. Trainer, server, and worker processes must be
upgraded and restarted together; the new runtime rejects legacy or mismatched
wire versions with an explicit error.

## Run inference

```shell
$ (cd mortal && ./mortal 0 < /absolute/path/to/log.json)
```

Set `MORTAL_CFG=/absolute/path/to/config.toml` when the configuration is not
in the current directory.

## Optional targets

> Working directory: the Mortal repository root.

### Run tests

```shell
$ cargo test --workspace --no-default-features --features flate2/zlib -- --nocapture
$ python -m unittest discover -s tests -v
```

### Run benchmarks

```shell
$ cargo test -p libriichi --no-default-features --bench bench
```

### Build executable utilities

```shell
$ cargo build -p libriichi --bins --no-default-features --release
$ cargo build -p exe-wrapper --release
```

### Build documentation

```shell
$ cd docs
$ cargo install mdbook mdbook-admonish mdbook-pagetoc
$ mdbook build
```
