# contour_net_rust

A command-line application written in Rust that trains and uses a small neural network to detect contour lines in images.

## How it works

The project uses a fully connected feedforward neural network with a **36–40–36** architecture: 36 input neurons, one hidden layer with 40 neurons, and 36 output neurons. Each training example is a pair of corresponding 6×6 patches from the source and target images. The network uses sigmoid activation and online backpropagation with a learning rate of 1.0. A weight update is performed after every patch.

Training samples random patches for 100,000 steps by default. It reads the low-order blue channel from each RGB pixel and divides it by 256. Conversion uses grayscale input, processes 6×6 patches with a stride of 3, and writes overlapping predictions in scan order.

The model is stored as JSON in the same format as the Python contour_net project. The included bin/trained.json can be used immediately.

## Requirements

- Rust and Cargo (install via https://rustup.rs/)

## Build

From the project directory:

    cargo build --release

## Train

Train with the bundled source and target images:

    cargo run --release -- train

By default, the command reads bin/keanu.png and bin/keanu_result.png, performs 100,000 training steps, and writes the model to bin/trained.json. Progress is printed every 50,000 cases. Training time is reported separately and does not include saving the model.

Options:

    --source PATH   Source image
    --target PATH   Target image
    --model PATH    Output model path
    --steps N       Number of training steps

For example:

    cargo run --release -- train --steps 100000 --model bin/trained.json

## Convert

Convert an image using the bundled trained model:

    cargo run --release -- convert bin/keanu3.png

By default, the result is saved beside the input as keanu3_result.png. Conversion time is reported without counting the output file save.

Options:

    --model PATH    Model JSON path
    --output PATH   Output image path

For example:

    cargo run --release -- convert bin/london.png --model bin/trained.json --output bin/london_contours.png

## Typical runtime

In one reported run under comparable conditions:

| Version | Training (100,000 cases) | Image conversion |
| --- | ---: | ---: |
| Python | 7.13 s | 0.97 s |
| Rust | 0.68 s | 0.15 s |

Runtime depends on hardware, compiler settings, and input images; these figures are indicative, not a benchmark guarantee. Random-number generators differ between the implementations, so retraining does not produce identical weights.

## Project structure

- src/main.rs — CLI, neural network, training, and conversion logic
- bin/ — sample images and the bundled trained model
- Cargo.toml — package metadata and dependencies
- Cargo.lock — locked dependency versions for reproducible builds

## License

No license has been selected yet.
