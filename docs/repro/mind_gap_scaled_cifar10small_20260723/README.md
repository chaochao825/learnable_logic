# Mind-the-Gap scaled CIFAR-10-small smoke run

This server-210 capture validates the image data path and all practical
baseline execution paths.

- Remote source commit: `d1a9354`
- Architecture: fixed-wiring width 1,600, depth 3, 4,800 gates
- Budget: 20 total epochs
- Seed: 0
- Split: 2,000 training and 500 test examples
- Device: CUDA
- Protocol: `--mind-gap-scaled`

Task-aware coordinate refit was omitted because its gate-by-gate search is not
appropriate for this bounded smoke budget. The result must not be compared as
a reproduction of the 61M-gate ConvLGN or 256K-wide Mind-the-Gap settings.
