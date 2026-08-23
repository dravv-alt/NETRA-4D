# Local GPU status

Observed on 2026-08-22:

- Installed discrete GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- Windows `Win32_VideoController.Status`: `Error`.
- Windows `ConfigManagerErrorCode`: `43`.
- `nvidia-smi`: failed because the GPU/driver was not available to the process.
- Installed PyTorch: `2.13.0+cpu`.
- `torch.cuda.is_available()`: `False`.

Therefore no local run may be labelled GPU-executed at this time. The verified
runner can execute on CPU, while the Colab notebook requires a real CUDA T4 and
stops if CUDA is unavailable.

For local CUDA execution, repair/reinstall the NVIDIA driver until Device
Manager no longer reports Code 43, reboot, install a CUDA-enabled PyTorch wheel,
and re-run with `--device cuda:0 --require-cuda`.
