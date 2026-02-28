[IMPORTANT]
> ** v1.1.0 Update: Improve caching and others I/O performance. Fix multi-GPU support on Windows with DPP. Check for more information below. Please consider update asap**

# Anima Standalone Trainer

A lightweight, decoupled training environment for circlestone-labs' Anima model, currently support Lora training only. Windows only, Linux compatibility is in work. Built upon [sd-scripts](https://github.com/kohya-ss/sd-scripts) implementation.

<img width="2554" height="1234" alt="image" src="https://github.com/user-attachments/assets/cb5ff930-ce8c-49d6-a77a-3da393fe719d" />


## Prerequisites

- **Python 3.10+** (Python 3.12 recommended)
- **Node.js** (Required for the Web UI)
- **CUDA fitting your system** (CUDA 12.7+ recommended)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/gazingstars123/Anima-Standalone-Trainer.git
cd Anima-Standalone-Trainer
```

### 2. Set up the environment

Run the provided setup script for your operating system:

**Windows:**
```powershell
.\setup_env.bat
```

*This will create a virtual environment (`venv`), install all Python dependencies (assuming you have met the prereqisites), and set up the Web UI.*

This script will probably install a pytorch with CPU only.
Depends on your system, you may want to install a specific version of Pytorch with CUDA.

```cmd
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
```

## Launching the UI

To start the training server and open the web interface:
```bash
.training_ui\start_training_ui_anima.bat
```
Once launched, open your browser to: `http://localhost:3000`

## Release

**v1.1.0. Improving caching and others I/O performance.**

## Multi-GPUs

Tested on torch2.7+cu128 and torch2.10+cu130 with [this fix](https://github.com/pytorch/pytorch/pull/175316) applied on Windows when encountered **libuv** error.

Seems to works best with torch<=2.3 and cuda <= 12.4 without directly applying the fix.

## Update

To update, simply run this command

```cmd
git pull
```

## Misc

Some features and settings from sd-scripts may not be available or working properly at the momment.

Linux are not tested.

Built and tested on Windows 11, 1x RTX 3090, CUDA 13.1, Pytorch 2.10 


