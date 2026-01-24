![Logo](https://i.ibb.co/LWY1P0D/resized.png)

# flask-image-upscaler
A minimal Flask web app that upscales images with Real-ESRGAN. Upload an image on the home page and download the upscaled result on the results page. Uses shared templates, a single stylesheet, and simple navigation.

## Features
- Upload → upscale → download flow
- Model selection dropdown (auto-detects models in `weights/` folder)
- Three routes: `/`, `/result/<job_id>`, `/about`
- Shared CSS with easy to tweak variables
- Supports PNG, JPG, JPEG, WEBP, BMP
- Optional tiling for large images (reduces memory usage)

## Requirements
- Tested on Python 3.12.*
- NVIDIA GPU recommended for best performance
- Real-ESRGAN model file (default path below)

## Quick Start
```bash
# create and activate a virtual env (example)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# install deps
pip install --upgrade pip
pip install -r requirements.txt
```

Conda users can do:
```bash
conda create -n flask312 python=3.12
conda activate flask312
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download a compatible Real-ESRGAN model

This app needs a pretrained ESRGAN model file. You can download multiple models and select between them in the UI.

**Download:**
- [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) - General purpose, works well on most images
- [RealESRGAN_x4plus_anime_6B.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) - Optimized for anime and line art

**Place the files:**
```bash
# Create a weights folder in your project root if it does not exist
mkdir -p weights
# Download models (macOS/Linux)
curl -L -o weights/RealESRGAN_x4plus.pth "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
curl -L -o weights/RealESRGAN_x4plus_anime_6B.pth "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
```

## Environment file
Create a `.env` file in the project root. **Do not commit** this file to source control.
```
# Default model (pre-selected in the dropdown)
REAL_ESRGAN_MODEL=weights/RealESRGAN_x4plus.pth
# Change this in production
DEV_SECRET=<your_secret_here>
```

## Run the project
Set your model path and dev secret in an `.env` file in the project root (see above), then run:
```bash
flask run
# open http://127.0.0.1:5000
```

## CUDA vs CPU
If your GPU supports CUDA, the provided requirements use the CUDA 12.4 PyTorch wheels.  
If your graphics card does not support CUDA, install the CPU builds instead:
```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

## Important note for Real-ESRGAN GPU device selection
If you have only one GPU, some dependency code in the `realesrgan` package may default to CPU unless you change the following in your local install.

```python
# fix for realesrgan gpu issue, make sure your module has this change
# CHANGE line 48 of utils.py to:
if gpu_id is not None:
```

## Troubleshooting
- If `basicsr` from PyPI fails on newer Python, the requirements pull it directly from GitHub.
- If installs complain about CUDA on unsupported hardware, switch to CPU PyTorch as shown above.
- Very large images can consume a lot of VRAM. Use the optional tiles input in the upload form to reduce memory usage.


## License
This project is for educational use. Check the licenses of Real-ESRGAN and its dependencies before redistribution.
