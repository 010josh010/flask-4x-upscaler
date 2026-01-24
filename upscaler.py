"""ESRGAN upscaler with tile slicing to work w/ image-slicer 3.1.0 and enable safe PIL large-image handling."""

import argparse
from math import ceil, sqrt
from tempfile import TemporaryDirectory

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from image_slicer import ImageSlicer, join_image

Image.MAX_IMAGE_PIXELS = None  # allow very large images (intentional for ESRGAN output)

GPU_ID = 0
SCALE = 4
NAMING_FORMAT = "tile_r{row}_c{col}.png"

# Model configurations: (num_block, num_grow_ch)
MODEL_CONFIGS = {
    "anime_6B": (6, 32),   # RealESRGAN_x4plus_anime_6B
    "default": (23, 32),   # RealESRGAN_x4plus and others
}


def _get_model_config(model_path: str) -> tuple[int, int]:
    """Determine model configuration based on filename."""
    model_name = model_path.lower()
    if "anime_6b" in model_name or "anime6b" in model_name:
        return MODEL_CONFIGS["anime_6B"]
    return MODEL_CONFIGS["default"]


def convert_from_image_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV ndarray (BGR)."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # pylint: disable=no-member


def _pyvips_to_numpy_rgb(vips_img) -> np.ndarray:
    """Convert a pyvips.Image to an RGB uint8 numpy array with shape (H, W, 3)."""
    height, width, bands = vips_img.height, vips_img.width, vips_img.bands
    mem = vips_img.write_to_memory()
    return np.frombuffer(mem, dtype=np.uint8).reshape(height, width, bands)


def _best_grid(n_tiles: int) -> tuple[int, int]:
    """Choose a roughly square (rows, cols) grid for about n_tiles tiles."""
    cols = ceil(sqrt(n_tiles))
    rows = ceil(n_tiles / cols)
    return rows, cols


def _make_upsampler(model_path: str) -> RealESRGANer:
    """Build a RealESRGAN upsampler for the RRDBNet x4 model."""
    num_block, num_grow_ch = _get_model_config(model_path)
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=num_grow_ch, scale=SCALE
    )
    return RealESRGANer(
        scale=SCALE,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=GPU_ID,
    )


def upscale(model_path: str, im_path: str) -> np.ndarray:
    """Upscale a whole image without slicing; returns OpenCV BGR ndarray."""
    upsampler = _make_upsampler(model_path)
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)  # pylint: disable=no-member
    result, _ = upsampler.enhance(img, outscale=SCALE)
    return result


def _process_and_save_tile(
    vips_tile, row: int, col: int, upsampler: RealESRGANer, out_dir: str
) -> None:
    """Enhance one vips tile with ESRGAN and save it to disk using NAMING_FORMAT."""
    rgb = _pyvips_to_numpy_rgb(vips_tile)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member
    up_bgr, _ = upsampler.enhance(bgr, outscale=SCALE)
    tile_path = f"{out_dir}/{NAMING_FORMAT.format(row=row, col=col)}"
    cv2.imwrite(tile_path, up_bgr)  # pylint: disable=no-member

def _generate_and_save_tiles(
    image_path: str,
    tile_size: tuple[int, int],
    model_path: str,
    out_dir: str,
    grid: tuple[int, int],
) -> None:
    """Generate tiles in-memory, ESRGAN each, and save to out_dir using NAMING_FORMAT."""
    slicer = ImageSlicer(image_path)
    upsampler = _make_upsampler(model_path)
    rows, cols = grid
    tw, th = tile_size

    for vips_tile, r, c in tqdm(
        slicer.generate_tiles(tile_width=tw, tile_height=th),
        desc="Upscaling tiles",
        total=rows * cols,
    ):
        _process_and_save_tile(vips_tile, r, c, upsampler, out_dir)



def upscale_slice(model_path: str, image_path: str, n_tiles: int) -> np.ndarray:
    """
    Slice ~n_tiles via ImageSlicer.generate_tiles(), ESRGAN each tile, join with join_image,
    and return OpenCV BGR ndarray.
    """
    # Size & grid packed to minimize locals
    with Image.open(image_path) as im:
        size = im.size  # (width, height)

    grid = _best_grid(n_tiles)  # (rows, cols)
    tile_size = (ceil(size[0] / grid[1]), ceil(size[1] / grid[0]))  # (tile_w, tile_h)

    with TemporaryDirectory(prefix="imgslice_up_") as tmpdir:
        _generate_and_save_tiles(image_path, tile_size, model_path, tmpdir, grid)

        stitched_path = f"{tmpdir}/stitched.png"
        join_image(tmpdir, stitched_path, naming_format=NAMING_FORMAT)
        stitched_pil = Image.open(stitched_path).convert("RGB")
        return convert_from_image_to_cv2(stitched_pil)



def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, help="REQUIRED: path to the ESRGAN model")
    parser.add_argument("-i", "--input", type=str, help="REQUIRED: path to the image to upscale")
    parser.add_argument("-o", "--output", type=str, help="REQUIRED: path to save the result")
    parser.add_argument("-v", "--visualize", action="store_true", help="OPTIONAL: preview")
    parser.add_argument(
        "-s",
        "--slice",
        nargs="?",
        type=int,
        const=4,
        help="OPTIONAL: slice into approximately this many tiles (auto grid)",
    )
    parser.add_argument(
        "-r",
        "--resize",
        nargs="?",
        type=str,
        const="1920x1080",
        help="OPTIONAL: resize final output to WIDTHxHEIGHT (e.g., 1920x1080)",
    )
    args = parser.parse_args()

    if args.model_path and args.input and args.output:
        if args.slice:
            result = upscale_slice(args.model_path, args.input, args.slice)
        else:
            result = upscale(args.model_path, args.input)

        if args.visualize:
            plt.imshow(mpimg.imread(args.input))
            plt.show()
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) # pylint: disable=no-member
            plt.show()

        if args.resize:
            size = tuple(int(i) for i in args.resize.split("x"))
            result = cv2.resize(result, size) # pylint: disable=no-member

        cv2.imwrite(args.output, result) # pylint: disable=no-member
    else:
        print("Error: Missing arguments, check -h / --help for details")


if __name__ == "__main__":
    main()
