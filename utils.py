import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageFilter
from PIL.ImageOps import invert, autocontrast


def expand2square(pil_img):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), list(pil_img.getdata())[1])
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), list(pil_img.getdata())[1])
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


def draw_grids(ax, **kwargs):
    ax.set_zorder(0)
    plt.grid(which='both', axis='both', **kwargs)


def maskgen(fname, shape=(256, 256)):
    
    msk = Image.open(f"{fname}.png")
    msk = expand2square(msk).resize(shape, resample=Image.LANCZOS)
    os.remove(f"{fname}.png")
    msk.save(f"{fname}.tiff")
    
def postprocessing(fname, shape=(256, 256),gray=False):
    img = Image.open(fname)
    img = expand2square(img).resize(shape, resample=Image.LANCZOS)
    if gray:
        img = img.convert('L')
    img.save(fname)
