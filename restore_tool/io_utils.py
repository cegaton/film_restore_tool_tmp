import os
import OpenImageIO as oiio
import numpy as np

DEFAULT_DIR_FILE = os.path.expanduser("~/.default_dir")

def load_default_dir():
    if os.path.exists(DEFAULT_DIR_FILE):
        p = open(DEFAULT_DIR_FILE).read().strip()
        if os.path.isdir(p):
            return p
    return os.path.expanduser("~")

def save_default_dir(path):
    with open(DEFAULT_DIR_FILE, "w") as f:
        f.write(path)

def read_exr(path):
    print("Reading:", path)
    inp = oiio.ImageInput.open(path)
    if not inp:
        return None
    img = inp.read_image(format=oiio.FLOAT)
    inp.close()
    return img[:, :, :3].astype(np.float32)

def write_exr(path, img):
    spec = oiio.ImageSpec(img.shape[1], img.shape[0], 3, oiio.HALF)
    out = oiio.ImageOutput.create(path)
    out.open(path, spec)
    out.write_image(img.astype(np.float16))
    out.close()
