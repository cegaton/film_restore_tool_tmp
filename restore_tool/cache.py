from collections import OrderedDict
from .io_utils import read_exr

class FrameCache:
    def __init__(self, size=50):
        self.cache = OrderedDict()
        self.size = size

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        img = read_exr(path)
        if img is None:
            return None

        self.cache[path] = img

        if len(self.cache) > self.size:
            self.cache.popitem(last=False)

        return img
