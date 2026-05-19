import numpy as np

def detect_dust(frames, center, thresh):
    diffs = [np.mean(np.abs(center - f), 2) for f in frames]
    diffs = np.stack(diffs)

    votes = np.sum(diffs > thresh, axis=0)
    return (votes >= 2).astype(np.float32)
