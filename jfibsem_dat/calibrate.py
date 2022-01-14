import numpy as np


def calibrate(orig, scaled, max_samples=100):
    x = orig.ravel()
    sort_idxs = np.argsort(x)
    x = x[sort_idxs]
    y = scaled.ravel()[sort_idxs]
    # explicit default args so we can use 'method' by position
    # rather than keyword, as it changed names (from 'interpolation')
    # in numpy 1.22
    y_samples = np.quantile(
        y,
        np.linspace(0, 1, max_samples, True),
        None,
        None,
        False,
        "closest_observation",
    )
    visited_x = set()
    left = []
    right = []

    for y_sample in y_samples:
        x_sample = x[y == y_sample][0]
        if x_sample in visited_x:
            continue
        left.append(x_sample)
        right.append(y_sample)
        visited_x.add(x_sample)

    return np.array(left, orig.dtype), np.array(right, scaled.dtype)
