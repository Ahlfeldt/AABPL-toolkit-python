import numpy as np


def batched_disk_sum(pxy, cand_xy, cand_vals, r2, max_bytes=None):
    """sum(cand_vals[j] for j where |pxy[i]-cand_xy[j]| <= r), per row i.

    Batches over pxy rows so the dense (batch_rows, n_cand) dx/dy/inside
    buffers stay under max_bytes -- avoids the multi-GB MemoryError crashes
    seen when a chunk/candidate group ends up far larger than intended
    (extreme forced-nd/ppc combos where the domain collapses relative to the
    search radius, or sparse real data forcing a coarse supercell nd).
    Assumes ~2 live float64 temporaries per (row, candidate) pair during
    the squared-distance computation (dx is squared and accumulated into
    in-place, so only dx and dy ever coexist as full buffers; the boolean
    `inside` matrix is 1 byte/pair and matmul against it auto-promotes to
    float without a separate .astype() copy).
    """
    from aabpl import config as _cfg
    if max_bytes is None:
        max_bytes = getattr(_cfg, 'MAX_DIST_MATRIX_BYTES', 1_000_000_000)

    n_p, n_c = len(pxy), len(cand_xy)
    out_shape = (n_p,) + cand_vals.shape[1:]
    out = np.zeros(out_shape, dtype=float)
    if n_p == 0 or n_c == 0:
        return out

    bytes_per_pair = 8 * 2
    max_pairs = max(1, int(max_bytes // bytes_per_pair))
    batch_rows = max(1, min(n_p, max_pairs // n_c))

    for start in range(0, n_p, batch_rows):
        end = min(start + batch_rows, n_p)
        sub = pxy[start:end]
        dx = sub[:, 0][:, None] - cand_xy[None, :, 0]
        dy = sub[:, 1][:, None] - cand_xy[None, :, 1]
        dx *= dx
        dy *= dy
        dx += dy          # dx now holds squared distance
        inside = dx <= r2  # bool, 1 byte/pair
        out[start:end] = inside @ cand_vals  # bool matmul auto-promotes, no .astype copy
    return out
