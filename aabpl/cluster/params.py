"""
Parameter-unpacking helpers for detect_cluster_pts / detect_cluster_cells.

Each function converts a user-facing shorthand (scalar, tuple, or list) into
the normalised internal form expected by the clustering and merging logic.
"""


def unpack_contingency(value):
    """Return (queen_contingency, rook_contingency) as ints."""
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    value = int(value)
    return value, value


def unpack_merge_dist(value):
    """
    Normalise ``merge_dist`` to a list of ``(centroid_dist, border_dist)``
    condition tuples used by ``merge_condition_distance_based_dnf``.

    Accepted forms
    --------------
    ``None``
        No distance-based merging → returns ``[]``.
    ``float`` or ``int``
        Single threshold applied to both centroid and border
        → ``[(value, value)]``.
    ``(centroid, border)``
        Single AND-condition → ``[(centroid, border)]``.
    ``[(c1, b1), (c2, b2), ...]``
        DNF: merge if ANY condition tuple is fully satisfied.
        OR between tuples; AND within each tuple.
        A ``None`` element inside a tuple disables that measure.
    """
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [(float(value), float(value))]
    if isinstance(value, (tuple, list)):
        if value and isinstance(value[0], (tuple, list)):
            return [(t[0], t[1]) for t in value]
        return [(value[0], value[1])]
    raise TypeError(
        f"merge_dist must be None, a number, a (centroid, border) tuple, "
        f"or a list of such tuples; got {type(value)}"
    )


def unpack_min_cluster_share(value):
    """Return (after_contingency, after_centroid_dist, after_convex) as floats."""
    if isinstance(value, (tuple, list)):
        return float(value[0]), float(value[1]), float(value[2])
    f = float(value)
    return f, f, f
