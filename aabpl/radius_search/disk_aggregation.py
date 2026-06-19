import numpy as np
from numpy import array as np_array, zeros as np_zeros, exp as np_exp
from numpy.linalg import norm as np_norm
from aabpl.utils.misc import flatten_list
from aabpl.utils.progress import SearchProgress
from aabpl.illustrations.plot_disk import illustrate_point_disk
from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
from aabpl.testing.test_performance import time_func_perf
from math import pi as math_pi
from .sample_area import compute_disk_cell_overlap
from aabpl import config as _cfg

# Points per block when forming the (block x candidates) distance matrix. Bounds
# the temporary matrix for very populous overlap groups; results are independent of it.
OVERLAP_BLOCK = 256


@time_func_perf
def search_and_aggregate(
    grid,
    pts_source,
    r,
    c=[],
    y='proj_lat',
    x='proj_lon',
    off_x='offset_x',
    off_y='offset_y',
    pts_target=None,
    row_name='id_y',
    col_name='id_x',
    cell_region_name='cell_region',
    sum_suffix=None,
    exclude_pt_itself=True,
    weight_valid_area=None,
    plot_pt_disk=None,
    silent=False,
    validate=False,
):
    """Aggregate point data within radius ``r`` around every source point.

    How it works
    ------------
    The search grid is keyed by packed integer cell keys (``grid.cell_codec``): a
    point's home cell is a single integer, and each region's set of relevant cells
    is a fixed integer *offset template*, so translating a template to a point is
    one vector add (``template + home_key`` -> absolute cell keys).

    Points are sorted by (cell, contain-region, overlap-region) and processed in
    groups rather than one by one:
      * the **contained-cell sums** are identical for every point in a group, so
        they are computed once and broadcast onto the whole group's slice;
      * the **overlap-cell sums** differ per point but share one candidate set, so
        a whole group is filtered against the radius as one (group x candidates)
        distance matrix.

    The optional example-disk plot is reconstructed once, after the loop, so it
    never touches the hot path.
    """
    
    if pts_target is None:
        pts_target = pts_source

    codec = grid.cell_codec
    sums_by_lvl = grid.id_to_sums_by_lvl
    vals_xy_by_lvl = grid.id_to_vals_xy_by_lvl
    nonempty_cell_keys = set(sums_by_lvl)
    grid_spacing = grid._search_spacing
    contain_region_mult = grid.search.contain_region_mult

    shared_cntd_cells = grid.search.shared_cntd_cells
    cntd_cells_by_region = grid.search.region_and_trgl_id_to_distinct_cntd_cells
    ovlpd_cells_by_region = grid.search.region_and_trgl_id_to_distinct_ovlpd_cells
    # level-0 (row,col) templates, used only for the valid-area invalid-cell test
    cntd_cells_by_cell_region = grid.search.region_id_to_cntd_cells
    ovlpd_cells_by_cell_region = grid.search.region_id_to_ovlpd_cells
    get_cell_centroid = grid.get_cell_centroid

    n_pts = len(pts_source)
    n_cols = len(c)
    r2 = r * r
    full_disk_area = math_pi * r2
    zero_sum = np_zeros(n_cols, dtype=int) if n_cols > 1 else 0

    # ---- edge weighting: which level-0 cells are NOT in the sampling grid ------
    cells_rndm_sample = grid.cells_rndm_sample
    if isinstance(cells_rndm_sample, bool) and cells_rndm_sample:
        weight_valid_area = False  # every cell is sampled -> the whole disk is valid
    if weight_valid_area not in ('precise', 'estimate', False, None):
        print("Value for 'weight_valid_area' must be in ['precise', 'estimate', False]. "
              f"Instead {weight_valid_area!r} was provided.")
        weight_valid_area = False

    # ---- integer offset templates (template + home_key == absolute cell keys) --
    shared_cntd_offset = codec.offset_int(shared_cntd_cells)
    cntd_offset_by_region = {rid: codec.offset_int(cells) for rid, cells in cntd_cells_by_region.items()}
    ovlpd_offset_by_region = {rid: codec.offset_int(cells) for rid, cells in ovlpd_cells_by_region.items()}

    if weight_valid_area:
        pad = -int(-grid_spacing // r)
        invalid_cells = set(
            (int(row_id), int(col_id))
            for row_id in range(min(grid._search_row_ids) - pad, max(grid._search_row_ids) + pad)
            for col_id in range(min(grid._search_col_ids) - pad, max(grid._search_col_ids) + pad)
            if (int(row_id), int(col_id)) not in cells_rndm_sample
        )
        invalid_keys = set(int(codec.key(0, rr, cc)) for rr, cc in invalid_cells)
        # invalid-cell membership is a level-0 (row,col) test, so force the level to 0
        cntd_l0_offset = {rid: codec.offset_int([(0, dc) for lvl, dc in cells])
                          for rid, cells in cntd_cells_by_cell_region.items()}
        ovlpd_l0_offset = {rid: codec.offset_int([(0, dc) for lvl, dc in cells])
                           for rid, cells in ovlpd_cells_by_cell_region.items()}

    # ---- helper paths --------------------------------------------------------
    # Set to True to run the optimized workflow

    if _cfg.USE_OPTIMIZED_METHOD:
        # ---- Build NumPy Sparse-to-Dense Hash Structures -------------------------
        unique_keys = np_array(sorted(sums_by_lvl.keys()), dtype=int)
        n_cells = len(unique_keys)

        hash_table_size = int(2 ** np.ceil(np.log2(n_cells * 2))) if n_cells > 0 else 1
        hash_mask = hash_table_size - 1

        hash_keys = np.full(hash_table_size, -1, dtype=int)
        hash_dense_indices = np.full(hash_table_size, -1, dtype=int)

        for dense_idx, k in enumerate(unique_keys):
            h = int(k) & hash_mask
            while hash_keys[h] != -1:
                h = (h + 1) & hash_mask
            hash_keys[h] = k
            hash_dense_indices[h] = dense_idx

        first_sum_val = next(iter(sums_by_lvl.values())) if sums_by_lvl else 0
        if isinstance(first_sum_val, np.ndarray):
            sum_shape = (n_cells, *first_sum_val.shape)
            sum_dtype = first_sum_val.dtype
        else:
            sum_shape = (n_cells,)
            sum_dtype = type(first_sum_val)

        fast_sums = np.zeros(sum_shape, dtype=sum_dtype)
        fast_vals_pointers = np.zeros((n_cells, 2), dtype=int)
        flat_vals_list = []

        current_idx = 0
        for dense_idx, k in enumerate(unique_keys):
            fast_sums[dense_idx] = sums_by_lvl[k]
            arr = vals_xy_by_lvl[k]
            n_pts_in_cell = arr.shape[0] if hasattr(arr, 'shape') else len(arr)
            flat_vals_list.append(arr)
            fast_vals_pointers[dense_idx] = [current_idx, current_idx + n_pts_in_cell]
            current_idx += n_pts_in_cell

        fast_vals_buffer = np.vstack(flat_vals_list) if flat_vals_list else np.zeros((0, n_cols + 2))

        def get_dense_indices(keys):
            indices = np.full(len(keys), -1, dtype=int)
            for i, k in enumerate(keys):
                h = int(k) & hash_mask
                while True:
                    hk = hash_keys[h]
                    if hk == k:
                        indices[i] = hash_dense_indices[h]
                        break
                    if hk == -1:
                        break
                    h = (h + 1) & hash_mask
            return indices

        def covered_cell_dense_indices(offset_template, home_key):
            abs_keys = offset_template + home_key
            dense_idxs = get_dense_indices(abs_keys)
            return dense_idxs[dense_idxs != -1]

        def covered_cell_keys(offset_template, home_key):
            """Non-empty grid keys covered by ``offset_template`` placed at ``home_key``."""
            abs_keys = offset_template + home_key
            return [int(k) for k in abs_keys if k in nonempty_cell_keys]

        def sum_over_cells(cell_keys):
            if not cell_keys:
                return zero_sum
            dense_idxs = get_dense_indices(np_array(cell_keys, dtype=int))
            dense_idxs = dense_idxs[dense_idxs != -1]
            if len(dense_idxs) == 0:
                return zero_sum
            if n_cols > 1:
                return fast_sums[dense_idxs].sum(axis=0)
            return fast_sums[dense_idxs].sum()

        max_cells_per_region = max(len(cells) for cells in ovlpd_cells_by_cell_region.values())
        max_candidates = sum(sorted(len(v) for v in grid.id_to_vals_xy.values())[-max_cells_per_region:])
        candidate_buffer = np_zeros((max_candidates, n_cols + 2), dtype=float)

        def gather_overlap_candidates(home_key, region_id):
            n = 0
            dense_idxs = covered_cell_dense_indices(ovlpd_offset_by_region[region_id], home_key)
            for idx in dense_idxs:
                start, end = fast_vals_pointers[idx]
                length = end - start
                if length > 0:
                    candidate_buffer[n:n + length] = fast_vals_buffer[start:end]
                    n += length
            return candidate_buffer[:n]

    else:
        # ORIGINAL BASELINE WORKFLOW
        def covered_cell_keys(offset_template, home_key):
            """Non-empty grid keys covered by ``offset_template`` placed at ``home_key``."""
            abs_keys = offset_template + home_key
            return [int(k) for k in abs_keys if k in nonempty_cell_keys]

        def sum_over_cells(cell_keys):
            if n_cols > 1:
                return np_array([sums_by_lvl[k] for k in cell_keys]).sum(axis=0) if cell_keys else zero_sum
            return sum(sums_by_lvl[k] for k in cell_keys)

        # reusable buffer to gather a region's overlap-candidate rows ([vals..., x, y])
        max_cells_per_region = max(len(cells) for cells in ovlpd_cells_by_cell_region.values())
        max_candidates = sum(sorted(len(v) for v in grid.id_to_vals_xy.values())[-max_cells_per_region:])
        candidate_buffer = np_zeros((max_candidates, n_cols + 2), dtype=float)

        def gather_overlap_candidates(home_key, region_id):
            n = 0
            for k in covered_cell_keys(ovlpd_offset_by_region[region_id], home_key):
                rows_xy = vals_xy_by_lvl[k]
                candidate_buffer[n:n + len(rows_xy)] = rows_xy
                n += len(rows_xy)
            return candidate_buffer[:n]

    # ---- valid-area term (per point; only built when weighting) ----------------
    if weight_valid_area:
        def invalid_cntd_area(cell_region_id, home_key):
            abs_keys = cntd_l0_offset[cell_region_id] + home_key
            n_invalid = len(set(int(k) for k in abs_keys) & invalid_keys)
            return n_invalid * grid_spacing ** 2

        def invalid_overlap_cells(cell_region_id, home_key):
            abs_keys = ovlpd_l0_offset[cell_region_id] + home_key
            cells = []
            for k in (set(int(k) for k in abs_keys) & invalid_keys):
                _, (rr, cc) = codec.decode_tuple(k)
                cells.append((int(rr), int(cc)))
            return cells

        if weight_valid_area == 'precise':
            if r2 < 2 * grid_spacing ** 2:
                print("WARNING: the precise valid-area method assumes r >= sqrt(2)*grid_spacing; "
                      "for smaller radii the valid area may be inaccurate.")

            def overlap_invalid_area(point_offset, point_xy, point_row, point_col, cells):
                return sum(compute_disk_cell_overlap(
                    point_offset,
                    row_col=(int(rr - point_row), int(cc - point_col)),
                    grid_spacing=grid_spacing, r=r, silent=True,
                ) for rr, cc in cells)
        else:  # 'estimate' — logit fit of overlap-area share vs. centroid distance
            logit_Q = 1 / (0.70628102 + np_exp(0.57266908 * (grid_spacing / r - 2)))
            logit_B = 1 / (-0.21443453 + np_exp(0.76899004 * (grid_spacing / r - 2)))

            def overlap_invalid_area(point_offset, point_xy, point_row, point_col, cells):
                if not cells:
                    return 0.0
                centroids = np_array([get_cell_centroid(int(rr), int(cc)) for rr, cc in cells])
                share = 1 - 1 / (1.0 + logit_Q * np_exp(-logit_B * (np_norm(point_xy - centroids, axis=1) / r - 1)))
                return share.sum() * grid_spacing ** 2

    # ---- sort points and pull the columns we loop over into arrays -------------
    if sum_suffix is None:
        sum_suffix = '_' + str(r)
    sum_radius_names = [cname + sum_suffix for cname in c]
    pts_source[sum_radius_names] = 0
    column_dtypes = pts_target[c].dtypes

    pts_source.sort_values([row_name, col_name, 'region_and_trgl_id'], inplace=True)
    point_xy = pts_source[[x, y]].values
    point_offset = pts_source[[off_x, off_y]].values
    rows = pts_source[row_name].values.astype(int)
    cols = pts_source[col_name].values.astype(int)
    cell_region = pts_source[cell_region_name].values
    region_and_trgl = pts_source['region_and_trgl_id'].values

    # one integer per point: its packed level-0 home-cell key (== codec.home(row,col))
    home_key = (rows * codec.scale - codec._rlo) * codec.row_stride + (cols * codec.scale - codec._clo)

    # group boundaries: a new group starts wherever the relevant key changes.
    # contain/overlap groups both nest under cell groups but cross-cut each other.
    cell_changed = np.ones(n_pts, dtype=bool)
    contain_changed = np.ones(n_pts, dtype=bool)
    overlap_changed = np.ones(n_pts, dtype=bool)
    if n_pts > 1:
        cell_changed[1:] = home_key[1:] != home_key[:-1]
        rt_changed = region_and_trgl[1:] != region_and_trgl[:-1]
        contain_changed[1:] = (cell_changed[1:]
                               | (cell_region[1:] // contain_region_mult != cell_region[:-1] // contain_region_mult)
                               | rt_changed)
        overlap_changed[1:] = (cell_changed[1:]
                               | (cell_region[1:] % contain_region_mult != cell_region[:-1] % contain_region_mult)
                               | rt_changed)

    sums_within_disks = np_zeros((n_pts, n_cols))
    invalid_area = np_zeros(n_pts) if weight_valid_area else None
    progress = SearchProgress(silent=silent, n_pts=n_pts)
    progress.start()
    next_threshold = progress.next_threshold

    # ---- contained sums: constant within a contain group, broadcast to its slice
    contain_starts = np.flatnonzero(contain_changed)
    contain_ends = np.append(contain_starts[1:], n_pts)
    cell_sum = zero_sum
    for start, end in zip(contain_starts, contain_ends):
        hk = int(home_key[start])
        if cell_changed[start]:
            cell_sum = sum_over_cells(covered_cell_keys(shared_cntd_offset, hk))
        sums_within_disks[start:end] += cell_sum + sum_over_cells(covered_cell_keys(cntd_offset_by_region[region_and_trgl[start]], hk))
        if weight_valid_area:
            invalid_area[start:end] += invalid_cntd_area(cell_region[start], hk)

    # ---- overlap sums: one (group x candidates) distance matrix per overlap group
    overlap_starts = np.flatnonzero(overlap_changed)
    overlap_ends = np.append(overlap_starts[1:], n_pts)
    for start, end in zip(overlap_starts, overlap_ends):
        hk = int(home_key[start])
        candidates = gather_overlap_candidates(hk, region_and_trgl[start])
        if len(candidates):
            candidate_xy = candidates[:, -2:]
            candidate_vals = candidates[:, :-2].astype(float)
            for block_start in range(start, end, OVERLAP_BLOCK):
                block_end = min(block_start + OVERLAP_BLOCK, end)
                block_xy = point_xy[block_start:block_end]
                dx = block_xy[:, 0][:, None] - candidate_xy[None, :, 0]
                dy = block_xy[:, 1][:, None] - candidate_xy[None, :, 1]
                inside = (dx * dx + dy * dy) <= r2
                sums_within_disks[block_start:block_end] += inside.astype(float) @ candidate_vals
        if weight_valid_area:
            bad_cells = invalid_overlap_cells(cell_region[start], hk)
            if bad_cells:
                for i in range(start, end):
                    invalid_area[i] += overlap_invalid_area(
                        point_offset[i], point_xy[i], rows[i], cols[i], bad_cells)
        if end - 1 >= next_threshold:
            next_threshold = progress.update(end - 1)
    progress.done()

    if weight_valid_area:
        valid_area_shares = (full_disk_area - invalid_area) / full_disk_area

    # ---- example-disk plot: reconstruct the chosen point once, off the hot path -
    if plot_pt_disk is not None:
        if 'pt_id' not in plot_pt_disk:
            plot_pt_disk['pt_id'] = sorted(
                (len(pt_ids), pt_ids[0] if len(pt_ids) > 0 else None)
                for pt_ids in grid.id_to_pt_ids_by_lvl.values())[-1][1]
        target_id = plot_pt_disk['pt_id']
        if target_id in pts_source.index:
            pos = pts_source.index.get_loc(target_id)
            hk = int(home_key[pos])
            region_id = region_and_trgl[pos]
            candidates = gather_overlap_candidates(hk, region_id)
            candidate_xy = candidates[:, -2:]
            dist = np_norm(candidate_xy - point_xy[pos], axis=1)
            # decode integer keys back to absolute (lvl, (row, col)) tuples for the figure
            decode = codec.decode_tuple
            shared_abs = [decode(int(k)) for k in (shared_cntd_offset + hk)]
            cntd_abs = [decode(int(k)) for k in (cntd_offset_by_region[region_id] + hk)]
            ovlpd_abs = [decode(int(k)) for k in (ovlpd_offset_by_region[region_id] + hk)]
            cntd_keys = covered_cell_keys(
                np.concatenate([shared_cntd_offset, cntd_offset_by_region[region_id]]), hk)
            pts_xy_in_cntd = (np_array(flatten_list([vals_xy_by_lvl[k] for k in cntd_keys]))[:, -2:]
                              if cntd_keys else np_zeros((0, 2)))
            # The offset-region overlay needs the point's offset-region id, which is
            # keyed differently from cell_region; pass it only when it is a real key,
            # otherwise let illustrate_point_disk skip that one overlay rather than
            # crash the search. Everything else in the figure is independent of it.
            region_id_for_plot = (int(cell_region[pos])
                                  if int(cell_region[pos]) in grid.id_to_offset_regions else None)
            try:
                illustrate_point_disk(
                    grid=grid, pts_source=pts_source, pts_target=pts_target, r=r, c=c, x=x, y=y,
                    shared_cntd_cells=shared_abs, shared_ovlpd_cells=[],
                    distinct_cntd_cells=cntd_abs, distinct_ovlpd_cells=ovlpd_abs,
                    pts_xy_in_cell_cntd_by_pt_region=pts_xy_in_cntd,
                    pts_xy_in_cells_ovlpd_by_pt_region=candidate_xy[dist > r],
                    pts_xy_in_radius=candidate_xy[dist <= r],
                    sums_within_disk=sums_within_disks[pos, :],
                    sum_names=sum_radius_names,
                    cell_region_id=cell_region[pos],
                    home_cell=(int(rows[pos]), int(cols[pos])),
                    region_id=region_id_for_plot,
                    **plot_pt_disk,
                )
            except Exception as plot_error:
                print(f"plot_pt_disk skipped (pt_id={target_id}): {type(plot_error).__name__}: {plot_error}")

    # ---- write results back, fix dtypes, exclude self, apply edge weighting -----
    pts_source[sum_radius_names] = pts_source[sum_radius_names].values + sums_within_disks
    pts_source = pts_source.astype({name: dt for name, dt in zip(sum_radius_names, column_dtypes)})

    if exclude_pt_itself and grid.search.tgt_df_contains_src_df:
        for sum_name, value_col in zip(sum_radius_names, c):
            pts_source[sum_name] = pts_source[sum_name].values - pts_source[value_col]

    if weight_valid_area:
        share_name = 'valid_area_share' + sum_suffix
        pts_source[share_name] = valid_area_shares
        for sum_name in sum_radius_names:
            pts_source[sum_name] = pts_source[sum_name].values / pts_source[share_name].values
        if not silent:
            print("Appended radius sum" + ("" if n_cols <= 1 else "s") + " (r=" + str(r) + ") for "
                  + ', '.join(f"'{cname}' as '{sname}'" for cname, sname in zip(c, sum_radius_names))
                  + " to pts DataFrame. (Sum names can be controlled by setting sum_suffix='...')")
            print("Appended valid area share as '" + share_name + "' to pts DataFrame.")

    if not validate and not _cfg.VALIDATE:
        return pts_source[sum_radius_names]

    # ---- brute-force validation (O(n^2)) on one representative point per region -
    id_to_pt_ids = grid.id_to_pt_ids
    all_xy = pts_target[[x, y]].values
    all_vals = pts_target[c].values if n_cols > 1 else pts_target[c[0]].values.reshape(-1, 1)
    lvl0_cells = {k: v for k, v in id_to_pt_ids.items() if k[0] == 0}
    cell_pop = {k: len(v) for k, v in lvl0_cells.items() if len(v) > 0}
    pts_source['_cell_pop'] = pts_source.apply(
        lambda row: cell_pop.get((0, (int(row[row_name]), int(row[col_name]))), 0), axis=1)
    rep_indices = (pts_source.sort_values('_cell_pop', ascending=False)
                   .groupby(cell_region_name, sort=False).apply(lambda g: g.index[0]))
    pts_source.drop(columns=['_cell_pop'], inplace=True)
    errors = []
    for cr, rep_idx in rep_indices.items():
        rep_xy = pts_source.loc[rep_idx, [x, y]].values.astype(float)
        dists = np_norm(all_xy - rep_xy, axis=1)
        brute_sums = all_vals[dists <= r].sum(axis=0)
        
        # Initialize own_vals as 0 or None in case exclude_pt_itself is False
        own_vals = 0.0 
        
        if exclude_pt_itself and grid.search.tgt_df_contains_src_df:
            if n_cols > 1:
                own_vals = pts_target.loc[rep_idx, c].values.astype(float)
            else:
                # Use c[0] and explicitly wrap it as a flat numpy array
                val_scalar = float(pts_target.loc[rep_idx, c[0]])
                own_vals = np.array([val_scalar])
            
            # Ensure brute_sums and own_vals are flat shapes matching axis format
            brute_sums = brute_sums.flatten() - own_vals.flatten()

        algo_sums = pts_source.loc[rep_idx, sum_radius_names].values.astype(float)
        diff = (brute_sums - algo_sums).max()
        
        if diff != 0:
            # FIX: Append own_vals directly into the tuple so it stays tied to this specific rep_idx
            errors.append((rep_idx, cr, brute_sums, algo_sums, diff, own_vals))
            
    if errors:
        print(f"VALIDATION FAILED: {len(errors)}/{len(rep_indices)} cell_region(s) have wrong sums:")
        # FIX: Unpack own_vals from the error tuple here
        for rep_idx, cr, bf, algo, diff, item_own_vals in errors:
            print(f"  pt_id={rep_idx} cell_region={cr} brute={bf} algo={algo} diff={diff} own_vals={item_own_vals}")
    else:
        print(f"VALIDATION OK: all {len(rep_indices)} cell_region(s) correct.")


    def plot_vars(self=grid, colnames=np_array([c, sum_radius_names]), filename='', **plot_kwargs):
        return create_plots_for_vars(grid=self, colnames=colnames, filename=filename, plot_kwargs=plot_kwargs)
    grid.plot.vars = plot_vars

    return pts_source[sum_radius_names]
