"""
Exploration: adaptive nest_depth and dual-spacing (dense/sparse) per chunk.

Runs at multiple spacing_ratios (sr = r/spacing) including sr<1 (spacing>r,
coarse sparse path) to understand cntd/ovlpd cell counts and cost model.

The dual-spacing idea:
  dense chunks  -> standard sr (e.g. 2.0)
  sparse chunks -> coarse sr = sr/2^k (e.g. 0.5 or 0.25, spacing > r)
                   disk touches only 1-4 coarse cells -> just area-fraction lookups

Run:
    python aabpl/testing/explore_adaptive_nd.py [sr1 sr2 ...]
    e.g.: python aabpl/testing/explore_adaptive_nd.py 0.5 1.0 1.5 2.0 3.0
"""
import sys, time
sys.path.insert(0, 'Z:/Algorithm/PL_python/AABPL-toolkit-python')
[sys.modules.pop(m) for m in list(sys.modules.keys()) if m.startswith('aabpl')]

import numpy as np
from aabpl.search.algorithm.disk_geometry import (
    build_disk_region_lookups, downgrade_disk_region_cache_entry
)
from aabpl.search.spacing_topology import estimate_template_cell_counts

R      = 500.0
SR_LIST = [float(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [0.5, 1.0, 1.5, 2.0, 3.0]
MAX_ND  = 4

C_c = 50e-9    # 50ns per contained cell lookup
C_o = 5e-6     # 5us per overlapped cell (Shapely intersection)

def analyse_sr(sr):
    spacing = R / sr
    print(f"\n{'='*80}")
    print(f"sr={sr:.2f}  spacing={spacing:.1f}m  (r={'%.2f'%sr}*spacing)")
    print(f"{'='*80}")

    t0 = time.perf_counter()
    entry_top = build_disk_region_lookups(
        None, grid_spacing=spacing, r=R, nest_depth=MAX_ND, silent=True
    )
    elapsed = time.perf_counter() - t0
    print(f"Built nd={MAX_ND} in {elapsed:.2f}s")

    entries = {MAX_ND: entry_top}
    for nd in range(MAX_ND - 1, -1, -1):
        entries[nd] = downgrade_disk_region_cache_entry(entries[nd + 1], nd + 1)

    print(f"\n{'nd':>4}  {'n_reg':>7}  {'shared':>7}  {'avg_cntd':>9}  "
          f"{'avg_ovlpd':>10}  {'cntd_frac':>10}  "
          f"{'cost_K1_us':>11}  {'cost_K100_us':>13}")
    print('-' * 82)

    rows = []
    for nd in range(MAX_ND + 1):
        e = entries[nd]
        rids = list(e['region_id_to_cntd_cells'].keys())
        n_reg = len(rids)
        shared_n = len(e['shared_cntd_cells'])

        cntd_counts  = [len(e['region_id_to_cntd_cells'][rid])  for rid in rids]
        ovlpd_counts = [len(e['region_id_to_ovlpd_cells'][rid]) for rid in rids]

        avg_cntd  = float(np.mean(cntd_counts))  if cntd_counts  else 0.0
        avg_ovlpd = float(np.mean(ovlpd_counts)) if ovlpd_counts else 0.0
        total_avg = avg_cntd + avg_ovlpd + shared_n
        cntd_frac = (avg_cntd + shared_n) / max(total_avg, 1e-9)

        cc = (avg_cntd + shared_n) * C_c * 1e6
        oc1   = avg_ovlpd * C_o * 1e6          # K=1
        oc100 = avg_ovlpd * C_o * 1e6 / 100    # K=100
        rows.append(dict(nd=nd, n_reg=n_reg, shared=shared_n,
                         avg_cntd=avg_cntd, avg_ovlpd=avg_ovlpd,
                         cntd_frac=cntd_frac, cc=cc, oc1=oc1, oc100=oc100))

        print(f"{nd:>4}  {n_reg:>7}  {shared_n:>7}  {avg_cntd:>9.1f}  "
              f"{avg_ovlpd:>10.1f}  {cntd_frac:>10.3f}  "
              f"{cc+oc1:>11.2f}  {cc+oc100:>13.2f}")
        sys.stdout.flush()

    # Breakeven table (nd->nd+1)
    print(f"\n  nd0->nd1   dcntd  dovlpd  breakeven_K")
    for i in range(len(rows)-1):
        r0, r1 = rows[i], rows[i+1]
        dc = r1['avg_cntd'] - r0['avg_cntd']
        do = r0['avg_ovlpd'] - r1['avg_ovlpd']
        if dc > 0 and do > 0:
            K = do * C_o / (dc * C_c)
            print(f"  {r0['nd']}->{r1['nd']}       {dc:>6.1f}  {do:>6.1f}  {K:>11.0f}")
        else:
            print(f"  {r0['nd']}->{r1['nd']}       {dc:>6.1f}  {do:>6.1f}  n/a")

    sys.stdout.flush()

# Summary comparison across all sr values (nd=0 only)
print(f"r={R}m  MAX_ND={MAX_ND}")
print(f"Comparing sr values at nd=0:")
print(f"{'sr':>6}  {'spacing_m':>10}  {'n_reg':>7}  {'shared':>7}  "
      f"{'avg_cntd':>9}  {'avg_ovlpd':>10}  {'cost_K1_us':>11}  {'cost_K100_us':>13}")
print('-' * 85)

summary_rows = []
for sr in SR_LIST:
    spacing = R / sr
    entry = build_disk_region_lookups(None, grid_spacing=spacing, r=R, nest_depth=0, silent=True)
    rids = list(entry['region_id_to_cntd_cells'].keys())
    shared_n = len(entry['shared_cntd_cells'])
    cntd_counts  = [len(entry['region_id_to_cntd_cells'][rid])  for rid in rids]
    ovlpd_counts = [len(entry['region_id_to_ovlpd_cells'][rid]) for rid in rids]
    avg_cntd  = float(np.mean(cntd_counts))  if cntd_counts  else 0.0
    avg_ovlpd = float(np.mean(ovlpd_counts)) if ovlpd_counts else 0.0
    cc   = (avg_cntd + shared_n) * C_c * 1e6
    oc1  = avg_ovlpd * C_o * 1e6
    oc100 = oc1 / 100
    summary_rows.append((sr, spacing, len(rids), shared_n, avg_cntd, avg_ovlpd, cc+oc1, cc+oc100))
    print(f"{sr:>6.2f}  {spacing:>10.1f}  {len(rids):>7}  {shared_n:>7}  "
          f"{avg_cntd:>9.1f}  {avg_ovlpd:>10.1f}  {cc+oc1:>11.2f}  {cc+oc100:>13.2f}")
    sys.stdout.flush()

# Detailed per-sr analysis
for sr in SR_LIST:
    analyse_sr(sr)

print("\nDone.")
