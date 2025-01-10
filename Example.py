
### Examples
#### Example 1:
path_to_your_csv = '../../cbsa_sample_data/plants_10180.txt'
crs_of_your_csv =  "EPSG:4326"

pts_df = read_csv(path_to_your_csv, sep=",", header=None)#[200:20000]
pts_df.columns = ["eid", "employment", "industry", "lat","lon","moved"]
convert_coords_to_local_crs(pts_df)

detect_clusters(
    pts_df=pts_df,
    radius=750,
    include_boundary=False,
    exclude_pt_itself=True,

    k_th_percentiles=[99.5],
    n_random_points=int(1e6),
    random_seed=0,

    sum_names=['employment'],
    plot_distribution={},
    plot_cluster_points={},
    silent = True,
)