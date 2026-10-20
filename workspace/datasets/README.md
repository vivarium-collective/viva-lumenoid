# Tyssue meshes (vendored)

`lumenoid.hf5` is a small closed-monolayer tyssue
epithelium meshes used as the `eptm` for the Tyssue composites (E3 / R5 / 2D
cross-section). They are vendored from vivarium-tyssue (PR #31,
`scripts/gen_geometry_datasets.py`) because tyssue's mesh *generation* needs a
compiled submodule (`tyssue._mesh_generation`) not in the pip wheel; *loading*
and running them does not. Regenerate upstream if the geometry changes.
