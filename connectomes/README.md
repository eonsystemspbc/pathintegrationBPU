# connectomes/ — prepared connectome substrates (git-ignored)

The connectome **inputs** used as `--matrix` across the scripts. Each dir holds an
`adjacency_unsigned.npz` (the recurrent wiring matrix) plus any region-native structure/training
artifacts. **Not tracked in git** (large; regenerable via `scripts/connectome/`). Moved here from
`outputs/` so inputs are no longer confused with results.

| dir | region / role | N (neurons) |
|---|---|---|
| `flywire_optic_lobe_bpu/` | optic lobe (vision / optic flow) | 96,816 |
| `flywire_mushroom_body/` | mushroom body (associative memory) | 14,025 |
| `cx_polar_bump_seed0/` | central complex (heading / path integration) | 7,349 |
| `cx_structure_polar_{frozen,observed}/` | CX structure-run variants | 7,349 |
| `hemibrain_mushroom_body_plume/` | hemibrain MB (plume task) | — |
| `ol_sub14025_s1/`, `ol_sub14025_s2/` | optic lobe subsampled to MB size (capacity control) | 14,025 |
| `larva_bpu/`, `flywire_whole_bump_seed0/` | additional substrates | — |

Used as e.g. `--matrix connectomes/flywire_mushroom_body/adjacency_unsigned.npz`.
