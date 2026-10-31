---
license: other
license_name: nyc-open-data-terms-of-use
license_link: https://www.nyc.gov/home/terms-of-use.page
pretty_name: NYC Facade Typologies (FISP buildings, clustered)
size_categories:
  - 10K<n<100K
task_categories:
  - tabular-classification
tags:
  - built-environment
  - facades
  - new-york-city
  - clustering
  - unsupervised-learning
  - iaac-maai
  - ai-methods-1
---

# NYC Facade Typologies (FISP buildings, clustered)

New York City requires every building taller than 6 stories to have its facade inspected every 5 years (Facade Inspection & Safety Program,
FISP, Local Law 11). This dataset has **one row per building: 15,763 buildings** that filed FISP reports in cycles 6 to 10 (about 2004–2026).
Each row describes **what the facade is made of** (materials and wall types), **how the building behaved over its inspections** (SAFE /
SWARMP / UNSAFE, follow-up filings, penalties), **where and when it was built** (from PLUTO), and the **facade typology** it belongs to. The
typology comes from a K-Means clustering into six groups that we named, such as *pre-war terra cotta & stone* or *repeatedly unsafe brick facades*.

- **GitHub repository (how it was made):** https://github.com/iaac-maai/group1_day3_assignment
- **Model fitted on it:** https://huggingface.co/iaac-maai/day3-team1-facade-typologies
- **Team:** Amelia, Jerome, Kazuko, Sahil

## Source and attribution

| | |
|---|---|
| **Original dataset 1** | [DOB NOW: Safety – Facades Compliance Filings](https://data.cityofnewyork.us/d/xubg-57si), NYC Department of Buildings, via NYC Open Data (downloaded 2026-10-23) |
| **Original dataset 2** | [Primary Land Use Tax Lot Output (PLUTO)](https://data.cityofnewyork.us/d/64uk-42ks), NYC Department of City Planning, via NYC Open Data (used for coordinates, year built and floors) |
| **Original licence** | NYC Open Data, free public use under the [NYC Terms of Use](https://www.nyc.gov/home/terms-of-use.page). No restriction on redistribution or derived data |

## What we changed

- **Cleaning** (our [Day 2 notebook](https://github.com/iaac-maai/group1_day2_assignment/blob/main/notebook/data_cleaning_and_EDA.ipynb)):
  87,483 raw filings → **67,659** filings (duplicates and records that are not real inspections removed, categories harmonised, materials
  and wall types turned into 0/1 columns, penalties cleaned).
- **From filings to buildings:** the filings of each building (BIN) are aggregated into one row. 12 filings without a BIN and 4 buildings
  without a block / lot were dropped → **15,763 buildings**. 22 buildings that list no material or wall type are kept with all flags at 0.
- **Feature engineering (23 new columns):** inspection history (`n_cycles`, first / last cycle and year, counts and shares of SAFE / SWARMP /
  UNSAFE reports, `ever_unsafe`, `latest_status`), repair activity (`n_followups`, `followups_per_cycle`), penalties (total, failure to correct,
  late filing, `log_penalty`), `n_materials`, `n_wall_types`, and `BBL`.
- **Join with PLUTO** on the BBL (borough × 10⁹ + block × 10⁴ + lot): `latitude`, `longitude`, `year_built`, `num_floors`. **96% of the
  buildings are matched**; the 626 others (mostly condominiums filed under a billing lot) have empty values in these four columns.
- **Clustering:** `cluster` and `cluster_name` come from the model above (K-Means, k = 6, on 13 standardised features); `cluster_ward` is a
  Ward hierarchical clustering for comparison; `pca_x/y` and `tsne_x/y` are the map coordinates.
- **Privacy:** **no street addresses, owner names or inspector names**. The source data lists only building owners and licensed professionals,
  and we dropped those columns. Buildings are identified by public DOB / tax-lot identifiers (BIN, BBL).

## The clusters

![Facade typologies on the map of New York](https://huggingface.co/datasets/iaac-maai/day3-team1-nyc-facade-typologies/resolve/main/24_map_typologies.png)

| Cluster | Name | Share of the rows | What makes it different |
|---|---|---|---|
| 0 | non-masonry or unreported facades | 6.5% (1,031) | No masonry; **49% list no material at all**. Median year 1964; 40% entered FISP after cycle 6. Partly a data-quality group |
| 1 | plain brick, low risk | 38.4% (6,054) | Brick only (1.3 materials), **lowest share UNSAFE (0.09)**, few follow-ups. ~1955, 7 floors, all boroughs |
| 2 | pre-war terra cotta & stone | 20.7% (3,263) | **94% terra cotta, 60% natural stone**. Median year **1920** (82% before 1930), **88% Manhattan** |
| 3 | post-war concrete & brick | 10.7% (1,693) | **100% concrete + masonry**, concrete / cavity walls. ~1964, 12 floors |
| 4 | repeatedly unsafe brick facades | 14.9% (2,347) | **UNSAFE in 68% of inspections**, 69% UNSAFE today, 0.69 follow-ups per cycle, median penalties **119,000 USD**. Bronx and Upper Manhattan |
| 5 | modern glass & metal curtain walls | 8.7% (1,375) | **89% glass, 92% metal**. Median year **1997**, 15 floors, 77% Manhattan, 53% new to FISP after cycle 6 |

![Cluster profiles](https://huggingface.co/datasets/iaac-maai/day3-team1-nyc-facade-typologies/resolve/main/19_cluster_profiles.png)

**How well separated are they?** Not sharply: the silhouette is **0.20**, and on the PCA map the groups overlap. They are **tendencies** in a
continuum, not islands. They are, however, **stable** (ARI 0.86 across random seeds and 0.83 across 80% subsamples). Five groups are
defined by materials, and the year built, number of floors, wall types and borough (none of which were used in the clustering) confirm them:
the terra-cotta group is pre-war Manhattan and the glass group is post-1990 towers. The sixth group (4) is defined by **behaviour**: brick
buildings like group 1, but UNSAFE again and again. A Ward clustering finds the same material groups but merges group 4 into group 1 (column
`cluster_ward`, ARI with K-Means 0.45).

![Clusters on the PCA and t-SNE maps](https://huggingface.co/datasets/iaac-maai/day3-team1-nyc-facade-typologies/resolve/main/23_clusters_pca_tsne.png)

## Columns

| Column | Type | Unit | Description |
|---|---|---|---|
| `BIN` | integer | – | DOB Building Identification Number (unique key) |
| `BBL` | integer | – | Borough-Block-Lot tax-lot identifier (key of PLUTO) |
| `BOROUGH` | text | – | MANHATTAN, BRONX, BROOKLYN, QUEENS, STATEN ISLAND |
| `BLOCK`, `LOT` | integer | – | Tax block and lot |
| `latitude`, `longitude` | float | degrees | Lot centroid from PLUTO (empty for 626 buildings) |
| `year_built` | integer | year | Year built, from PLUTO (empty if unknown) |
| `num_floors` | float | floors | Number of floors, from PLUTO |
| `n_cycles` | integer | cycles | FISP cycles with an initial report (1–5) |
| `first_cycle`, `last_cycle` | integer | – | First / last FISP cycle with an initial report (6–10) |
| `first_filing_year`, `last_filing_year` | integer | year | Year of the first / last initial report |
| `n_safe`, `n_swarmp`, `n_unsafe` | integer | reports | Initial reports with each status |
| `share_swarmp`, `share_unsafe` | float | 0–1 | Share of initial reports that were SWARMP / UNSAFE |
| `ever_unsafe` | integer | 0/1 | At least one UNSAFE report |
| `latest_status` | text | – | Status of the latest initial report: SAFE, SWARMP (safe with repair and maintenance programme) or UNSAFE |
| `n_followups` | integer | filings | Amended and subsequent filings (after repairs or an UNSAFE result) |
| `followups_per_cycle` | float | filings / cycle | `n_followups / n_cycles` |
| `total_penalty` | float | USD | Sum of all FISP penalties |
| `failure_to_correct_penalty`, `late_filing_penalty` | float | USD | Two components of the penalties |
| `log_penalty` | float | log10 USD | `log10(total_penalty + 1)` |
| `n_materials` | integer | – | Number of facade materials listed (excluding "other") |
| `n_wall_types` | integer | – | Number of wall types listed (excluding "other") |
| `MATERIAL_*` (8 columns) | integer | 0/1 | Material listed in any report: masonry, other, terra cotta, natural stone, concrete, stucco / EIFS, glass, metal |
| `WALL_TYPE_*` (8 columns) | integer | 0/1 | Wall type listed in any report: brick masonry, other, masonry load-bearing wall, terra cotta, masonry cavity wall, steel frame, concrete, concrete frame |
| `cluster` | integer | – | Cluster number from the K-Means model (0–5) |
| `cluster_name` | text | – | Name of the cluster (our interpretation, see the table above) |
| `cluster_ward` | integer | – | Cluster from Ward hierarchical clustering (k = 6, fitted on a 4,000-building sample and extended with 5-nearest neighbours); its numbers do **not** correspond to `cluster` |
| `pca_x`, `pca_y` | float | – | Position on the first two principal components of the 13 standardised features (35% of the variance) |
| `tsne_x`, `tsne_y` | float | – | Position on the t-SNE map (`perplexity=50`, `init="pca"`, `random_state=42`); only for plotting, distances between groups are not meaningful |

The 13 features the clustering used: `share_unsafe`, `share_swarmp`, `followups_per_cycle`, `log_penalty`, `n_cycles`, `n_materials` and the
material flags for masonry, terra cotta, natural stone, concrete, stucco / EIFS, glass and metal.

## Limitations and biases

- **Coverage:** only buildings taller than 6 stories that filed FISP reports, cycles 6–10. Cycle 10 is still in progress, so the latest
  buildings have fewer reports.
- **Self-reported materials:** materials and wall types are tick boxes filled by the inspector; "other" is very common and a building can be
  described differently from one cycle to the next (we keep every material ever listed).
- **PLUTO join:** 4% of buildings are not geolocated, and a tax lot can contain more than one building, so the year built and floors may
  describe the lot's main building.
- **The clusters are one possible reading:** they depend on our feature choice and on k = 6, chosen for interpretability and stability rather
  than a clear optimum. Group 0 is partly defined by missing information.
- **Not a safety assessment:** a building in the "low risk" group can still be UNSAFE at its next inspection. Do not use these labels to skip
  or replace an inspection.

## How to load it

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hf://datasets/iaac-maai/day3-team1-nyc-facade-typologies/nyc_facade_typologies.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for name, rows in df.groupby("cluster_name"):
    axes[0].scatter(rows["tsne_x"], rows["tsne_y"], s=3, label=name)       # the t-SNE map
    axes[1].scatter(rows["longitude"], rows["latitude"], s=2, label=name)  # the map of New York
axes[0].legend(markerscale=5)
axes[1].set_aspect(1.32)
plt.show()
```

## Related

- GitHub repository with the notebook: https://github.com/iaac-maai/group1_day3_assignment
- The K-Means model that assigns new buildings: https://huggingface.co/iaac-maai/day3-team1-facade-typologies
- Our Day 2 model that predicts UNSAFE inspections from the same data: https://huggingface.co/iaac-maai/day2-team1-unsafe-facade

## Authors and licence

Amelia, Jerome, Kazuko, Sahil. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Released under the same terms as the source data, the [NYC Open Data Terms of Use](https://www.nyc.gov/home/terms-of-use.page). The dataset is based on
*DOB NOW: Safety – Facades Compliance Filings* (NYC Department of Buildings) and *PLUTO* (NYC Department of City Planning). Please credit both.
