
# Spatially Distributed and Regionally Unbound Cellular-Resolution Brain-Wide Processing Loops in Mice

This repository contains the main analysis code supporting the publication:

> **Schartner M, Ashwood Z, Acerbi L, Harris KD, Carandini M, International Brain Laboratory**  
> *Spatially Distributed and Regionally Unbound Cellular-Resolution Brain-Wide Processing Loops in Mice*  
> [bioRxiv 2025.07.30.667641](https://www.biorxiv.org/content/10.1101/2025.07.30.667641v1)

The code implements data processing, dimensionality reduction, anatomical mapping, and specialization analyses for the IBL Brain-Wide Map dataset.  

---

## Key Figures Documented Here

### 1. Rastermap Figure (Population Dynamics)
- **What it does:**  
  Applies **Rastermap**, a nonlinear dimensionality reduction algorithm tailored for neural population recordings.  
- **Purpose:**  
  - Orders neurons by similarity in activity across trials.  
  - Produces smooth embeddings that reveal structure in population-level dynamics.  
- **Implementation details:**  
  - Uses concatenated z-scored PETHs (`concat_z`).  
  - Rastermap parameters: `n_PCs=200`, `n_clusters=100`, `locality=0.75`, `time_lag_window=5`.  
  - Outputs a sorting index (`isort`) used to reorder neurons for visualization.  
- **Interpretation:**  
  Highlights how distributed neurons align into coherent task-related trajectories, exposing functional structure across regions.

---

### 2. Swanson Plot (Regional Specialization)
- **What it does:**  
  Generates **Swanson-style flatmap plots** of the mouse brain to visualize regional specialization.  
- **Purpose:**  
  - Projects decoding or clustering results onto a simplified 2D anatomical schematic.  
  - Colors encode anatomical regions from Allen Atlas / Cosmos / Beryl mappings.  
- **Implementation details:**  
  - Uses `plot_swanson_vector` from `iblatlas.plots`.  
  - Region membership derived from aligned spike sorting data.  
- **Interpretation:**  
  Provides an intuitive view of which brain areas exhibit stronger specialization, situating functional signals within anatomy.

---

### 3. Correlation Plot (Decoding vs Cluster Specialization)
- **What it does:**  
  Quantifies the relationship between **decoding specialization** and **cluster specialization**.  
- **Purpose:**  
  - Tests whether regions that show distinctive decoding profiles also exhibit distinctive clustering signatures.  
- **Implementation details:**  
  - Decoding scores: from the **Brain-Wide Map paper** (Spitmaan et al., 2024) compared against decoding results from the **Supersession paper** (International Brain Laboratory, 2025).  
  - Cluster specialization: enrichment analysis of cluster membership (`regional_group`).  
  - Correlation: Pearson or Spearman statistics.  
- **Interpretation:**  
  A strong positive correlation links functional specialization (decoding across studies) with anatomical specialization (clusters), reinforcing the distributed yet structured organization of brain-wide processing loops.

---

## Requirements
- Python 3.10+  
- Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `umap-learn`, `hdbscan`, `rastermap`, `iblatlas`, `brainbox`, `one.api`.  

---

## Usage
Typical workflow:
1. **Preprocessing:** Concatenate and stack PETHs with `stack_concat`.  
2. **Dimensionality reduction:** Compute Rastermap, UMAP, PCA embeddings.  
3. **Regional specialization:** Generate Swanson-style anatomical plots.  
4. **Correlation analysis:** Compare Brain-Wide Map decoding with Supersession decoding, and relate to cluster specialization.  

Figures are saved into the local ONE cache directory (`dmn/imgs/`).  
