
# Spatially Distributed and Regionally Unbound Cellular-Resolution Brain-Wide Processing Loops in Mice

This repository contains the main analysis code supporting the publication:

Schartner, Michael, et al. "Spatially distributed and regionally unbound cellular resolution brain-wide processing loops in mice." bioRxiv (2025): 2025-07.
https://www.biorxiv.org/content/10.1101/2025.07.30.667641v1

The script `dmn_bwm.py` implements data processing, dimensionality reduction, anatomical mapping, and specialization analyses for the IBL Brain-Wide Map dataset (paper about the data and basic correlates: https://www.nature.com/articles/s41586-025-09235-0).
Data accessible here: https://docs.internationalbrainlab.org/notebooks_external/data_structure.html  

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
  Highlights the diversity of neural responses across the whole hemisphere. 

---

### 2. Swanson Plot (Regional Specialization)
- **What it does:**  
  Generates **Swanson-style flatmap plots** of the mouse brain to visualize regional specialization.  
- **Purpose:**  
  - Projects decoding or clustering results onto a simplified 2D anatomical schematic.  
  - Colors encode anatomical regions from Allen (Beryl) mapping.  
- **Implementation details:**  
  - Uses `plot_swanson_vector` from `iblatlas.plots`.  
  - Region membership derived from histology.
- **Interpretation:**  
  Provides an overview of which brain areas exhibit stronger specialization; also using a specialisation scroe from the decoding analysis of **A Brain-Wide Map of Neural Activity during Complex Behaviour** (https://www.biorxiv.org/content/10.1101/2023.07.04.547681v2).

---

### 3. Correlation Plot (Decoding vs Cluster Specialization)
- **What it does:**  
  Quantifies the relationship between **decoding specialization** and **cluster specialization**.  
- **Purpose:**  
  - Tests whether regions that show broad decodability across main variables (low specialisation) also show a broad distribution across neuronal response types, based on clustering concatenated PETHs.
- **Implementation details:**  
  - Decoding scores: from **A Brain-Wide Map of Neural Activity during Complex Behaviour** (https://www.biorxiv.org/content/10.1101/2023.07.04.547681v2) compared against decoding results from **Spatially Distributed and Regionally Unbound Cellular-Resolution Brain-Wide Processing Loops in Mice** (https://www.biorxiv.org/content/10.1101/2025.07.30.667641v1).    
- **Interpretation:**  
  A strong positive correlation links decoding specialization (decoding of task variables) with neural response specialization (clusters), supporting each as a measure of functional specialisation per region.

---

## Requirements
- Python 3.10+  
- Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `umap-learn`, `rastermap`, `iblatlas`, `brainbox`, `one.api`.  

---

## Usage
Typical workflow:
1. **Preprocessing:** Concatenate and stack PETHs with `stack_concat`.  
2. **Dimensionality reduction:** Compute Rastermap, UMAP, PCA embeddings.  
3. **Regional specialization:** Generate Swanson-style anatomical plots.  
4. **Correlation analysis:** Compare Brain-Wide Map decoding with Supersession decoding, and relate to cluster specialization.  

Figures are saved into the local ONE cache directory (`dmn/imgs/`).  
