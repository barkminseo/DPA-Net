# DPA-net: Deformable-Point-Attention-for-LiDAR-Place-Recognition-with-Weighted-GeM-Aggregation

by  MinSeo Park

### Abstract
We present DPA-Net, a lightweight yet highly dis- criminative LiDAR place recognition network that combines sparse 3D convolution with deformable point-based attention. Unlike conventional voxel-based architectures that suffer from in- formation loss during quantization, DPA-Net reconstructs dense point features through an interpolation module and applies Deformable Point Attention (DPA) to adaptively aggregate ge- ometric structures. The proposed attention mechanism predicts position-aware offsets, samples local key–value features through a single 3-NN interpolation step, and incorporates relative posi- tional encoding to enhance spatial sensitivity while maintaining computational efficiency. A top-down feature propagation module further refines point-wise features by injecting high-level con- textual information back to the original point distribution. A Weighted Generalized Mean Pooling aggregates the refined point features into a global descriptor for retrieval. Experiments con- firm that DPA-Net achieves competitive performance compared to existing point-based and voxel-based LiDAR place recognition methods.

![Overview](media/Overview.png)

### Oxford & In-house (Baseline Setting)
![Oxford Results](media/Results1.png)

---

### Refined Setting
![Refined Results](media/Results2.png)

---

### MulRan Benchmark
![MulRan Results](media/Results3.png)

---

### RoboLoc Benchmark
![RoboLoc Results](media/Results4.png)

---

### Model Efficiency
![Efficiency](media/parameters.png)

### Datasets
dataset can be downloaded here:

🔗 [Download Dataset (Google Drive)](링크)

* Oxford dataset
* NUS (in-house) Datasets
  * university sector (U.S.)
  * residential area (R.A.)
  * business district (B.D.)
* Mulran dataset
* roboloc dataset

### Training and Evaluation
Note that our training code refers to PTCnet. For more details of the training code please refer to [here](https://github.com/LeegoChen/PTC-Net).

For training and evaluation on the **RoboLoc dataset**, please replace the default dataset loader with:

- `base_datasets_ajou.py`
- `pnv_raw_ajou.py`

These files contain the dataset configuration and loading pipeline adapted for the RoboLoc benchmark.

