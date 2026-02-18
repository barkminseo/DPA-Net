# DPA-net: Deformable-Point-Attention-for-LiDAR-Place-Recognition-with-Weighted-GeM-Aggregation

by  MinSeo Park

### Abstract
We present DPA-Net, a lightweight yet highly dis- criminative LiDAR place recognition network that combines sparse 3D convolution with deformable point-based attention. Unlike conventional voxel-based architectures that suffer from in- formation loss during quantization, DPA-Net reconstructs dense point features through an interpolation module and applies Deformable Point Attention (DPA) to adaptively aggregate ge- ometric structures. The proposed attention mechanism predicts position-aware offsets, samples local key–value features through a single 3-NN interpolation step, and incorporates relative posi- tional encoding to enhance spatial sensitivity while maintaining computational efficiency. A top-down feature propagation module further refines point-wise features by injecting high-level con- textual information back to the original point distribution. A Weighted Generalized Mean Pooling aggregates the refined point features into a global descriptor for retrieval. Experiments con- firm that DPA-Net achieves competitive performance compared to existing point-based and voxel-based LiDAR place recognition methods.

![Overview](media/Overview.jpg)

## Main Results
![Results](media/Results1.png)
![Results](media/Results2.png)
![Results](media/Results3.png)
![Results](media/Results4.png)

![Results](media/parameters.png)

### Datasets
* Oxford dataset
* NUS (in-house) Datasets
  * university sector (U.S.)
  * residential area (R.A.)
  * business district (B.D.)

Following [PointNetVLAD](https://arxiv.org/abs/1804.03492) the datasets can be downloaded [here](https://drive.google.com/open?id=1H9Ep76l8KkUpwILY-13owsEMbVCYTmyx).
Run the below code to generate pickles with positive and negative point clouds for each anchor point cloud. 

```generate pickles
cd generating_queries/ 

# Generate training tuples for the Baseline Dataset
python generate_training_tuples_baseline.py --dataset_root <dataset_root_path>

# Generate training tuples for the Refined Dataset
python generate_training_tuples_refine.py --dataset_root <dataset_root_path>

# Generate evaluation tuples
python generate_test_sets.py --dataset_root <dataset_root_path>
```
`<dataset_root_path>` is a path to dataset root folder, e.g. `/data/pointnetvlad/benchmark_datasets/`.
Before running the code, ensure you have read/write rights to `<dataset_root_path>`, as training and evaluation pickles
are saved there. 

### Environment and Dependencies
Our code was tested using Python 3.8.12 with PyTorch 1.10.2 and MinkowskiEngine 0.5.4 on Ubuntu 18.04 with CUDA 10.2.

The following Python packages are required:
* PyTorch (version 1.10.1)
* [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) (version 0.5.4)
* pytorch_metric_learning (version 1.1 or above)
* pandas
* tqdm


### Training and Evaluation
Note that our training code refers to MinkLoc3Dv2. For more details of the training code please refer to [here](https://github.com/jac99/MinkLoc3Dv2).

* Modify the `PYTHONPATH` environment variable to include absolute path to the project root folder: 
    ```
    export PYTHONPATH=$PYTHONPATH:/home/.../PTC-Net-main
    ```

* build the pointops

  ```
  cd libs/pointops && python setup.py install && cd ../../
  ```
  
* Train the network

    ```
    cd training
    
    # To train model on the Baseline Dataset
    python train.py --config ../config/config_baseline.txt --model_config ../models/config_model.txt
    
    # To train model on the Refined Dataset
    python train.py --config ../config/config_refined.txt --model_config ../models/config_model.txt
    ```

* Evaluate the network

    ```eval baseline
    cd eval
    
    # To evaluate the model trained on the Baseline Dataset
    python pnv_evaluate.py --config ../config/config_baseline.txt --model_config ../models/config_model.txt --weights ../weights/*.pth
    
    # To evaluate the model trained on the Refined Dataset
    python pnv_evaluate.py --config ../config/config_refined.txt --model_config ../models/config_model.txt --weights ../weights/*.pth
    ```


