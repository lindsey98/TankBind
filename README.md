![header](imgs/header.png)

Source code for the *NeurIPS 2022* paper [TANKBind: Trigonometry-Aware Neural NetworKs for Drug-Protein Binding Structure Prediction](https://biorxiv.org/cgi/content/short/2022.06.06.495043v1)

# TankBind
TankBind could predict both the protein-ligand binding structure and their affinity.

The primary purpose of this repository is to enable the reproduction of the results reported in the paper, as well as to facilitate the work of others who wish to build upon it.
To experience the latest version, which includes various improvements made to the model, simply create an account at https://m1.galixir.com/public/login_en/index.html.

If you have any question or suggestion, please feel free to open an issue or email me at [wei.lu@galixir.com](wei.lu@galixir.com) or shuangjia zheng at [shuangjia.zheng@galixir.com](shuangjia.zheng@galixir.com).

## Installation
Original
````
conda create -n tankbind_py38 python=3.8
conda activate tankbind_py38
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install torchdrug=0.1.2 pyg=2.1.0 biopython nglview jupyterlab -c milagraph -c conda-forge -c pytorch -c pyg
pip install biopython tqdm mlcrate pyarrow
rdkit version used: 2021.03.4
````

I have tried CUDA: => but then torch-cluster is not compatible => the root cause is my nvcc -v is 10.1 but my pytorch is using cuda 11.6. 
However, there is no compatible lower CUDA for pytorch 1.13.1. 
```bash
conda create -n tankbind_py38 python=3.8
conda activate tankbind_py38
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch_geometric
pip install torchdrug==0.1.2
pip install rdkit
```

What about CPU? Finally working. torch=1.13.1 and torchdrug=0.1.2 are must.
```
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install torch-cluster
pip install torchdrug==0.1.2
pip install rdkit
sudo apt-get install ninja-build
```

p2rank v2.3 could be downloaded from here:

https://github.com/rdk/p2rank/releases/download/2.3/p2rank_2.3.tar.gz


## Test set evaluation
We include the script for reproducing the self-dock result in

    examples/testset_evaluation_cleaned.ipynb

The test_dataset is constructed using the notebook in "Dataset construction" section.


## Prediction
We use the prediction of the structure of protein ABL1 in complex with two drugs, Imatinib and compound6 (PDB: 6HD6) as an example for predicting the drug-protein binding structure. 

    examples/prediction_example_using_PDB_6hd6.ipynb

<img src="imgs/example_6hd6.png" width="200">


## Dataset construction
Scripts for training/test dataset construction is provided in:

    examples/construction_PDBbind_training_and_test_dataset.ipynb.ipynb

The Script I used to train the model is 

    python main.py -d 0 -m 0 --batch_size 5 --label baseline --addNoise 5 --use_equivalent_native_y_mask


## High-throughput virtual screening
TankBind also support virtual screening. In our example here, for the WDR domain of LRRK2 protein, we can screen 10,000 drug candidates in 2 minutes (or 1M in around 3 hours) with a single GPU. Check out

    examples/high_throughput_virtual_screening_LRRK2_WDR.ipynb


## Citation
    @article{lu2022tankbind,
    	title={Tankbind: Trigonometry-aware neural networks for drug-protein binding structure prediction},
    	author={Lu, Wei and Wu, Qifeng and Zhang, Jixian and Rao, Jiahua and Li, Chengtao and Zheng, Shuangjia},
    	journal={Advances in Neural Information Processing Systems},
    	year={2022}
    }
