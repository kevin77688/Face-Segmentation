## Environment Setup

1. **CUDA 11.8 Installation:**

    For optimal performance, install CUDA 11.8 from the official [Nvidia website](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04). We recommend using the `runfile(local)` option for installation.

2. **Anaconda Installation:**

    Download and install Anaconda from the official website: [Anaconda Download](https://docs.anaconda.com/free/anaconda/install/index.html).

3. **Environment Setup:**

    Open a terminal and run the following commands to create an environment using the provided `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    conda activate face
    ```

## Data Preparation

1. Download the `CelebAMask-HQ.zip` file from the dataset source and unzip it into the `./data` directory. Ensure the extracted folder structure matches the one shown below.

2. Download the `test_idx.txt` and `train_idx.txt` files from the following [Kaggle dataset Link](https://www.kaggle.com/competitions/cs6550-face-parsing/data). Place these files in the `./data` directory.

3. After extracting the files and preparing the directory structure, run the script `dataset/preprocess.py` to preprocess the images. This script will automatically process and combine the images, placing them in the `./data/CelebAMask-HQ/CelebAMask-HQ-combined_mask` folder.

    ### Tree Structure Example
    ```
    Face_Competition
    ├── checkpoint
    │ └── unet_model.pth
    ├── data
    │ ├── CelebAMask-HQ
    │ │ ├── CelebA-HQ-img
    │ │ │ ├── 0.jpg
    │ │ │ ├── 1.jpg
    │ │ │ ├── 2.jpg
    │ │ │ ├── ...
    │ │ │ └── 2999.jpg
    │ │ ├── CelebAMask-HQ-mask-anno
    │ │ │ ├── 0
    │ │ │ ├── 1
    │ │ │ ├── 2
    │ │ │ ├── ...
    │ │ │ └── 14
    │ │ ├── CelebAMask-HQ-combined_mask
    │ │ │ ├── 0.jpg
    │ │ │ ├── 1.jpg
    │ │ │ ├── 2.jpg
    │ │ │ ├── ...
    │ │ │ └── 2999.jpg
    │ │ ├── CelebA-HQ-to-CelebA-mapping.txt
    │ │ ├── CelebAMask-HQ-attribute-anno.txt
    │ │ └── CelebAMask-HQ-pose-anno.txt
    │ ├── test_idx.txt
    │ ├── train_idx.txt
    ├── dataset
    │ ├── dataset.py
    │ ├── preprocess.py
    │ └── transform.py
    ├── model
    │ ├── unet_parts.py
    │ └── unet.py
    ├── out
    │ └── test.png
    ├── Readme.md
    ├── run.py
    └── visualize.py
    ```

## Training and Testing

**1. Run the Notebook:**

Open the `run.ipynb` file and select the `face` kernel. Click `Run All` to execute the entire notebook.

**2. Parameter Explanation:**

The notebook defines several key parameters:

**Paths:**

* `CHECKPOINT_PATH`: Path to the model weights for resuming training or performing inference.
* `TRAIN_INDEX_PATH`: Path to the list of training data indexes.
* `TEST_INDEX_PATH`: Path to the list of testing data indexes.

**Global Variables:**

* `MODE`: Training mode (`train`) or testing mode (`test`).
* `RECORD`: Whether to use Weights & Biases (wandb) to record training losses.
* `SAVE_MODEL_NAME`: Name of the saved model in the `checkpoint/` folder.

**Hyperparameters:**

* `EPOCHS`: Maximum number of training epochs.
* `BATCH_SIZE`: Batch size for both the training and testing data loaders.
    * **Note:** Set `BATCH_SIZE = 1` when `MODE = 'test'`.
* `LEARNING_RATE`: Optimizer learning rate.
