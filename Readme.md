## Data Preparation
- Unzip the `CelebAMask-HQ.zip` file into the `./data` directory, ensuring that the resulting structure matches the one shown below. **Don't forget to create an empty folder named "CelebAMask-HQ-combined_mask" inside the `./data` directory for the model.**
- The `test_idx.txt` and `train_idx.txt` files should be downloaded from the following [Kaggle dataset link](https://www.kaggle.com/competitions/cs6550-face-parsing/data).
- After extracting the files, run `dataset/preprocess.py` to preprocess the images. The script will process and place the combined images into the `data/CelebAMask-HQ/CelebAMask-HQ-combined_mask` folder automatically.


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