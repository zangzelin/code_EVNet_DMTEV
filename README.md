
# EVNet: An Explainable Deep Network for Dimension Reduction (EVNet)

The code includes the following modules:
* Datasets (digits, coil20, coil100, Mnist, EMNIST, KMnsit, Colon, Activity, MCA, Gast10K, Samusik, HCL)
* Training for EVNet
* Evaluation metrics 
* Visualisation

## Requirements

* pytorch == 2.1.2
* pytorch-lightning == 1.9.0
* torchvision == 0.16.2
* numpy == 1.26.2
* scikit-learn == 1.3.2
* wandb == 0.16.1

## Description

* ./EVNet_main.py -- End-to-end training of the EVNet model
* ./eval
  * eval/eval_core.py -- The code for evaluate the embedding 
* ./Loss -- Calculate losses
  * ./Loss/dmt_loss_aug.py -- The EVNet loss
  * ./Loss/dmt_loss_source.py -- The template of loss function 
* ./eval -- The yaml file for gird search
* ./dataloader -- the dataloader
  * ./dataloader/source.py -- The template of dataset 
  * ./dataloader/data_base.py -- The EVNet dataset 

## Dataset

The datasets include six simple image datasets ([Digits](https://scikit-learn.org/stable/auto\_examples/datasets/plot\_digits\_last\_image.html), [Coil20](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php), [Coil100](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php), [Mnist](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits), [EMnist](https://www.tensorflow.org/datasets/catalog/emnist), [KMnist](https://www.tensorflow.org/datasets/catalog/kmnist)) and six biological datasets ([Colon](https://figshare.com/articles/dataset/The\_microarray\_dataset\_of\_colon\_cancer\_in\_csv\_format\_/13658790/1), [Activity](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones), [MCA](https://figshare.com/articles/dataset/MCA\_DGE\_Data/5435866), [Gast10k](http://biogps.org/dataset/tag/gastric\%20carcinoma/), [SAMUSIK](https://github.com/abbioinfo/CyAnno), and HCL).

## Running the code

``` bash
bash main.sh
```

## 
If you want to use your own dataset, you can use the following command:
``` bash
CUDA_VISIBLE_DEVICES=-1 python EVNet_main.py --data_name CSV --num_fea_aim 64 --nu 5e-3 --epochs 90 --data_path=data/niu/
```

If you use 'data/niu/'. The data file should named as data/niu/data.csv, the label file file should named as data/niu/label.csv.

## Cite the paper

```
@article{zang2023evnet,
  title={Evnet: An explainable deep network for dimension reduction},
  author={Zang, Zelin and Cheng, Shenghui and Lu, Linyan and Xia, Hanchen and Li, Liangyu and Sun, Yaoting and Xu, Yongjie and Shang, Lei and Sun, Baigui and Li, Stan Z},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2023},
  publisher={IEEE}
}
```



## License

EVNet is released under the MIT license.
