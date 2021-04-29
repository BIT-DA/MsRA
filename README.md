# End-to-End Transfer Anomaly Detection via Multi-spectral Cross-domain Representation Alignment
Code release for "End-to-End Transfer Anomaly Detection via Multi-spectral Cross-domain Representation Alignment"

In this paper, we propose a Multi-spectral Cross-domain Representation Alignment (MsRA) method for the anomaly detection in the domain adaptation setting, where we can only access a set of normal source data and a limited number of normal target data.

## Prerequisites
The code is implemented with **CUDA 10.0.130**, **Python 3.6.13** and **Pytorch 1.2.0**.

To install the required python packages, run

```pip install -r requirements.txt```

## Datasets

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

## Running the code

Office-Home
```
python3 train_TSA.py --gpu_id 4 --arch resnet50 --seed 0 --dset office-home --output_dir log/office-home --s_dset_path data/list/home/Art_65.txt --t_dset_path data/list/home/Product_65.txt --epochs 40 --iters-per-epoch 500 --lambda0 0.25 --MI 0.1
```

## Acknowledgements
Some codes are adapted from [DSVDD](https://github.com/lukasruff/Deep-SVDD-PyTorch), [DANN](https://github.com/fungtion/DANN) and 
[BiOST](https://github.com/tomercohen11/BiOST). We thank them for their excellent projects.

## Contact
If you have any problem about our code, feel free to contact
- shuangli@bit.edu.cn
- shugangli@bit.edu.cn
- mxxie@bit.edu.cn
- kxgong@bit.edu.cn

or describe your problem in Issues.
