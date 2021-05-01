# End-to-End Transferable Anomaly Detection via Multi-spectral Cross-domain Representation Alignment

Code release for "End-to-End Transferable Anomaly Detection via Multi-spectral Cross-domain Representation Alignment"

# Paper

<div align=center><img src="./figures/Fig_method.pdf" width="100%"></div>

In this paper, we propose a Multi-spectral Cross-domain Representation Alignment (MsRA) method for the anomaly detection in the domain adaptation setting, where we can only access a set of normal source data and a limited number of normal target data.

## Prerequisites
The code is implemented with **CUDA 10.0.130**, **Python 3.6.13** and **Pytorch 1.2.0**.

To install the required python packages, run

```pip install -r requirements.txt```

## Datasets

Download the dataset and place the images to the corresponding folder.

Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

## Running the code

```
python MsRA.py --dataset OfficeHomeDataset --source Product --target Clipart --c_cls Bike
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
