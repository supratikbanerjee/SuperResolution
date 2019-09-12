# SuperResolution-Base
 
 **SuperResolution-Base** is a Single Image Super-Resolution (SISR) framework inspired by the [Basic-SR](https://github.com/xinntao/BasicSR) framework for research in Super Resolution using modern Neural Network Architectures.
 This repository is Pytorch code for our proposed SubPixel-BackProjection Network.
 
### Dependencies
  * python 3.x
  * pytorch 1.1.0
  * cuda10
  * torch
  * torchvision
  * scikit-image
  * pillow
  * pyyaml
  * visdom
  * tqdm
  * robust loss (https://github.com/jonbarron/robust_loss_pytorch.git)
  * pacnet (https://github.com/jonbarron/robust_loss_pytorch.git) [already included in this repo!!!]

#### Install Dependencies
```
# Create virtual environment
conda create -n sr_env

# Install torch
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

# Install skimage
conda install -c conda-forge scikit-image

# Install visdom
conda install -c conda-forge visdom

# Install pyyaml
conda install -c conda-forge pyyaml

# Install tqdm
conda install -c conda-forge tqdm

# Install Robust Loss
pip install git+https://github.com/jonbarron/robust_loss_pytorch
```

## Datasets
The networks are being trained on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K). . The pretrained models released in this repository have been trained with DIV2K as of now.

Evaluating the performance of networks on the following benchmark datasets:

* [Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
* [Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests)
* [B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
* [Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr)
* [Manga109](http://www.manga109.org/en/#)
* historical


Direct Download Links:
[Train](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
[Test](http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip)

To setup the datasets, use the direct links to download the datasets and place them under [data/datasets/ALL_DOWNLOADED_DATASETS](https://github.com/supratikbanerjee/SuperResolution/tree/master/data/datasets). No other operation is required to setup the datasets.

The data can be loaded in two ways during trainig, either from the Hard Drive or from the RAM. This option can be switched by changing 'read' parameter in config file to 'ram' or 'disk', 
NOTE: Loading from disk is currently not well optimized, it is recommended to load the entire dataset into RAM at the begining for optimal training performance. (Min 16GB RAM)


## Training
Train a model using the script `train.py`:
```
# Train with configuration file
python train.py -config options/train/CONFIG.yaml
```
Checkpoints and log files are stored in `experiments/`. The `-config` flag reads configuration files in `yaml` format. In `options/train/` config files for training various architectures are provided.
```
# Train SubPixel-BackProjection Network (SPBP)
python train.py -config options/train/SPBP.yaml
```

#### Visualization
To visualize intermediate results (optional) run the `python -m visdom.server -port PORT_NUMBER_IN_CONFIG_FILE` in a separate terminal and enable visualization in the config file by setting 'use_visdom' to 'true'.

```
# Run the server in a separate terminal
python -m visdom.server -port 8067
```

## Testing
Run:
```
python test.py -config options/test/CONFIG.yaml 
```
The script will compute the resulting PSNR, SSIM and Inference Time for every dataset in the config file. It will also store the `LR` and `SR` images under `trained_models/experiments/` along with a log file.



