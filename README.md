# SuperResolution-Base
 
 **SuperResolution-Base** is a Single Image Super-Resolution (SISR) framework derived from the Basic-SR framework for research in Super Resolution using modern Neural Network Architectures. It is currently under active development for the fullfilment of my MSc. thesis at Trinity College Dublin. Soon the repository will be updated with my results.
 
### Dependencies
  * python 3.x
  * pytorch 1.1.0
  * cuda92
  * torch
  * torchvision
  * scikit-image
  * opencv
  * pillow
  * pyyaml
  * visdom
  * tqdm

#### Install Dependencies
```
# Create virtual environment
conda create -n sr_env

# Install torch
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# Install image libraries
conda install -c conda-forge scikit-image
conda install -c conda-forge opencv

# Install visdom
conda install -c conda-forge visdom

# Install pyyaml
conda install -c conda-forge pyyaml

# Install tqdm
conda install -c conda-forge tqdm
```

### Datasets
The networks are being trained on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K). . The pretrained models released in this repository have been trained with DIV2K as of now.

Evaluating the performance of networks on the following benchmark datasets:

* [Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
* [Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests)
* [B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
* [Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr)

### Loading the dataset
Coming Soon...

## Training
Train your own model using the script `train.py`:
```
# Train with configuration file
python train.py --config CONFIG.yaml
```

Checkpoints and log files are stored in `experiments/EXPERIMENT_DIR`. Alternatively, the `--config` flag reads configuration files in `yaml` format. In `PROJECT_ROOT/options` config files for various architectures are provided.

#### Visualization
To visualize intermediate results (optional) run the `python -m visdom.server -port PORT_NUMBER_IN_CONFIG_FILE` in a separate terminal and enable visualization in the config file by setting 'use_visdom' to 'true'.

```
# Run the server in a separate terminal
python -m visdom.server -port 8067
```

## Testing
Run:
```
python test.py --config CONFIG.yaml 
```
The script will compute the resulting PSNR, SSIM and Inference Time for every dataset in the config file. It will also store the `LR` and `SR` images under `PROJECT_ROOT/trained_models/experiments` along with a log file.


