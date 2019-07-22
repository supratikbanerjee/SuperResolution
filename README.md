# SuperResolution-Base
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
