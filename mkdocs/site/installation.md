# Installation

This section explains how to install ``pytorch-widedeep``.

For the latest stable release, execute:

```bash
pip install pytorch-widedeep
```

For the bleeding-edge version, execute:

```bash
pip install git+https://github.com/jrzaurin/pytorch-widedeep.git
```

For developer install

```bash
# Clone the repository
git clone https://github.com/jrzaurin/pytorch-widedeep
cd pytorch-widedeep

# Install in dev mode
pip install -e .
```

## Dependencies

* pandas>=1.3.5
* numpy>=1.21.6
* scipy>=1.7.3
* scikit-learn>=1.0.2
* gensim
* spacy
* opencv-contrib-python
* imutils
* tqdm
* torch
* torchvision
* einops
* wrapt
* torchmetrics
* pyarrow
* fastparquet>=0.8.1