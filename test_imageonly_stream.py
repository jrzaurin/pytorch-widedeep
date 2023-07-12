#%%

import glob
import pandas as pd
from pytorch_widedeep.models import BasicRNN, WideDeep, Vision

from pytorch_widedeep.stream.preprocessing.image_preprocessor import StreamImagePreprocessor
from pytorch_widedeep.stream.training.trainer import StreamTrainer

from sklearn.datasets import fetch_olivetti_faces

import shutil

# Prepare dataset
# img_path = '/home/krish/Downloads/MNIST Dataset JPG format/MNIST - JPG - training'
# samples = 200
# imgs = []
# labels = []
# for f in glob.glob(f"{img_path}/*"):
#     for im in glob.glob(f"{f}/*")[:samples]:
#         imgs.append(im.split('/')[-1])
#         labels.append(f.split('/')[-1])
#         # shutil.copy(im, '/home/krish/Downloads/MNIST Dataset JPG format/sample_imgs')

# X = pd.DataFrame({'imgs': imgs, 'labels': labels})
# X.to_csv('mnist_samples.csv')

# %%

# User is responsible for shuffling their inputs?
# We should give an option to pass X as either a path or a dataframe
# Images can be grabbed from disk as needed by the WDIterator

'''
TODO:
1. Enable shuffling inputs
2. Get vision-only mnist training working
3. enable validation in training loop
4. enable callbacks
5. add unit tests
6. add support for image transforms
7. review with javier for next steps
'''

img_col = 'imgs'
target_col = 'labels'
data_path = 'mnist_samples.csv'
img_path = '/home/krish/Downloads/MNIST Dataset JPG format/sample_imgs'

image_processor = StreamImagePreprocessor(img_col=img_col, img_path=img_path)
image_processor.fit()

deepimage = Vision(
     pretrained_model_name="resnet18", n_trainable=0, head_hidden_dims=[200, 100]
)

# %%

wd = WideDeep(deepimage=deepimage, pred_dim=10)
trainer = StreamTrainer(
    model=wd, 
    objective='multiclass',
    X_path=data_path,
    img_col=img_col,
    target_col=target_col,
    img_preprocessor=image_processor,
    fetch_size=250
)
trainer.fit(
     n_epochs=5,
     batch_size=32
 )
# %%
