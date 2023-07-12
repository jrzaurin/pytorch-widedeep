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

wd = WideDeep(deepimage=deepimage)
trainer = StreamTrainer(
    model=wd, 
    objective='regression',
    X_path=data_path,
    img_col=img_col,
    target_col=target_col,
    img_preprocessor=image_processor,
    fetch_size=250
)
trainer.fit(
     n_epochs=20,
     batch_size=64
 )
# %%
