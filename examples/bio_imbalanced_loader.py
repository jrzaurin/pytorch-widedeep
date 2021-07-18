import time
import datetime
import warnings

import numpy as np
import pandas as pd
from torch.optim import SGD, lr_scheduler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.metrics import Recall, F1Score, Accuracy, Precision
from pytorch_widedeep.dataloaders import DataLoaderImbalanced
from pytorch_widedeep.initializers import XavierNormal
from pytorch_widedeep.preprocessing import TabPreprocessor

warnings.filterwarnings("ignore", category=DeprecationWarning)

# increase displayed columns in jupyter notebook
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 300)


header_list = ["EXAMPLE_ID", "BLOCK_ID", "target"] + [str(i) for i in range(4, 78)]
df = pd.read_csv("data/kddcup04/bio_train.dat", sep="\t", names=header_list)
df.head()

# drop columns we won't need in this example
df.drop(columns=["EXAMPLE_ID", "BLOCK_ID"], inplace=True)

df_train, df_valid = train_test_split(
    df, test_size=0.2, stratify=df["target"], random_state=1
)
df_valid, df_test = train_test_split(
    df_valid, test_size=0.5, stratify=df_valid["target"], random_state=1
)

continuous_cols = df.drop(columns=["target"]).columns.values.tolist()

# deeptabular
tab_preprocessor = TabPreprocessor(continuous_cols=continuous_cols, scale=True)
X_tab_train = tab_preprocessor.fit_transform(df_train)
X_tab_valid = tab_preprocessor.transform(df_valid)
X_tab_test = tab_preprocessor.transform(df_test)

# target
y_train = df_train["target"].values
y_valid = df_valid["target"].values
y_test = df_test["target"].values

# Define the model
input_layer = len(tab_preprocessor.continuous_cols)
output_layer = 1
hidden_layers = np.linspace(
    input_layer * 2, output_layer, 5, endpoint=False, dtype=int
).tolist()

deeptabular = TabMlp(
    mlp_hidden_dims=hidden_layers,
    column_idx=tab_preprocessor.column_idx,
    continuous_cols=tab_preprocessor.continuous_cols,
)
model = WideDeep(deeptabular=deeptabular)
model

# Metrics from pytorch-widedeep
accuracy = Accuracy(top_k=2)
precision = Precision(average=False)
recall = Recall(average=True)
f1 = F1Score(average=False)

# Optimizers
deep_opt = SGD(model.deeptabular.parameters(), lr=0.1)

# LR Scheduler
deep_sch = lr_scheduler.StepLR(deep_opt, step_size=3)

trainer = Trainer(
    model,
    objective="binary",
    lr_schedulers={"deeptabular": deep_sch},
    initializers={"deeptabular": XavierNormal},
    optimizers={"deeptabular": deep_opt},
    metrics=[accuracy, precision],  # , recall, f1],
    verbose=1,
)

start = time.time()
trainer.fit(
    X_train={"X_tab": X_tab_train, "target": y_train},
    X_val={"X_tab": X_tab_valid, "target": y_valid},
    n_epochs=1,
    batch_size=32,
    custom_dataloader=DataLoaderImbalanced,
    oversample_mul=5,
)
print(
    "Training time[s]: {}".format(
        datetime.timedelta(seconds=round(time.time() - start))
    )
)

pd.DataFrame(trainer.history)

df_pred = trainer.predict(X_tab=X_tab_test)
print(classification_report(df_test["target"].to_list(), df_pred))
print("Actual predicted values:\n{}".format(np.unique(df_pred, return_counts=True)))
