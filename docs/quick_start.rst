Quick Start
***********

This is an example of a binary classification with the `adult census
<https://www.kaggle.com/wenruliu/adult-income-dataset?select=adult.csv>`__
dataset using a combination of a wide and deep model (in this case a so called
``deeptabular`` model) with defaults settings.


Read and split the dataset
--------------------------

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/adult/adult.csv.zip")
    df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop("income", axis=1, inplace=True)
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df.income_label)



Prepare the wide and deep columns
---------------------------------

.. code-block:: python

    import torch
    from pytorch_widedeep import Trainer
    from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
    from pytorch_widedeep.models import Wide, TabMlp, WideDeep
    from pytorch_widedeep.metrics import Accuracy

    # prepare wide, crossed, embedding and continuous columns
    wide_cols = [
        "education",
        "relationship",
        "workclass",
        "occupation",
        "native-country",
        "gender",
    ]
    cross_cols = [("education", "occupation"), ("native-country", "occupation")]
    embed_cols = [
        ("education", 16),
        ("workclass", 16),
        ("occupation", 16),
        ("native-country", 32),
    ]
    cont_cols = ["age", "hours-per-week"]
    target_col = "income_label"

    # target
    target = df_train[target_col].values

Preprocessing and model components definition
---------------------------------------------

.. code-block:: python

    # wide
    wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=cross_cols)
    X_wide = wide_preprocessor.fit_transform(df_train)
    wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=1)

    # deeptabular
    tab_preprocessor = TabPreprocessor(cat_embed_cols=embed_cols, continuous_cols=cont_cols)
    X_tab = tab_preprocessor.fit_transform(df_train)
    deeptabular = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=cont_cols,
        mlp_hidden_dims=[64, 32],
    )

    # wide and deep
    model = WideDeep(wide=wide, deeptabular=deeptabular)


Fit and predict
-------------------------------

.. code-block:: python

    # train the model
    trainer = Trainer(model, objective="binary", metrics=[Accuracy])
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=5,
        batch_size=256,
        val_split=0.1,
    )

    # predict
    X_wide_te = wide_preprocessor.transform(df_test)
    X_tab_te = tab_preprocessor.transform(df_test)
    preds = trainer.predict(X_wide=X_wide_te, X_tab=X_tab_te)


Save and load
-------------------------------

.. code-block:: python

    # Option 1: this will also save training history and lr history if the
    # LRHistory callback is used

    # Day 0, you have trained your model, save it using the trainer.save
    # method
    trainer.save(path="model_weights", save_state_dict=True)

    # Option 2: save as any other torch model

    # Day 0, you have trained your model, save as any other torch model
    torch.save(model.state_dict(), "model_weights/wd_model.pt")

    # From here in advance, Option 1 or 2 are the same

    # Few days have passed...I assume the user has prepared the data and
    # defined the model components:
    # 1. Build the model
    model_new = WideDeep(wide=wide, deeptabular=deeptabular)
    model_new.load_state_dict(torch.load("model_weights/wd_model.pt"))

    # 2. Instantiate the trainer
    trainer_new = Trainer(
        model_new,
        objective="binary",
    )

    # 3. Either fit or directly predict
    preds = trainer_new.predict(X_wide=X_wide, X_tab=X_tab)


Of course, one can do **much more**. See the Examples folder in the repo, this
documentation or the companion posts for a better understanding of the content
of the package and its functionalities.
