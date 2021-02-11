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
    wide = Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=1)

    # deeptabular
    tab_preprocessor = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
    X_tab = tab_preprocessor.fit_transform(df_train)
    deeptabular = TabMlp(
        mlp_hidden_dims=[64, 32],
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=cont_cols,
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

    # save and load
    trainer.save_model("model_weights/model.t")

Of course, one can do **much more**. See the Examples folder in the repo, this
documentation or the companion posts for a better understanding of the content
of the package and its functionalities.
