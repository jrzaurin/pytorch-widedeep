Quick Start
***********

This is an example of a binary classification with the `adult census
<https://www.kaggle.com/wenruliu/adult-income-dataset?select=adult.csv>`__
dataset using a combination of a wide and deep model (in this case a so called
``DeepDense model``) with defaults settings.


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

    from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
    from pytorch_widedeep.models import Wide, DeepDense, WideDeep
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

    # wide (linear) model
    preprocess_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=cross_cols)
    X_wide = preprocess_wide.fit_transform(df_train)
    wide = Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=1)

    # deeptabular component is a DeepDense model
    preprocess_deep = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
    X_tab = preprocess_deep.fit_transform(df_train)
    deeptabular = DeepDense(
        hidden_layers=[64, 32],
        column_idx=preprocess_deep.column_idx,
        embed_input=preprocess_deep.embeddings_input,
        continuous_cols=cont_cols,
    )


Build, compile, fit and predict
-------------------------------

.. code-block:: python

    # build, compile and fit
    model = WideDeep(wide=wide, deeptabular=deeptabular)
    model.compile(method="binary", metrics=[Accuracy])
    model.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=5,
        batch_size=256,
        val_split=0.1,
    )

    # predict
    X_wide_te = preprocess_wide.transform(df_test)
    X_tab_te = preprocess_deep.transform(df_test)
    preds = model.predict(X_wide=X_wide_te, X_tab=X_tab_te)

Of course, one can do much more, such as using different initializations,
optimizers or learning rate schedulers for each component of the overall
model. Adding FC-Heads to the Text and Image components. Using the Focal Loss,
warming up individual components before joined training, using the
`TabTransformer <https://arxiv.org/pdf/2012.06678.pdf>`__, etc. See the
`examples <https://github.com/jrzaurin/pytorch-widedeep/tree/build_docs/examples>`__
directory for a better understanding of the content of the package and its
functionalities.
