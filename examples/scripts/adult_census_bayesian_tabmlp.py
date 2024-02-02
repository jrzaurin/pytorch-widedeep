import numpy as np
import torch
import pandas as pd

from pytorch_widedeep import BayesianTrainer
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor
from pytorch_widedeep.bayesian_models import BayesianWide, BayesianTabMlp

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    df: pd.DataFrame = load_adult(as_frame=True)
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["age_buckets"] = pd.cut(
        df.age, bins=[16, 25, 30, 35, 40, 45, 50, 55, 60, 91], labels=np.arange(9)
    )
    df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop("income", axis=1, inplace=True)
    df.head()

    for model_name in ["linear", "mlp"]:
        for objective in ["binary", "multiclass", "regression"]:
            cat_cols = [
                "workclass",
                "education",
                "marital_status",
                "occupation",
                "relationship",
                "native_country",
                "race",
                "gender",
            ]

            if model_name == "linear":
                crossed_cols = [
                    ("education", "occupation"),
                    ("native_country", "occupation"),
                ]

            if objective == "binary":
                continuous_cols = ["age", "hours_per_week"]
                target_name = "income_label"
                target = df[target_name].values
            elif objective == "multiclass":
                continuous_cols = ["hours_per_week"]
                target_name = "age_buckets"
                target = np.array(df[target_name].tolist())
            elif objective == "regression":
                continuous_cols = ["hours_per_week"]
                target_name = "age"
                target = df[target_name].values

            if model_name == "linear":
                prepare_wide = WidePreprocessor(
                    wide_cols=cat_cols, crossed_cols=crossed_cols
                )
                X_tab = prepare_wide.fit_transform(df)

                model = BayesianWide(
                    input_dim=np.unique(X_tab).shape[0],
                    pred_dim=df["age_buckets"].nunique()
                    if objective == "multiclass"
                    else 1,
                    prior_sigma_1=1.0,
                    prior_sigma_2=0.002,
                    prior_pi=0.8,
                    posterior_mu_init=0,
                    posterior_rho_init=-7.0,
                )

            if model_name == "mlp":
                prepare_tab = TabPreprocessor(
                    cat_embed_cols=cat_cols, continuous_cols=continuous_cols, scale=True  # type: ignore[arg-type]
                )
                X_tab = prepare_tab.fit_transform(df)

                model = BayesianTabMlp(  # type: ignore[assignment]
                    column_idx=prepare_tab.column_idx,
                    cat_embed_input=prepare_tab.cat_embed_input,
                    continuous_cols=continuous_cols,
                    # embed_continuous=True,
                    mlp_hidden_dims=[128, 64],
                    prior_sigma_1=1.0,
                    prior_sigma_2=0.002,
                    prior_pi=0.8,
                    posterior_mu_init=0,
                    posterior_rho_init=-7.0,
                    pred_dim=df["age_buckets"].nunique()
                    if objective == "multiclass"
                    else 1,
                )

            model_checkpoint = ModelCheckpoint(
                filepath="model_weights/wd_out",
                save_best_only=True,
                max_save=1,
            )
            early_stopping = EarlyStopping(patience=2)
            callbacks = [early_stopping, model_checkpoint]
            metrics = [Accuracy] if objective != "regression" else None

            trainer = BayesianTrainer(
                model,
                objective=objective,
                optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                callbacks=callbacks,
                metrics=metrics,
            )

            trainer.fit(
                X_tab=X_tab,
                target=target,
                val_split=0.2,
                n_epochs=1,
                batch_size=256,
            )

            # # simply to check predicts functions as expected
            # preds = trainer.predict(X_tab=X_tab)
