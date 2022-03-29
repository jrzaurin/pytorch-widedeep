import warnings
from copy import copy, deepcopy

import shap
import numpy as np
import torch
from torch.nn import Module, Softmax, Sequential

from pytorch_widedeep.training import Trainer

from ..wdtypes import *


class Sigmoid2d(Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = torch.sigmoid(input)
        return torch.cat((1 - out, out), dim=1)


class ShapExplainer(object):
    def __init__(
        self,
    ):
        super(ShapExplainer, self).__init__()

    def fit(
        self,
        tab_trainer: Trainer,
        X_tab_train: np.ndarray,
        explainer_type: Literal["kernel", "deep"],
        background_sample_count: Optional[int] = 100,
    ):
        """Fit SHAP explainer.

        Args:
            tab_trainer (np.ndarray): deep tabular model component input data
            X_tab_train (Trainer): pytorch widedeep model trainer
            explainer_type (str): type of SHAP explainer
            background_sample_count (int): 'The background dataset to use for integrating out features'
                see:
                    * https://shap-lrjball.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
                    * https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html
        Returns:
            self.model (): model in a form integratable with SHAP explainer
            self.explainer (): fitted SHAP explainer
        """
        self.explainer_type = explainer_type
        self.objective = copy(tab_trainer.objective)

        if self.explainer_type == "kernel":
            self.model: Union[Trainer, Module] = deepcopy(tab_trainer)
            self.explainer = shap.KernelExplainer(
                model=self._kernel_explain_model_predict,
                data=X_tab_train[
                    np.random.choice(
                        X_tab_train.shape[0],
                        size=background_sample_count,
                        replace=False,
                    )
                ],
            )
        elif self.explainer_type == "deep":
            warnings.warn(
                """
            WARNING:
            This is not a native functionality of the current version of SHAP.
            According to SHAP code '(commented out because self.data could be huge for GradientExpliner)', see:
            https://github.com/slundberg/shap/blob/895a796b20cb2ab6b158a4cd4326d8f4d00ca615/shap/explainers/_gradient.py#L165,
            we use same expected value calculation as used in DeepExplainer pytorch implementation, see:
            https://github.com/slundberg/shap/blob/895a796b20cb2ab6b158a4cd4326d8f4d00ca615/shap/explainers/_deep/deep_pytorch.py#L53"""
            )
            self.model = self._deep_explain_data_model(tab_trainer)
            self.explainer = shap.DeepExplainer(
                model=self.model,
                data=torch.tensor(
                    X_tab_train[
                        np.random.choice(
                            X_tab_train.shape[0],
                            size=background_sample_count,
                            replace=False,
                        )
                    ]
                ),
            )
        elif self.explainer_type == "gradient":
            # this causes model to be save in 2 places as attribute but it simplifies/unifies
            # the approach with other explainers and ShapExplainer methods, eg. computation
            # of SHAP values and predictions
            self.model = self._deep_explain_data_model(tab_trainer)
            self.explainer = shap.GradientExplainer(
                model=self.model,
                data=torch.tensor(X_tab_train),
            )
            warnings.warn(
                """
            WARNING:
            This is not a native functionality of the current version of SHAP.
            According to SHAP code '(commented out because self.data could be huge for GradientExpliner)', see:
            https://github.com/slundberg/shap/blob/895a796b20cb2ab6b158a4cd4326d8f4d00ca615/shap/explainers/_gradient.py#L165,
            we use same expected value calculation as used in DeepExplainer pytorch implementation, see:
            https://github.com/slundberg/shap/blob/895a796b20cb2ab6b158a4cd4326d8f4d00ca615/shap/explainers/_deep/deep_pytorch.py#L53"""
            )
            # expected value is needed for force and decision plots
            self.explainer.expected_value = (
                self.explainer.explainer.model(*self.explainer.explainer.model_inputs)
                .mean(0)
                .detach()
                .numpy()
            )

    def _kernel_explain_model_predict(self, data: np.ndarray) -> np.ndarray:
        """Function to predict values inrepreted by Kernel Explainer

        Args:
            data (np.ndarray): deep tabular model component input data
        Returns:
            predictions (np.ndarray): predictions
        """
        if self.objective == "regression":
            predictions = self.model.predict(X_tab=data)
        elif self.objective in ["binary", "multiclass"]:
            predictions = self.model.predict_proba(X_tab=data)
        return predictions

    @staticmethod
    def _deep_explain_data_model(tab_trainer: Trainer) -> Module:
        """Simple function to extract the deeptabular component of the model for explanation using Deep Explainer

        Args:
            tab_trainer (Trainer): fitted model Trainer
        Returns:
            explain_model (Sequential): deep tabular component model of the fitted trainer
        """
        tab_trainer_dpc = deepcopy(tab_trainer)
        if tab_trainer_dpc.objective == "regression":
            explain_model = tab_trainer_dpc.model.deeptabular
        elif tab_trainer_dpc.objective == "binary":
            explain_model = Sequential(
                tab_trainer_dpc.model.deeptabular,
                Sigmoid2d(),
            )
        elif tab_trainer_dpc.objective == "multiclass":
            explain_model = Sequential(
                tab_trainer_dpc.model.deeptabular,
                Softmax(dim=1),
            )
        return explain_model

    def explain_decision_plot(
        self, X_tab_explain: np.ndarray, feature_names: Optional[list] = None
    ):
        """Process the data and pick proper decision plot based on model output and SHAP explainer type.

        Args:
            X_tab_explain (np.ndarray): deep tabular model component input data
        Returns:
            shap_decision_plot (): decision plot
        """
        if (self.objective in ["binary", "multiclass"]) and (
            len(X_tab_explain.shape) > 1
        ):
            raise ValueError(
                "Multioutput decision plots for classification support only per value analysis."
            )
        elif len(X_tab_explain.shape) == 1:
            X_tab_explain = np.expand_dims(X_tab_explain, 0)

        predictions = self.compute_preds(X_tab_explain)

        if self.objective == "regression":
            shap_values = self.compute_shap_values(X_tab_explain)
            shap_decision_plot = shap.decision_plot(
                base_value=self.explainer.expected_value,
                shap_values=shap_values,
                features=X_tab_explain,
                feature_names=feature_names,
                legend_labels=self._labels(predictions),
                link="identity",
            )
        elif self.objective in ["binary", "multiclass"]:
            shap_values = self.compute_shap_values(X_tab_explain)
            shap_decision_plot = shap.multioutput_decision_plot(
                base_values=list(self.explainer.expected_value),
                shap_values=shap_values,
                features=X_tab_explain,
                feature_names=feature_names,
                legend_labels=self._labels(predictions),
                link="identity",
                row_index=0,
            )
        return shap_decision_plot

    def _labels(self, predictions: np.ndarray):
        if self.objective == "regression":
            labels = [
                f"Sample {i} ({predictions[i].round(2):.2f})"
                for i in range(len(predictions))
            ]
        elif self.objective in ["binary", "multiclass"]:
            labels = [
                f"Class {i} ({predictions[i].round(2):.2f})"
                for i in range(len(predictions))
            ]
        return labels

    def compute_shap_values(
        self,
        X_tab_explain: np.ndarray,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Helper method to compute SHAP values for other shap plots not
        included in the ShapExplainer object.

        Args:
            X_tab_explain (np.ndarray): array of values to explain
        Returns:
            shap_values (np.ndarray): computed shap values
        """
        if self.explainer_type == "kernel":
            shap_values = self.explainer.shap_values(X_tab_explain)
        elif self.explainer_type in ["deep", "gradient"]:
            X_tab_explain_tensor = torch.tensor(X_tab_explain)
            shap_values = self.explainer.shap_values(X_tab_explain_tensor)
        return shap_values

    def compute_preds(
        self,
        X_tab_explain: np.ndarray,
    ) -> np.ndarray:
        """Helper method to compute SHAP values for other shap plots not
        included in the ShapExplainer object.

        Args:
            X_tab_explain (np.ndarray): array of values to explain
        Returns:
            preds (np.ndarray): computed shap values
        """
        if self.explainer_type == "kernel":
            preds = self._kernel_explain_model_predict(data=X_tab_explain)
        elif self.explainer_type in ["deep", "gradient"]:
            X_tab_explain_tensor = torch.tensor(X_tab_explain)
            preds = self.model(X_tab_explain_tensor).detach().numpy()

        if self.objective in ["binary", "multiclass"]:
            # classification predictions produce arrays of arrays
            preds = np.squeeze(preds)
        elif self.objective == "regression" and self.explainer_type in [
            "deep",
            "gradient",
        ]:
            preds = np.squeeze(preds, axis=1)
        return preds
