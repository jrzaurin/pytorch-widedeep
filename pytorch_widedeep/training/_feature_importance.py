from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.sparse import csc_matrix
from torch.utils.data import DataLoader

from pytorch_widedeep.wdtypes import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Tensor,
    Optional,
    WideDeep,
)
from pytorch_widedeep.models.tabular import (
    SAINT,
    TabPerceiver,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
    SelfAttentionMLP,
    ContextAttentionMLP,
)
from pytorch_widedeep.utils.general_utils import alias
from pytorch_widedeep.training._wd_dataset import WideDeepDataset
from pytorch_widedeep.models.tabular.tabnet._utils import create_explain_matrix

TransformerBasedModels = (
    SAINT,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
    SelfAttentionMLP,
    ContextAttentionMLP,
)

__all__ = ["FeatureImportance", "Explainer"]


# TO DO: review typing for WideDeep (in particular the deeptabular part) The
# issue with the typing of the deeptabular part is the following: the
# deeptabular part is typed as Optional[BaseWDModelComponent]. While that is
# correct, is not fully informative, and the most correct approach would
# perhaps be [BaseWDModelComponent, BaseTabularModelWithAttention,
# BaseTabularModelWithoutAttention]. Perhaps this way as we do
# model.deeptabular._modules["0"] it would be understood that the type of
# this value is such that has a property called `attention_weights` and I
# would not have to use type ignores here and there. For the time being I am
# going to leave it as it is (since it is not wrong), but this needs to be
# revisited


class FeatureImportance:
    def __init__(self, device: str, n_samples: int = 1000):
        self.device = device
        self.n_samples = n_samples

    def feature_importance(
        self, loader: DataLoader, model: WideDeep
    ) -> Dict[str, float]:
        if model.is_tabnet:
            model_feat_importance: ModelFeatureImportance = TabNetFeatureImportance(
                self.device, self.n_samples
            )
        elif isinstance(model.deeptabular._modules["0"], TransformerBasedModels):
            model_feat_importance = TransformerBasedFeatureImportance(
                self.device, self.n_samples
            )
        else:
            raise ValueError(
                "The computation of feature importance is not supported for this particular model"
            )

        return model_feat_importance.feature_importance(loader, model)


class Explainer:
    def __init__(self, device: str):
        self.device = device

    def explain(
        self,
        model: WideDeep,
        X_tab: np.ndarray,
        num_workers: int,
        batch_size: Optional[int] = None,
        save_step_masks: Optional[bool] = None,
    ) -> Union[Tuple, np.ndarray]:
        if model.is_tabnet:
            assert (
                save_step_masks is not None
            ), "If the model is TabNet, please set 'save_step_masks' to True/False"
            model_explainer: ModelExplainer = TabNetExplainer(self.device)
            res = model_explainer.explain(
                model,
                X_tab,
                num_workers,
                batch_size,
                save_step_masks,
            )
        elif isinstance(model.deeptabular._modules["0"], TransformerBasedModels):
            model_explainer = TransformerBasedExplainer(self.device)
            res = model_explainer.explain(
                model,
                X_tab,
                num_workers,
                batch_size,
            )
        else:
            raise ValueError(
                "The computation of feature importance is not supported for this particular model"
            )

        return res


class BaseFeatureImportance(ABC):
    def __init__(self, device: str, n_samples: int = 1000):
        self.device = device
        self.n_samples = n_samples

    @abstractmethod
    def feature_importance(
        self, loader: DataLoader, model: WideDeep
    ) -> Dict[str, float]:
        raise NotImplementedError(
            "Any Feature Importance technique must implement this method"
        )

    def _sample_data(self, loader: DataLoader) -> Tensor:
        n_iterations = self.n_samples // loader.batch_size

        batches = []
        for i, (data, _, _) in enumerate(loader):
            if i < n_iterations:
                batches.append(data["deeptabular"].to(self.device))
            else:
                break

        return torch.cat(batches, dim=0)


class TabNetFeatureImportance(BaseFeatureImportance):
    def __init__(self, device: str, n_samples: int = 1000):
        super().__init__(
            device=device,
            n_samples=n_samples,
        )

    def feature_importance(
        self, loader: DataLoader, model: WideDeep
    ) -> Dict[str, float]:
        model.eval()

        reducing_matrix = create_explain_matrix(model)
        model_backbone = list(model.deeptabular.children())[0]
        feat_imp = np.zeros((model_backbone.embed_out_dim))  # type: ignore[arg-type]

        X = self._sample_data(loader)
        M_explain, _ = model_backbone.forward_masks(X)  # type: ignore[operator]
        feat_imp += M_explain.sum(dim=0).cpu().detach().numpy()
        feat_imp = csc_matrix.dot(feat_imp, reducing_matrix)
        feat_imp = feat_imp / np.sum(feat_imp)

        return {k: v for k, v in zip(model_backbone.column_idx.keys(), feat_imp)}  # type: ignore


class TransformerBasedFeatureImportance(BaseFeatureImportance):
    def __init__(self, device: str, n_samples: int = 1000):
        super().__init__(
            device=device,
            n_samples=n_samples,
        )

    def feature_importance(
        self, loader: DataLoader, model: WideDeep
    ) -> Dict[str, float]:
        self._check_inputs(model)
        self.model_type = self._model_type(model)

        X = self._sample_data(loader)

        model.eval()
        _ = model.deeptabular(X)

        feature_importance, column_idx = self._feature_importance(model)

        agg_feature_importance = feature_importance.mean(0).cpu().detach().numpy()

        return {k: v for k, v in zip(column_idx, agg_feature_importance)}

    def _model_type_attention_weights(self, model: WideDeep) -> Tensor:
        if self.model_type == "saint":
            attention_weights = torch.stack(
                [aw[0] for aw in model.deeptabular[0].attention_weights],  # type: ignore[index]
                dim=0,
            )
        elif self.model_type == "tabfastformer":
            alpha_weights, beta_weights = zip(*model.deeptabular[0].attention_weights)  # type: ignore[index]
            attention_weights = torch.stack(alpha_weights + beta_weights, dim=0)
        else:
            attention_weights = torch.stack(
                model.deeptabular[0].attention_weights, dim=0  # type: ignore[index]
            )

        return attention_weights

    def _model_type_feature_importance(
        self, model: WideDeep, attention_weights: Tensor
    ) -> Tuple[Tensor, List[str]]:
        model_backbone = list(model.deeptabular.children())[0]
        with_cls_token = model.deeptabular._modules["0"].with_cls_token

        column_idx = (
            list(model_backbone.column_idx.keys())[1:]  # type: ignore
            if with_cls_token
            else list(model_backbone.column_idx.keys())  # type: ignore
        )

        if self.model_type == "contextattentionmlp":
            feat_imp = (
                attention_weights.mean(0)
                if not with_cls_token
                else attention_weights.mean(0)[:, 1:]
            )
        elif self.model_type == "tabfastformer":
            feat_imp = (
                attention_weights.mean(0).mean(1)
                if not with_cls_token
                else attention_weights.mean(0).mean(1)[:, 1:]
            )
        else:
            feat_imp = (
                attention_weights.mean(0).mean(1).mean(1)
                if not with_cls_token
                else attention_weights.mean(0).mean(1)[:, 0, 1:]
            )

        return feat_imp, column_idx

    @staticmethod
    def _model_type(model: WideDeep) -> str:
        if isinstance(model.deeptabular._modules["0"], SAINT):
            model_type = "saint"
        if isinstance(model.deeptabular._modules["0"], FTTransformer):
            model_type = "fttransformer"
        if isinstance(model.deeptabular._modules["0"], TabFastFormer):
            model_type = "tabfastformer"
        if isinstance(model.deeptabular._modules["0"], TabTransformer):
            model_type = "tabtransformer"
        if isinstance(model.deeptabular._modules["0"], SelfAttentionMLP):
            model_type = "selfattentionmlp"
        if isinstance(model.deeptabular._modules["0"], ContextAttentionMLP):
            model_type = "contextattentionmlp"

        return model_type

    def _feature_importance(self, model: WideDeep) -> Tuple[Tensor, List[str]]:
        attention_weights = self._model_type_attention_weights(model)

        feature_importance, column_idx = self._model_type_feature_importance(
            model, attention_weights
        )

        return feature_importance, column_idx

    def _check_inputs(self, model: WideDeep):
        if isinstance(model.deeptabular._modules["0"], TabPerceiver):
            raise ValueError(
                "At this stage the feature importance is not supported for the 'TabPerceiver'"
            )
        if isinstance(model.deeptabular._modules["0"], FTTransformer) and (
            model.deeptabular._modules["0"].kv_compression_factor != 1
        ):
            raise ValueError(
                "Feature importance can only be computed if the compression factor "
                "'kv_compression_factor' is set to 1"
            )


class TabNetExplainer:
    def __init__(self, device: str, n_samples: int = 1000):
        self.device = device

    @alias("X_tab", ["X"])
    def explain(
        self,
        model: WideDeep,
        X_tab: np.ndarray,
        num_workers: int,
        batch_size: Optional[int] = None,
        save_step_masks: bool = False,
    ) -> Union[Tuple, np.ndarray]:
        model.eval()
        model_backbone = list(model.deeptabular.children())[0]
        reducing_matrix = create_explain_matrix(model)

        loader = DataLoader(
            dataset=WideDeepDataset(X_tab=X_tab),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        m_explain_l = []
        for batch_nb, data in enumerate(loader):
            X = data["deeptabular"].to(self.device)
            M_explain, masks = model_backbone.forward_masks(X)  # type: ignore[operator]
            m_explain_l.append(
                csc_matrix.dot(M_explain.cpu().detach().numpy(), reducing_matrix)
            )
            if save_step_masks:
                for key, value in masks.items():
                    masks[key] = csc_matrix.dot(
                        value.cpu().detach().numpy(), reducing_matrix
                    )
                if batch_nb == 0:
                    m_explain_step = masks
                else:
                    for key, value in masks.items():
                        m_explain_step[key] = np.vstack([m_explain_step[key], value])

        m_explain_agg = np.vstack(m_explain_l)
        m_explain_agg_norm = m_explain_agg / m_explain_agg.sum(axis=1)[:, np.newaxis]

        res: Union[Tuple, np.ndarray] = (
            (m_explain_agg_norm, m_explain_step)
            if save_step_masks
            else np.vstack(m_explain_agg_norm)
        )

        return res


class TransformerBasedExplainer(TransformerBasedFeatureImportance):
    def __init__(self, device: str):
        super().__init__(
            device=device,
            n_samples=1000,  # irrelevant
        )

    @alias("X_tab", ["X"])
    def explain(
        self,
        model: WideDeep,
        X_tab: np.ndarray,
        num_workers: int,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        model.eval()

        self.model_type = self._model_type(model)

        loader = DataLoader(
            dataset=WideDeepDataset(X_tab=X_tab),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        batch_feat_imp: Any = []
        for _, data in enumerate(loader):
            X = data["deeptabular"].to(self.device)
            _ = model.deeptabular(X)

            feat_imp, _ = self._feature_importance(model)

            batch_feat_imp.append(feat_imp)

        return torch.cat(batch_feat_imp).cpu().detach().numpy()


ModelFeatureImportance = Union[
    TabNetFeatureImportance, TransformerBasedFeatureImportance
]
ModelExplainer = Union[TabNetExplainer, TransformerBasedExplainer]
