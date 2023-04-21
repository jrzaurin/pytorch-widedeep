from typing import Tuple, Union, Optional

import numpy as np
import torch
from scipy.sparse import csc_matrix
from torch.utils.data import DataLoader

from pytorch_widedeep.models.tabular import (
    SAINT,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
    SelfAttentionMLP,
    ContextAttentionMLP,
)
from pytorch_widedeep.utils.general_utils import Alias
from pytorch_widedeep.training._wd_dataset import WideDeepDataset
from pytorch_widedeep.models.tabular.tabnet._utils import create_explain_matrix


class FeatureImportance:
    def __init__(self, device, n_samples=1000):
        self.device = device
        self.n_samples = n_samples

    def feature_importance(self, loader, model):
        self._check_inputs(model)
        self.model_type = self._model_type(model)

        X = self._sample_data(loader)

        model.eval()
        _ = model.deeptabular(X)

        attention_weights = self._model_type_attention_weights(model)

        feature_importance, column_idx = self._model_type_feature_importance(
            model, attention_weights
        )

        return {k: v for k, v in zip(column_idx, feature_importance)}

    @Alias("X_tab", "X")
    def explain(
        self,
        model,
        X_tab,
        num_workers: int,
        batch_size: Optional[int] = None,
    ):
        pass

    def _sample_data(self, loader):
        n_iterations = self.n_samples // loader.batch_size

        batches = []
        for i, (data, _, _) in enumerate(loader):
            if i < n_iterations:
                batches.append(data["deeptabular"].to(self.device))
            else:
                break

        return torch.cat(batches, dim=0)

    def _model_type_attention_weights(self, model):
        if self.model_type == "saint":
            attention_weights = torch.stack(
                [aw[0] for aw in model.deeptabular[0].attention_weights], dim=0
            )
        elif self.model_type == "tabfastformer":
            alpha_weights, beta_weights = zip(*model.deeptabular[0].attention_weights)
            attention_weights = torch.stack(alpha_weights + beta_weights, dim=0)
        else:
            attention_weights = torch.stack(
                model.deeptabular[0].attention_weights, dim=0
            )

        return attention_weights

    def _model_type_feature_importance(self, model, attention_weights):
        model_backbone = list(model.deeptabular.children())[0]
        with_cls_token = model.deeptabular[0].with_cls_token

        column_idx = (
            list(model_backbone.column_idx.keys())[1:]
            if with_cls_token
            else list(model_backbone.column_idx.keys())
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
    def _model_type(model):
        if isinstance(model.deeptabular[0], SAINT):
            model_type = "saint"
        # if isinstance(model.deeptabular[0], TabNet):
        #     model_type = "tabnet"
        if isinstance(model.deeptabular[0], FTTransformer):
            model_type = "fttransformer"
        if isinstance(model.deeptabular[0], TabFastFormer):
            model_type = "tabfastformer"
        if isinstance(model.deeptabular[0], TabTransformer):
            model_type = "tabtransformer"
        if isinstance(model.deeptabular[0], SelfAttentionMLP):
            model_type = "selfattentionmlp"
        if isinstance(model.deeptabular[0], ContextAttentionMLP):
            model_type = "contextattentionmlp"

        return model_type

    def _check_inputs(self, model):
        pass


class TabNetFeatureImportance(object):
    def __init__(self, device, n_samples=1000):
        self.device = device
        self.n_samples = n_samples

    def feature_importance(self, loader, model):
        model.eval()

        reducing_matrix = create_explain_matrix(model)
        model_backbone = list(model.deeptabular.children())[0]
        feat_imp = np.zeros((model_backbone.embed_out_dim))

        X = self._sample_data(loader)
        M_explain, _ = model_backbone.forward_masks(X)
        feat_imp += M_explain.sum(dim=0).cpu().detach().numpy()
        feat_imp = csc_matrix.dot(feat_imp, reducing_matrix)
        feat_imp = feat_imp / np.sum(feat_imp)

        return {k: v for k, v in zip(model_backbone.column_idx.keys(), feat_imp)}

    def _sample_data(self, loader):
        n_iterations = self.n_samples // loader.batch_size

        batches = []
        for i, (data, _, _) in enumerate(loader):
            if i < n_iterations:
                batches.append(data["deeptabular"].to(self.device))
            else:
                break

        return torch.cat(batches, dim=0)

    @Alias("X_tab", "X")
    def _explain(
        self,
        model,
        X_tab,
        num_workers: int,
        batch_size: Optional[int] = None,
        save_step_masks: bool = False,
    ):
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
            M_explain, masks = model_backbone.forward_masks(X)
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
        res = np.vstack(m_explain_agg_norm)

        return res
