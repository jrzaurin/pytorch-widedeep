import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm,trange
from sklearn.utils import Bunch

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau

from ..wdtypes import *

from .wide import Wide
from .deep_dense import DeepDense
from .deep_text import DeepText
from .deep_image import DeepImage

from .radam import RAdam

import pdb

use_cuda = torch.cuda.is_available()


def set_method(method:str):
    if method =='regression':
        return None, F.mse_loss
    if method =='logistic':
        return torch.sigmoid, F.binary_cross_entropy
    if method=='multiclass':
        return F.softmax, F.cross_entropy


def set_optimizer(model_params:ModelParams, opt_name, opt_params:Optional[Dict]=None):
    if opt_name == "RAdam":
        return RAdam(model_params, **opt_params) if opt_params else RAdam(model_params)
    if opt_name == "Adam":
        return torch.optim.Adam(model_params, **opt_params) if opt_params else torch.optim.Adam(model_params)
    if opt_name == "SGD":
        return torch.optim.SGD(model_params, **opt_params) if opt_params else torch.optim.SGD(model_params)


def set_scheduler(optimizer:Optimizer, sch_name:str, sch_params:Optional[Dict]=None):
    if sch_name == "StepLR":
        return StepLR(optimizer, **sch_params) if sch_params else StepLR(optimizer)
    if sch_name == "MultiStepLR":
        return MultiStepLR(optimizer, **sch_params) if sch_params else MultiStepLR(optimizer)
    if sch_name == "ExponentialLR":
        return ExponentialLR(optimizer, **sch_params) if sch_params else ExponentialLR(optimizer)
    if sch_name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, **sch_params) if sch_params else ReduceLROnPlateau(optimizer)


class EarlyStopping:
    def __init__(self, patience:int=10, delta:float=0.):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss:Tensor):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class MultipleOptimizer(object):
    def __init__(self, opts:List[Optimizer]):
        self.optimizers = opts

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class MultipleLRScheduler(object):
    def __init__(self, scheds:List[LRScheduler]):
        self.schedulers = scheds

    def step(self):
        for sc in self.schedulers:
            sc.step()


class WideDeepLoader(Dataset):
    def __init__(self, X_wide:np.ndarray, X_deep_dense:np.ndarray,
        X_deep_text:Optional[np.ndarray]=None,
        X_deep_img:Optional[np.ndarray]=None,
        target:Optional[np.ndarray]=None, transform:Optional=None):

        self.X_wide = X_wide
        self.X_deep_dense = X_deep_dense
        self.X_deep_text = X_deep_text
        self.X_deep_img = X_deep_img
        self.transform = transform
        self.Y = target

    def __getitem__(self, idx:int):

        xw = self.X_wide[idx]
        X = Bunch(wide=xw)
        xdd = self.X_deep_dense[idx]
        X.deep_dense= xdd
        if self.X_deep_text is not None:
            xdt = self.X_deep_text[idx]
            X.deep_text = xdt
        if self.X_deep_img is not None:
            xdi = (self.X_deep_img[idx]/255).astype('float32')
            if self.transform is not None:
                xdi = self.transform(xdi)
            X.deep_img = xdi
        if self.Y is not None:
            y  = self.Y[idx]
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X_wide)


class WideDeep(nn.Module):

    def __init__(self, output_dim:int, wide_dim:int, embeddings_input:List[Tuple[str,int,int]],
        embeddings_encoding_dict:Dict[str,Any], continuous_cols:List[str],
        deep_column_idx:Dict[str,int], hidden_layers:List[int]=[64,32],
        dropout:List[float]=[0.], nlp_model:Optional[nn.Module]=None,
        vocab_size:Optional[int]=None, word_embedding_dim:Optional[int]=64,
        rnn_hidden_dim:Optional[int]=32, rnn_n_layers:Optional[int]=3,
        rnn_dropout:Optional[float]=0.,emb_spatial_dropout:Optional[float]=0.,
        padding_idx:Optional[int]=1, bidirectional:Optional[bool]=False,
        embedding_matrix:Optional[np.ndarray]=None,
        vision_model:Optional[nn.Module]=None, pretrained:Optional[bool]=True,
        resnet:Optional[int]=18, freeze:Optional[Union[str,int]]=6):

        super(WideDeep, self).__init__()

        self.output_dim = output_dim

        # WIDE
        self.wide_dim = wide_dim
        self.wide = Wide(self.wide_dim, self.output_dim)

        # DEEP DENSE
        self.embeddings_input = embeddings_input
        self.embeddings_encoding_dict = embeddings_encoding_dict
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.deep_dense = DeepDense( self.embeddings_input, self.embeddings_encoding_dict,
            self.continuous_cols, self.deep_column_idx, self.hidden_layers, self.dropout,
            self.output_dim)

        # DEEP TEXT
        if nlp_model is not None:
            self.deep_text = nlp_model
        else:
            self.vocab_size = vocab_size
            self.word_embedding_dim = word_embedding_dim
            self.rnn_hidden_dim = rnn_hidden_dim
            self.rnn_n_layers = rnn_n_layers
            self.rnn_dropout = rnn_dropout
            self.emb_spatial_dropout = emb_spatial_dropout
            self.padding_idx = padding_idx
            self.bidirectional = bidirectional
            self.embedding_matrix = embedding_matrix
            self.deep_text = DeepText(self.vocab_size, self.word_embedding_dim, self.rnn_hidden_dim,
                self.rnn_n_layers, self.rnn_dropout, self.emb_spatial_dropout, self.padding_idx,
                self.output_dim, self.bidirectional, self.embedding_matrix)

        # DEEP IMAGE
        if vision_model is not None:
            self.deep_img = vision_model
        else:
            self.pretrained = pretrained
            self.resnet = resnet
            self.freeze = freeze
            self.deep_img = DeepImage(self.output_dim, self.pretrained, self.resnet,
                self.freeze)

    def compile(self, method:str, optimizer:str=None, optimizer_param:Optional[Dict]=None,
        lr_scheduler:Optional[str]=None, lr_scheduler_params:Optional[Dict]=None,
        wide_optimizer:Optional[str]=None, wide_optimizer_param:Optional[Dict]=None,
        deep_dense_optimizer:Optional[str]=None, deep_dense_optimizer_param:Optional[Dict]=None,
        deep_text_optimizer:Optional[str]=None, deep_text_optimizer_param:Optional[Dict]=None,
        deep_img_optimizer:Optional[str]=None, deep_img_optimizer_param:Optional[Dict]=None,
        wide_scheduler:Optional[str]=None, wide_scheduler_param:Optional[Dict]=None,
        deep_dense_scheduler:Optional[str]=None, deep_dense_scheduler_param:Optional[Dict]=None,
        deep_text_scheduler:Optional[str]=None, deep_text_scheduler_param:Optional[Dict]=None,
        deep_img_scheduler:Optional[str]=None, deep_img_scheduler_param:Optional[Dict]=None):

        self.method = method
        self.activation, self.criterion = set_method(method)

        if optimizer is not None:
            self.optimizer = set_optimizer(self.parameters(), optimizer, optimizer_param)
            self.lr_scheduler = set_scheduler(self.optimizer, lr_scheduler, lr_scheduler_params)
        else:
            if wide_optimizer is None:
                wide_opt = set_optimizer(self.wide.parameters(), "Adam")
            else:
                wide_opt = set_optimizer(self.wide.parameters(), wide_optimizer, wide_optimizer_param)
            wide_sch = set_scheduler(self.wide_opt, wide_scheduler, wide_scheduler_param) if wide_scheduler else None
            if deep_dense_optimizer is None:
                deep_dense_opt = set_optimizer(self.deep_dense.parameters(), "Adam")
            else:
                deep_dense_opt = set_optimizer(self.deep_dense.parameters(), deep_dense_optimizer, deep_dense_optimizer_param)
            deep_dense_sch = set_scheduler(self.deep_dense_opt, deep_dense_scheduler, deep_dense_scheduler_param) if deep_dense_scheduler else None
            if deep_text_optimizer is None:
                deep_text_opt = set_optimizer(self.deep_text.parameters(), "Adam")
            else:
                deep_text_opt = set_optimizer(self.deep_text.parameters(), deep_text_optimizer, deep_text_optimizer_param)
            deep_text_sch = set_scheduler(self.deep_text_opt, deep_text_scheduler, deep_text_scheduler_param) if deep_text_scheduler else None
            if deep_img_optimizer is None:
                deep_img_opt = set_optimizer(self.deep_img.parameters(), "Adam")
            else:
                deep_img_opt = set_optimizer(self.deep_img.parameters(), deep_img_optimizer, deep_img_optimizer_param)
            deep_img_sch = set_scheduler(self.deep_img_opt, deep_img_scheduler, deep_img_scheduler_param) if deep_img_scheduler else None

            self.optimizer = MultipleOptimizer([wide_opt, deep_dense_opt, deep_text_opt, deep_img_opt])
            self.lr_scheduler = MultipleLRScheduler([wide_sch, deep_dense_sch, deep_text_sch, deep_img_sch])

    def forward(self, X:Tuple[Dict[str,Tensor],Tensor])->Tensor:

        wide_deep = self.wide(X['wide'])
        wide_deep.add_(self.deep_dense(X['deep_dense']))

        if 'deep_text' in X.keys():
            wide_deep.add_(self.deep_text(X['deep_text']))

        if 'deep_img' in X.keys():
            wide_deep.add_(self.deep_img(X['deep_img']))

        if not self.activation:
            return wide_deep
        else:
            if (self.activation==F.softmax):
                out = self.activation(wide_deep, dim=1)
            else:
                out = self.activation(wide_deep)
            return out

    def training_step(self, data:Dict[str, Tensor], target:Tensor, batch_nb:int):

        X = {k:v.cuda() for k,v in data.keys()} if use_cuda else data
        y = target.float() if self.method != 'multiclass' else target
        y = y.cuda() if use_cuda else y

        self.optimizer.zero_grad()
        y_pred =  self.forward(X)
        if(self.criterion == F.cross_entropy):
            loss = self.criterion(y_pred, y)
        else:
            loss = self.criterion(y_pred, y.view(-1, 1))
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss/(batch_nb+1)
        if self.method != "regression":
            total+= y.size(0)
            if self.method == 'logistic':
                y_pred_cat = (y_pred > 0.5).squeeze(1).float()
            if self.method == "multiclass":
                _, y_pred_cat = torch.max(y_pred, 1)
            correct+= float((y_pred_cat == y).sum().item())
            acc = correct/total

        if self.method != 'regression':
            return avg_loss
        else:
            return acc, avg_loss

    def validation_step(self, data:Dict[str, Tensor], target:Tensor, batch_nb:int):

        with torch.no_grad():
            X = {k:v.cuda() for k,v in data.keys()} if use_cuda else data
            y = target.float() if self.method != 'multiclass' else target
            y = y.cuda() if use_cuda else y

            y_pred =  self.forward(X)
            if(self.criterion == F.cross_entropy):
                loss = self.criterion(y_pred, y)
            else:
                loss = self.criterion(y_pred, y.view(-1, 1))
            running_loss += loss.item()
            avg_loss = running_loss/(i+1)
            if self.method != "regression":
                total+= y.size(0)
                if self.method == 'logistic':
                    y_pred_cat = (y_pred > 0.5).squeeze(1).float()
                if self.method == "multiclass":
                    _, y_pred_cat = torch.max(y_pred, 1)
                correct+= float((y_pred_cat == y).sum().item())

        if self.method != 'regression':
            return avg_loss
        else:
            return acc, avg_loss

    def fit(self, n_epochs:int, train_loader:DataLoader, eval_loader:Optional[DataLoader]=None,
        patience:Optional[int]=10):

        train_steps =  (len(train_loader.dataset) // train_loader.batch_size) + 1
        if eval_loader:
            early_stopping = EarlyStopping(patience=patience)
            eval_steps =  (len(eval_loader.dataset) // eval_loader.batch_size) + 1
        for epoch in range(n_epochs):

            total, correct, running_loss = 0,0,0
            with trange(train_steps) as t:
                for i, (data,target) in zip(t, train_loader):
                    t.set_description('epoch %i' % (epoch+1))
                    if self.method != 'regression':
                        acc, avg_loss = self.training_step(data, target, i)
                        t.set_postfix(acc=correct/total, loss=avg_loss)
                    else:
                        avg_loss = self.training_step(data, target, i)
                        t.set_postfix(loss=np.sqrt(avg_loss))

            if eval_loader:
                current_best_loss, stopping_step, should_stop = 1e3, 0, False
                total, correct, running_loss = 0,0,0
                with trange(eval_steps) as v:
                    for i, (data,target) in zip(v, eval_loader):
                        v.set_description('valid')
                        if self.method != 'regression':
                            acc, avg_loss = self.validation_step(data, target, i)
                            v.set_postfix(acc=correct/total, loss=avg_loss)
                        else:
                            avg_loss = self.validation_step(data, target, i)
                            v.set_postfix(loss=np.sqrt(avg_loss))
                early_stopping(avg_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if self.lr_scheduler:
                if isinstance(scheduler, ReduceLROnPlateau): self.lr_scheduler.step(avg_loss)
            else:
                self.lr_scheduler.step()

    def predict(self, dataloader:DataLoader)->np.ndarray:
        test_steps =  (len(dataloader.dataset) // dataloader.batch_size) + 1
        net = self.eval()
        preds_l = []
        with torch.no_grad():
            with trange(test_steps) as t:
                for i, data in zip(t, dataloader):
                    t.set_description('predict')
                    X = tuple(x.cuda() for x in data) if use_cuda else data
                    # This operations is cheap in terms of computing time, but
                    # would be more efficient to append Tensors and then cat
                    preds_l.append(net(X).cpu().data.numpy())
            if self.method == "regression":
                return np.vstack(preds_l).squeeze(1)
            if self.method == "logistic":
                preds = np.vstack(preds_l).squeeze(1)
                return (preds > 0.5).astype('int')
            if self.method == "multiclass":
                preds = np.vstack(preds_l)
                return np.argmax(preds, 1)

    def predict_proba(self, dataloader:DataLoader)->np.ndarray:
        test_steps =  (len(dataloader.dataset) // dataloader.batch_size) + 1
        net = self.eval()
        preds_l = []
        with torch.no_grad():
            with trange(test_steps) as t:
                for i, data in zip(t, dataloader):
                    t.set_description('predict')
                    X = tuple(x.cuda() for x in data) if use_cuda else data
                    preds_l.append(net(X).cpu().data.numpy())
            if self.method == "logistic":
                preds = np.vstack(preds_l).squeeze(1)
                probs = np.zeros([preds.shape[0],2])
                probs[:,0] = 1-preds
                probs[:,1] = preds
                return probs
            if self.method == "multiclass":
                return np.vstack(preds_l)

    def get_embeddings(self, col_name:str) -> Dict[str,np.ndarray]:
        """Extract the embeddings for the embedding columns.

        Parameters:
        -----------
        col_name: str
            column we want to extract the embedding for

        Returns:
        --------
        embeddings_dict: Dict
            Dict with the column values and the corresponding embeddings
        """

        params = list(self.named_parameters())
        emb_layers = [p for p in params if 'emb_layer' in p[0]]
        emb_layer  = [layer for layer in emb_layers if col_name in layer[0]][0]
        embeddings = emb_layer[1].cpu().data.numpy()
        col_label_encoding = self.embeddings_encoding_dict[col_name]
        inv_dict = {v:k for k,v in col_label_encoding.items()}
        embeddings_dict = {}
        for idx,value in inv_dict.items():
            embeddings_dict[value] = embeddings[idx]
        return embeddings_dict