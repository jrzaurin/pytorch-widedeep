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
from ..radam import RAdam

from ..initializers import MultipleInitializers
from ..optimizers import MultipleOptimizers
from ..lr_schedulers import MultipleLRScheduler

from .wide import Wide
from .deep_dense import DeepDense
from .deep_text import DeepText
from .deep_image import DeepImage

from .callbacks import History, CallbackContainer
from .metrics import MultipleMetrics, MetricCallback

import pdb

use_cuda = torch.cuda.is_available()


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

    def set_method(self, method:str):
        self.method = method
        if self.method =='regression':
            self.activation, self.criterion = None, F.mse_loss
        if self.method =='logistic':
            self.activation, self.criterion = torch.sigmoid, F.binary_cross_entropy
        if self.method=='multiclass':
            self.activation, self.criterion = F.softmax, F.cross_entropy

    def compile(self, method, callbacks=None, initializers=None, optimizers=None, lr_schedulers=None,
        metrics=None, global_optimizer=None, global_optimizer_params=None, global_lr_scheduler=None,
        global_lr_scheduler_params=None):

        self.set_method(method)

        if initializers is not None:
            self.initializer = MultipleInitializers(initializers)
            self.initializer.apply(self)

        if optimizers is not None:
            self.optimizer = MultipleOptimizers(optimizers)
            self.optimizer.apply(self)
        elif global_optimizer is not None:
            self.optimizer = global_optimizer(self)
        else:
            print('bla bla...')
            self.optimizer = torch.optim.Adam(self.parameters())

        if lr_schedulers is not None:
            self.lr_scheduler = MultipleLRScheduler(lr_schedulers)
            self.lr_scheduler.apply(self.optimizer._optimizers)
        elif global_lr_scheduler is not None:
            self.lr_scheduler = global_lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None

        self.history = History
        self.callbacks = [self.history]
        if callbacks is not None:
            self.callbacks.append(callbacks)

        if metrics is not None:
            self.metric = MultipleMetrics(metrics)
            self.callbacks.append(MetricCallback(self.metric))

        callback_container = CallbackContainer(self.callbacks)

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

        self.running_loss += loss.item()
        avg_loss = self.running_loss/(batch_nb+1)
        if self.method != "regression":
            self.total+= y.size(0)
            if self.method == 'logistic':
                y_pred_cat = (y_pred > 0.5).squeeze(1).float()
            if self.method == "multiclass":
                _, y_pred_cat = torch.max(y_pred, 1)
            self.correct+= float((y_pred_cat == y).sum().item())
            acc = self.correct/self.total

        if self.method != 'regression':
            return acc, avg_loss
        else:
            return avg_loss

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
            self.running_loss += loss.item()
            avg_loss = self.running_loss/(batch_nb+1)
            if self.method != "regression":
                self.total+= y.size(0)
                if self.method == 'logistic':
                    y_pred_cat = (y_pred > 0.5).squeeze(1).float()
                if self.method == "multiclass":
                    _, y_pred_cat = torch.max(y_pred, 1)
                self.correct+= float((y_pred_cat == y).sum().item())
                acc = self.correct/self.total

        if self.method != 'regression':
            return acc, avg_loss
        else:
            return avg_loss


    def fit(self, n_epochs:int, train_loader:DataLoader, eval_loader:Optional[DataLoader]=None,
        patience:Optional[int]=10):

        train_steps =  (len(train_loader.dataset) // train_loader.batch_size) + 1
        if eval_loader:
            eval_steps =  (len(eval_loader.dataset) // eval_loader.batch_size) + 1
        for epoch in range(n_epochs):

            self.total, self.correct, self.running_loss = 0,0,0
            with trange(train_steps) as t:
                for i, (data,target) in zip(t, train_loader):
                    t.set_description('epoch %i' % (epoch+1))
                    if self.method != 'regression':
                        acc, avg_loss = self.training_step(data, target, i)
                        t.set_postfix(acc=acc, loss=avg_loss)
                    else:
                        avg_loss = self.training_step(data, target, i)
                        t.set_postfix(loss=np.sqrt(avg_loss))

            if eval_loader:
                self.total, self.correct, self.running_loss = 0,0,0
                current_best_loss, stopping_step, should_stop = 1e3, 0, False
                with trange(eval_steps) as v:
                    for i, (data,target) in zip(v, eval_loader):
                        v.set_description('valid')
                        if self.method != 'regression':
                            acc, avg_loss = self.validation_step(data, target, i)
                            v.set_postfix(acc=self.correct/self.total, loss=avg_loss)
                        else:
                            avg_loss = self.validation_step(data, target, i)
                            v.set_postfix(loss=np.sqrt(avg_loss))

            if self.lr_scheduler: self.lr_scheduler.step()

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