import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..wdtypes import *

from ..initializers import Initializer, MultipleInitializer
from ..callbacks import Callback, History, CallbackContainer
from ..metrics import Metric, MultipleMetrics, MetricCallback
from ..losses import FocalLoss

from ._wd_dataset import WideDeepDataset
from ._multiple_optimizer import MultipleOptimizer
from ._multiple_lr_scheduler import MultipleLRScheduler
from ._multiple_transforms import MultipleTransforms
from .deep_dense import dense_layer

from tqdm import tqdm,trange
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import pdb


use_cuda = torch.cuda.is_available()


class WideDeep(nn.Module):

    def __init__(self,
        wide:nn.Module,
        deepdense:nn.Module,
        output_dim:int=1,
        deeptext:Optional[nn.Module]=None,
        deepimage:Optional[nn.Module]=None,
        deephead:Optional[nn.Module]=None,
        head_layers:Optional[List]=None,
        head_dropout:Optional[List]=None,
        head_batchnorm:Optional[bool]=None):

        super(WideDeep, self).__init__()

        # The main 5 components of the wide and deep assemble
        self.wide = wide
        self.deepdense = deepdense
        self.deeptext  = deeptext
        self.deepimage = deepimage
        self.deephead = deephead

        if self.deephead is None:
            if head_layers is not None:
                input_dim = self.deepdense.output_dim + self.deeptext.output_dim + self.deepimage.output_dim
                head_layers = [input_dim] + head_layers
                if not head_dropout: head_dropout = [0.] * (len(head_layers)-1)
                self.deephead = nn.Sequential()
                for i in range(1, len(head_layers)):
                    self.deephead.add_module(
                        'head_layer_{}'.format(i-1),
                        dense_layer( head_layers[i-1], head_layers[i], head_dropout[i-1], head_batchnorm))
                self.deephead.add_module('head_out', nn.Linear(head_layers[-1], output_dim))
            else:
                self.deepdense = nn.Sequential(
                    self.deepdense,
                    nn.Linear(self.deepdense.output_dim, output_dim))
                if self.deeptext is not None:
                    self.deeptext = nn.Sequential(
                        self.deeptext,
                        nn.Linear(self.deeptext.output_dim, output_dim))
                if self.deepimage is not None:
                    self.deepimage = nn.Sequential(
                        self.deepimage,
                        nn.Linear(self.deepimage.output_dim, output_dim))

    def forward(self, X:List[Dict[str,Tensor]])->Tensor:
        # Wide output: direct connection to the output neuron(s)
        out = self.wide(X['wide'])

        # Deep output: either connected directly to the output neuron(s) or
        # passed through a head first
        if self.deephead:
            deepside = self.deepdense(X['deepdense'])
            if self.deeptext is not None:
                deepside = torch.cat( [deepside, self.deeptext(X['deeptext'])], axis=1 )
            if self.deepimage is not None:
                deepside = torch.cat( [deepside, self.deepimage(X['deepimage'])], axis=1 )
            deepside_out = self.head(deepside)
            return out.add_(deepside_out)
        else:
            out.add_(self.deepdense(X['deepdense']))
            if self.deeptext is not None:
                out.add_(self.deeptext(X['deeptext']))
            if self.deepimage is not None:
                out.add_(self.deepimage(X['deepimage']))
            return out

    def compile(self,
        method:str,
        initializers:Optional[Dict[str,Initializer]]=None,
        optimizers:Optional[Dict[str,Optimizer]]=None,
        global_optimizer:Optional[Optimizer]=None,
        lr_schedulers:Optional[Dict[str,LRScheduler]]=None,
        global_lr_scheduler:Optional[LRScheduler]=None,
        transforms:Optional[List[Transforms]]=None,
        callbacks:Optional[List[Callback]]=None,
        metrics:Optional[List[Metric]]=None,
        class_weight:Optional[Union[float,List[float],Tuple[float]]]=None,
        with_focal_loss:bool=False,
        alpha:float=0.25,
        gamma:float=1,
        verbose=1):

        self.verbose = verbose
        self.early_stop = False
        self.method = method
        self.with_focal_loss = with_focal_loss
        if self.with_focal_loss:
            self.alpha, self.gamma = alpha, gamma

        if isinstance(class_weight, float):
            self.class_weight = torch.tensor([class_weight, 1.-class_weight])
        elif isinstance(class_weight, (List, Tuple)):
            self.class_weight =  torch.tensor(class_weight)
        else:
            self.class_weight = None

        if initializers is not None:
            self.initializer = MultipleInitializer(initializers, verbose=self.verbose)
            self.initializer.apply(self)

        if optimizers is not None:
            self.optimizer = MultipleOptimizer(optimizers)
        elif global_optimizer is not None:
            self.optimizer = global_optimizer
        else:
            self.optimizer = torch.optim.Adam(self.parameters())

        if lr_schedulers is not None:
            self.lr_scheduler = MultipleLRScheduler(lr_schedulers)
            scheduler_names = [sc.__class__.__name__.lower() for _,sc in self.lr_scheduler._schedulers.items()]
            self.cyclic = any(['cycl' in sn for sn in scheduler_names])
        elif global_lr_scheduler is not None:
            self.lr_scheduler = global_lr_scheduler
            self.cyclic = 'cycl' in self.lr_scheduler.__class__.__name__.lower()
        else:
            self.lr_scheduler, self.cyclic = None, False

        if transforms is not None:
            self.transforms = MultipleTransforms(transforms)()
        else:
            self.transforms = None

        self.history = History()
        self.callbacks = [self.history]
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, type): callback = callback()
                self.callbacks.append(callback)

        if metrics is not None:
            self.metric = MultipleMetrics(metrics)
            self.callbacks += [MetricCallback(self.metric)]
        else:
            self.metric = None

        self.callback_container = CallbackContainer(self.callbacks)
        self.callback_container.set_model(self)

    def _activation_fn(self, inp:Tensor) -> Tensor:
        if self.method == 'regression':
            return inp
        if self.method == 'logistic':
            return torch.sigmoid(inp)
        if self.method == 'multiclass':
            return F.softmax(inp, dim=1)

    def _loss_fn(self, y_pred:Tensor, y_true:Tensor) -> Tensor:
        if self.with_focal_loss:
            return FocalLoss(self.alpha, self.gamma)(y_pred, y_true)
        if self.method == 'regression':
            return F.mse_loss(y_pred, y_true.view(-1, 1))
        if self.method == 'logistic':
            return F.binary_cross_entropy(y_pred, y_true.view(-1, 1), weight=self.class_weight)
        if self.method == 'multiclass':
            return F.cross_entropy(y_pred, y_true, weight=self.class_weight)

    def _training_step(self, data:Dict[str, Tensor], target:Tensor, batch_idx:int):

        X = {k:v.cuda() for k,v in data.items()} if use_cuda else data
        y = target.float() if self.method != 'multiclass' else target
        y = y.cuda() if use_cuda else y

        self.optimizer.zero_grad()
        y_pred =  self._activation_fn(self.forward(X))
        loss = self._loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss/(batch_idx+1)

        if self.metric is not None:
            acc = self.metric(y_pred, y)
            return acc, avg_loss
        else:
            return None, avg_loss

    def _validation_step(self, data:Dict[str, Tensor], target:Tensor, batch_idx:int):

        with torch.no_grad():
            X = {k:v.cuda() for k,v in data.item()} if use_cuda else data
            y = target.float() if self.method != 'multiclass' else target
            y = y.cuda() if use_cuda else y

            y_pred = self._activation_fn(self.forward(X))
            loss = self._loss_fn(y_pred, y)
            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss/(batch_idx+1)

        if self.metric is not None:
            acc = self.metric(y_pred, y)
            return acc, avg_loss
        else:
            return None, avg_loss

    def _lr_scheduler_step(self, step_location:str):

        if self.lr_scheduler.__class__.__name__ == 'MultipleLRScheduler' and self.cyclic:
            if step_location == 'on_batch_end':
                for model_name, scheduler in self.lr_scheduler._schedulers.items():
                    if 'cycl' in scheduler.__class__.__name__.lower(): scheduler.step()
            elif step_location == 'on_epoch_end':
                for scheduler_name, scheduler in self.lr_scheduler._schedulers.items():
                    if 'cycl' not in scheduler.__class__.__name__.lower(): scheduler.step()
        elif self.cyclic:
            if step_location == 'on_batch_end': self.lr_scheduler.step()
            else: pass
        elif self.lr_scheduler.__class__.__name__ == 'MultipleLRScheduler':
            if step_location == 'on_epoch_end': self.lr_scheduler.step()
            else: pass
        elif step_location == 'on_epoch_end': self.lr_scheduler.step()
        else: pass

    def _train_val_split(self,
        X_wide:Optional[np.ndarray]=None,
        X_deep:Optional[np.ndarray]=None,
        X_text:Optional[np.ndarray]=None,
        X_img:Optional[np.ndarray]=None,
        X_train:Optional[Dict[str,np.ndarray]]=None,
        X_val:Optional[Dict[str,np.ndarray]]=None,
        val_split:Optional[float]=None,
        target:Optional[np.ndarray]=None,
        seed:int=1):

        # No evaluation set
        if X_val is None and val_split is None:
            if X_train is not None:
                X_wide, X_deep, target = X_train['X_wide'], X_train['X_deep'], X_train['target']
                if 'X_text' in X_train.keys(): X_text = X_train['X_text']
                if 'X_img' in X_train.keys(): X_img = X_train['X_img']
            X_train={'X_wide': X_wide, 'X_deep': X_deep, 'target': target}
            try: X_train.update({'X_text': X_text})
            except: pass
            try: X_train.update({'X_img': X_img})
            except: pass
            train_set = WideDeepDataset(**X_train, transforms=self.transforms)
            eval_set = None
        else:
            # evaluation set will be used. Either X_val or val_split are not None
            if X_val is not None:
                if X_train is None:
                    X_train = {'X_wide':X_wide, 'X_deep': X_deep, 'target': target}
                    if X_text is not None: X_train.update({'X_text': X_text})
                    if X_img is not None:  X_train.update({'X_img': X_img})
            else:
                if X_train is not None:
                    X_wide, X_deep, target = X_train['X_wide'], X_train['X_deep'], X_train['target']
                    if 'X_text' in X_train.keys(): X_text = X_train['X_text']
                    if 'X_img' in X_train.keys(): X_img = X_train['X_img']
                X_tr_wide, X_val_wide, X_tr_deep, X_val_deep, y_tr, y_val = train_test_split(X_wide,
                    X_deep, target, test_size=val_split, random_state=seed)
                X_train = {'X_wide':X_tr_wide, 'X_deep': X_tr_deep, 'target': y_tr}
                X_val = {'X_wide':X_val_wide, 'X_deep': X_val_deep, 'target': y_val}
                try:
                    X_tr_text, X_val_text = train_test_split(X_text, test_size=val_split,
                        random_state=seed)
                    X_train.update({'X_text': X_tr_text}), X_val.update({'X_text': X_val_text})
                except: pass
                try:
                    X_tr_img, X_val_img = train_test_split(X_img, test_size=val_split,
                        random_state=seed)
                    X_train.update({'X_img': X_tr_img}), X_val.update({'X_img': X_val_img})
                except: pass
            # Train and validation dictionaries have been built
            train_set = WideDeepDataset(**X_train, transforms=self.transforms)
            eval_set = WideDeepDataset(**X_val, transforms=self.transforms)
        return train_set, eval_set

    def fit(self,
        X_wide:Optional[np.ndarray]=None,
        X_deep:Optional[np.ndarray]=None,
        X_text:Optional[np.ndarray]=None,
        X_img:Optional[np.ndarray]=None,
        X_train:Optional[Dict[str,np.ndarray]]=None,
        X_val:Optional[Dict[str,np.ndarray]]=None,
        val_split:Optional[float]=None,
        target:Optional[np.ndarray]=None,
        n_epochs:int=1,
        batch_size:int=32,
        patience:int=10,
        seed:int=1):

        if X_train is None and (X_wide is None or X_deep is None or target is None):
            raise ValueError(
                "training data is missing. Either a dictionary (X_train) with "
                "the training data or at least 3 arrays (X_wide, X_deep, "
                "target) must be passed to the fit method")

        self.batch_size = batch_size
        train_set, eval_set = self._train_val_split(X_wide, X_deep, X_text, X_img,
            X_train, X_val, val_split, target, seed)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=8)
        train_steps =  (len(train_loader.dataset) // batch_size) + 1
        self.callback_container.on_train_begin({'batch_size': batch_size,
            'train_steps': train_steps, 'n_epochs': n_epochs})

        for epoch in range(n_epochs):
            # train step...
            epoch_logs={}
            self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)
            self.train_running_loss = 0.
            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, (data,target) in zip(t, train_loader):
                    t.set_description('epoch %i' % (epoch+1))
                    acc, train_loss = self._training_step(data, target, batch_idx)
                    if acc is not None:
                        t.set_postfix(metrics=acc, loss=train_loss)
                    else:
                        t.set_postfix(loss=np.sqrt(train_loss))
                    if self.lr_scheduler: self._lr_scheduler_step(step_location='on_batch_end')
                    self.callback_container.on_batch_end(batch=batch_idx)
            epoch_logs['train_loss'] = train_loss
            if acc is not None: epoch_logs['train_acc'] = acc['acc']
            # eval step...
            if eval_set is not None:
                eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size, num_workers=8,
                    shuffle=False)
                eval_steps =  (len(eval_loader.dataset) // batch_size) + 1
                self.valid_running_loss = 0.
                with trange(eval_steps, disable=self.verbose != 1) as v:
                    for i, (data,target) in zip(v, eval_loader):
                        v.set_description('valid')
                        acc, val_loss = self._validation_step(data, target, i)
                        if acc is not None:
                            v.set_postfix(metrics=acc, loss=val_loss)
                        else:
                            v.set_postfix(loss=np.sqrt(val_loss))
                epoch_logs['val_loss'] = val_loss
                if acc is not None: epoch_logs['val_acc'] = acc['acc']
            if self.lr_scheduler: self._lr_scheduler_step(step_location='on_epoch_end')
            # log and check if early_stop...
            self.callback_container.on_epoch_end(epoch, epoch_logs)
            if self.early_stop:
                self.callback_container.on_train_end(epoch)
                break
            self.callback_container.on_train_end(epoch)

    def predict(self, X_wide:np.ndarray, X_deep:np.ndarray, X_text:Optional[np.ndarray]=None,
        X_img:Optional[np.ndarray]=None, X_test:Optional[Dict[str, np.ndarray]]=None)->np.ndarray:

        if X_test is not None:
            test_set = WideDeepDataset(**X_test)
        else:
            load_dict = {'X_wide': X_wide, 'X_deep': X_deep}
            if X_text is not None: load_dict.update({'X_text': X_text})
            if X_img is not None:  load_dict.update({'X_img': X_img})
            test_set = WideDeepDataset(**load_dict)

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
            batch_size=self.batch_size,shuffle=False)
        test_steps =  (len(test_loader.dataset) // test_loader.batch_size) + 1

        preds_l = []
        with torch.no_grad():
            with trange(test_steps, disable=self.verbose != 1) as t:
                for i, data in zip(t, test_loader):
                    t.set_description('predict')
                    X = {k:v.cuda() for k,v in data.items()} if use_cuda else data
                    preds = self._activation_fn(self.forward(X)).cpu().data.numpy()
                    preds_l.append(preds)
            if self.method == "regression":
                return np.vstack(preds_l).squeeze(1)
            if self.method == "logistic":
                preds = np.vstack(preds_l).squeeze(1)
                return (preds > 0.5).astype('int')
            if self.method == "multiclass":
                preds = np.vstack(preds_l)
                return np.argmax(preds, 1)

    def predict_proba(self, X_wide:np.ndarray, X_deep:np.ndarray, X_text:Optional[np.ndarray]=None,
        X_img:Optional[np.ndarray]=None, X_test:Optional[Dict[str, np.ndarray]]=None)->np.ndarray:

        if X_test is not None:
            test_set = WideDeepDataset(**X_test)
        else:
            load_dict = {'X_wide': X_wide, 'X_deep': X_deep}
            if X_text is not None: load_dict.update({'X_text': X_text})
            if X_img is not None:  load_dict.update({'X_img': X_img})
            test_set = WideDeepDataset(**load_dict)

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
            batch_size=self.batch_size,shuffle=False)
        test_steps =  (len(test_loader.dataset) // test_loader.batch_size) + 1

        preds_l = []
        with torch.no_grad():
            with trange(test_steps, disable=self.verbose != 1) as t:
                for i, data in zip(t, test_loader):
                    t.set_description('predict')
                    X = {k:v.cuda() for k,v in data.items()} if use_cuda else data
                    preds = self._activation_fn(self.forward(X)).cpu().data.numpy()
                    preds_l.append(preds)
            if self.method == "logistic":
                preds = np.vstack(preds_l).squeeze(1)
                probs = np.zeros([preds.shape[0],2])
                probs[:,0] = 1-preds
                probs[:,1] = preds
                return probs
            if self.method == "multiclass":
                return np.vstack(preds_l)

    def get_embeddings(self, col_name:str,
        cat_encoding_dict:Dict[str,Dict[str,int]]) -> Dict[str,np.ndarray]:
        for n,p in self.named_parameters():
            if 'embed_layers' in n and col_name in n:
                embed_mtx = p.cpu().data.numpy()
        encoding_dict = cat_encoding_dict[col_name]
        inv_encoding_dict = {v:k for k,v in encoding_dict.items()}
        cat_embed_dict = {}
        for idx,value in inv_encoding_dict.items():
            cat_embed_dict[value] = embed_mtx[idx]
        return cat_embed_dict