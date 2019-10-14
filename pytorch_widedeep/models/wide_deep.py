import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..wdtypes import *
from ..initializers import Initializer, MultipleInitializers
from ..optimizers import MultipleOptimizers
from ..lr_schedulers import MultipleLRScheduler
from ..callbacks import Callback, History, CallbackContainer
from ..metrics import Metric, MultipleMetrics, MetricCallback
from ..transforms import MultipleTransforms
from ..losses import FocalLoss

from .wide import Wide
from .deep_dense import DeepDense
from .deep_text import DeepText
from .deep_image import DeepImage

from tqdm import tqdm,trange
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

use_cuda = torch.cuda.is_available()


import pdb


class WideDeepLoader(Dataset):
    def __init__(self, X_wide:np.ndarray, X_deep:np.ndarray, target:np.ndarray,
        X_text:Optional[np.ndarray]=None, X_img:Optional[np.ndarray]=None,
        transforms:Optional=None):

        self.X_wide = X_wide
        self.X_deep = X_deep
        self.X_text = X_text
        self.X_img  = X_img
        self.transforms = transforms
        if self.transforms:
            self.transforms_names = [tr.__class__.__name__ for tr in self.transforms.transforms]
        else: self.transforms_names = []
        self.Y = target

    def __getitem__(self, idx:int):

        X = Bunch(wide=self.X_wide[idx])
        X.deepdense= self.X_deep[idx]
        if self.X_text is not None:
            X.deeptext = self.X_text[idx]
        if self.X_img is not None:
            xdi = self.X_img[idx]
            if 'int' in str(xdi.dtype) and 'uint8' != str(xdi.dtype): xdi = xdi.astype('uint8')
            if 'float' in str(xdi.dtype) and 'float32' != str(xdi.dtype): xdi = xdi.astype('float32')
            if not self.transforms or 'ToTensor' not in self.transforms_names:
                xdi = xdi.transpose(2,0,1)
                if 'int' in str(xdi.dtype): xdi = (xdi/xdi.max()).astype('float32')
            if 'ToTensor' in self.transforms_names: xdi = self.transforms(xdi)
            elif self.transforms: xdi = self.transforms(torch.Tensor(xdi))
            X.deepimage = xdi
        if self.Y is not None:
            y  = self.Y[idx]
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X_wide)


class WideDeep(nn.Module):

    def __init__(self,
        wide:TorchModel,
        deepdense:TorchModel,
        deeptext:Optional[TorchModel]=None,
        deepimage:Optional[TorchModel]=None):

        super(WideDeep, self).__init__()
        self.wide = wide
        self.deepdense = deepdense
        self.deeptext  = deeptext
        self.deepimage = deepimage

    def forward(self, X:List[Dict[str,Tensor]])->Tensor:
        wide_deep = self.wide(X['wide'])
        wide_deep.add_(self.deepdense(X['deepdense']))
        if self.deeptext is not None:
            wide_deep.add_(self.deeptext(X['deeptext']))
        if self.deepimage is not None:
            wide_deep.add_(self.deepimage(X['deepimage']))
        return wide_deep

    def compile(self,method:str,
        initializers:Optional[Dict[str,Initializer]]=None,
        optimizers:Optional[Dict[str,Optimizer]]=None,
        global_optimizer:Optional[Optimizer]=None,
        lr_schedulers:Optional[Dict[str,LRScheduler]]=None,
        global_lr_scheduler:Optional[LRScheduler]=None,
        transforms:Optional[List[Transforms]]=None,
        callbacks:Optional[List[Callback]]=None,
        metrics:Optional[List[Metric]]=None,
        class_weight:Optional[Union[float,List[float],Tuple[float]]]=None,
        focal_loss:bool=False, alpha:float=0.25, gamma:float=1):

        self.early_stop = False
        self.method = method
        self.focal_loss = focal_loss
        if self.focal_loss:
            self.alpha, self.gamma = alpha, gamma

        if isinstance(class_weight, float):
            self.class_weight = torch.tensor([class_weight, 1.-class_weight])
        elif isinstance(class_weight, (List, Tuple)):
            self.class_weight =  torch.tensor(class_weight)
        else:
            self.class_weight = None

        if initializers is not None:
            self.initializer = MultipleInitializers(initializers)
            self.initializer.apply(self)

        if optimizers is not None:
            self.optimizer = MultipleOptimizers(optimizers)
            self.optimizer.apply(self)
        elif global_optimizer is not None:
            if isinstance(global_optimizer, type): self.optimizer = global_optimizer()
            self.optimizer = global_optimizer(self.parameters())
        else:
            self.optimizer = torch.optim.Adam(self.parameters())

        if lr_schedulers is not None:
            self.lr_scheduler = MultipleLRScheduler(lr_schedulers)
            self.lr_scheduler.apply(self.optimizer._optimizers)
            if 'cycl' in [sc.__class__.__name__.lower() for _,sc in self.lr_scheduler._schedulers.items()]:
                self.cyclic = True
            else: self.cyclic = False
        elif global_lr_scheduler is not None:
            if isinstance(global_optimizer, type): self.lr_scheduler = global_lr_scheduler()
            self.lr_scheduler = global_lr_scheduler(self.optimizer)
            if 'cycl' in self.lr_scheduler.__class__.__name__.lower(): self.cyclic = True
            else: self.cyclic = False
        else:
            self.lr_scheduler = None

        if transforms is not None:
            self.transforms = MultipleTransforms(transforms)()
        else:
            self.transforms = None

        self.history = History()
        self.callbacks = [self.history]
        if callbacks is not None:
            self.callbacks += callbacks

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
        if self.focal_loss:
            return self.FocalLoss(self.alpha, self.gamma)(y_pred, y_true)
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
                for scheduler_name, scheduler in self.lr_scheduler._schedulers.items():
                    if 'cycl' in scheduler_name.lower(): scheduler.step()
            elif step_location == 'on_epoch_end':
                for scheduler_name, scheduler in self.lr_scheduler._schedulers.items():
                    if 'cycl' not in scheduler_name.lower(): scheduler.step()
        elif self.cyclic:
            if step_location == 'on_batch_end': self.lr_scheduler.step()
            else: pass
        elif self.lr_scheduler.__class__.__name__ == 'MultipleLRScheduler':
            if step_location == 'on_epoch_end': self.lr_scheduler.step()
            else: pass
        elif step_location == 'on_epoch_end': self.lr_scheduler.step()
        else: pass

    def _train_val_split(self, X_wide:Optional[np.ndarray]=None,
        X_deep:Optional[np.ndarray]=None, X_text:Optional[np.ndarray]=None,
        X_img:Optional[np.ndarray]=None,
        X_train:Optional[Dict[str,np.ndarray]]=None,
        X_val:Optional[Dict[str,np.ndarray]]=None,
        val_split:Optional[float]=None, target:Optional[np.ndarray]=None,
        seed:int=1):

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
            train_set = WideDeepLoader(**X_train, transforms=self.transforms)
            eval_set = None
        else:
            if X_val is not None:
                if X_train is None:
                    X_train = {'X_wide':X_wide, 'X_deep': X_deep, 'target': target}
                    if X_text is not None: X_train.update({'X_text': X_text})
                    if X_img is not None:  X_train.update({'X_img': X_img})
            else:
                if X_train is not None:
                    X_wide, X_deep, target = X_train['X_wide'], X_train['X_deep'], X_train['target']
                    if 'X_text' in X_train.keys():
                        X_text = X_train['X_text']
                    if 'X_img' in X_train.keys():
                        X_img = X_train['X_img']
                X_tr_wide, X_val_wide, X_tr_deep, X_val_deep, y_tr, y_val = train_test_split(X_wide, X_deep, target, test_size=val_split, random_state=seed)
                X_train = {'X_wide':X_tr_wide, 'X_deep': X_tr_deep, 'target': y_tr}
                X_val = {'X_wide':X_val_wide, 'X_deep': X_val_deep, 'target': y_val}
                try:
                    X_tr_text, X_val_text = train_test_split(X_text, test_size=val_split, random_state=seed)
                    X_train.update({'X_text': X_tr_text}), X_val.update({'X_text': X_val_text})
                except: pass
                try:
                    X_tr_img, X_val_img = train_test_split(X_img, test_size=val_split, random_state=seed)
                    X_train.update({'X_img': X_tr_img}), X_val.update({'X_img': X_val_img})
                except: pass
            train_set = WideDeepLoader(**X_train, transforms=self.transforms)
            eval_set = WideDeepLoader(**X_val, transforms=self.transforms)
        return train_set, eval_set

    def fit(self, X_wide:Optional[np.ndarray]=None, X_deep:Optional[np.ndarray]=None,
        X_text:Optional[np.ndarray]=None, X_img:Optional[np.ndarray]=None,
        X_train:Optional[Dict[str,np.ndarray]]=None,
        X_val:Optional[Dict[str,np.ndarray]]=None,
        val_split:Optional[float]=None, target:Optional[np.ndarray]=None,
        n_epochs:int=1, batch_size:int=32, patience:int=10, seed:int=1,
        verbose:int=1):

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
            epoch_logs = {}
            self.callback_container.on_epoch_begin(epoch+1, epoch_logs)
            self.train_running_loss = 0.
            with trange(train_steps, disable=verbose != 1) as t:
                for batch_idx, (data,target) in zip(t, train_loader):
                    t.set_description('epoch %i' % (epoch+1))
                    acc, train_loss = self._training_step(data, target, batch_idx)
                    if acc is not None:
                        t.set_postfix(metrics=acc, loss=train_loss)
                    else:
                        t.set_postfix(loss=np.sqrt(train_loss))
                    if self.lr_scheduler: self._lr_scheduler_step(step_location="on_batch_end")
            epoch_logs['train_loss'] = train_loss
            if acc is not None: epoch_logs['train_acc'] = acc

            # eval step...
            if eval_set is not None:
                eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size, num_workers=8,
                    shuffle=False)
                eval_steps =  (len(eval_loader.dataset) // batch_size) + 1
                self.valid_running_loss = 0.
                with trange(eval_steps, disable=verbose != 1) as v:
                    for i, (data,target) in zip(v, eval_loader):
                        v.set_description('valid')
                        acc, val_loss = self._validation_step(data, target, i)
                        if acc is not None:
                            v.set_postfix(metrics=acc, loss=val_loss)
                        else:
                            v.set_postfix(loss=np.sqrt(val_loss))
                epoch_logs['val_loss'] = val_loss
                if acc is not None: epoch_logs['val_acc'] = acc

            self.callback_container.on_epoch_end(epoch+1, epoch_logs)
            if self.early_stop:
                break
            if self.lr_scheduler: self._lr_scheduler_step(step_location="on_epoch_end")

    def predict(self, X_wide:np.ndarray, X_deep:np.ndarray, X_text:Optional[np.ndarray]=None,
        X_img:Optional[np.ndarray]=None, X_test:Optional[Dict[str, np.ndarray]]=None)->np.ndarray:

        if X_test is not None:
            test_set = WideDeepLoader(**X_test)
        else:
            load_dict = {'X_wide': X_wide, 'X_deep': X_deep}
            if X_text is not None: load_dict.update({'X_text': X_text})
            if X_img is not None:  load_dict.update({'X_img': X_img})
            test_set = WideDeepLoader(**load_dict)

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
            batch_size=self.batch_size,shuffle=False)
        test_steps =  (len(test_loader.dataset) // test_loader.batch_size) + 1

        preds_l = []
        with torch.no_grad():
            with trange(test_steps) as t:
                for i, data in zip(t, test_loader):
                    t.set_description('predict')
                    X = {k:v.cuda() for k,v in data.items()} if use_cuda else data
                    preds = self._activation_fn(self.forward(X).cpu().data.numpy())
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
            test_set = WideDeepLoader(**X_test)
        else:
            load_dict = {'X_wide': X_wide, 'X_deep': X_deep}
            if X_text is not None: load_dict.update({'X_text': X_text})
            if X_img is not None:  load_dict.update({'X_img': X_img})

        test_set = WideDeepLoader(**load_dict)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
            batch_size=self.batch_size,shuffle=False)
        test_steps =  (len(test_loader.dataset) // test_loader.batch_size) + 1

        preds_l = []
        with torch.no_grad():
            with trange(test_steps) as t:
                for i, data in zip(t, test_loader):
                    t.set_description('predict')
                    X = {k:v.cuda() for k,v in data.items()} if use_cuda else data
                    preds = self._activation_fn(self.forward(X).cpu().data.numpy())
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
        cat_embed_encoding_dict:Dict[str,Dict[str,int]]) -> Dict[str,np.ndarray]:

        params = list(self.named_parameters())
        emb_layers = [p for p in params if 'emb_layer' in p[0]]
        emb_layer  = [layer for layer in emb_layers if col_name in layer[0]][0]
        embeddings = emb_layer[1].cpu().data.numpy()
        col_label_encoding = cat_embed_encoding_dict[col_name]
        inv_dict = {v:k for k,v in col_label_encoding.items()}
        embeddings_dict = {}
        for idx,value in inv_dict.items():
            embeddings_dict[value] = embeddings[idx]
        return embeddings_dict