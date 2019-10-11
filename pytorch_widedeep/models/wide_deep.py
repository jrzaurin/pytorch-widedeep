import numpy as np
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


class WideDeepLoader(Dataset):
    def __init__(self, X_wide:np.ndarray, X_deep:np.ndarray,
        X_text:Optional[np.ndarray]=None, X_img:Optional[np.ndarray]=None,
        target:Optional[np.ndarray]=None, transforms:Optional=None):

        self.X_wide = X_wide
        self.X_deep = X_deep
        self.X_text = X_text
        self.X_img = X_img
        self.transforms = transforms
        self.Y = target

    def __getitem__(self, idx:int):

        xw = self.X_wide[idx]
        X = Bunch(wide=xw)
        xdd = self.X_deep[idx]
        X.deep_dense= xdd
        if self.X_text is not None:
            xdt = self.X_text[idx]
            X.deep_text = xdt
        if self.X_img is not None:
            xdi = (self.X_img[idx]/255).astype('float32')
            if self.transforms is not None:
                xdi = self.transforms(xdi)
            X.deep_img = xdi
        if self.Y is not None:
            y  = self.Y[idx]
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X_wide)


class WideDeep(nn.Module):

    def __init__(self, output_dim:int, wide_dim:int,
        cat_embed_input:List[Tuple[str,int,int]], continuous_cols:List[str],
        deep_column_idx:Dict[str,int], hidden_layers:List[int]=[64,32],
        dropout:List[float]=[0.], add_text:bool=False,
        nlp_model:Optional[nn.Module]=None, vocab_size:Optional[int]=None,
        word_embed_dim:Optional[int]=64, rnn_hidden_dim:Optional[int]=32,
        rnn_num_layers:Optional[int]=3, rnn_dropout:Optional[float]=0.,
        embed_spatial_dropout:Optional[float]=0., padding_idx:Optional[int]=1,
        bidirectional:Optional[bool]=False,
        word_embed_matrix:Optional[np.ndarray]=None, add_image:bool=False,
        vision_model:Optional[nn.Module]=None, pretrained:Optional[bool]=True,
        resnet:Optional[int]=18, freeze:Optional[Union[str,int]]=6):

        super(WideDeep, self).__init__()

        self.output_dim = output_dim
        self.add_text = add_text
        self.add_image = add_image

        # WIDE
        self.wide_dim = wide_dim
        self.wide = Wide(self.wide_dim, self.output_dim)

        # DEEP DENSE
        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.deep_dense = DeepDense( self.cat_embed_input,
            # self.cat_embed_encoding_dict,
            self.continuous_cols, self.deep_column_idx, self.hidden_layers, self.dropout,
            self.output_dim)

        # DEEP TEXT
        if (add_text) and (nlp_model is not None):
            self.deep_text = nlp_model
        elif add_text:
            self.vocab_size = vocab_size
            self.word_embed_dim = word_embed_dim
            self.rnn_hidden_dim = rnn_hidden_dim
            self.rnn_num_layers = rnn_num_layers
            self.rnn_dropout = rnn_dropout
            self.embed_spatial_dropout = embed_spatial_dropout
            self.padding_idx = padding_idx
            self.bidirectional = bidirectional
            self.word_embed_matrix = word_embed_matrix
            self.deep_text = DeepText(self.vocab_size, self.word_embed_dim, self.rnn_hidden_dim,
                self.rnn_num_layers, self.rnn_dropout, self.embed_spatial_dropout, self.padding_idx,
                self.output_dim, self.bidirectional, self.word_embed_matrix)

        # DEEP IMAGE
        if (add_image) and (vision_model is not None):
            self.deep_img = vision_model
        elif add_image:
            self.pretrained = pretrained
            self.resnet = resnet
            self.freeze = freeze
            self.deep_img = DeepImage(self.output_dim, self.pretrained, self.resnet,
                self.freeze)

        self.early_stop = False

    def forward(self, X:List[Dict[str,Tensor]])->Tensor:
        wide_deep = self.wide(X['wide'])
        wide_deep.add_(self.deep_dense(X['deep_dense']))

        if self.add_text:
            wide_deep.add_(self.deep_text(X['deep_text']))

        if self.add_image:
            wide_deep.add_(self.deep_img(X['deep_img']))

        if not self.activation:
            return wide_deep
        else:
            if (self.activation==F.softmax):
                out = self.activation(wide_deep, dim=1)
            else:
                out = self.activation(wide_deep)
            return out

    def set_activation(self):
        if self.method == 'regression': self.activation = None
        if self.method == 'logistic':   self.activation = torch.sigmoid
        if self.method == 'multiclass': self.activation = F.softmax

    def loss_fn(self, y_pred:Tensor, y_true:Tensor) -> Tensor:
        if self.focal_loss:
            return self.FocalLoss(self.alpha, self.gamma)(y_pred, y_true)
        if self.method == 'regression':
            return F.mse_loss(y_pred, y_true.view(-1, 1))
        if self.method == 'logistic':
            return F.binary_cross_entropy(y_pred, y_true.view(-1, 1), weight=self.class_weight)
        if self.method == 'multiclass':
            return F.cross_entropy(y_pred, y_true, weight=self.class_weight)

    def compile(self, method:str,
        callbacks:Optional[List[Callback]]=None,
        initializers:Optional[List[Initializer]]=None,
        optimizers:Optional[List[Optimizer]]=None,
        lr_schedulers:Optional[List[LRScheduler]]=None,
        transforms:Optional[List[Transforms]]=None,
        metrics:Optional[List[Metric]]=None,
        global_optimizer:Optional[Optimizer]=None,
        global_optimizer_params:Optional[Dict]=None,
        global_lr_scheduler:Optional[LRScheduler]=None,
        global_lr_scheduler_params:Optional[Dict]=None,
        class_weight:Union[float, List[float], Tuple[float]]=None,
        focal_loss:bool=False, alpha:float=0.25, gamma:float=1,):

        self.method = method
        self.set_activation()
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
            self.optimizer = global_optimizer(self)
        else:
            self.optimizer = torch.optim.Adam(self.parameters())

        if lr_schedulers is not None:
            self.lr_scheduler = MultipleLRScheduler(lr_schedulers)
            self.lr_scheduler.apply(self.optimizer._optimizers)
        elif global_lr_scheduler is not None:
            self.lr_scheduler = global_lr_scheduler(self.optimizer)
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

    def training_step(self, data:Dict[str, Tensor], target:Tensor, batch_idx:int):

        X = {k:v.cuda() for k,v in data.items()} if use_cuda else data
        y = target.float() if self.method != 'multiclass' else target
        y = y.cuda() if use_cuda else y

        self.optimizer.zero_grad()
        y_pred =  self.forward(X)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss/(batch_idx+1)

        if self.metric is not None:
            acc = self.metric(y_pred, y)
            return acc, avg_loss
        else:
            return None, avg_loss

    def validation_step(self, data:Dict[str, Tensor], target:Tensor, batch_idx:int):

        with torch.no_grad():
            X = {k:v.cuda() for k,v in data.item()} if use_cuda else data
            y = target.float() if self.method != 'multiclass' else target
            y = y.cuda() if use_cuda else y

            y_pred =  self.forward(X)
            loss = self.loss_fn(y_pred, y)
            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss/(batch_idx+1)

        if self.metric is not None:
            acc = self.metric(y_pred, y)
            return acc, avg_loss
        else:
            return None, avg_loss

    def fit(self, X_wide:np.ndarray, X_deep:np.ndarray, target:np.ndarray,
        X_text:Optional[np.ndarray]=None, X_img:Optional[np.ndarray]=None,
        n_epochs:int=1, batch_size:int=32, X_train:Optional[Dict[str,
        np.ndarray]]=None, X_val:Optional[Dict[str, np.ndarray]]=None,
        val_split:float=0., seed:int=1, patience:int=10, verbose:int=1,
        shuffle:bool=True):

        self.with_validation = True
        self.batch_size = batch_size

        if X_train is not None: # if train dict is passed takes priority
            X_wide, X_deep = X_train['wide'], X_train['deepdense']
            if 'deepimage' in X_train.keys(): X_img = X['deepimage']
            if 'deeptext' in X_train.keys(): X_text = X['deeptext']
            train_set = WideDeepLoader(**X_train, transforms=self.transforms)
            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=8,
                shuffle=False)
            train_steps =  (len(train_loader.dataset) // train_loader.batch_size) + 1
        if X_val is not None:  # if valid dict is passed takes priority
            valid_set = WideDeepLoader(**X_val, transforms=self.transforms)
            eval_loader = DataLoader(dataset=valid_set, batch_size=batch_size, num_workers=8,
                shuffle=False)
            eval_steps =  (len(eval_loader.dataset) // eval_loader.batch_size) + 1
        elif val_split > 0.:
            X_tr_wide, X_val_wide = train_test_split(X_wide, test_size=val_split, random_state=seed)
            X_tr_deep, X_val_deep = train_test_split(X_deep, test_size=val_split, random_state=seed)
            y_tr, y_val = train_test_split(target, test_size=val_split, random_state=seed)
            train_load_dict = {'X_wide':X_tr_wide, 'X_deep': X_tr_deep, 'target': y_tr}
            val_load_dict = {'X_wide':X_val_wide, 'X_deep': X_val_deep, 'target': y_val}
            if X_text is not None:
                X_tr_text, X_val_text = train_test_split(X_text, test_size=val_split, random_state=seed)
                train_load_dict.update({'X_text': X_tr_text})
                val_load_dict.update({'X_text': X_val_text})
            if X_img is not None:
                X_tr_img, X_val_img = train_test_split(X_img, test_size=val_split, random_state=seed)
                train_load_dict.update({'X_img': X_tr_img})
                val_load_dict.update({'X_img': X_val_img})
            train_set = WideDeepLoader(**train_load_dict, transforms=self.transforms)
            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=8,
                shuffle=shuffle)
            valid_set = WideDeepLoader(**val_load_dict, transforms=self.transforms)
            eval_loader = DataLoader(dataset=valid_set, batch_size=batch_size, num_workers=8,
                shuffle=False)
        else:
            train_load_dict = {'X_wide':X_wide, 'X_deep': X_deep, 'target': target}
            if X_text is not None:
                train_load_dict.update({'X_text': X_text})
            if X_img is not None:
                train_load_dict.update({'X_img': X_img})
            train_set = WideDeepLoader(**train_load_dict, transforms=self.transforms)
            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=8,
                shuffle=shuffle)
            self.with_validation = False

        train_steps =  (len(train_loader.dataset) // batch_size) + 1
        if self.with_validation:
            eval_steps =  (len(eval_loader.dataset) // batch_size) + 1

        self.callback_container.on_train_begin({'batch_size': batch_size,
            'train_steps': train_steps, 'n_epochs': n_epochs})
        for epoch in range(n_epochs):
            epoch_logs = {}
            self.callback_container.on_epoch_begin(epoch, epoch_logs)
            self.train_running_loss = 0.
            with trange(train_steps, disable=verbose != 1) as t:
                for batch_idx, (data,target) in zip(t, train_loader):
                    t.set_description('epoch %i' % (epoch+1))
                    acc, train_loss = self.training_step(data, target, batch_idx)
                    if acc is not None:
                        t.set_postfix(metrics=acc, loss=train_loss)
                    else:
                        t.set_postfix(loss=np.sqrt(train_loss))
            epoch_logs['train_loss'] = train_loss
            if acc is not None: epoch_logs['train_acc'] = acc

            if self.with_validation:
                self.valid_running_loss = 0.
                with trange(eval_steps, disable=verbose != 1) as v:
                    for i, (data,target) in zip(v, eval_loader):
                        v.set_description('valid')
                        acc, val_loss = self.validation_step(data, target, i)
                        if acc is not None:
                            v.set_postfix(metrics=acc, loss=val_loss)
                        else:
                            v.set_postfix(loss=np.sqrt(val_loss))
                epoch_logs['val_loss'] = val_loss
                if acc is not None: epoch_logs['val_acc'] = acc

            self.callback_container.on_epoch_end(epoch, epoch_logs)
            if self.early_stop:
                break
            if self.lr_scheduler: self.lr_scheduler.step()

    def predict(self, X_wide:np.ndarray, X_deep:np.ndarray, X_text:Optional[np.ndarray]=None,
        X_img:Optional[np.ndarray]=None, X_test:Optional[Dict[str, np.ndarray]]=None)->np.ndarray:

        if X_test is not None:
            test_set = WideDeepLoader(**X_test)
            test_loader = torch.utils.data.DataLoader(dataset=test_set,
                batch_size=self.batch_size,shuffle=False)
            test_steps =  (len(test_loader.dataset) // test_loader.batch_size) + 1
        else:
            load_dict = {'X_wide': X_wide, 'X_deep': X_deep}
            if X_text is not None:
                load_dict.update({'X_text': X_text})
            if X_img is not None:
                load_dict.update({'X_img': X_img})
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
                    preds_l.append(self.forward(X).cpu().data.numpy())
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
            test_loader = torch.utils.data.DataLoader(dataset=test_set,
                batch_size=self.batch_size,shuffle=False)
            test_steps =  (len(test_loader.dataset) // test_loader.batch_size) + 1
        else:
            load_dict = {'X_wide': X_wide, 'X_deep': X_deep}
            if X_text is not None:
                load_dict.update({'X_text': X_text})
            if X_img is not None:
                load_dict.update({'X_img': X_img})
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
                    preds_l.append(self.forward(X).cpu().data.numpy())
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
        col_label_encoding = self.cat_embed_encoding_dict[col_name]
        inv_dict = {v:k for k,v in col_label_encoding.items()}
        embeddings_dict = {}
        for idx,value in inv_dict.items():
            embeddings_dict[value] = embeddings[idx]
        return embeddings_dict