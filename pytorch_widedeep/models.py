import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wdtypes import *
from .initializers import MultipleInitializers
from .optimizers import MultipleOptimizers
from .lr_schedulers import MultipleLRScheduler
from .callbacks import History, CallbackContainer
from .metrics import MultipleMetrics, MetricCallback
from .transforms import MultipleTransforms
from .losses import FocalLoss

from tqdm import tqdm,trange
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from torchvision import models
from torch.utils.data import Dataset, DataLoader

from copy import deepcopy

use_cuda = torch.cuda.is_available()


def dense_layer(inp:int, out:int, dropout:float, batchnorm=False):
    if batchnorm:
        return nn.Sequential(
            nn.Linear(inp, out),
            nn.BatchNorm1d(out),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
            )
    else:
        return nn.Sequential(
            nn.Linear(inp, out),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
            )


def conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, maxpool:bool=True,
    adaptiveavgpool:bool=False):
    layer = nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=True, stride=stride, padding=ks//2),
        nn.BatchNorm2d(nf, momentum=0.01),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))
    if maxpool: layer.add_module('maxpool', nn.MaxPool2d(2, 2))
    if adaptiveavgpool: layer.add_module('adaptiveavgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    return layer


class Wide(nn.Module):
    def __init__(self,wide_dim:int, output_dim:int=1):
        super(Wide, self).__init__()
        # (Wide Linear, wlinear)
        self.wlinear = nn.Linear(wide_dim, output_dim)

    def forward(self, X:Tensor)->Tensor:
        out = self.wlinear(X.float())
        return out


class DeepDense(nn.Module):
    def __init__(self,
        deep_column_idx:Dict[str,int],
        hidden_layers:List[int],
        dropout:List[float]=0.,
        embed_input:Optional[List[Tuple[str,int,int]]]=None,
        continuous_cols:Optional[List[str]]=None,
        batchnorm:bool=False,
        output_dim:int=1):

        super(DeepDense, self).__init__()

        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx

        # Embeddings
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict({'emb_layer_'+col: nn.Embedding(val, dim)
                for col, val, dim in self.embed_input})
            emb_inp_dim = np.sum([embed[2] for embed in self.embed_input])
        else:
            emb_inp_dim = 0

        # Continuous
        if self.continuous_cols is not None: cont_inp_dim = len(self.continuous_cols)
        else: cont_inp_dim = 0

        # Dense Layers
        input_dim = emb_inp_dim + cont_inp_dim
        hidden_layers = [input_dim] + hidden_layers
        dropout = [0.0] + dropout
        self.dense = nn.Sequential()
        for i in range(1, len(hidden_layers)):
            self.dense.add_module(
                'dense_layer_{}'.format(i-1),
                dense_layer( hidden_layers[i-1], hidden_layers[i], dropout[i-1], batchnorm))

        # Last Linear (Deep Dense Linear ddlinear)
        self.dense.add_module('ddlinear', nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, X:Tensor)->Tensor:
        if self.embed_input is not None:
            embed = [self.embed_layers['emb_layer_'+col](X[:,self.deep_column_idx[col]].long())
                for col,_,_ in self.embed_input]
        if self.continuous_cols is not None:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = X[:, cont_idx].float()
        try:
            out = self.dense(torch.cat(embed+[cont], 1))
        except:
            try:
                out = self.dense(torch.cat(embed, 1))
            except:
                out = self.dense(cont)
        return out


class DeepText(nn.Module):
    def __init__(self,
        vocab_size:int,
        embed_dim:Optional[int]=None,
        hidden_dim:int=64,
        n_layers:int=3,
        rnn_dropout:float=0.,
        spatial_dropout:float=0.,
        padding_idx:int=1,
        output_dim:int=1,
        bidirectional:bool=False,
        embedding_matrix:Optional[np.ndarray]=None):
        super(DeepText, self).__init__()
        """
        Standard Text Classifier/Regressor with a stack of RNNs.
        """

        if embed_dim is not None and embedding_matrix is not None and not embed_dim==embedding_matrix.shape[1]:
            warnings.warn(
                'the input embedding dimension {} and the dimension of the '
                'pretrained embeddings {} do not match. The pretrained embeddings '
                'dimension ({}) will be used'.format(embed_dim, embedding_matrix.shape[1],
                    embedding_matrix.shape[1]), UserWarning)

        self.bidirectional = bidirectional
        self.spatial_dropout = spatial_dropout
        self.word_embed_dropout = nn.Dropout2d(spatial_dropout)

        # Pre-trained Embeddings
        if isinstance(embedding_matrix, np.ndarray):
            self.word_embed = nn.Embedding(vocab_size, embedding_matrix.shape[1], padding_idx = padding_idx)
            self.word_embed.weight = nn.Parameter(torch.Tensor(embedding_matrix))
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx = padding_idx)

        # stack of GRUs
        self.rnn = nn.GRU(embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
            batch_first=True)
        input_dim = hidden_dim*2 if bidirectional else hidden_dim

        # Deep Text Linear (dtlinear)
        self.dtlinear = nn.Linear(input_dim, output_dim)

    def forward(self, X:Tensor)->Tensor:

        embed = self.word_embed(X.long())
        # Spatial dropout: dropping an entire channel (word-vector dimension)
        if self.spatial_dropout > 0.:
            sd_embed = embed.unsqueeze(2)
            sd_embed = sd_embed.permute(0, 3, 2, 1)
            sd_embed = self.word_embed_dropout(sd_embed)
            sd_embed = sd_embed.permute(0, 3, 2, 1)
            embed = sd_embed.squeeze(2)
        o, h = self.rnn(embed)
        if self.bidirectional:
            last_h = torch.cat((h[-2], h[-1]), dim = 1)
        else:
            last_h = h[-1]
        out = self.dtlinear(last_h)
        return out


class DeepImage(nn.Module):

    def __init__(self,
        output_dim:int=1,
        pretrained:bool=True,
        resnet=18,
        freeze:Union[str,int]=6):
        super(DeepImage, self).__init__()
        """
        Standard image classifier/regressor using a pretrained network
        freezing some of the  first layers (or all layers).

        I use Resnets which have 9 "components" before the last dense layers.
        The first 4 are: conv->batchnorm->relu->maxpool.

        After that we have 4 additional 'layers' (so 4+4=8) comprised by a
        series of convolutions and then the final AdaptiveAvgPool2d (8+1=9).

        The parameter freeze sets the last layer to be frozen. For example,
        freeze=6 will freeze all but the last 2 Layers and AdaptiveAvgPool2d
        layer. If freeze='all' it freezes the entire network.
        """
        if pretrained:
            if resnet==18:
                vision_model = models.resnet18(pretrained=True)
            elif resnet==34:
                vision_model = models.resnet34(pretrained=True)
            elif resnet==50:
                vision_model = models.resnet50(pretrained=True)

            backbone_layers = list(vision_model.children())[:-1]

            if isinstance(freeze, str):
                frozen_layers = []
                for layer in backbone_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                    frozen_layers.append(layer)
                self.backbone = nn.Sequential(*frozen_layers)
            if isinstance(freeze, int):
                assert freeze < 8, 'freeze must be less than 8 when using resnet architectures'
                frozen_layers = []
                trainable_layers = backbone_layers[freeze:]
                for layer in backbone_layers[:freeze]:
                    for param in layer.parameters():
                        param.requires_grad = False
                    frozen_layers.append(layer)

                backbone_layers = frozen_layers + trainable_layers
                self.backbone = nn.Sequential(*backbone_layers)
        else:
            self.backbone = nn.Sequential(
                conv_layer(3, 64, 3),
                conv_layer(64, 128, 1, maxpool=False),
                conv_layer(128, 256, 1, maxpool=False),
                conv_layer(256, 512, 1, maxpool=False, adaptiveavgpool=True),
                )
        self.dilinear = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, output_dim)
            )

    def forward(self, x:Tensor)->Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        out = self.dilinear(x)
        return out


class WideDeepLoader(Dataset):
    def __init__(self, X_wide:np.ndarray, X_deep:np.ndarray,
        target:Optional[np.ndarray]=None, X_text:Optional[np.ndarray]=None,
        X_img:Optional[np.ndarray]=None, transforms:Optional=None):

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
        wide:nn.Module,
        deepdense:nn.Module,
        deeptext:Optional[nn.Module]=None,
        deepimage:Optional[nn.Module]=None):

        super(WideDeep, self).__init__()
        self.wide = wide
        self.deepdense = deepdense
        self.deeptext  = deeptext
        self.deepimage = deepimage

    def forward(self, X:List[Dict[str,Tensor]])->Tensor:
        out = self.wide(X['wide'])
        out.add_(self.deepdense(X['deepdense']))
        if self.deeptext is not None:
            out.add_(self.deeptext(X['deeptext']))
        if self.deepimage is not None:
            out.add_(self.deepimage(X['deepimage']))
        return out

    def compile(self,method:str,
        initializers:Optional[Dict[str,Initializer]]=None,
        optimizers:Optional[Dict[str,Optimizer]]=None,
        global_optimizer:Optional[Optimizer]=None,
        param_groups:Optional[Union[List[Dict],Dict[str,List[Dict]]]]=None,
        lr_schedulers:Optional[Dict[str,LRScheduler]]=None,
        global_lr_scheduler:Optional[LRScheduler]=None,
        transforms:Optional[List[Transforms]]=None,
        callbacks:Optional[List[Callback]]=None,
        metrics:Optional[List[Metric]]=None,
        class_weight:Optional[Union[float,List[float],Tuple[float]]]=None,
        with_focal_loss:bool=False, alpha:float=0.25, gamma:float=1,
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
            self.initializer = MultipleInitializers(initializers, verbose=self.verbose)
            self.initializer.apply(self)

        if optimizers is not None:
            self.optimizer = MultipleOptimizers(optimizers)
            self.optimizer.apply(self, param_groups)
        elif global_optimizer is not None:
            if isinstance(global_optimizer, type): global_optimizer = global_optimizer()
            self.optimizer = global_optimizer(self, param_groups)
        else:
            self.optimizer = torch.optim.Adam(self.parameters())

        if lr_schedulers is not None:
            self.lr_scheduler = MultipleLRScheduler(lr_schedulers)
            self.lr_scheduler.apply(self.optimizer._optimizers)
            scheduler_names = [sc.__class__.__name__.lower() for _,sc in self.lr_scheduler._schedulers.items()]
            self.cyclic = any(['cycl' in sn for sn in scheduler_names])
        elif global_lr_scheduler is not None:
            if isinstance(global_lr_scheduler, type): global_lr_scheduler = global_lr_scheduler()
            try: self.lr_scheduler = global_lr_scheduler(self.optimizer)
            except:
                raise TypeError(
                    "{} is not an Optimizer. If a global learning rate scheduler "
                    "is used then a single global optimizer must also be used".format(
                        type(self.optimizer).__name__)
                    )
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
        n_epochs:int=1, batch_size:int=32, patience:int=10, seed:int=1):

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