import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm,trange

from ..wdtypes import *
from .wide import Wide
from .deep_dense import DeepDense
from .deep_text import DeepText
from .deep_image import DeepImage


use_cuda = torch.cuda.is_available()


def set_method(method:str):
    if method =='regression':
        return None, F.mse_loss
    if method =='logistic':
        return torch.sigmoid, F.binary_cross_entropy
    if method=='multiclass':
        return F.softmax, F.cross_entropy


def set_optimizer(model_params:ModelParams, opt_params:List[Union[str,float]]):
    try:
        opt, lr, m = opt_params
    except:
        opt, lr = opt_params
    if opt == "Adam":
        return torch.optim.Adam(model_params, lr=lr)
    if opt == "Adagrad":
        return torch.optim.Adam(model_params, lr=lr)
    if opt == "RMSprop":
        return torch.optim.RMSprop(model_params, lr=lr, momentum=m)
    if opt == "SGD":
        return torch.optim.SGD(model_params, lr=lr, momentum=m)


def set_scheduler(optimizer:Optimizer, sch_params:List[Any]):
    sch,s,g = sch_params
    if sch == "StepLR":
        return StepLR(optimizer, step_size=s, gamma=g)
    if sch == "MultiStepLR":
        return MultiStepLR(optimizer, milestones=s, gamma=g)


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
    def __init__(self, data:Dict[str,Any], transform:Optional=None, mode:str='train'):

        self.mode = mode
        input_types = list(data.keys())
        self.input_types = input_types
        self.X_wide = data['wide']
        if 'deep_dense' in self.input_types: self.X_deep_dense = data['deep_dense']
        if 'deep_text' in self.input_types: self.X_deep_text = data['deep_text']
        if 'deep_img' in self.input_types:
            self.X_deep_img = data['deep_img']
            self.transform = transform
        if self.mode is 'train':
            self.Y = data['target']
        elif self.mode is 'test':
            self.Y = None

    def __getitem__(self, idx:int):

        xw = self.X_wide[idx]
        X = (xw, )
        if 'deep_dense' in self.input_types:
            xdd = self.X_deep_dense[idx]
            X += (xdd,)
        if 'deep_text' in self.input_types:
            xdt = self.X_deep_text[idx]
            X += (xdt,)
        if 'deep_img' in self.input_types:
            xdi = (self.X_deep_img[idx]/255).astype('float32')
            xdi = self.transform(xdi)
            X += (xdi,)
        if self.mode is 'train':
            y  = self.Y[idx]
            return X, y
        elif self.mode is 'test':
            return X

    def __len__(self):
        return len(self.X_wide)


class WideDeep(nn.Module):
    """ Wide and Deep model (Heng-Tze Cheng et al., 2016), adjusted to take also
    text and images.
    """
    def __init__(self, output_dim:int, **params:Any):
        super(WideDeep, self).__init__()

        self.datasets = {}
        self.n_datasets = 1
        self.output_dim = output_dim

        for k,v in params['wide'].items():
            setattr(self, k, v)
        self.wide = Wide(
            self.wide_dim,
            self.output_dim
            )
        if 'deep_dense' in params.keys():
            self.datasets['deep_dense'] = self.n_datasets
            self.n_datasets+=1
            for k,v in params['deep_dense'].items():
                setattr(self, k, v)
            self.deep_dense = DeepDense(
                self.embeddings_input,
                self.embeddings_encoding_dict,
                self.continuous_cols,
                self.deep_column_idx,
                self.hidden_layers,
                self.dropout,
                self.output_dim
                )
        if 'deep_text' in params.keys():
            self.datasets['deep_text'] = self.n_datasets
            self.n_datasets+=1
            for k,v in params['deep_text'].items():
                setattr(self, k, v)
            self.deep_text = DeepText(
                self.vocab_size,
                self.embedding_dim,
                self.hidden_dim,
                self.n_layers,
                self.rnn_dropout,
                self.spatial_dropout,
                self.padding_idx,
                self.output_dim,
                self.attention,
                self.bidirectional,
                self.embedding_matrix
                )
        if 'deep_img' in params.keys():
            self.datasets['deep_img'] = self.n_datasets
            self.n_datasets+=1
            for k,v in params['deep_img'].items():
                setattr(self, k, v)
            self.deep_img = DeepImage(
                self.output_dim,
                self.pretrained,
                self.freeze
                )

    def compile(self, method:str, optimizer:Dict[str,List[Any]],
        lr_scheduler:Optional[Dict[str,List[Any]]]=None):
        """Set the activation, loss and the optimizer.

        Parameters:
        ----------
        method: str
            'regression', 'logistic' or 'multiclass'
        optimizer: str or Dict
            if str one of the following: 'SGD', 'Adam', or 'RMSprop'
            if Dict must contain  elements for different models
            e.g. optimizer = {'wide: ['SGD', 0.001, 0.3]', 'deep':['Adam', 0.001]}
        """

        self.method = method
        self.activation, self.criterion = set_method(method)

        if len(optimizer)==1:
            self.optimizer = set_optimizer(self.parameters(), optimizer['widedeep'])
            self.lr_scheduler = set_scheduler(self.optimizer, lr_scheduler['widedeep']) if lr_scheduler else None
        else:
            wide_opt = set_optimizer(self.wide.parameters(), optimizer['wide'])
            wide_sch = set_scheduler(wide_opt, lr_scheduler['wide']) if lr_scheduler else None
            optimizers, schedulers = [wide_opt], [wide_sch]
            if 'deep_dense' in optimizer.keys():
                deep_dense_opt = set_optimizer(self.deep_dense.parameters(), optimizer['deep_dense'])
                deep_dense_sch = set_scheduler(deep_dense_opt, lr_scheduler['deep_dense']) if lr_scheduler else None
                optimizers+=[deep_dense_opt]
                schedulers+=[deep_dense_sch]
            if 'deep_text' in optimizer.keys():
                deep_text_opt = set_optimizer(self.deep_text.parameters(), optimizer['deep_text'])
                deep_text_sch = set_scheduler(deep_text_opt, lr_scheduler['deep_text']) if lr_scheduler else None
                optimizers+=[deep_text_opt]
                schedulers+=[deep_text_sch]
            if 'deep_img' in optimizer.keys():
                deep_img_opt = set_optimizer(self.deep_img.parameters(), optimizer['deep_img'])
                deep_img_sch = set_scheduler(deep_img_opt, lr_scheduler['deep_img']) if lr_scheduler else None
                optimizers+=[deep_img_opt]
                schedulers+=[deep_img_sch]
            self.optimizer = MultipleOptimizer(optimizers)
            self.lr_scheduler = MultipleLRScheduler(schedulers)

    def forward(self, X:Tuple[Tensor,...])->Tensor:

        wide_inp = X[0]
        wide_deep = self.wide(wide_inp)
        if 'deep_dense' in self.datasets.keys():
            deep_dense_idx = self.datasets['deep_dense']
            deep_dense_out = self.deep_dense(X[deep_dense_idx])
            wide_deep.add_(deep_dense_out)
        if 'deep_text' in self.datasets.keys():
            deep_text_idx = self.datasets['deep_text']
            deep_text_out = self.deep_text(X[deep_text_idx])
            wide_deep.add_(deep_text_out)
        if 'deep_img' in self.datasets.keys():
            deep_img_idx = self.datasets['deep_img']
            deep_img_out = self.deep_img(X[deep_img_idx])
            wide_deep.add_(deep_img_out)

        if not self.activation:
            return wide_deep
        else:
            if (self.activation==F.softmax):
                out = self.activation(wide_deep, dim=1)
            else:
                out = self.activation(wide_deep)
            return out

    def fit(self, n_epochs:int, train_loader:DataLoader, eval_loader:Optional[DataLoader]=None):

        train_steps =  (len(train_loader.dataset) // train_loader.batch_size) + 1
        if eval_loader:
            eval_steps =  (len(eval_loader.dataset) // eval_loader.batch_size) + 1
        for epoch in range(n_epochs):
            if self.lr_scheduler: self.lr_scheduler.step()
            net = self.train()
            total, correct, running_loss = 0,0,0
            with trange(train_steps) as t:
                for i, (data,target) in zip(t, train_loader):
                    t.set_description('epoch %i' % (epoch+1))
                    X = tuple(x.cuda() for x in data) if use_cuda else data
                    y = target.float() if self.method != 'multiclass' else target
                    y = y.cuda() if use_cuda else y

                    self.optimizer.zero_grad()
                    y_pred =  net(X)
                    if(self.criterion == F.cross_entropy):
                        loss = self.criterion(y_pred, y)
                    else:
                        loss = self.criterion(y_pred, y.view(-1, 1))
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    avg_loss = running_loss/(i+1)

                    if self.method != "regression":
                        total+= y.size(0)
                        if self.method == 'logistic':
                            y_pred_cat = (y_pred > 0.5).squeeze(1).float()
                        if self.method == "multiclass":
                            _, y_pred_cat = torch.max(y_pred, 1)
                        correct+= float((y_pred_cat == y).sum().item())
                        t.set_postfix(acc=correct/total, loss=avg_loss)
                    else:
                        t.set_postfix(loss=np.sqrt(avg_loss))

            if eval_loader:
                total, correct, running_loss = 0,0,0
                net = self.eval()
                with torch.no_grad():
                    with trange(eval_steps) as v:
                        for i, (data,target) in zip(v, eval_loader):
                            v.set_description('valid')
                            X = tuple(x.cuda() for x in data) if use_cuda else data
                            y = target.float() if self.method != 'multiclass' else target
                            y = y.cuda() if use_cuda else y
                            y_pred =  net(X)
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
                                v.set_postfix(acc=correct/total, loss=avg_loss)
                            else:
                                v.set_postfix(loss=np.sqrt(avg_loss))

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