# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


use_cuda = torch.cuda.is_available()


class MultipleOptimizer(object):
    """Helper to use multiple optimizers as one.

    Parameters:
    ----------
    opts: List
        List with the names of the optimizers to use
    """
    def __init__(self, opts):
        self.optimizers = opts

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class WideDeepLoader(Dataset):
    """Helper to facilitate loading the data to the pytorch models.

    Parameters:
    ----------
    data: namedtuple with 3 elements - (wide_input_data, deep_inp_data, target)
    """
    def __init__(self, data):

        self.X_wide = data.wide
        self.X_deep = data.deep
        self.Y = data.labels

    def __getitem__(self, idx):

        xw = self.X_wide[idx]
        xd = self.X_deep[idx]
        y  = self.Y[idx]

        return xw, xd, y

    def __len__(self):
        return len(self.Y)


class WideDeep(nn.Module):
    """ Wide and Deep model. As explained in Heng-Tze Cheng et al., 2016, the
    model taked the wide features and the deep features after being passed through
    the hidden layers and connects them to an output neuron. For details, please
    refer to the paper and the corresponding tutorial in the tensorflow site:
    https://www.tensorflow.org/tutorials/wide_and_deep

    Parameters:
    --------
    wide_dim: Int
        dim of the wide-side input tensor
    embeddings_input: Tuple.
        3-elements tuple with the embeddings "set-up" - (col_name,
        unique_values, embeddings dim)
    continuous_cols: List.
        list with the name of the continuum columns
    deep_column_idx: Dict
        dictionary where the keys are column names and the values their
        corresponding index in the deep-side input tensor
    hidden_layers: List
        list with the number of units per hidden layer
    encoding_dict: Dict
        dictionary with the label-encode mapping
    n_class: Int
        number of classes. Defaults to 1 if logistic or regression
    dropout: Float
    """

    def __init__(self,
                 wide_dim,
                 embeddings_input,
                 continuous_cols,
                 deep_column_idx,
                 hidden_layers,
                 dropout,
                 encoding_dict,
                 n_class):

        super(WideDeep, self).__init__()
        self.wide_dim = wide_dim
        self.deep_column_idx = deep_column_idx
        self.embeddings_input = embeddings_input
        self.continuous_cols = continuous_cols
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.encoding_dict = encoding_dict
        self.n_class = n_class

        # Build the embedding layers to be passed through the deep-side
        for col,val,dim in self.embeddings_input:
            setattr(self, 'emb_layer_'+col, nn.Embedding(val, dim))

        # Build the deep-side hidden layers with dropout if specified
        input_emb_dim = np.sum([emb[2] for emb in self.embeddings_input])
        self.linear_1 = nn.Linear(input_emb_dim+len(continuous_cols), self.hidden_layers[0])
        if self.dropout:
            self.linear_1_drop = nn.Dropout(self.dropout[0])
        for i,h in enumerate(self.hidden_layers[1:],1):
            setattr(self, 'linear_'+str(i+1), nn.Linear( self.hidden_layers[i-1], self.hidden_layers[i] ))
            if self.dropout:
                setattr(self, 'linear_'+str(i+1)+'_drop', nn.Dropout(self.dropout[i]))

        # Connect the wide- and dee-side of the model to the output neuron(s)
        self.output = nn.Linear(self.hidden_layers[-1]+self.wide_dim, self.n_class)


    @staticmethod
    def set_optimizer(model_params, optimizer, learning_rate, momentum=0.0):
        """

        Simple helper so we can set the optimizers with a string, which will
        be convenient later. Add more parameters if you need.
        """
        if optimizer == "Adam":
            return torch.optim.Adam(model_params, lr=learning_rate)
        if optimizer == "Adagrad":
            return torch.optim.Adam(model_params, lr=learning_rate)
        if optimizer == "RMSprop":
            return torch.optim.RMSprop(model_params, lr=learning_rate, momentum=momentum)
        if optimizer == "SGD":
            return torch.optim.SGD(model_params, lr=learning_rate, momentum=momentum)


    @staticmethod
    def set_method(method):
        """
        Simple helper so we can set the method with a string, which will
        be convenient later.
        """
        if method =='regression':
            return None, F.mse_loss
        if method =='logistic':
            return torch.sigmoid, F.binary_cross_entropy
        if method=='multiclass':
            return F.softmax, F.cross_entropy


    def compile(self, method="logistic", optimizer="Adam", learning_rate=0.001, momentum=0.0):
        """Wrapper to set the activation, loss and the optimizer.

        Parameters:
        ----------
        method: String
            'regression', 'logistic' or 'multiclass'
        optimizer: String or Dict
            if string one of the following: 'SGD', 'Adam', or 'RMSprop'
            if Dictionary must contain two elements for the wide and deep
            parts respectively with keys 'wide' and 'deep'. E.g.
            optimizer = {'wide: ['SGD', 0.001, 0.3]', 'deep':['Adam', 0.001]}
        """
        self.method = method
        self.activation, self.criterion = self.set_method(method)

        if type(optimizer) is dict:
            params = list(self.parameters())
            # last two sets of parameters are the weights and bias of the last
            # linear layer
            last_linear_weights = params[-2]
            # by construction, if the weights from wide_dim "in advance"
            # correspond to the weight side and will use one optimizer
            wide_params = [nn.Parameter(last_linear_weights[:, -self.wide_dim:])]
            # The weights from the deep side and the bias will use the other
            # optimizer
            deep_weights = last_linear_weights[:, :-self.wide_dim]
            deep_params = params[:-2] + [nn.Parameter(deep_weights)] + [params[-1]]
            # Very inelegant, but will do for now
            if len(optimizer['wide'])>2:
                wide_opt = self.set_optimizer(wide_params, optimizer['wide'][0], optimizer['wide'][1], optimizer['wide'][2])
            else:
                wide_opt = self.set_optimizer(wide_params, optimizer['wide'][0], optimizer['wide'][1])
            if len(optimizer['deep'])>2:
                deep_opt = self.set_optimizer(deep_params, optimizer['deep'][0], optimizer['deep'][1], optimizer['deep'][2])
            else:
                deep_opt = self.set_optimizer(deep_params, optimizer['deep'][0], optimizer['deep'][1])
            self.optimizer = MultipleOptimizer([wide_opt, deep_opt])
        elif type(optimizer) is str:
            self.optimizer = self.set_optimizer(self.parameters(), optimizer, learning_rate, momentum)


    def forward(self, X_w, X_d):
        """Implementation of the forward pass.

        Parameters:
        ----------
        X_w (torch.tensor) : wide-side input tensor
        X_d (torch.tensor) : deep-side input tensor

        Returns:
        --------
        out (torch.tensor) : result of the output neuron(s)
        """
        # Deep Side
        emb = [getattr(self, 'emb_layer_'+col)(X_d[:,self.deep_column_idx[col]].long())
               for col,_,_ in self.embeddings_input]
        if self.continuous_cols:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = [X_d[:, cont_idx].float()]
            deep_inp = torch.cat(emb+cont, 1)
        else:
            deep_inp = torch.cat(emb, 1)

        x_deep = F.relu(self.linear_1(deep_inp))
        if self.dropout:
            x_deep = self.linear_1_drop(x_deep)
        for i in range(1,len(self.hidden_layers)):
            x_deep = F.relu( getattr(self, 'linear_'+str(i+1))(x_deep) )
            if self.dropout:
                x_deep = getattr(self, 'linear_'+str(i+1)+'_drop')(x_deep)

        # Deep + Wide sides
        wide_deep_input = torch.cat([x_deep, X_w.float()], 1)
        if not self.activation:
            out = self.output(wide_deep_input)
        else:
            if (self.activation==F.softmax):
                out = self.activation(self.output(wide_deep_input), dim=1)
            else:
                out = self.activation(self.output(wide_deep_input))

        return out


    def fit(self, dataset, n_epochs, batch_size):
        """Run the model for the training set at dataset.

        Parameters:
        ----------
        dataset (dict): dictionary with the training sets -
        X_wide_train, X_deep_train, target
        n_epochs (int)
        batch_size (int)
        """
        widedeep_dataset = WideDeepLoader(dataset)
        train_loader = torch.utils.data.DataLoader(dataset=widedeep_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # set the model in training mode
        net = self.train()
        for epoch in range(n_epochs):
            total=0
            correct=0
            for i, (X_wide, X_deep, target) in enumerate(train_loader):
                X_w = Variable(X_wide)
                X_d = Variable(X_deep)
                y = (Variable(target).float() if self.method != 'multiclass' else Variable(target))

                if use_cuda:
                    X_w, X_d, y = X_w.cuda(), X_d.cuda(), y.cuda()

                self.optimizer.zero_grad()
                y_pred =  net(X_w, X_d)
                loss = None
                if(self.criterion == F.cross_entropy):
                    loss = self.criterion(y_pred, y) #[batch_size, 1]
                else:
                    loss = self.criterion(y_pred, y.view(-1, 1)) #[batch_size, 1]
                loss.backward()
                self.optimizer.step()

                if self.method != "regression":
                    total+= y.size(0)
                    if self.method == 'logistic':
                        y_pred_cat = (y_pred > 0.5).squeeze(1).float()
                    if self.method == "multiclass":
                        _, y_pred_cat = torch.max(y_pred, 1)
                    correct+= float((y_pred_cat == y).sum().item())

            if self.method != "regression":
                print ('Epoch {} of {}, Loss: {}, accuracy: {}'.format(epoch+1,
                    n_epochs, round(loss.item(),3), round(correct/total,4)))
            else:
                print ('Epoch {} of {}, Loss: {}'.format(epoch+1, n_epochs,
                    round(loss.item(),3)))


    def predict(self, dataset):
        """Predict target for dataset.

        Parameters:
        ----------
        dataset (dict): dictionary with the testing dataset -
        X_wide_test, X_deep_test, target

        Returns:
        --------
        array-like with the target for dataset
        """

        X_w = Variable(torch.from_numpy(dataset.wide)).float()
        X_d = Variable(torch.from_numpy(dataset.deep))

        if use_cuda:
            X_w, X_d = X_w.cuda(), X_d.cuda()

        # set the model in evaluation mode so dropout is not applied
        net = self.eval()
        pred = net(X_w,X_d).cpu()
        if self.method == "regression":
            return pred.squeeze(1).data.numpy()
        if self.method == "logistic":
            return (pred > 0.5).squeeze(1).data.numpy()
        if self.method == "multiclass":
            _, pred_cat = torch.max(pred, 1)
            return pred_cat.data.numpy()


    def predict_proba(self, dataset):
        """Predict predict probability for dataset.
        This method will only work with method logistic/multiclass

        Parameters:
        ----------
        dataset (dict): dictionary with the testing dataset -
        X_wide_test, X_deep_test, target

        Returns:
        --------
        array-like with the probability for dataset.
        """

        X_w = Variable(torch.from_numpy(dataset.wide)).float()
        X_d = Variable(torch.from_numpy(dataset.deep))

        if use_cuda:
            X_w, X_d = X_w.cuda(), X_d.cuda()

        # set the model in evaluation mode so dropout is not applied
        net = self.eval()
        pred = net(X_w,X_d).cpu()
        if self.method == "logistic":
            pred = pred.squeeze(1).data.numpy()
            probs = np.zeros([pred.shape[0],2])
            probs[:,0] = 1-pred
            probs[:,1] = pred
            return probs
        if self.method == "multiclass":
            return pred.data.numpy()


    def get_embeddings(self, col_name):
        """Extract the embeddings for the embedding columns.

        Parameters:
        -----------
        col_name (str) : column we want the embedding for

        Returns:
        --------
        embeddings_dict (dict): dictionary with the column values and the embeddings
        """

        params = list(self.named_parameters())
        emb_layers = [p for p in params if 'emb_layer' in p[0]]
        emb_layer  = [layer for layer in emb_layers if col_name in layer[0]][0]
        embeddings = emb_layer[1].cpu().data.numpy()
        col_label_encoding = self.encoding_dict[col_name]
        inv_dict = {v:k for k,v in col_label_encoding.items()}
        embeddings_dict = {}
        for idx,value in inv_dict.items():
            embeddings_dict[value] = embeddings[idx]

        return embeddings_dict


