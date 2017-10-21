from __future__ import print_function
import torch
import numpy as np
import pandas as pd
from wide_deep.torch_model import WideDeep
from wide_deep.data_utils import prepare_data


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    DF = pd.read_csv('data/adult_data.csv')
    DF['income_label'] = (DF["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    age_groups = [0, 25, 50, 90]
    age_labels = range(len(age_groups) - 1)
    DF['age_group'] = pd.cut(DF['age'], age_groups, labels=age_labels)

    # EXPERIMENT SET UP
    # wide_cols = ['age','hours_per_week','education', 'relationship','workclass',
    #              'occupation','native_country','gender']
    # crossed_cols  = (['education', 'occupation'], ['native_country', 'occupation'])
    # embeddings_cols  = [('education',10), ('relationship',8), ('workclass',10),
    #                     ('occupation',10),('native_country',10)]
    # continuous_cols = ["age","hours_per_week"]
    # target = 'income_label'
    # hidden_layers = [100,50]
    # method = 'logistic'

    wide_cols = ['hours_per_week','education', 'relationship','workclass',
                 'occupation','native_country','gender']
    crossed_cols  = (['education', 'occupation'], ['native_country', 'occupation'])
    embeddings_cols  = [('education',10), ('relationship',8), ('workclass',10),
                        ('occupation',10),('native_country',10)]
    continuous_cols = ["hours_per_week"]
    target = 'age_group'
    hidden_layers = [100,50]
    method = 'multiclass'

    wd_dataset = prepare_data(
        DF, wide_cols,
        crossed_cols,
        embeddings_cols,
        continuous_cols,
        target)

    wide_dim = wd_dataset['train_dataset'].wide.shape[1]
    n_unique = len(np.unique(wd_dataset['train_dataset'].labels))

    if (method=="regression") or (method=="logistic"):
        n_class = 1
    else:
        n_class = n_unique

    deep_column_idx = wd_dataset['deep_column_idx']
    embeddings_input= wd_dataset['embeddings_input']
    encoding_dict   = wd_dataset['encoding_dict']

    model = WideDeep(
        wide_dim,
        embeddings_input,
        continuous_cols,
        deep_column_idx,
        hidden_layers,
        encoding_dict,
        n_class)
    model.compile(method=method)
    if use_cuda:
        model = model.cuda()
    train_dataset = wd_dataset['train_dataset']
    test_dataset  = wd_dataset['test_dataset']
    model.fit(dataset=train_dataset, n_epochs=10, batch_size=64)
    print(model.predict(dataset=test_dataset)[:10])
    print(model.predict_proba(dataset=test_dataset)[:10])
    print(model.get_embeddings('workclass'))

    # save and load
    # torch.save(model.state_dict(), 'test.pkl')
    # model = WideDeep(wide_dim, embeddings_input, continuous_cols, deep_column_idx, hidden_layers, encoding_dict)
    # model.load_state_dict(torch.load("test.pkl"))
