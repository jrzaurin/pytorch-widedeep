import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import pdb

from hyperopt import hp, tpe, fmin, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from widedeep.utils.data_utils import label_encode


def objective(params):
	"""
	objective function for lightgbm.
	"""
	# hyperopt casts as float
	params['num_boost_round'] = int(params['num_boost_round'])
	params['num_leaves'] = int(params['num_leaves'])

	# need to be passed as parameter
	params['is_unbalance'] = True
	params['verbose'] = -1
	params['seed'] = 1

	cv_result = lgb.cv(
		params,
		dtrain,
		num_boost_round=params['num_boost_round'],
		metrics='binary_logloss',
		nfold=3,
		early_stopping_rounds=20,
		stratified=False)
	early_stop_dict[objective.i] = len(cv_result['binary_logloss-mean'])
	error = round(cv_result['binary_logloss-mean'][-1], 4)
	objective.i+=1
	return error


space = {
	'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
	'num_boost_round': hp.quniform('num_boost_round', 50, 500, 10),
	'num_leaves': hp.quniform('num_leaves', 31, 255, 4),
    'min_child_weight': hp.uniform('min_child_weight', 0.1, 5),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
    'subsample': hp.uniform('subsample', 0.5, 1.),
    'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
    'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
}

df = pd.read_csv("data/adult/adult.csv")
df.columns = [c.replace("-", "_") for c in df.columns]
df['income_label'] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
drop_cols = ['fnlwgt','educational_num','income']
df.drop(drop_cols, axis=1, inplace=True)

categorical_columns = list(df.select_dtypes(include=['object']).columns)
df, encoding_d = label_encode(df, categorical_columns)
all_columns =  df.columns.tolist()[:-1]

df_tr, df_te = train_test_split(df, test_size=0.3, random_state=1)
dtrain = lgb.Dataset(
	df_tr.iloc[:, :-1],
	label=df_tr['income_label'],
	categorical_feature=categorical_columns,
	feature_name=all_columns,
	free_raw_data=False)

maxevals=100
early_stop_dict = {}
objective.i=0
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=maxevals,
            trials=trials)

best['num_boost_round'] = early_stop_dict[trials.best_trial['tid']]
best['num_leaves'] = int(best['num_leaves'])
best['verbose'] = -1
model = lgb.LGBMClassifier(**best)
model.fit(dtrain.data,
	dtrain.label,
	feature_name=all_columns,
	categorical_feature=categorical_columns)

X_te = df_te.iloc[:,:-1].values
y_te = df_te['income_label'].values
preds = model.predict(X_te)
acc = accuracy_score(y_te, preds)

print(acc)

pickle.dump(model, open("data/models/adult_lgb_model.p", 'wb'))
pickle.dump(best, open("data/models/adult_lgb_best_params.p", 'wb'))
