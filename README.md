# ce_tune
tune ml model's superparameters

```
import pandas as pd
import lightgbm as lgb

from tune_util import *


def get_values(x):
    return x.values if hasattr(x, 'values') else x


class LgbTrainer:
    def __init__(self, params):
        self.params = params
        self.model = None

    def set_params(self, **params):
        self.params.update(params)

    def fit(self, x, y):
        self.model = lgb.train(self.params, lgb.Dataset(get_values(x), label=get_values(y)),
                               num_boost_round=self.params['num_boost_round'])
        return self

    def predict(self, x):
        return self.model.predict(get_values(x))


def get_data(data_src_dir):
    train_pickle_path = data_src_dir + 'data\\select'
    train_df = pd.read_pickle(train_pickle_path, compression='gzip')
    target = train_df['target']
    x = train_df.drop('target', axis=1)

    return x, target


data_dir = 'E:\\work\\kaggle\\porto\\'
X, y = get_data(data_dir)

train_x, train_y, holdout_data_set = insample_outsample_split(X, y, train_size=0.8, random_state=853)
print(X.shape, train_x.shape)
del X, y

num_boost_round = 1128
# origin_params = {'objective': ['binary'], 'boosting_type': 'gbdt', 'metric': ['auc'], 'verbose': -1, 'nthread': 3,
#                  'learning_rate': 0.0085, 'num_leaves': 16, 'max_depth': 8, 'max_bin': 255, 'min_data': 1187,
#                  'bagging_fraction': 0.9, 'feature_fraction': 0.7, 'bagging_freq': 3, 'lambda_l1': 48,
#                  'lambda_l2': 0.4, 'is_unbalance': True}
origin_params = {'objective': ['binary'], 'boosting_type': 'gbdt', 'metric': ['auc'], 'verbose': -1, 'nthread': 3,
                 'learning_rate': 0.00875, 'num_leaves': 20, 'max_depth': 8, 'max_bin': 255, 'min_data': 1187,
                 'bagging_fraction': 0.9, 'feature_fraction': 0.5, 'bagging_freq': 1, 'lambda_l1': 32,
                 'lambda_l2': 0.45875, 'is_unbalance': True}
lgb_model = LgbTrainer(params=origin_params)

# init_param = [('learning_rate', 0.0085), ('num_leaves', 16), ('max_depth', 8), ('max_bin', 255), ('min_data', 1187),
#               ('bagging_fraction', 0.9), ('feature_fraction', 0.7), ('bagging_freq', 3), ('lambda_l1', 48.0),
#               ('lambda_l2', 0.4)]
init_param = [('learning_rate', 0.00875), ('num_leaves', 20), ('max_depth', 8), ('max_bin', 255), ('min_data', 1187),
              ('bagging_fraction', 0.9), ('feature_fraction', 0.5), ('bagging_freq', 1), ('lambda_l1', 32.0),
              ('lambda_l2', 0.45875)]
param_dic = {'learning_rate': [.001, .002, .004, .008, .02, .04, .08, .2],
             'num_leaves': [8, 16, 32, 64, 128],
             'max_depth': [0, 1, 2, 4, 8],
             'max_bin': [None, 255],
             'min_data': [64, 128, 256, 512, 1024],
             'bagging_fraction': [None, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
             'bagging_freq': [2, 4, 8],
             'feature_fraction': [None, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
             'lambda_l1': [0.0, 4.0, 8, 16, 32],
             'lambda_l2': [0.0, .02, .04, .08, .2, .4, .8, 2.0]}

tune(lgb_model, (train_x, train_y), init_param, param_dic, measure_func=metrics.roc_auc_score, detail=True,
     data_dir=data_dir, kc=(3, 1), random_state=853, task_id='porto_lgb_select')
```
