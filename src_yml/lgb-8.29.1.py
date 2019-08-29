# %%
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import record_evaluation
from sklearn.metrics import *
import nni
# %%
path_fts_valid = 'tmp/fts_valid.npy'
path_lbs_valid = 'tmp/lbs_valid.npy'
path_fts_train = 'tmp/fts_train.npy'
path_lbs_train = 'tmp/lbs_train.npy'


# %%
fts_valid = np.load(path_fts_valid)
lbs_valid = np.load(path_lbs_valid)
fts_train = np.load(path_fts_train)
lbs_train = np.load(path_lbs_train)
fts_train.shape, lbs_train.shape, fts_valid.shape, lbs_valid.shape
# %%


def report_intermediate_result(env):
    nni.report_intermediate_result(env.evaluation_result_list[0][2])
    # print(env.evaluation_result_list)


# %%
params = nni.get_next_parameter()
lgb = LGBMClassifier(n_jobs=-1, **params)

lgb.fit(fts_train, lbs_train,
        eval_set=[(fts_valid, lbs_valid)],
        eval_metric='multi_error',
        verbose=100,
        callbacks=[report_intermediate_result],
        early_stopping_rounds=100)


# %%
preds = lgb.predict(fts_valid)
score = accuracy_score(lbs_valid, preds)
nni.report_final_result(1-score)

# %%
