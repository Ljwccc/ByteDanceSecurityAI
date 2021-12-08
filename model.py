from catboost import CatBoostClassifier

from sklearn.metrics import f1_score,roc_auc_score
import pandas as pd
import numpy as np
import gc


useless_cols = ['user','user_status']

def train_model_cat(data_, test_, y_, folds_, cat_cols, semi_data_=None):
    oof_preds = np.zeros(data_.shape[0])  # 验证集预测结果
    sub_preds = np.zeros(test_.shape[0])  # 测试集预测结果
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in useless_cols]
    
    # 半监督每批训练数据
    if not semi_data_ is None:
        semi_num = semi_data_.shape[0]/5
        semi_y = semi_data_['user_status']
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        
        if not semi_data_ is None:
            semi_data_batch = semi_data_[feats].iloc[int(n_fold*semi_num):int((n_fold+1)*semi_num)]
            semi_y_batch = semi_y.iloc[int(n_fold*semi_num):int((n_fold+1)*semi_num)]
        
            trn_x, trn_y = pd.concat([data_[feats].iloc[trn_idx],semi_data_batch]), pd.concat([y_.iloc[trn_idx],semi_y_batch])
        else:
            trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]   # 训练集数据
            
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]   # 验证集数据
       
        clf = CatBoostClassifier(
            iterations=6000,
            learning_rate=0.08,  # 0.08
            # num_leaves=2**5,
            eval_metric='AUC',
            task_type="CPU",
            loss_function='Logloss',
            colsample_bylevel = 0.8,
            
            subsample=0.9,   # 0.9
            max_depth=7,
            reg_lambda = 0.3,
            verbose=-1,
        )
        
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                verbose_eval=300, early_stopping_rounds=100,  # 这个参数有点小，可以再大一点
                cat_features = cat_cols
               )
        oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]   # 验证集结果
        
        sub_preds += clf.predict_proba(test_[feats])[:, 1] / folds_.n_splits  # 测试集结果
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    
    oof_preds = [1 if i >= 0.4 else 0 for i in oof_preds]
    print('Full F1 score %.6f' % f1_score(y_, oof_preds))
    
    test_['label'] = sub_preds

    return oof_preds, test_[['user', 'label']], feature_importance_df