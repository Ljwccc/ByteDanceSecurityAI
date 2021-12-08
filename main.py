#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from sklearn.model_selection import KFold
import os

from gensim.models import Word2Vec

from utils import *
from model import *

# # 读取数据

# In[2]:


data_dir = './'

user_request = pd.read_csv(data_dir+'1-data_feature1_date1.txt')
user_request = pd.concat([user_request, pd.read_csv(data_dir+'2-data_feature1_date2.txt')],ignore_index=True)
user_info = pd.read_csv(data_dir+'3-data_feature2_date1.txt')
user_info = pd.concat([user_info, pd.read_csv(data_dir+'4-data_feature2_date2.txt')],ignore_index=True)

test = pd.read_csv(data_dir+'to_prediction_model.csv',header=None,names=['user'])

print('user_request shape:',user_request.shape)
print('user_info shape:',user_info.shape)
print('test shape:',test.shape)

# 测试集数据
test = test.merge(user_info, how='left', on='user',)
print('test merge shape:',test.shape)
# 带label的数据
user_info = user_info[user_info['user_status'].notna()]
print('user_info with label shape:',user_info.shape)
# 合并所有数据
user_info = pd.concat([user_info,test],ignore_index=True)

print('user_info with label and test shape:',user_info.shape)


# In[3]:


# 正负样本数量
print('正样本数量：',user_info[user_info['user_status']==1].shape[0])
print('负样本数量：',user_info[user_info['user_status']==0].shape[0])
print('无标签数量：',user_info[user_info['user_status'].isnull()].shape[0])


# # 特征工程

# In[4]:


# 从序列特征中提取用户的设备信息、channel信息和app_version信息

# 先group
user_request_cat_group = user_request.groupby(['request_user'],as_index=False)

user_request_device_type = user_request_cat_group['request_device_type'].agg({'device_type_list':list})
user_request_channel = user_request_cat_group['request_app_channel'].agg({'channel_list':list})
user_request_app_version = user_request_cat_group['request_app_version'].agg({'app_version_list':list})

user_request_device_type['device_type'] = user_request_device_type['device_type_list'].apply(lambda x:x[0])
user_request_channel['channel'] = user_request_channel['channel_list'].apply(lambda x:x[0])
user_request_app_version['app_version'] = user_request_app_version['app_version_list'].apply(lambda x:x[-1])

user_feat_from_action = pd.concat([user_request_device_type[['request_user','device_type']],user_request_channel[['channel']]
                                 ,user_request_app_version[['app_version']]],axis=1).rename(columns={'request_user':'user'})

del user_request_device_type
del user_request_channel
del user_request_app_version
del user_request_cat_group
gc.collect()


# ## 用户基础特征

# In[5]:


# 类别特征
cat_cols = ['user_name','user_profile','user_register_type','user_register_app','user_least_login_app', 'user_freq_ip',
            'user_freq_ip_3','device_type','channel','app_version','user_freq_ip_2','user_freq_ip_1',]


# 手工类别特征
user_info['user_freq_ip_3'] = user_info['user_freq_ip'].apply(lambda x:'.'.join(str(x).split('.')[:3])) # 常用ip取前3位
user_info['user_freq_ip_2'] = user_info['user_freq_ip'].apply(lambda x:'.'.join(str(x).split('.')[:2])) # 常用ip取前2位
user_info['user_freq_ip_1'] = user_info['user_freq_ip'].apply(lambda x:'.'.join(str(x).split('.')[:1])) # 常用ip取前1位

# 合并从request中提取的基础特征
user_info = user_info.merge(user_feat_from_action,on='user',how='left')
del user_feat_from_action

# 类别特征的频次
for col in cat_cols:
    user_info = freq_enc(user_info,col)
    
# 对所有类别特征做label_encoder
user_info = label_enc(user_info,cat_cols)

# 点赞量，关注量等交叉特征，直接梭哈所有乘除法
num_cols = ['user_fans_num','user_follow_num','user_post_num','user_post_like_num']

for col1 in num_cols:
    for col2 in [col for col in num_cols if col!=col1]:
        user_info[f'{col1}_{col2}_mul'] = user_info[col1]*user_info[col2]
        user_info[f'{col1}_{col2}_div'] = user_info[col1]/(user_info[col2]+1e-3)


# In[6]:


# 类别特征下粉丝量、关注量、发帖量、被点赞量、请求数量的统计值
num_cols = ['user_fans_num','user_follow_num','user_post_num','user_post_like_num']

for cat_col in cat_cols:
    
    cat_group = user_info.groupby(cat_col)[num_cols]
    # 平均值
    cat_col_stat = cat_group.transform(np.mean)
    cat_col_stat.rename(columns={name:f'{name}_{cat_col}_mean' for name in cat_col_stat.columns},inplace=True)
    user_info = pd.concat([user_info,cat_col_stat],axis=1)
    # 和
    cat_col_stat = cat_group.transform(np.sum)
    cat_col_stat.rename(columns={name:f'{name}_{cat_col}_sum' for name in cat_col_stat.columns},inplace=True)
    user_info = pd.concat([user_info,cat_col_stat],axis=1)
    # 方差
    cat_col_stat = cat_group.transform(np.std)
    cat_col_stat.rename(columns={name:f'{name}_{cat_col}_std' for name in cat_col_stat.columns},inplace=True)
    user_info = pd.concat([user_info,cat_col_stat],axis=1)
    
del cat_col_stat


# # 序列特征

# In[7]:


# 用户请求序列特征


user_request_list = user_request.groupby(['request_user'],as_index=False)['request_target'].agg({'request_list':list})

# 先按照时间进行排序
user_request = user_request.sort_values(by='request_time',)
# 请求的数量
user_action_feat = user_request.groupby(['request_user'],as_index=False)['request_user'].agg({'request_num':'count'})

# 用户请求的时间统计量，但是80%的用户只有一次请求行为
user_action_feat_temp = user_request.groupby(['request_user'],as_index=False)['request_time'].agg({'time_list':list})
user_action_feat = user_action_feat.merge(user_action_feat_temp,on='request_user',how='left')
user_action_feat['time_min'] = user_action_feat['time_list'].apply(min)
user_action_feat['time_max'] = user_action_feat['time_list'].apply(max)
user_action_feat['time_var'] = user_action_feat['time_list'].apply(np.var)
user_action_feat['time_max-min'] = user_action_feat['time_list'].apply(lambda x:np.max(x)-np.min(x))

# 时间间隔的平均值，最大值，最小值，方差
def diff_value(time):
    time_shift = list(time[1:])
    time_shift.append(time[-1])

    diff_time = time_shift-time
    return diff_time
user_action_feat['diff_time'] = user_action_feat['time_list'].apply(lambda x: diff_value(np.array(x)))
user_action_feat['diff_time_max'] = user_action_feat['diff_time'].apply(max)
user_action_feat['diff_time_var'] = user_action_feat['diff_time'].apply(np.var)
user_action_feat['diff_time_mean'] = user_action_feat['diff_time'].apply(np.mean)
user_action_feat['diff_time_min'] = user_action_feat['diff_time'].apply(min)


# In[8]:
# 用户请求序列做一个embedding
user_request_list = user_request.groupby(['request_user'],as_index=False)['request_target'].agg({'request_list':list})
del user_request
sentences = user_request_list['request_list'].values.tolist()
emb_size = 64
for i in range(len(sentences)):
    sentences[i] = [str(x) for x in sentences[i]]  # 数字转化为字符串用于训练w2v

model = Word2Vec(sentences, size=emb_size, window=5, min_count=5, sg=0, hs=0, seed=1, iter=5, workers=8)

emb_matrix = []
for seq in sentences:
    vec = []
    for w in seq:
        if w in model.wv.vocab:
            vec.append(model.wv[w])
    if len(vec) > 0:
        emb_matrix.append(np.mean(vec, axis=0))
    else:
        emb_matrix.append([0] * emb_size)
emb_matrix = np.array(emb_matrix)

emb_size = 64
for i in range(emb_size):
    user_request_list['action_emb_{}'.format(i)] = emb_matrix[:, i]

user_request_list = user_request_list.drop(['request_list'],axis=1)
user_action_feat = user_action_feat.merge(user_request_list,how='left',on='request_user')

# 合并基础特征和序列特征
user_info = user_info.merge(user_action_feat,how='left',on='user')

del user_action_feat
gc.collect()


# In[9]:


# 制作训练集、测试集和Label
train = user_info[user_info['user_status'].notna()]
test = user_info[user_info['user_status'].isna()]
y = train['user_status']
print('train shape:',train.shape)
print('test shape:',test.shape)


# In[10]:


folds = KFold(n_splits=10, shuffle=True, random_state=546789)
oof_preds, test_preds, importances = train_model_cat(train, test, y, folds, cat_cols)


# In[11]:


test_preds['label'] = test_preds['label'].apply(lambda x:0 if x<0.4 else 1)

test_preds = test_preds.drop_duplicates(subset=['user'])   # 去除相同的user
# 生成结果
test_preds[['user', 'label']].to_csv('submission.csv', index=False, header=None)

