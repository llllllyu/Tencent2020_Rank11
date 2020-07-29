'''
特征模块，负责将用户和点击序列转换成特征
数据集模块data.py调用此模块来构建特征作为网络的输入
'''

import torch
import numpy as np
import sys
sys.path.append('../../')
from models.lyu.config.config import _C as cfg

# 如果使用deepwalk特征，则将其与w2v特征拼接起来
if cfg.deepwalk:
	vec_creative = np.hstack((np.load('../../models/lyu/data/vec_final/creative.npy'), np.load('../../models/lyu/data/vec_final/dw_creative.npy')))
	vec_ad = np.hstack((np.load('../../models/lyu/data/vec_final/ad.npy'), np.load('../../models/lyu/data/vec_final/dw_ad.npy')))
	vec_pro = np.hstack((np.load('../../models/lyu/data/vec_final/pro.npy'), np.load('../../models/lyu/data/vec_final/dw_pro.npy')))
	vec_adver = np.hstack((np.load('../../models/lyu/data/vec_final/adver.npy'), np.load('../../models/lyu/data/vec_final/dw_adver.npy')))
	vec_industry = np.hstack((np.load('../../models/lyu/data/vec_final/industry.npy'), np.load('../../models/lyu/data/vec_final/dw_industry.npy')))
else:
	vec_creative = np.load('../../models/lyu/data/vec_final/creative.npy')
	vec_ad = np.load('../../models/lyu/data/vec_final/ad.npy')
	vec_pro = np.load('../../models/lyu/data/vec_final/pro.npy')
	vec_adver = np.load('../../models/lyu/data/vec_final/adver.npy')
	vec_industry = np.load('../../models/lyu/data/vec_final/industry.npy')

# 根据任务加载tfidf特征
if cfg.task == 'age':
	creative_tfidf = np.load('../../models/lyu/data/vec_final/creative_tfidf_age.npy')
	ad_tfidf = np.load('../../models/lyu/data/vec_final/ad_tfidf_age.npy')
	adver_tfidf = np.load('../../models/lyu/data/vec_final/adver_tfidf_age.npy')
else:
	creative_tfidf = np.load('../../models/lyu/data/vec_final/creative_tfidf_gender.npy')
	ad_tfidf = np.load('../../models/lyu/data/vec_final/ad_tfidf_gender.npy')
	adver_tfidf = np.load('../../models/lyu/data/vec_final/adver_tfidf_gender.npy')

# 3个ID的tfidf特征拼接成为统计特征
statistics = np.hstack((creative_tfidf, ad_tfidf, adver_tfidf))

import torch.nn as nn
# 用embedding特征构建embedding层
emb_creative = nn.Embedding.from_pretrained(torch.from_numpy(vec_creative))
emb_ad = nn.Embedding.from_pretrained(torch.from_numpy(vec_ad))
emb_pro = nn.Embedding.from_pretrained(torch.from_numpy(vec_pro))
emb_adver = nn.Embedding.from_pretrained(torch.from_numpy(vec_adver))
emb_industry = nn.Embedding.from_pretrained(torch.from_numpy(vec_industry))

def emb_feature(data, time, mask_rate = 0):
	'''
	特征转化函数
	输入用户及点击序列，输出tfidf与embedding特征
	feature1为[creative，ad，pro，adver，industry]拼接的序列embedding特征
	feature2为[ad，pro，adver，industry]拼接的序列embedding特征
	feature3为tfidf特征
	'''
	data = torch.tensor(data)
	feature3 = statistics[data[0][0] - 1]
	feature1 = torch.cat((emb_creative(data[:,1]),emb_ad(data[:,2]),emb_pro(data[:,3]),emb_adver(data[:,5]),emb_industry(data[:,6])),-1)
	feature2 = torch.cat((emb_ad(data[:,2]),emb_pro(data[:,3]),emb_adver(data[:,5]),emb_industry(data[:,6])),-1)
	return feature1, feature2, feature3

