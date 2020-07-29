'''
将队友生成的deepwalk特征按照ID排列存为numpy矩阵
'''

import numpy as np
import joblib

emb = np.load('../../models/data/deepwalk/emb_creative_id.npy')  # 原始deepwalk特征向量
index = joblib.load('../../models/data/deepwalk/index/cre.pkl')  # deepwalk向量的ID与索引的映射
vec = np.zeros((4445721,192)).astype(np.float32)  # 使用float32减少内存

for i in range(1,4445721):
	vec[i] = emb[index[str(i)]]  # 按照ID重新排列

np.save('../../models/lyu/data/vec_final/dw_creative.npy', vec)

emb = np.load('../../models/data/deepwalk/emb_ad_id.npy')
index = joblib.load('../../models/data/deepwalk/index/ad.pkl')
vec = np.zeros((3812203,192)).astype(np.float32)

for i in range(1,3812203):
	vec[i] = emb[index[str(i)]]

np.save('../../models/lyu/data/vec_final/dw_ad.npy', vec)

emb = np.load('../../models/data/deepwalk/emb_advvertiser_id.npy')
index = joblib.load('../../models/data/deepwalk/index/adv.pkl')
vec = np.zeros((62966,128)).astype(np.float32)

for i in range(1,62966):
	vec[i] = emb[index[str(i)]]

np.save('../../models/lyu/data/vec_final/dw_adv.npy', vec)

emb = np.load('../../models/data/deepwalk/emb_product_id.npy')
index = joblib.load('../../models/data/deepwalk/index/pro.pkl')
vec = np.zeros((44315,128)).astype(np.float32)

for i in range(1,44315):
	vec[i] = emb[index[str(i)]]

np.save('../../models/lyu/data/vec_final/dw_product.npy', vec)

emb = np.load('../../models/data/deepwalk/emb_industry.npy')
index = joblib.load('../../models/data/deepwalk/index/ind.pkl')
vec = np.zeros((336,64)).astype(np.float32)

for i in range(1,336):
	vec[i] = emb[index[str(i)]]

np.save('../../models/lyu/data/vec_final/dw_industry.npy', vec)