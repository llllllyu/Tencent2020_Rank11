'''
将队友生成的tfidf特征存为numpy矩阵
'''

import numpy as np
import pickle

f = open('../../models/data/tfidf/final_tfidf11_creative_id_age.pkl','rb')
dfm = pickle.load(f)
dfm.fillna(0.1, inplace=True)  # 用均值概率来代替缺失值
tfidf = dfm.values.astype(np.float32)  # 使用float32减少内存
np.save('../../models/lyu/data/vec_final/creative_tfidf_age.npy',tfidf)

f = open('../../models/data/tfidf/final_tfidf11_creative_id_gender.pkl','rb')
dfm = pickle.load(f)
dfm.fillna(0.5, inplace=True)
tfidf = dfm.values.astype(np.float32)
np.save('../../models/lyu/data/vec_final/creative_tfidf_gender.npy',tfidf)

f = open('../../models/data/tfidf/final__tfidf11_ad_id_age.pkl','rb')
dfm = pickle.load(f)
dfm.fillna(0.1, inplace=True)
tfidf = dfm.values.astype(np.float32)
np.save('../../models/lyu/data/vec_final/ad_tfidf_age.npy',tfidf)

f = open('../../models/data/tfidf/final__tfidf11_ad_id_gender.pkl','rb')
dfm = pickle.load(f)
dfm.fillna(0.5, inplace=True)
tfidf = dfm.values.astype(np.float32)
np.save('../../models/lyu/data/vec_final/ad_tfidf_gender.npy',tfidf)

f = open('../../models/data/tfidf/final__tfidf11_advertiser_id_gender.pkl','rb')
dfm = pickle.load(f)
dfm.fillna(0.5, inplace=True)
tfidf = dfm.values.astype(np.float32)
np.save('../../models/lyu/data/vec_final/adver_tfidf_gender.npy',tfidf)

f = open('../../models/data/tfidf/final__tfidf11_advertiser_id_age.pkl','rb')
dfm = pickle.load(f)
dfm.fillna(0.1, inplace=True)
tfidf = dfm.values.astype(np.float32)
np.save('../../models/lyu/data/vec_final/adver_tfidf_age.npy',tfidf)