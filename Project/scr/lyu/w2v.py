'''
w2v训练
'''

import logging
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
import time
import h5py

# 定义callback类用于观察每个epoch结束后的loss情况
class EpochLogger(CallbackAny2Vec):
	def __init__(self):
		self.epoch = 1
		self.loss_save = 0
		self.time = time.time()
	def on_epoch_begin(self, model):
		print("Epoch #{} start".format(self.epoch))
	def on_epoch_end(self, model):
		print("Epoch #{} end, used time: {}min".format(self.epoch,(time.time()-self.time)/60))
		print('loss: {}'.format(Word2Vec.get_latest_training_loss(model) - self.loss_save))
		self.epoch += 1
		self.loss_save = Word2Vec.get_latest_training_loss(model)

# 加载用户点击序列
user_dict_1 = h5py.File('../../models/lyu/data/npy_final/user_1.h5','r')
user_dict_2 = h5py.File('../../models/lyu/data/npy_final/user_2.h5','r')
# 加载广告ID属性
ad_list = np.load('../../models/lyu/data/npy_final/ad_list.npy')

# 训练creative
item = 'creative'
print(item)
word_list = []
starttime = time.time()

# 将需要训练的ID序列拼成列表
for user in tqdm(range(1, 2000001)):
	l = list(user_dict_1[str(user)][:])
# 	l = [ad_list[i-1][4] for i in l]
# 	l = list(filter(lambda number: number > 0, l))
	l = [str(i) for i in l]
	word_list.append(l)

for user in tqdm(range(2000001, 4000001)):
	l = list(user_dict_2[str(user)][:])
# 	l = [ad_list[i-1][4] for i in l]
# 	l = list(filter(lambda number: number > 0, l))
	l = [str(i) for i in l]
	word_list.append(l)

print(len(word_list))

epoch_logger = EpochLogger()
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)  # 这一行可以使得w2v的训练过程打印出来
model = Word2Vec(word_list,sg=1,size=300,window=20,min_count=1,workers=6,iter=10,compute_loss=True,callbacks=[epoch_logger])  # 训练w2v

# 将训练好的特征按照ID顺序排列拼成矩阵，其中索引为0的代表缺失值特征
vec = np.zeros((4445721,300)).astype(np.float32)  # 使用float32以减少内存
for i in range(1,4445721):
	vec[i] = model.wv[str(i)]

np.save('../../models/lyu/data/vec_final/'+item+'.npy', vec)

# 训练ad
item = 'ad'
print(item)
word_list = []
starttime = time.time()
for user in tqdm(range(1, 2000001)):
	l = list(user_dict_1[str(user)][:])
	l = [ad_list[i-1][0] for i in l]
# 	l = list(filter(lambda number: number > 0, l))
	l = [str(i) for i in l]
	word_list.append(l)

for user in tqdm(range(2000001, 4000001)):
	l = list(user_dict_2[str(user)][:])
	l = [ad_list[i-1][0] for i in l]
# 	l = list(filter(lambda number: number > 0, l))
	l = [str(i) for i in l]
	word_list.append(l)

print(len(word_list))

epoch_logger = EpochLogger()
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
model = Word2Vec(word_list,sg=1,size=300,window=20,min_count=1,workers=6,iter=10,compute_loss=True,callbacks=[epoch_logger])

vec = np.zeros((3812203,300)).astype(np.float32)
for i in range(1,3812203):
	vec[i] = model.wv[str(i)]

np.save('../../models/lyu/data/vec_final/'+item+'.npy', vec)

# 训练product
item = 'product'
print(item)
word_list = []
starttime = time.time()
for user in tqdm(range(1, 2000001)):
	l = list(user_dict_1[str(user)][:])
	l = [ad_list[i-1][1] for i in l]
	l = list(filter(lambda number: number > 0, l))  # 去掉0的项
	l = [str(i) for i in l]
	word_list.append(l)

for user in tqdm(range(2000001, 4000001)):
	l = list(user_dict_2[str(user)][:])
	l = [ad_list[i-1][1] for i in l]
	l = list(filter(lambda number: number > 0, l))
	l = [str(i) for i in l]
	word_list.append(l)

print(len(word_list))

epoch_logger = EpochLogger()
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
model = Word2Vec(word_list,sg=1,size=150,window=20,min_count=1,workers=6,iter=15,compute_loss=True,callbacks=[epoch_logger])

vec = np.zeros((44315,150)).astype(np.float32)
for i in range(1,44315):
	vec[i] = model.wv[str(i)]

np.save('../../models/lyu/data/vec_final/'+item+'.npy', vec)

# 训练advertiser
item = 'adver'
print(item)
word_list = []
starttime = time.time()
for user in tqdm(range(1, 2000001)):
	l = list(user_dict_1[str(user)][:])
	l = [ad_list[i-1][3] for i in l]
# 	l = list(filter(lambda number: number > 0, l))
	l = [str(i) for i in l]
	word_list.append(l)

for user in tqdm(range(2000001, 4000001)):
	l = list(user_dict_2[str(user)][:])
	l = [ad_list[i-1][3] for i in l]
# 	l = list(filter(lambda number: number > 0, l))
	l = [str(i) for i in l]
	word_list.append(l)

print(len(word_list))

epoch_logger = EpochLogger()
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
model = Word2Vec(word_list,sg=1,size=150,window=20,min_count=1,workers=6,iter=15,compute_loss=True,callbacks=[epoch_logger])

vec = np.zeros((62966,150)).astype(np.float32)
for i in range(1,62966):
	vec[i] = model.wv[str(i)]

np.save('../../models/lyu/data/vec_final/'+item+'.npy', vec)

# 训练industry
item = 'industry'
print(item)
word_list = []
starttime = time.time()
for user in tqdm(range(1, 2000001)):
	l = list(user_dict_1[str(user)][:])
	l = [ad_list[i-1][4] for i in l]
	l = list(filter(lambda number: number > 0, l))
	l = [str(i) for i in l]
	word_list.append(l)

for user in tqdm(range(2000001, 4000001)):
	l = list(user_dict_2[str(user)][:])
	l = [ad_list[i-1][4] for i in l]
	l = list(filter(lambda number: number > 0, l))
	l = [str(i) for i in l]
	word_list.append(l)

print(len(word_list))

epoch_logger = EpochLogger()
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
model = Word2Vec(word_list,sg=1,size=50,window=20,min_count=1,workers=6,iter=30,compute_loss=True,callbacks=[epoch_logger])

vec = np.zeros((336,50)).astype(np.float32)
for i in range(1,336):
	vec[i] = model.wv[str(i)]

np.save('../../models/lyu/data/vec_final/'+item+'.npy', vec)