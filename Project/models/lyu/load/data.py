'''
数据集模块，负责训练集和验证集生成
训练主文件train.py调用此模块来构建dataloader
'''

import time
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sys
sys.path.append('../../')
from models.lyu.config.config import _C as cfg  # 引用配置文件

print('开始加载字典')
startload = time.time()
from models.lyu.load import feature  # 引用特征模块
# 加载用户点击序列
user_dict_1 = h5py.File('../../models/lyu/data/npy_final/user_1.h5','r')
user_dict_2 = h5py.File('../../models/lyu/data/npy_final/user_2.h5','r')
# 加载用户标签
label_list = np.load('../../models/lyu/data/npy_final/label.npy')
# 加载广告ID属性
ad_list = np.load('../../models/lyu/data/npy_final/ad_list.npy')

# 用户按照ID顺序分为5折
L = list(range(1, 3000001))
l = [0] * 5
l[0] = list(range(1, 600001))
l[1] = list(range(600001, 1200001))
l[2] = list(range(1200001, 1800001))
l[3] = list(range(1800001, 2400001))
l[4] = list(range(2400001, 3000001))

class TFDataset_age(Dataset):
	'''
	age任务数据集，与gender的区别在于标签
	包括用户序列提取和特征提取
	输出每个用户的embedding序列和tfidf特征
	'''
	def __init__(self, train = True, mask_rate = 0):
		'''
		将用户的点击序列提取出来并放在内存
		'''
		datas = []
		self.mask_rate = mask_rate
		self.train = train
		if train:  # 对训练集采用数据增强
			for user in list(set(L) - set(l[cfg.fold-1])):
				label = label_list[user - 1][0]  # age的标签
				# 提取用户点击的creative序列，只截取后max_len长度
				if user < 2000001:
					data = user_dict_1[str(user)][-cfg.max_len:]
				else:
					data = user_dict_2[str(user)][-cfg.max_len:]
				time = []
				# 所有序列放在内存
				datas.append((user, data, time, label))
				if cfg.augmentation:  # 采用数据叠加的情况下，将序列倒序以及打乱，一同放入训练集中，默认false
					data_rever = data.copy()
					data_rand = data.copy()
					data_rever = data_rever[::-1]
					np.random.shuffle(data_rand)
					datas.append((user, data_rever, time, label))
					datas.append((user, data_rand, time, label))
		else:
			# 验证集不采用数据增强
			for user in l[cfg.fold-1]:
				label = label_list[user - 1][0]
				if user < 2000001:
					data = user_dict_1[str(user)][-cfg.max_len:]
				else:
					data = user_dict_2[str(user)][-cfg.max_len:]
				time = []
				datas.append((user, data, time, label))

		self.datas = datas

	def __getitem__(self, index):
		'''
		将数据集中的用户和点击序列转化为embedding和tfidf特征
		'''
		user, data, time, label = self.datas[index]
		if self.train and not cfg.augmentation:  # 不采用数据叠加的情况下，根据概率将序列倒序以及打乱，只针对训练集
			rate = random.random()
			if rate < cfg.threshold:
				data = data[::-1]
			elif rate > cfg.threshold * 2:
				np.random.shuffle(data)
		# 将用户点击creative序列扩展为[user, creative，ad, pro, pro_cate, adver, industry]
		inputs = list(map(lambda creative: [user, creative] + list(ad_list[creative - 1]), data))
		# 序列转化为特征
		inputs1, inputs2, inputs3 = feature.emb_feature(inputs, time, self.mask_rate)
		return inputs1, inputs2, inputs3, label

	def __len__(self):
		return len(self.datas)

class TFDataset_gender(Dataset):
	'''
	gender任务数据集，与age的区别在于标签
	包括用户序列提取和特征提取
	输出每个用户的embedding序列和tfidf特征
	'''
	def __init__(self, train = True, mask_rate = 0):
		'''
		将用户的点击序列提取出来并放在内存
		'''
		datas = []
		self.mask_rate = mask_rate
		self.train = train
		if train:  # 对训练集采用数据增强
			for user in list(set(L) - set(l[cfg.fold-1])):
				label = label_list[user - 1][1]  # age的标签
				# 提取用户点击的creative序列，只截取后max_len长度
				if user < 2000001:
					data = user_dict_1[str(user)][-cfg.max_len:]
				else:
					data = user_dict_2[str(user)][-cfg.max_len:]
				time = []
				# 所有序列放在内存
				datas.append((user, data, time, label))
				if cfg.augmentation:  # 采用数据叠加的情况下，将序列倒序以及打乱，一同放入训练集中，默认false
					data_rever = data.copy()
					data_rand = data.copy()
					data_rever = data_rever[::-1]
					np.random.shuffle(data_rand)
					datas.append((user, data_rever, time, label))
					datas.append((user, data_rand, time, label))
		else:
			# 验证集不采用数据增强
			for user in l[cfg.fold-1]:
				label = label_list[user - 1][0]
				if user < 2000001:
					data = user_dict_1[str(user)][-cfg.max_len:]
				else:
					data = user_dict_2[str(user)][-cfg.max_len:]
				time = []
				datas.append((user, data, time, label))

		self.datas = datas

	def __getitem__(self, index):
		'''
		将数据集中的用户和点击序列转化为embedding和tfidf特征
		'''
		user, data, time, label = self.datas[index]
		if self.train and not cfg.augmentation:  # 不采用数据叠加的情况下，根据概率将序列倒序以及打乱，只针对训练集
			rate = random.random()
			if rate < cfg.threshold:
				data = data[::-1]
			elif rate > cfg.threshold * 2:
				np.random.shuffle(data)
		# 将用户点击creative序列扩展为[user, creative，ad, pro, pro_cate, adver, industry]
		inputs = list(map(lambda creative: [user, creative] + list(ad_list[creative - 1]), data))
		# 转化为特征，inputs1和inputs2为序列embedding，inputs3为用户tfidf
		inputs1, inputs2, inputs3 = feature.emb_feature(inputs, time, self.mask_rate)
		return inputs1, inputs2, inputs3, label

	def __len__(self):
		return len(self.datas)

def tfcollate_fn(train_data_all):
	'''
	dataloader的collate_fn，用于将同一个batch中的数据按照序列长度排序并且pad
	'''
	train_data_all.sort(key=lambda data: len(data[0]), reverse=True)  # 数据按照序列长度从小到大排序
	train_data1 = [data[0] for data in train_data_all]  # inputs1特征单独提取成列表
	train_data2 = [data[1] for data in train_data_all]  # inputs2特征单独提取成列表
	train_data3 = torch.from_numpy(np.array([data[2] for data in train_data_all])).float()  # inputs3特征转化为tensor
	label = torch.LongTensor([data[3] for data in train_data_all])  # label转化为tensor
	data_length = [len(data) for data in train_data1]  # 保存原有序列长度以用于LSTM计算
	train_data1 = pad_sequence(train_data1, batch_first=True, padding_value=0)  # inputs1特征padding
	train_data2 = pad_sequence(train_data2, batch_first=True, padding_value=0)  # inputs2特征padding
	return train_data1, train_data2, train_data3, data_length, label

print('开始加载数据')

if cfg.task == 'age':
	train_data = TFDataset_age(mask_rate=cfg.mask_rate)
	test_data = TFDataset_age(False)
else:
	train_data = TFDataset_gender(mask_rate=cfg.mask_rate)
	test_data = TFDataset_gender(False)

trainloader = DataLoader(dataset=train_data, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers, collate_fn=tfcollate_fn)
testloader = DataLoader(dataset=test_data, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, collate_fn=tfcollate_fn)
print('数据集加载完成')
L_train = len(trainloader)
print('训练集长度:',L_train)
L_test = len(testloader)
print('测试集长度:',L_test)
endload = time.time() - startload
print('加载时间：%.2fmin' % (endload / 60))


user_dict_1.close()
user_dict_2.close()