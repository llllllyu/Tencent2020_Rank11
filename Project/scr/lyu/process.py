'''
推理生成训练集输出概率文件
'''
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import h5py

import sys
sys.path.append('../../')

from models.lyu.config.config import _C as cfg

import os
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda
print('cuda:',cfg.cuda)
print('task:',cfg.task)

if cfg.cudnn_benchmark:
	torch.backends.cudnn.benchmark = True
	print('启用cudnn加速')

starttime = time.time()
print('开始加载字典')
import load.feature as feature
# 加载用户点击序列
user_dict_1 = h5py.File('../../models/lyu/data/npy_final/user_1.h5','r')
user_dict_2 = h5py.File('../../models/lyu/data/npy_final/user_2.h5','r')
# 加载广告ID属性
ad_list = np.load('../../models/lyu/data/npy_final/ad_list.npy')
print('加载完成')
print('时间:{:.2f}min'.format((time.time() - starttime) / 60))

# 用户按照ID顺序分为5折
L = list(range(1, 3000001))
l = [0] * 5
l[0] = list(range(1, 600001))
l[1] = list(range(600001, 1200001))
l[2] = list(range(1200001, 1800001))
l[3] = list(range(1800001, 2400001))
l[4] = list(range(2400001, 3000001))

class TFDataset(Dataset):
	'''
	测试集的类，按照不同的fold加载不同的用户数据
	包括用户序列提取和特征提取
	输出每个用户的embedding序列和tfidf特征
	'''
	def __init__(self, fold):
		'''
		将用户的点击序列提取出来并放在内存
		'''
		datas = []
		for user in l[fold - 1]:
			label = user - 1  # 这里要记住每一个序列所对应的用户，-1是为了直接得到储存矩阵的索引
			# 提取用户点击的creative序列，只截取后max_len长度
			if user < 2000001:
				data = user_dict_1[str(user)][-cfg.max_len:]
			else:
				data = user_dict_2[str(user)][-cfg.max_len:]
			data_rever = data.copy()
			data_rever = data_rever[::-1]  # 对序列进行倒序
			time = []
			datas.append((user, data, data_rever, time, label))  # 原始序列和倒序序列一同放入数据集

		self.datas = datas

	def __getitem__(self, index):
		'''
		将数据集中的用户和点击序列转化为embedding和tfidf特征
		'''
		user, data, data_rever, time, label = self.datas[index]
		# 将用户点击creative序列扩展为[user, creative，ad, pro, pro_cate, adver, industry]
		inputs = list(map(lambda creative: [user, creative] + list(ad_list[creative - 1]), data))
		# 转化为特征，inputs1和inputs2为序列embedding，inputs5为用户tfidf
		inputs1, inputs2, inputs5 = feature.tf_feature(inputs, time, 0)

		# 对于倒序序列也同样提取特征
		inputs = list(map(lambda creative: [user, creative] + list(ad_list[creative - 1]), data_rever))
		inputs3, inputs4, _ = feature.tf_feature(inputs, time, 0)
		return inputs1, inputs2, inputs3, inputs4, inputs5, label

	def __len__(self):
		return len(self.datas)

def tfcollate_fn(train_data_all):
	'''
	dataloader的collate_fn，用于将同一个batch中的数据按照序列长度排序并且pad
	'''
	train_data_all.sort(key=lambda data: len(data[0]), reverse=True)  # 数据按照序列长度从小到大排序
	# inputs1到inputs4的特征分别单独提取成列表
	train_data1 = [data[0] for data in train_data_all]
	train_data2 = [data[1] for data in train_data_all]
	train_data3 = [data[2] for data in train_data_all]
	train_data4 = [data[3] for data in train_data_all]
	train_data5 = torch.from_numpy(np.array([data[4] for data in train_data_all])).float()  # inputs3特征转化为tensor
	label = torch.LongTensor([data[5] for data in train_data_all])  # label（这里是user-1）转化为tensor
	data_length = [len(data) for data in train_data1]  # 保存原有序列长度以用于LSTM计算
	# 各个特征padding
	train_data1 = pad_sequence(train_data1, batch_first=True, padding_value=0)
	train_data2 = pad_sequence(train_data2, batch_first=True, padding_value=0)
	train_data3 = pad_sequence(train_data3, batch_first=True, padding_value=0)
	train_data4 = pad_sequence(train_data4, batch_first=True, padding_value=0)
	return train_data1, train_data2, train_data3, train_data4, train_data5, data_length, label

# 加载每折的训练集
print('开始加载数据')
test_data = TFDataset(1)
testloader1 = DataLoader(dataset=test_data, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, collate_fn=tfcollate_fn)
test_data = TFDataset(2)
testloader2 = DataLoader(dataset=test_data, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, collate_fn=tfcollate_fn)
test_data = TFDataset(3)
testloader3 = DataLoader(dataset=test_data, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, collate_fn=tfcollate_fn)
test_data = TFDataset(4)
testloader4 = DataLoader(dataset=test_data, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, collate_fn=tfcollate_fn)
test_data = TFDataset(5)
testloader5 = DataLoader(dataset=test_data, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, collate_fn=tfcollate_fn)

print('加载完成')
print('时间:{:.2f}min'.format((time.time() - starttime) / 60))

from models.lyu.model import model
# 分别加载五折模型
net1 = model.Advanced_Trans_LSTM(256,32,16,encoder_num=1,head=2,mode=cfg.trans_mode)
net1 = nn.DataParallel(net1)
net1.cuda()
net1.load_state_dict(torch.load('../../models/lyu/save/' + cfg.task + '/Advanced_' + cfg.trans_mode + '_LSTM_'+ cfg.dw + cfg.a + '_params_1.pkl'))
net1.eval()

net2 = model.Advanced_Trans_LSTM(256,32,16,encoder_num=1,head=2,mode=cfg.trans_mode)
net2 = nn.DataParallel(net2)
net2.cuda()
net2.load_state_dict(torch.load('../../models/lyu/save/' + cfg.task + '/Advanced_' + cfg.trans_mode + '_LSTM_'+ cfg.dw + cfg.a + '_params_2.pkl'))
net2.eval()

net3 = model.Advanced_Trans_LSTM(256,32,16,encoder_num=1,head=2,mode=cfg.trans_mode)
net3 = nn.DataParallel(net3)
net3.cuda()
net3.load_state_dict(torch.load('../../models/lyu/save/' + cfg.task + '/Advanced_' + cfg.trans_mode + '_LSTM_'+ cfg.dw + cfg.a + '_params_3.pkl'))
net3.eval()

net4 = model.Advanced_Trans_LSTM(256,32,16,encoder_num=1,head=2,mode=cfg.trans_mode)
net4 = nn.DataParallel(net4)
net4.cuda()
net4.load_state_dict(torch.load('../../models/lyu/save/' + cfg.task + '/Advanced_' + cfg.trans_mode + '_LSTM_'+ cfg.dw + cfg.a + '_params_4.pkl'))
net4.eval()

net5 = model.Advanced_Trans_LSTM(256,32,16,encoder_num=1,head=2,mode=cfg.trans_mode)
net5 = nn.DataParallel(net5)
net5.cuda()
net5.load_state_dict(torch.load('../../models/lyu/save/' + cfg.task + '/Advanced_' + cfg.trans_mode + '_LSTM_'+ cfg.dw + cfg.a + '_params_5.pkl'))
net5.eval()

# 生成储存结果的矩阵
if cfg.task == 'age':
	raw_prob = np.zeros((3000000,10))
else:
	raw_prob = np.zeros((3000000,2))
# 每一折只用对应的模型来推理
with torch.no_grad():
	for i, data in enumerate(testloader1, 0):
		inputs1, inputs2, inputs3, inputs4, inputs5, length, labels = data
		inputs1 = inputs1.cuda()
		inputs2 = inputs2.cuda()
		inputs3 = inputs3.cuda()
		inputs4 = inputs4.cuda()
		inputs5 = inputs5.cuda()
		outputs = net1(inputs1, inputs2, inputs5, length) + net1(inputs3, inputs4, inputs5, length)
		# 正反向结果取平均，并求softmax得到最终概率
		outputs /= 2
		batch = outputs.size(0)
		label = labels.numpy()
		outputs = F.softmax(outputs, dim=1)
		predicted = outputs.cpu().numpy()
		# 按照user的索引放入矩阵
		for j in range(batch):
			raw_prob[label[j]] = predicted[j]
		if i % 100 == 0:
			print('num:%.4f,time:%.2fm' % (i / len(testloader1) / 5, (time.time() - starttime) / 60))
	for i, data in enumerate(testloader2, 0):
		inputs1, inputs2, inputs3, inputs4, inputs5, length, labels = data
		inputs1 = inputs1.cuda()
		inputs2 = inputs2.cuda()
		inputs3 = inputs3.cuda()
		inputs4 = inputs4.cuda()
		inputs5 = inputs5.cuda()
		outputs = net2(inputs1, inputs2, inputs5, length) + net2(inputs3, inputs4, inputs5, length)
		outputs /= 2
		batch = outputs.size(0)
		label = labels.numpy()
		outputs = F.softmax(outputs, dim=1)
		predicted = outputs.cpu().numpy()
		for j in range(batch):
			raw_prob[label[j]] = predicted[j]
		if i % 100 == 0:
			print('num:%.4f,time:%.2fm' % (i / len(testloader2) / 5 + 0.2, (time.time() - starttime) / 60))
	for i, data in enumerate(testloader3, 0):
		inputs1, inputs2, inputs3, inputs4, inputs5, length, labels = data
		inputs1 = inputs1.cuda()
		inputs2 = inputs2.cuda()
		inputs3 = inputs3.cuda()
		inputs4 = inputs4.cuda()
		inputs5 = inputs5.cuda()
		outputs = net3(inputs1, inputs2, inputs5, length) + net3(inputs3, inputs4, inputs5, length)
		outputs /= 2
		batch = outputs.size(0)
		label = labels.numpy()
		outputs = F.softmax(outputs, dim=1)
		predicted = outputs.cpu().numpy()
		for j in range(batch):
			raw_prob[label[j]] = predicted[j]
		if i % 100 == 0:
			print('num:%.4f,time:%.2fm' % (i / len(testloader3) / 5 + 0.4, (time.time() - starttime) / 60))
	for i, data in enumerate(testloader4, 0):
		inputs1, inputs2, inputs3, inputs4, inputs5, length, labels = data
		inputs1 = inputs1.cuda()
		inputs2 = inputs2.cuda()
		inputs3 = inputs3.cuda()
		inputs4 = inputs4.cuda()
		inputs5 = inputs5.cuda()
		outputs = net4(inputs1, inputs2, inputs5, length) + net4(inputs3, inputs4, inputs5, length)
		outputs /= 2
		batch = outputs.size(0)
		label = labels.numpy()
		outputs = F.softmax(outputs, dim=1)
		predicted = outputs.cpu().numpy()
		for j in range(batch):
			raw_prob[label[j]] = predicted[j]
		if i % 100 == 0:
			print('num:%.4f,time:%.2fm' % (i / len(testloader4) / 5 + 0.6, (time.time() - starttime) / 60))
	for i, data in enumerate(testloader5, 0):
		inputs1, inputs2, inputs3, inputs4, inputs5, length, labels = data
		inputs1 = inputs1.cuda()
		inputs2 = inputs2.cuda()
		inputs3 = inputs3.cuda()
		inputs4 = inputs4.cuda()
		inputs5 = inputs5.cuda()
		outputs = net5(inputs1, inputs2, inputs5, length) + net5(inputs3, inputs4, inputs5, length)
		outputs /= 2
		batch = outputs.size(0)
		label = labels.numpy()
		outputs = F.softmax(outputs, dim=1)
		predicted = outputs.cpu().numpy()
		for j in range(batch):
			raw_prob[label[j]] = predicted[j]
		if i % 100 == 0:
			print('num:%.4f,time:%.2fm' % (i / len(testloader5) / 5 + 0.8, (time.time() - starttime) / 60))

np.save('../../models/data/stacking/' + cfg.task + '/' + cfg.task+'_' + cfg.trans_mode + '_' + cfg.dw + cfg.a + '_train.npy',raw_prob)