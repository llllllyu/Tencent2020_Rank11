'''
训练主程序
'''
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import random
import torchcontrib

import sys
sys.path.append('../../')
# 加载配置文件
from models.lyu.config.config import _C as cfg
print('task:',cfg.task)
print('fold:',cfg.fold)
# 设定随机种子
def setup_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	if not cfg.cudnn_benchmark:
		torch.backends.cudnn.deterministic = True

seed = cfg.base_seed + cfg.fold
print('seed:',seed)
setup_seed(seed)
# 若保存临时模型，则用当前时刻来命名以防止重复
temp = str(int(time.time()))
if cfg.save:
	print('保存模型')
else:
	print('保存临时模型:' + temp)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda
print('cuda:',cfg.cuda)

if cfg.cudnn_benchmark:
	torch.backends.cudnn.benchmark = True
	print('启用cudnn加速')
else:
	print('不启用cudnn加速')

if cfg.swa:
	print('使用SWA')
else:
	print('不使用SWA')

if cfg.deepwalk:
	print('使用deepwalk')
else:
	print('不使用deepwalk')

if cfg.adversarial:
	print('使用对抗训练')
else:
	print('不使用对抗训练')

if cfg.augmentation:
	print('使用数据叠加')
else:
	print('不使用数据叠加')

print('序列长度:',cfg.max_len)

from models.lyu.model import model  # 加载模型文件
from models.lyu.load import data  # 加载数据读取模块
trainloader = data.trainloader
testloader = data.testloader
L_train = data.L_train
L_test = data.L_test
log_time = int(L_train / 10)  # 训练集数据每训练十分之一就输出一次信息

acc_save = 0  # 保存最好的acc
loss_save = 100  # 保存最好的loss
net = model.Advanced_Trans_LSTM(256,32,16,encoder_num=1,head=2,mode=cfg.trans_mode)
net = nn.DataParallel(net)
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=0)
# 使用学习率下降的周期学习率
scheduler = optim.lr_scheduler.CyclicLR(optimizer, 3e-5, 5e-4, int(L_train * cfg.step_epoch), mode='triangular2', cycle_momentum=False)
if cfg.swa:
	optimizer = torchcontrib.optim.SWA(optimizer)  # 随机加权平均SWA
starttime = time.time()

for epoch in range(cfg.epoch):
	# 为了加速收敛，后期需要降低循环周期
	if epoch == cfg.down_epoch:
		print('降低循环周期')
		if cfg.adversarial:  # 若后续还会进行对抗训练，则最小学习率可以先不用下降
			scheduler = optim.lr_scheduler.CyclicLR(optimizer, 3e-5, 1.25e-4, int(L_train * cfg.step_epoch_down),
		                                        mode='triangular2', cycle_momentum=False)
		else:  # 若只是正常训练，则同时降低最小学习率
			if cfg.task == 'age':
				scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1.25e-4, int(L_train * cfg.step_epoch_down),
				                                        mode='triangular2', cycle_momentum=False)
			else:
				scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 2.5e-4, int(L_train * cfg.step_epoch_down),
			                                        mode='triangular2', cycle_momentum=False)
	if epoch == cfg.adversarial_epoch and cfg.adversarial:  # 训练到一定epoch后开始对抗训练，最小学习率下降
		print('开始对抗训练')
		scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1.25e-4, int(L_train * cfg.step_epoch_adversarial),
		                                        mode='triangular2', cycle_momentum=False)

	running_loss = 0.0
	train_total = 0
	train_correct = 0
	total = 0
	correct = 0
	iter = 0
	net.train()
	for i, data in enumerate(trainloader, 0):
		inputs1, inputs2, inputs3, length, labels = data
		inputs1 = inputs1.cuda()
		inputs2 = inputs2.cuda()
		inputs3 = inputs3.cuda()
		labels = labels.cuda()
		optimizer.zero_grad()
		# 为了减少训练时间，训练到一定epoch后才开始对抗训练
		if epoch + 1 > cfg.adversarial_epoch and cfg.adversarial:
			outputs, loss = model.FGM(inputs1, inputs2, inputs3, length, labels, net, criterion, epsilon=2.)
		else:
			outputs = net(inputs1, inputs2, inputs3, length)
			loss = criterion(outputs, labels)
			loss.backward()
		optimizer.step()
		running_loss += loss.item() / log_time
		_, predicted = torch.max(outputs.data, 1)
		train_total += labels.size(0)
		train_correct += (predicted == labels).sum().item()
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		scheduler.step()
		# 每十分之一训练集输出acc和loss
		if (i + 1) % log_time == 0:
			iter += 1
			ltime = time.time() - starttime
			print('[%d,%d]loss:%.4f acc:%.4f time:%.2fmin'
			      %(epoch + 1, iter, running_loss, train_correct / train_total, ltime / 60))
			running_loss = 0.0
			train_total = 0
			train_correct = 0

	testing_loss = 0.0
	test_total = 0
	test_correct = 0
	with torch.no_grad():
		net.eval()
		for i, data in enumerate(testloader, 0):
			inputs1, inputs2, inputs3, length, labels = data
			inputs1 = inputs1.cuda()
			inputs2 = inputs2.cuda()
			inputs3 = inputs3.cuda()
			labels = labels.cuda()
			outputs = net(inputs1, inputs2, inputs3, length)
			loss = criterion(outputs, labels)
			testing_loss += loss.item() / L_test
			_, predicted = torch.max(outputs.data, 1)
			test_total += labels.size(0)
			test_correct += (predicted == labels).sum().item()

	acc_test = test_correct / test_total
	print('lr:',optimizer.state_dict()['param_groups'][0]['lr'])  # 输出当前学习率

	# 模型达到更优时保存
	if acc_save < acc_test:
		loss_save = testing_loss
		acc_save = acc_test
		print('保存成功')
		if cfg.save:
			torch.save(net.state_dict(), '../../models/lyu/save/' + cfg.task + '/Advanced_' + cfg.trans_mode + '_LSTM_'+ cfg.dw + cfg.a + '_params_' + str(cfg.fold) + '.pkl')
		else:
			torch.save(net.state_dict(), '../../models/lyu/save/temp/' + temp + '.pkl')

	# 输出整个epoch的训练集和测试集acc和loss
	ltime = time.time() - starttime
	print('[%d] loss:%.4f loss_save:%.4f trainacc:%.4f acc:%.4f acc_save:%.4f time:%.2fmin'
	      % (epoch + 1, testing_loss, loss_save, correct / total, acc_test, acc_save, ltime / 60))

	# 在使用swa的情况下，需要在每个学习率谷底更新参数，并在训练结束时进行参数平均
	if cfg.swa:
		if epoch > 0 and epoch % 6 == 1:
			print('更新SWA')
			optimizer.update_swa()
			if epoch == cfg.epoch - 1:
				optimizer.swap_swa_sgd()
				optimizer.bn_update_trans(trainloader, net)  # 在有bn的情况下，还需要对bn的参数进行更新
				print('保存SWA模型')
				torch.save(net.state_dict(),
				           '../../models/lyu/save/' + cfg.task + '/Advanced_' + cfg.trans_mode + '_LSTM_' + cfg.dw + cfg.a + '_params_SWA_' + str(
					           cfg.fold) + '.pkl')
