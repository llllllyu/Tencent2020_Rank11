'''
数据预处理
将初始数据处理成序列、广告特征和标签，并用合适方式储存
'''

import pandas as pd
import time
import numpy as np
import h5py

starttime = time.time()

label = []
# 读取初赛和复赛的用户数据
user_p = pd.read_csv('../../models/data/train_preliminary/user.csv', sep=',')
user_f = pd.read_csv('../../models/data/train_final/user.csv', sep=',')

user = pd.concat([user_p, user_f]).reset_index()
# 将用户标签提取出来
for i in range(3000000):
	label.append([user['age'][i]-1, user['gender'][i]-1])
	if i % 10000 == 0:
		print('1:%.4f,time:%.2fm'%(i / 3000000, (time.time()-starttime)/60))
# label储存为numpy矩阵
np.save('../../models/lyu/data/npy_final/label.npy', np.array(label))

# 读取初赛复赛和测试集的点击数据
click_p = pd.read_csv('../../models/data/train_preliminary/click_log.csv', sep=',')
click_f = pd.read_csv('../../models/data/train_final/click_log.csv', sep=',')
click_t = pd.read_csv('../../models/data/test/click_log.csv', sep=',')

click = pd.concat([click_p, click_f, click_t])
click = click.sort_values(by=['user_id', 'time']).reset_index()  # 按照用户和点击时间排序

click_dict = {}  # 建一个包括4000000用户点击序列的字典
for i in range(4000000):
	click_dict[i+1] = []

for i in range(len(click['user_id'])):
	user_id = click['user_id'][i]
	creative_id = click['creative_id'][i]
	times = click['click_times'][i]
	# 将每一个用户的点击按照时间顺序排列，点击次数大于1的重复排列
	for j in range(times):
		click_dict[user_id].append(creative_id)
	if i % 10000 == 0:
		print('2:%.4f,time:%.2fm'%(i / len(click['user_id']), (time.time()-starttime)/60))

# 将用户点击序列储存为h5py数据，为了避免单个h5py中键的数量过多，分为两个文件储存
f = h5py.File('../../models/lyu/data/npy_final/user_1.h5','w')
for user in range(1, 2000001):
	click_list = click_dict[user]
	f[str(user)] = np.array(click_list)  # 键名为用户ID，键值为点击的creative的ID组成的numpy列表
	if user % 10000 == 0:
		print('3:%.4f,time:%.2fm'%(user / 2000000, (time.time()-starttime)/60))
f.close()

f = h5py.File('../../models/lyu/data/npy_final/user_2.h5','w')
for user in range(2000001, 4000001):
	click_list = click_dict[user]
	f[str(user)] = np.array(click_list)
	if user % 10000 == 0:
		print('4:%.4f,time:%.2fm'%(user / 2000000, (time.time()-starttime)/60))
f.close()

# 读取初赛复赛和测试集的广告属性数据
ad_p = pd.read_csv('../../models/data/train_preliminary/ad.csv', sep=',')
ad_f = pd.read_csv('../../models/data/train_final/ad.csv', sep=',')
ad_t = pd.read_csv('../../models/data/test/ad.csv', sep=',')

ad = pd.concat([ad_p, ad_f, ad_t])
ad = ad.drop_duplicates().reset_index()  # 去掉重复的creative

ad_list = [0] * len(ad['ad_id'])
# 按照creative的ID的顺序，将ad、product、product_category、advertiser、industry的ID拼接成一行，组成4445720*5的矩阵
for i in range(len(ad['ad_id'])):
	creative_id = ad['creative_id'][i]
	ad_id = ad['ad_id'][i]
	product_id = ad['product_id'][i]
	product_category = ad['product_category'][i]
	advertiser_id = ad['advertiser_id'][i]
	industry = ad['industry'][i]
	if product_id == '\\N':
		product_id = 0
	else:
		product_id = int(product_id)
	if industry == '\\N':
		industry = 0
	else:
		industry = int(industry)
	ad_list[creative_id - 1] = [ad_id, product_id, product_category, advertiser_id, industry]
	if i % 10000 == 0:
		print('5:%.4f,time:%.2fm'%(i / len(ad['ad_id']), (time.time()-starttime)/60))
# 储存为numpy矩阵
np.save('../../models/lyu/data/npy_final/ad_list.npy', np.array(ad_list))