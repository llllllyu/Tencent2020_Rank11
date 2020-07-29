'''
配置文件
打注释的为可修改项，包括模型，特征，训练设置等
未注释项为模型参数，建议不修改
目前单模最优为LSTM + deepwalk + 对抗训练，age单折0.519
'''

from yacs.config import CfgNode as CN

_C = CN()
_C.cuda = '0'  # 只支持单GPU，多GPU效率下降
_C.fold = 1  # 选择fold 1-5
_C.task = 'age'  # 任务为age或gender
_C.cudnn_benchmark = True
_C.deepwalk = True  # 是否使用deepwalk特征
_C.adversarial = True  # 是否使用对抗训练
_C.swa = False
_C.trans_mode = 'LSTM'  # 模型的encoder模式，LSTM效果最好，Res其次
if _C.trans_mode == 'LSTM':
	_C.base_seed = 0
elif _C.trans_mode == 'Linear':
	_C.base_seed = 10
elif _C.trans_mode == 'GRU':
	_C.base_seed = 20
elif _C.trans_mode == 'Res':
	_C.base_seed = 30
else:
	_C.base_seed = 0
_C.augmentation = False
_C.save = True  # 是否保存模型，false则保存临时模型
if _C.augmentation:
	_C.step_epoch = 1
else:
	_C.step_epoch = 3
if _C.augmentation:
	_C.step_epoch_down = 0.5
	_C.step_epoch_adversarial = 0.5
else:
	_C.step_epoch_down = 1.5
	_C.step_epoch_adversarial = 1.5
if _C.task == 'age':
	_C.down_epoch = _C.step_epoch * 4
	_C.adversarial_epoch = _C.step_epoch * 6
	_C.epoch = _C.step_epoch * 11
	_C.adversarial = True
else:
	_C.down_epoch = _C.step_epoch * 2
	_C.adversarial_epoch = _C.step_epoch * 3
	_C.epoch = _C.step_epoch * 5
	_C.adversarial = False
if _C.deepwalk:
	_C.dw = 'dw'
else:
	_C.dw = ''
if _C.adversarial:
	_C.a = 'a'
else:
	_C.a = ''
_C.batch = 512
_C.num_workers = 3
_C.creative_dim = 128
_C.ad_dim = 128
_C.pro_dim = 128
_C.pro_cate_dim = 32
_C.adver_dim = 128
_C.industry_dim = 64
_C.input_dim = _C.creative_dim + _C.ad_dim + _C.pro_dim + _C.pro_cate_dim + _C.adver_dim + _C.industry_dim
if _C.task == 'gender':
	_C.output_dim = 2
else:
	_C.output_dim = 10
_C.creative_num = 4445720
_C.ad_num = 3812202
_C.pro_num = 44314
_C.pro_cate_num = 18
_C.adver_num = 62965
_C.industry_num = 335
_C.input_num = _C.creative_num + _C.ad_num + _C.pro_num + _C.pro_cate_num + _C.adver_num + _C.industry_num
_C.user_tfdif = 332
_C.user_emb = 256
if _C.deepwalk:
	_C.creative_emb = 300 + 192
	_C.ad_emb =300 + 192
	_C.pro_emb = 150 + 128
	_C.pro_cate_emb = 64
	_C.adver_emb = 150 + 128
	_C.industry_emb = 50 + 64
else:
	_C.creative_emb = 300
	_C.ad_emb =300
	_C.pro_emb = 150
	_C.pro_cate_emb = 64
	_C.adver_emb = 150
	_C.industry_emb = 50
_C.week_vec = 7
_C.click_vec = 1
_C.input_emb = _C.creative_emb + _C.ad_emb + _C.pro_emb + _C.adver_emb + _C.industry_emb
_C.input_emb1 = _C.creative_emb + _C.ad_emb + _C.pro_emb + _C.adver_emb + _C.industry_emb
_C.input_emb2 = _C.ad_emb + _C.pro_emb + _C.adver_emb + _C.industry_emb
_C.input_emb3 = _C.pro_emb + _C.adver_emb + _C.industry_emb

_C.lstm_hidden_size = 256
_C.lstm_num_layers = 1
_C.lstm_dropout = 0
_C.lstm_bidirectional = False
if _C.lstm_bidirectional:
	_C.lstm_output_size = _C.lstm_hidden_size * 2
else:
	_C.lstm_output_size = _C.lstm_hidden_size
_C.max_len = 100
_C.mask_rate = 0.2
_C.threshold = 0.2
_C.bert_hidden_size = 256
_C.cnn_size = 5
_C.cnn_pad = int((_C.cnn_size - 1) / 2)
_C.dropout = 0.3
_C.creative_count = 70
_C.ad_count = 70
_C.adver_count = 70
if _C.task == 'age':
	_C.creative_tfidf = 70
	_C.ad_tfidf = 70
	_C.adver_tfidf = 70
else:
	_C.creative_tfidf = 14
	_C.ad_tfidf = 14
	_C.adver_tfidf = 14
_C.industry_tfidf = 50
_C.target = 37
_C.input_tfidf = _C.creative_tfidf + _C.ad_tfidf + _C.adver_tfidf