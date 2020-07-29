'''
模型文件，最后只使用了Advanced_Trans_LSTM模型
通过更改配置文件里面的trans_mode来更改模型使用的encoder种类
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.lyu.config.config import _C as cfg
import math

class FC_emb(nn.Module):
	def __init__(self, layer1, layer2):
		print('当前模型:FC_emb')
		super(FC_emb, self).__init__()
		in_channel = cfg.input_emb
		out_channel = cfg.output_dim
		self.fc1 = nn.Linear(in_channel, layer1)
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.constant_(self.fc1.bias, 0)
		self.fc2 = nn.Linear(layer1, layer2)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.fc2.bias, 0)
		self.fc3 = nn.Linear(layer2, out_channel)
		nn.init.xavier_uniform_(self.fc3.weight)
		nn.init.constant_(self.fc3.bias, 0)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		return x

class LSTM(nn.Module):
	def __init__(self, layer1, layer2):
		print('当前模型:LSTM')
		super(LSTM, self).__init__()
		in_channel = cfg.input_emb
		out_channel = cfg.output_dim
		self.lstm = nn.LSTM(input_size=in_channel,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.fc1 = nn.Linear(cfg.lstm_output_size, layer1)
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.constant_(self.fc1.bias, 0)
		self.fc2 = nn.Linear(layer1, layer2)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.fc2.bias, 0)
		self.fc3 = nn.Linear(layer2, out_channel)
		nn.init.xavier_uniform_(self.fc3.weight)
		nn.init.constant_(self.fc3.bias, 0)

	def forward(self, x, length):
		data = pack_padded_sequence(x, length, batch_first=True)
		output, hidden = self.lstm(data)
		output, out_len = pad_packed_sequence(output, batch_first=True)
		# x = torch.sum(output, dim=1)
		# x /= out_len.cuda().unsqueeze(1)
		mask1 = torch.zeros(output.size(0), output.size(1))
		# mask0 = torch.zeros(output.size(0), output.size(1))
		for i in range(len(out_len)):
			mask1[i, out_len[i] - 1] = 1
			# mask0[i, 0] = 1
		mask1 = mask1.bool()
		# mask0 = mask0.bool()
		x = output[mask1]
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		return x

class GRU(nn.Module):
	def __init__(self, layer1, layer2):
		print('当前模型:GRU')
		super(GRU, self).__init__()
		in_channel = cfg.input_emb
		out_channel = cfg.output_dim
		self.lstm = nn.GRU(input_size=in_channel,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		# self.lstm2 = nn.GRU(input_size=cfg.lstm_hidden_size, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)
		self.fc1 = nn.Linear(cfg.lstm_output_size, layer1)
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.constant_(self.fc1.bias, 0)
		self.fc2 = nn.Linear(layer1, layer2)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.fc2.bias, 0)
		self.fc3 = nn.Linear(layer2, out_channel)
		nn.init.xavier_uniform_(self.fc3.weight)
		nn.init.constant_(self.fc3.bias, 0)

	def forward(self, x, length):
		data = pack_padded_sequence(x, length, batch_first=True)
		output, hidden = self.lstm(data)
		# output, hidden = self.lstm2(output)
		output, out_len = pad_packed_sequence(output, batch_first=True)
		# x = torch.sum(output, dim=1)
		# x /= out_len.cuda().unsqueeze(1)
		mask1 = torch.zeros(output.size(0), output.size(1))
		# mask0 = torch.zeros(output.size(0), output.size(1))
		for i in range(len(out_len)):
			mask1[i, out_len[i] - 1] = 1
			# mask0[i, 0] = 1
		mask1 = mask1.bool()
		# mask0 = mask0.bool()
		x = output[mask1]
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		return x

class Multi_LSTM(nn.Module):
	def __init__(self, layer1, layer2):
		print('当前模型:Multi_LSTM')
		super(Multi_LSTM, self).__init__()
		in_channel1 = cfg.input_emb1
		in_channel2 = cfg.input_emb2
		out_channel = cfg.output_dim
		self.lstm1 = nn.LSTM(input_size=in_channel1,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm2 = nn.LSTM(input_size=in_channel2, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)
		self.maxpool = nn.AdaptiveMaxPool2d((1,cfg.lstm_hidden_size))
		# self.avgpool = nn.AdaptiveAvgPool2d((1, cfg.lstm_hidden_size))
		self.dropout = nn.Dropout(p=cfg.dropout)
		self.bn = nn.BatchNorm1d(cfg.lstm_output_size * 2)
		self.fc01 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc02 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc1 = nn.Linear(cfg.lstm_output_size * 2, layer1)
		self.fc2 = nn.Linear(layer1, layer2)
		self.fc3 = nn.Linear(layer2, out_channel)

	def forward(self, x1, x2, length):
		x1 = self.dropout(x1)
		x2 = self.dropout(x2)
		x1 = pack_padded_sequence(x1, length, batch_first=True)
		x1, hidden = self.lstm1(x1)
		x1, out_len = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, hidden = self.lstm2(x2)
		x2, out_len = pad_packed_sequence(x2, batch_first=True)

		# mask1 = torch.zeros(x1.size(0), x1.size(1))
		# mask2 = torch.zeros(x2.size(0), x2.size(1))
		# for i in range(len(out_len)):
		# 	mask1[i, out_len[i] - 1] = 1
		# 	mask2[i, out_len[i] - 1] = 1
		# mask1 = mask1.bool()
		# mask2 = mask2.bool()
		# x13 = x1[mask1]
		# x23 = x2[mask2]

		x1 = self.fc01(x1)
		x2 = self.fc02(x2)

		# out_len = out_len.cuda().unsqueeze(1)
		# x11 = torch.sum(x1, dim=1) / out_len
		# x21 = torch.sum(x2, dim=1) / out_len
		# x11 = self.avgpool(x1).squeeze(1)
		# x21 = self.avgpool(x2).squeeze(1)

		x12 = self.maxpool(x1).squeeze(1)
		x22 = self.maxpool(x2).squeeze(1)

		x = torch.cat((x12, x22), 1)
		x = self.bn(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fc1(x)
		# x = F.relu(x)
		# x = self.fc2(x)
		# x = F.relu(x)
		# x = self.fc3(x)
		return x

class Multi_GRU(nn.Module):
	def __init__(self, layer1, layer2):
		print('当前模型:Multi_GRU')
		super(Multi_GRU, self).__init__()
		in_channel1 = cfg.input_emb1
		in_channel2 = cfg.input_emb2
		# in_channel3 = cfg.input_emb3
		out_channel = cfg.output_dim
		self.lstm1 = nn.GRU(input_size=in_channel1,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm2 = nn.GRU(input_size=in_channel2, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)
		# self.lstm3 = nn.GRU(input_size=in_channel3, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		#                     dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)
		# self.lstm = nn.GRU(input_size=cfg.lstm_hidden_size * 3, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		#                     dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)
		self.maxpool = nn.AdaptiveMaxPool2d((1,cfg.lstm_hidden_size))
		self.dropout = nn.Dropout(p=cfg.dropout)
		self.fc1 = nn.Linear(cfg.lstm_output_size * 2, layer1)
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.constant_(self.fc1.bias, 0)
		self.fc2 = nn.Linear(layer1, layer2)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.fc2.bias, 0)
		self.fc3 = nn.Linear(layer2, out_channel)
		nn.init.xavier_uniform_(self.fc3.weight)
		nn.init.constant_(self.fc3.bias, 0)

	def forward(self, x1, x2, length):
		x1 = self.dropout(x1)
		x2 = self.dropout(x2)
		x1 = pack_padded_sequence(x1, length, batch_first=True)
		x1, hidden = self.lstm1(x1)
		x1, out_len = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, hidden = self.lstm2(x2)
		x2, out_len = pad_packed_sequence(x2, batch_first=True)
		# x3 = pack_padded_sequence(x3, length, batch_first=True)
		# x3, hidden = self.lstm3(x3)
		# x3, out_len = pad_packed_sequence(x3, batch_first=True)
		# x = torch.cat((x1, x2, x3), 2)
		# x = pack_padded_sequence(x, length, batch_first=True)
		# x, hidden = self.lstm(x)
		# x, out_len = pad_packed_sequence(x, batch_first=True)

		# mask1 = torch.zeros(x1.size(0), x1.size(1))
		# mask2 = torch.zeros(x2.size(0), x2.size(1))
		# for i in range(len(out_len)):
		# 	mask1[i, out_len[i] - 1] = 1
		# 	mask2[i, out_len[i] - 1] = 1
		# mask1 = mask1.bool()
		# mask2 = mask2.bool()
		# x1 = x1[mask1]
		# x2 = x2[mask2]
		x1 = self.maxpool(x1).squeeze(1)
		x2 = self.maxpool(x2).squeeze(1)

		x = torch.cat((x1, x2), 1)
		# x = self.dropout(x)
		# x = x[mask1]
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		return x

class self_attention(nn.Module):
	def __init__(self, in_channel, attention_head):
		pe = torch.zeros(cfg.max_len, cfg.input_emb)
		position = torch.arange(0, cfg.max_len).unsqueeze(1)
		div_term = torch.exp((torch.arange(0, cfg.input_emb, 2, dtype=torch.float) * -(math.log(100000.0) / cfg.input_emb)))
		pe[:, 0::2] = torch.sin(position.float() * div_term)
		pe[:, 1::2] = torch.cos(position.float() * div_term)
		pe = pe.unsqueeze(0)
		super(self_attention, self).__init__()
		# self.pe = nn.Parameter(pe, requires_grad=False)
		self.in_channel = in_channel
		# self.in_channel = 1
		self.hidden_size = cfg.bert_hidden_size
		self.head = attention_head
		self.linearQ = nn.Linear(self.in_channel, self.head * self.hidden_size, bias=False)
		self.linearK = nn.Linear(self.in_channel, self.head * self.hidden_size, bias=False)
		self.linearV = nn.Linear(self.in_channel, self.head * self.hidden_size, bias=False)
		self.linearX = nn.Linear(self.head * self.hidden_size, self.in_channel, bias=False)

	def forward(self, x, length):

		# seq_len = x.size(1)
		batch = x.size(0)
		# pe = self.pe[:,:seq_len,:].expand(batch,seq_len,self.in_channel).cuda()
		# x += pe
		Q = self.linearQ(x)
		Q = Q.view(batch, -1, self.head, self.hidden_size).transpose(1, 2)
		# Q = F.normalize(Q, dim=3)
		# K = Q.transpose(2, 3)
		K = self.linearK(x)
		K = K.view(batch, -1, self.head, self.hidden_size).transpose(1, 2).transpose(2, 3)
		V = self.linearV(x)
		V = V.view(batch, -1, self.head, self.hidden_size).transpose(1, 2)
		x = Q @ K / math.sqrt(self.hidden_size)
		y = torch.zeros(x.size()).cuda()
		for i in range(x.size(0)):
			y[i,:,:length[i],:length[i]] = F.softmax(x[i,:,:length[i],:length[i]], dim=2)
		# length = torch.tensor(length).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(batch, self.head, seq_len, seq_len)
		# scale = torch.sqrt(length + 1e-8).cuda()
		# y = x / scale
		x = y @ V
		x = x.transpose(1, 2).contiguous().view(batch, -1, self.head * self.hidden_size)
		x = self.linearX(x)

		return x

class BERT_LSTM(nn.Module):
	def __init__(self, layer1, layer2, attention_head):
		print('当前模型:BERT_LSTM')
		super(BERT_LSTM, self).__init__()
		self.attention = self_attention(cfg.input_emb, attention_head)
		self.lstm = LSTM(layer1,layer2)
		self.normalization = nn.LayerNorm(cfg.input_emb)

	def forward(self, x, length):

		x = self.normalization(x + self.attention(x, length))
		x = self.lstm(x, length)

		return x

class BERT_GRU(nn.Module):
	def __init__(self, layer1, layer2, attention_head):
		print('当前模型:BERT_GRU')
		super(BERT_GRU, self).__init__()
		self.attention = self_attention(cfg.input_emb, attention_head)
		self.lstm = GRU(layer1,layer2)
		self.normalization = nn.LayerNorm(cfg.input_emb)

	def forward(self, x, length):

		x = self.normalization(x + self.attention(x, length))
		x = self.lstm(x, length)

		return x

class BERT(nn.Module):
	def __init__(self, layer1, layer2, attention_head):
		print('当前模型:BERT')
		super(BERT, self).__init__()
		in_channel = cfg.input_emb
		out_channel = cfg.output_dim
		self.attention = self_attention(cfg.input_emb, attention_head)
		self.normalization = nn.LayerNorm(in_channel)
		self.maxpool = nn.AdaptiveMaxPool2d((1, in_channel))
		self.fc1 = nn.Linear(in_channel, layer1)
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.constant_(self.fc1.bias, 0)
		self.fc2 = nn.Linear(layer1, layer2)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.fc2.bias, 0)
		self.fc3 = nn.Linear(layer2, out_channel)
		nn.init.xavier_uniform_(self.fc3.weight)
		nn.init.constant_(self.fc3.bias, 0)

	def forward(self, x, length):

		x = self.normalization(x + self.attention(x, length))
		x = self.maxpool(x).squeeze(1)
		# x = torch.sum(x, dim=1)
		# x /= torch.tensor(length).cuda().unsqueeze(1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)

		return x

class Multi_BERT(nn.Module):
	def __init__(self, layer1, layer2, attention_head):
		print('当前模型:Multi_BERT')
		super(Multi_BERT, self).__init__()
		in_channel1 = cfg.input_emb1
		in_channel2 = cfg.input_emb2
		out_channel = cfg.output_dim
		self.attention1 = self_attention(in_channel1, attention_head)
		self.normalization1 = nn.LayerNorm(in_channel1)
		self.maxpool1 = nn.AdaptiveMaxPool2d((1, in_channel1))
		self.attention2 = self_attention(in_channel2, attention_head)
		self.normalization2 = nn.LayerNorm(in_channel2)
		self.maxpool2 = nn.AdaptiveMaxPool2d((1, in_channel2))
		self.dropout = nn.Dropout(p=cfg.dropout)
		self.fc1 = nn.Linear(in_channel1 + in_channel2, layer1)
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.constant_(self.fc1.bias, 0)
		self.fc2 = nn.Linear(layer1, layer2)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.fc2.bias, 0)
		self.fc3 = nn.Linear(layer2, out_channel)
		nn.init.xavier_uniform_(self.fc3.weight)
		nn.init.constant_(self.fc3.bias, 0)

	def forward(self, x1, x2, length):

		x1 = self.dropout(x1)
		x2 = self.dropout(x2)
		x1 = self.normalization1(x1 + self.attention1(x1, length))
		x1 = self.maxpool1(x1).squeeze(1)
		x2 = self.normalization2(x2 + self.attention2(x2, length))
		x2 = self.maxpool2(x2).squeeze(1)
		# x = torch.sum(x, dim=1)
		# x /= torch.tensor(length).cuda().unsqueeze(1)
		x = torch.cat((x1, x2), 1)
		# x = self.dropout(x)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)

		return x

class TF_LSTM(nn.Module):
	def __init__(self, layer1, layer2, layer3):
		print('当前模型:TF_LSTM')
		super(TF_LSTM, self).__init__()
		in_channel1 = cfg.input_emb1
		in_channel2 = cfg.input_emb2
		in_channel3 = cfg.input_tfidf
		out_channel = cfg.output_dim
		self.lstm1 = nn.LSTM(input_size=in_channel1,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm2 = nn.LSTM(input_size=in_channel2, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)
		self.maxpool = nn.AdaptiveMaxPool2d((1,cfg.lstm_hidden_size))
		self.dropout = nn.Dropout(p=cfg.dropout)
		self.fc01 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc02 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc03 = nn.Linear(in_channel3, layer3)
		self.bn = nn.BatchNorm1d(cfg.lstm_output_size * 2 + layer3)
		self.fc1 = nn.Linear(cfg.lstm_output_size * 2 + layer3, layer1)
		self.fc2 = nn.Linear(layer1, layer2)
		self.fc3 = nn.Linear(layer2, out_channel)

	def forward(self, x1, x2, x3, length):
		x1 = self.dropout(x1)
		x2 = self.dropout(x2)
		x1 = pack_padded_sequence(x1, length, batch_first=True)
		x1, hidden = self.lstm1(x1)
		x1, out_len = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, hidden = self.lstm2(x2)
		x2, out_len = pad_packed_sequence(x2, batch_first=True)

		x1 = self.fc01(x1)
		x2 = self.fc02(x2)

		x12 = self.maxpool(x1).squeeze(1)
		x22 = self.maxpool(x2).squeeze(1)

		x3 = self.fc03(x3)
		x = torch.cat((x12, x22, x3), 1)
		x = self.bn(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		return x

class Trans_LSTM(nn.Module):
	def __init__(self, layer1, layer2, layer3):
		print('当前模型:Trans_LSTM')
		super(Trans_LSTM, self).__init__()
		in_channel1 = cfg.input_emb1
		in_channel2 = cfg.input_emb2
		in_channel3 = cfg.input_tfidf
		out_channel = cfg.output_dim
		self.lstm11 = nn.LSTM(input_size=in_channel1,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm21 = nn.LSTM(input_size=in_channel2, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)
		self.ffn11 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.ffn21 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.ffn12 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.ffn22 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.ln1 = nn.LayerNorm(cfg.lstm_output_size)
		self.ln2 = nn.LayerNorm(cfg.lstm_output_size)
		self.lstm12 = nn.LSTM(input_size=cfg.lstm_output_size,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm22 = nn.LSTM(input_size=cfg.lstm_output_size, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)
		self.maxpool = nn.AdaptiveMaxPool2d((1,cfg.lstm_hidden_size))
		self.dropout = nn.Dropout(p=cfg.dropout)
		self.fc01 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc02 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc03 = nn.Linear(in_channel3, layer3)
		self.bn = nn.BatchNorm1d(cfg.lstm_output_size * 2 + layer3)
		self.fc1 = nn.Linear(cfg.lstm_output_size * 2 + layer3, layer1)
		self.fc2 = nn.Linear(layer1, layer2)
		self.fc3 = nn.Linear(layer2, out_channel)

	def forward(self, x1, x2, x3, length):
		x1 = self.dropout(x1)
		x2 = self.dropout(x2)

		x1 = pack_padded_sequence(x1, length, batch_first=True)
		x1, hidden = self.lstm11(x1)
		x1, out_len = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, hidden = self.lstm21(x2)
		x2, out_len = pad_packed_sequence(x2, batch_first=True)
		x1 = self.ln1(self.ffn12(F.relu(self.ffn11(x1))) + x1)
		x2 = self.ln2(self.ffn22(F.relu(self.ffn21(x2))) + x2)

		x1 = pack_padded_sequence(x1, length, batch_first=True)
		x1, hidden = self.lstm12(x1)
		x1, out_len = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, hidden = self.lstm22(x2)
		x2, out_len = pad_packed_sequence(x2, batch_first=True)

		x1 = self.fc01(x1)
		x2 = self.fc02(x2)

		x12 = self.maxpool(x1).squeeze(1)
		x22 = self.maxpool(x2).squeeze(1)

		x3 = self.fc03(x3)
		x = torch.cat((x12, x22, x3), 1)
		x = self.bn(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		return x

class lstm_block(nn.Module):
	'''
	encoder内部的特征提取器，用于提取序列特征
	'''
	def __init__(self, in_channel1, in_channel2):
		super(lstm_block, self).__init__()
		self.lstm11 = nn.LSTM(input_size=in_channel1,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm21 = nn.LSTM(input_size=in_channel2, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)


	def forward(self, x1, x2, length):
		'''
		输入：
		x1.size(): batch, seq_length, embedding_size(in_channel1)
		x2.size(): batch, seq_length, embedding_size(in_channel2)
		length: 长度为batch的list，储存每一个序列的真实长度

		输出：
		x1.size(): batch, seq_length, cfg.lstm_output_size
		x2.size(): batch, seq_length, cfg.lstm_output_size
		'''
		# self.lstm11.flatten_parameters()
		# self.lstm21.flatten_parameters()

		x1 = pack_padded_sequence(x1, length, batch_first=True)  # 使用pytorch的pack操作，避免计算pad部分
		x1, _ = self.lstm11(x1)
		x1, _ = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, _ = self.lstm21(x2)
		x2, _ = pad_packed_sequence(x2, batch_first=True)

		return x1, x2

class multi_block(nn.Module):
	def __init__(self, in_channel1, in_channel2):
		super(multi_block, self).__init__()
		self.lstm11 = nn.LSTM(input_size=in_channel1,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm21 = nn.GRU(input_size=in_channel2, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)


	def forward(self, x1, x2, length):
		# self.lstm11.flatten_parameters()
		# self.lstm21.flatten_parameters()

		x1 = pack_padded_sequence(x1, length, batch_first=True)
		x1, _ = self.lstm11(x1)
		x1, _ = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, _ = self.lstm21(x2)
		x2, _ = pad_packed_sequence(x2, batch_first=True)

		return x1, x2

class gru_block(nn.Module):
	def __init__(self, in_channel1, in_channel2):
		super(gru_block, self).__init__()
		self.lstm11 = nn.GRU(input_size=in_channel1,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm21 = nn.GRU(input_size=in_channel2, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)

	def forward(self, x1, x2, length):

		x1 = pack_padded_sequence(x1, length, batch_first=True)
		x1, _ = self.lstm11(x1)
		x1, _ = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, _ = self.lstm21(x2)
		x2, _ = pad_packed_sequence(x2, batch_first=True)

		return x1, x2

class rnn_block(nn.Module):
	def __init__(self, in_channel1, in_channel2):
		super(rnn_block, self).__init__()
		self.lstm11 = nn.RNN(input_size=in_channel1,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm21 = nn.RNN(input_size=in_channel2, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)

	def forward(self, x1, x2, length):

		x1 = pack_padded_sequence(x1, length, batch_first=True)
		x1, _ = self.lstm11(x1)
		x1, _ = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, _ = self.lstm21(x2)
		x2, _ = pad_packed_sequence(x2, batch_first=True)

		return x1, x2

class cnn_block(nn.Module):
	def __init__(self, in_channel1, in_channel2):
		super(cnn_block, self).__init__()
		self.fc1 = nn.Linear(in_channel1, cfg.lstm_output_size)
		self.fc2 = nn.Linear(in_channel2, cfg.lstm_output_size)
		self.conv11 = nn.Conv2d(1, 1, kernel_size = (cfg.cnn_size, 1),stride=1, padding=(cfg.cnn_pad, 0))
		self.conv21 = nn.Conv2d(1, 1, kernel_size=(cfg.cnn_size, 1), stride=1, padding=(cfg.cnn_pad, 0))
		self.conv12 = nn.Conv2d(1, 1, kernel_size=(cfg.cnn_size, 1), stride=1, padding=(cfg.cnn_pad, 0))
		self.conv22 = nn.Conv2d(1, 1, kernel_size=(cfg.cnn_size, 1), stride=1, padding=(cfg.cnn_pad, 0))
		# self.conv13 = nn.Conv2d(1, 1, kernel_size=(cfg.cnn_size, 1), stride=1, padding=(cfg.cnn_pad, 0))
		# self.conv23 = nn.Conv2d(1, 1, kernel_size=(cfg.cnn_size, 1), stride=1, padding=(cfg.cnn_pad, 0))

	def forward(self, x1, x2, length):

		x1 = x1.unsqueeze(1)
		x2 = x2.unsqueeze(1)
		x1 = self.fc1(x1)
		x2 = self.fc2(x2)
		x1 = self.conv11(x1)
		# x1 = F.relu(x1)
		x1 = self.conv12(x1)
		# x1 = F.relu(x1)
		# x1 = self.conv13(x1)
		x2 = self.conv21(x2)
		# x2 = F.relu(x2)
		x2 = self.conv22(x2)
		# x2 = F.relu(x2)
		# x2 = self.conv23(x2)
		x1 = x1.squeeze(1)
		x2 = x2.squeeze(1)

		return x1, x2

class attention_block(nn.Module):
	def __init__(self, in_channel1, in_channel2):
		super(attention_block, self).__init__()
		self.q1 = nn.Linear(in_channel1, cfg.lstm_output_size)
		self.q2 = nn.Linear(in_channel2, cfg.lstm_output_size)
		self.k1 = nn.Linear(in_channel1, cfg.lstm_output_size)
		self.k2 = nn.Linear(in_channel2, cfg.lstm_output_size)
		self.v1 = nn.Linear(in_channel1, cfg.lstm_output_size)
		self.v2 = nn.Linear(in_channel2, cfg.lstm_output_size)

	def forward(self, x1, x2, length):

		x1q = self.q1(x1)
		x2q = self.q2(x2)
		x1k = self.k1(x1).permute(0,2,1)
		x2k = self.k2(x2).permute(0,2,1)
		x1v = self.v1(x1)
		x2v = self.v2(x2)
		x1 = F.softmax(x1q @ x1k / math.sqrt(cfg.lstm_output_size) + attention_mask(length), dim=2) @ x1v
		x2 = F.softmax(x2q @ x2k / math.sqrt(cfg.lstm_output_size) + attention_mask(length), dim=2) @ x2v

		return x1, x2

def attention_mask(length, value = -1e5):
	mask = torch.ones(len(length), max(length), max(length)) * value
	for batch in range(len(length)):
		mask[batch,:length[batch],:length[batch]] = 0
	return mask.cuda()

class cross_block(nn.Module):
	def __init__(self, in_channel1, in_channel2):
		super(cross_block, self).__init__()
		self.fc11 = nn.Linear(in_channel1, cfg.lstm_output_size)
		self.fc21 = nn.Linear(in_channel2, cfg.lstm_output_size)
		self.fc12 = nn.Linear(cfg.lstm_output_size * 2, cfg.lstm_output_size)
		self.fc22 = nn.Linear(cfg.lstm_output_size * 2, cfg.lstm_output_size)

	def forward(self, x1, x2, length):

		x1 = self.fc11(x1)
		x2 = self.fc21(x2)
		attention = x1 @ x2.transpose(1, 2) + attention_mask(length)
		X1 = torch.cat((x1, F.softmax(attention, 2) @ x2), 2)
		X2 = torch.cat((x2, F.softmax(attention.transpose(1, 2), 2) @ x1), 2)
		x1 = self.fc12(X1)
		x2 = self.fc22(X2)

		return x1, x2

class linear_block(nn.Module):
	def __init__(self, in_channel1, in_channel2):
		super(linear_block, self).__init__()
		self.fc11 = nn.Linear(in_channel1, cfg.lstm_output_size)
		self.fc21 = nn.Linear(in_channel2, cfg.lstm_output_size)
		self.fc12 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc22 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)

	def forward(self, x1, x2, length):

		x1 = self.fc11(x1)
		x2 = self.fc21(x2)
		x1 = self.fc12(F.relu(x1))
		x2 = self.fc22(F.relu(x2))

		return x1, x2

class se_block(nn.Module):
	def __init__(self, in_channel1, in_channel2):
		super(se_block, self).__init__()
		self.fc11 = nn.Linear(in_channel1, cfg.lstm_output_size)
		self.fc21 = nn.Linear(in_channel2, cfg.lstm_output_size)
		self.fc12 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc22 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc1e1 = nn.Linear(cfg.lstm_output_size, int(math.sqrt(cfg.lstm_output_size)))
		self.fc2e1 = nn.Linear(cfg.lstm_output_size, int(math.sqrt(cfg.lstm_output_size)))
		self.fc1e2 = nn.Linear(int(math.sqrt(cfg.lstm_output_size)), cfg.lstm_output_size)
		self.fc2e2 = nn.Linear(int(math.sqrt(cfg.lstm_output_size)), cfg.lstm_output_size)
		self.ln1 = nn.LayerNorm(cfg.lstm_output_size)
		self.ln2 = nn.LayerNorm(cfg.lstm_output_size)

	def forward(self, x1, x2, length):

		x1 = self.fc11(x1)
		x2 = self.fc21(x2)

		x1se = torch.mean(self.fc12(x1),1)
		x2se = torch.mean(self.fc22(x1), 1)
		x1se = torch.sigmoid(self.fc1e2(F.relu(self.fc1e1(x1se))))
		x2se = torch.sigmoid(self.fc2e2(F.relu(self.fc2e1(x2se))))

		x1 = self.ln1(x1 * x1se.unsqueeze(1) + x1)
		x2 = self.ln2(x2 * x2se.unsqueeze(1) + x2)

		return x1, x2

class res_block(nn.Module):
	def __init__(self, in_channel1, in_channel2):
		super(res_block, self).__init__()
		self.fc11 = nn.Linear(in_channel1 + cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc21 = nn.Linear(in_channel2 + cfg.lstm_output_size, cfg.lstm_output_size)
		self.lstm11 = nn.LSTM(input_size=in_channel1,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm21 = nn.LSTM(input_size=in_channel2, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)

	def forward(self, x1, x2, length):

		x1res = x1
		x2res = x2

		x1 = pack_padded_sequence(x1, length, batch_first=True)
		x1, _ = self.lstm11(x1)
		x1, _ = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, _ = self.lstm21(x2)
		x2, _ = pad_packed_sequence(x2, batch_first=True)

		x1 = F.relu(self.fc11(torch.cat((x1res,x1),-1)))
		x2 = F.relu(self.fc21(torch.cat((x2res,x2),-1)))

		return x1, x2

class encoder(nn.Module):
	'''
	仿transformer的encoder结构
	包含多头的特征提取器和ffn
	'''
	def __init__(self, in_channel1, in_channel2, head, mode):
		super(encoder, self).__init__()
		self.head = head
		if mode == 'LSTM':
			self.blocks = nn.ModuleList([lstm_block(in_channel1, in_channel2) for _ in range(head)])
		elif mode == 'CNN':
			self.blocks = nn.ModuleList([cnn_block(in_channel1, in_channel2) for _ in range(head)])
		elif mode == 'Attention':
			self.blocks = nn.ModuleList([attention_block(in_channel1, in_channel2) for _ in range(head)])
		elif mode == 'GRU':
			self.blocks = nn.ModuleList([gru_block(in_channel1, in_channel2) for _ in range(head)])
		elif mode == 'Linear':
			self.blocks = nn.ModuleList([linear_block(in_channel1, in_channel2) for _ in range(head)])
		elif mode == 'Cross':
			self.blocks = nn.ModuleList([cross_block(in_channel1, in_channel2) for _ in range(head)])
		elif mode == 'RNN':
			self.blocks = nn.ModuleList([rnn_block(in_channel1, in_channel2) for _ in range(head)])
		elif mode == 'SE':
			self.blocks = nn.ModuleList([se_block(in_channel1, in_channel2) for _ in range(head)])
		elif mode == 'Res':
			self.blocks = nn.ModuleList([res_block(in_channel1, in_channel2) for _ in range(head)])
		elif mode == 'Multi':
			self.blocks = nn.ModuleList([lstm_block(in_channel1, in_channel2), gru_block(in_channel1, in_channel2)])
		self.shrink1 = nn.Linear(cfg.lstm_output_size * head, cfg.lstm_output_size)
		self.shrink2 = nn.Linear(cfg.lstm_output_size * head, cfg.lstm_output_size)
		self.ffn11 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.ffn21 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.ffn12 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.ffn22 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.ln1 = nn.LayerNorm(cfg.lstm_output_size)
		self.ln2 = nn.LayerNorm(cfg.lstm_output_size)


	def forward(self, x1, x2, length):
		'''
		输入：
		x1.size(): batch, seq_length, embedding_size(in_channel1)
		x2.size(): batch, seq_length, embedding_size(in_channel2)
		length: 长度为batch的list，储存每一个序列的真实长度

		输出：
		x1.size(): batch, seq_length, cfg.lstm_output_size
		x2.size(): batch, seq_length, cfg.lstm_output_size
		'''
		x10, x20 = self.blocks[0](x1, x2, length)
		for i in range(self.head - 1):
			x11, x21 = self.blocks[i + 1](x1, x2, length)
			x10 = torch.cat((x10, x11), 2)
			x20 = torch.cat((x20, x21), 2)
		x1 = self.shrink1(x10)
		x2 = self.shrink2(x20)
		x1 = self.ln1(self.ffn12(F.relu(self.ffn11(x1))) + x1)
		x2 = self.ln2(self.ffn22(F.relu(self.ffn21(x2))) + x2)
		# x1 = self.ffn12(F.relu(self.ffn11(self.ln1(x1)))) + x1
		# x2 = self.ffn22(F.relu(self.ffn21(self.ln2(x1)))) + x2

		return x1, x2

class LSTM_Module(nn.Module):
	'''
	将前面encoder提取的两个序列特征x1，x2通过LSTM和maxpool提取出向量
	'''
	def __init__(self, inchannel1, inchannel2):
		super(LSTM_Module, self).__init__()
		in_channel1 = inchannel1
		in_channel2 = inchannel2
		self.lstm1 = nn.LSTM(input_size=in_channel1,hidden_size=cfg.lstm_hidden_size,num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout,bidirectional=cfg.lstm_bidirectional,batch_first=True)
		self.lstm2 = nn.LSTM(input_size=in_channel2, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers,
		                    dropout=cfg.lstm_dropout, bidirectional=cfg.lstm_bidirectional, batch_first=True)
		self.fc1 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.fc2 = nn.Linear(cfg.lstm_output_size, cfg.lstm_output_size)
		self.maxpool = nn.AdaptiveMaxPool2d((1, cfg.lstm_output_size))

	def forward(self, x1, x2, length):
		'''
		输入：
		x1.size(): batch, seq_length, cfg.lstm_output_size
		x2.size(): batch, seq_length, cfg.lstm_output_size
		length: 长度为batch的list，储存每一个序列的真实长度

		输出：
		x1.size(): batch, cfg.lstm_output_size
		x2.size(): batch, cfg.lstm_output_size
		'''
		# self.lstm1.flatten_parameters()
		# self.lstm2.flatten_parameters()

		x1 = pack_padded_sequence(x1, length, batch_first=True)
		x1, _ = self.lstm1(x1)
		x1, _ = pad_packed_sequence(x1, batch_first=True)
		x2 = pack_padded_sequence(x2, length, batch_first=True)
		x2, _ = self.lstm2(x2)
		x2, _ = pad_packed_sequence(x2, batch_first=True)

		x1 = self.fc1(x1)
		x2 = self.fc2(x2)

		x1 = self.maxpool(x1).squeeze(1)
		x2 = self.maxpool(x2).squeeze(1)

		return x1, x2

class FC_Module(nn.Module):
	'''
	将提取出来的x1，x2特征与输入的统计特征x3交叉，产生模型输出
	'''
	def __init__(self, layer1, layer2, layer3):
		super(FC_Module, self).__init__()
		out_channel = cfg.output_dim
		in_channel3 = cfg.input_tfidf
		self.fc03 = nn.Linear(in_channel3, layer3)
		self.bn = nn.BatchNorm1d(cfg.lstm_output_size * 2 + layer3)
		self.fc1 = nn.Linear(cfg.lstm_output_size * 2 + layer3, layer1)
		self.fc2 = nn.Linear(layer1, layer2)
		self.fc3 = nn.Linear(layer2, out_channel)
		self.dropout = nn.Dropout(p=cfg.dropout)

	def forward(self, x1, x2, x3):
		'''
		输入：
		x1.size(): batch, cfg.lstm_output_size
		x2.size(): batch, cfg.lstm_output_size
		x3.size(): batch, cfg.input_tfidf

		输出：
		x.size(): batch, cfg.output_dim
		'''
		x3 = self.fc03(x3)
		x = torch.cat((x1, x2, x3), 1)
		x = self.bn(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)

		return x

class Advanced_Trans_LSTM(nn.Module):
	'''
	主模型，输入x1，x2，x3对应特征inputs1, inputs2, inputs3
	输出为分类的logit
	'''
	def __init__(self, layer1, layer2, layer3, encoder_num=1, head=2, mode='LSTM'):
		'''
		仿transformer结构
		layer1与layer2为最后的fc层宽度
		layer3为统计特征x3进来的fc层宽度
		encoder_num为使用的encoder数量
		head为encoder头数
		mode为encoder内部特征提取器的类型
		'''
		print('当前模型:Advanced_Trans_LSTM, 模式:' + mode)
		super(Advanced_Trans_LSTM, self).__init__()
		in_channel1 = cfg.input_emb1
		in_channel2 = cfg.input_emb2
		self.encoder_num = encoder_num
		self.encoders = nn.ModuleList([encoder(in_channel1, in_channel2, head, mode)])
		for _ in range(encoder_num - 1):  # 将多个encoder串行拼接，用于特征提取
			self.encoders.append(encoder(cfg.lstm_output_size, cfg.lstm_output_size, head, mode))
		self.lstm = LSTM_Module(cfg.lstm_output_size, cfg.lstm_output_size)  # LSTM_Module用于序列提取为向量
		self.fc = FC_Module(layer1, layer2, layer3)  # FC_Module将向量特征拼接转为最后的logit输出
		self.dropout = nn.Dropout(p=cfg.dropout)

	def forward(self, x1, x2, x3, length):
		'''
		x1与x2为两种序列特征，前期并行处理，最后在fc加上x3进行交叉
		输入：
		x1.size(): batch, seq_length, embedding_size(in_channel1)
		x2.size(): batch, seq_length, embedding_size(in_channel2)
		x3.size(): batch, cfg.input_tfidf
		length: 长度为batch的list，储存每一个序列的真实长度

		输出：
		x.size(): batch, cfg.output_dim
		'''
		x1 = self.dropout(x1)
		x2 = self.dropout(x2)

		for i in range(self.encoder_num):
			x1, x2 = self.encoders[i](x1, x2, length)

		x1, x2 = self.lstm(x1, x2, length)
		x = self.fc(x1, x2, x3)

		return x

def FGM(inputs1, inputs2, inputs3, length, labels, net, criterion, epsilon=1.):
	'''
	对抗训练函数，替代原始训练流程中的前向和后向过程
	'''
	inputs1.requires_grad = True
	inputs2.requires_grad = True
	outputs = net(inputs1, inputs2, inputs3, length)  # 原始样本前向传播
	loss = criterion(outputs, labels)
	loss.backward(retain_graph=True)  # 原始样本后向传播
	# 提取x1和x2的embedding的梯度
	grad_1 = inputs1.grad.detach()
	grad_2 = inputs2.grad.detach()
	norm_1 = torch.norm(grad_1)
	norm_2 = torch.norm(grad_2)
	# 对x1和x2的embedding的梯度进行标准化并生成对抗样本
	if norm_1 != 0:
		r_at = epsilon * grad_1 / norm_1
		inputs1 = inputs1 + r_at
	if norm_2 != 0:
		r_at = epsilon * grad_2 / norm_2
		inputs2 = inputs2 + r_at
	outputs_adv = net(inputs1, inputs2, inputs3, length)  # 对抗样本前向传播
	loss_adv = criterion(outputs_adv, labels)
	loss_adv.backward()  # 对抗样本反向传播
	return outputs, loss