# Tencent2020_Rank11

2020腾讯广告算法大赛复赛rank11（lyu）

队名：日晨

分数：复赛B榜1.479998

排名：初赛rank6，复赛rank11

本项目为我个人（lyu）部分代码，队友代码参考[istar]()、[wujie](https://github.com/wujiekd/2020-Tencent-advertising-algorithm-contest-rank11)，
其中本项目需要用到istar的tfidf部分特征和wujie的deepwalk部分特征，缺少这部分特征效果会有下降

## 项目环境

pytorch 1.3.0

cuda 10.1

gensim

yacs

torchcontrib

h5py

## 文件目录

```
Project
├─models
│  ├─data
│  │  ├─deepwalk
│  │  │  └─index
│  │  ├─stacking
│  │  │  ├─age
│  │  │  └─gender
│  │  ├─test
│  │  ├─tfidf
│  │  ├─train_final
│  │  └─train_preliminary
│  ├─istar
│  ├─lyu
│  │  ├─config
│  │  │  ├─config.py
│  │  │  └─__init__.py
│  │  │
│  │  ├─data
│  │  │  ├─npy_final
│  │  │  └─vec_final
│  │  ├─load
│  │  │  ├─data.py
│  │  │  ├─feature.py
│  │  │  └─__init__.py
│  │  │
│  │  ├─model
│  │  │  ├─model.py
│  │  │  └─__init__.py
│  │  │
│  │  └─save
│  │      ├─age
│  │      ├─gender
│  │      └─temp
│  └─wujie
└─scr
    ├─istar
    ├─lyu
    │  ├─data_process.py
    │  ├─inference.py
    │  ├─n2v.py
    │  ├─process.py
    │  ├─tfidf.py
    │  ├─train.py
    │  └─w2v.py
    │
    └─wujie
```

## 处理流程

将初赛、复赛、测试数据集分别放在models/data/train_preliminary/、models/data/train_final/、models/data/test/ 文件夹

运行 src/lyu/data_process.py 进行数据处理

运行 src/lyu/w2v.py 训练w2v

运行 src/lyu/n2v.py 处理wujie生成的deepwalk特征

运行 src/lyu/tfidf.py 处理istar生成的tfidf特征

修改 models/lyu/config/config.py 里面参数来调整任务和模型，修改项为cuda、fold、task、deepwalk、adversarial、trans_mode

运行 src/lyu/train.py 训练模型

运行 src/lyu/process.py 和 src/lyu/inference.py 推理模型
