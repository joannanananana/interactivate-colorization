# interactive colorization
openbayes 交互上色

## 简介
colorization_pytorch复现版



## 数据库
本项目共使用一个数据库进行训练
1.ILSVRC数据集,openbayes数据集ID: ZuU1TX2iW3N
将自己的数据库链接到本项目的data文件夹下，对齐config中对data root字段即可
其中，train_small文件夹包含从ILSVRC中随机选取的10张图，train文件夹为ILSVRC训练集
示例文件夹结构如下：
```
color
--data
---ILSVRC
------train_samll
--------train
-----------xxx.jpg
-----------xxx.jpg
------train
--------xxx.jpg
--------xxx.jpg
------test
--------xxx.jpg
--------xxx.jpg
---- MyDataset
------train
------test
```

## 训练过程
使用colorization_pytorch原作中的训练方式,请使用config/mmcolor中四个训练阶段的默认配置（如在tutorial中测试单图效果时修改过，请改回默认配置）
1.先用train_samll中10张图训练classification
2.用全部训练集训练classifacation
3.用全部数据集训练regression1
4.用全部数据集训练regression2
5.(实验中)用电影3的数据库上进行finetune


## 训练指令
四阶段训练指令，按顺序执行
```
python ./tools/train.py ./configs/mmcolor/color_siggraph_train_small.py

python ./tools/train.py ./configs/mmcolor/color_siggraph_train.py

python ./tools/train.py ./configs/mmcolor/color_siggraph_regre1.py

python ./tools/train.py ./configs/mmcolor/color_siggraph_regre2.py
```

### 图片交互上色demo
在obvisionflow/Interactive-demo.ipynb上色

### 测试psnr
在tools/metric_psnr，可以修改测试的图片路径、数量、pth等参数,执行
```
python tools/metric_psnr.py

```
