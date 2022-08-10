# Obvisionflow
OpenBayes 视频处理

## 简介
DeOldify PyTorch复现版(强化对中国黑白战争片上色)



## 数据库
本项目共使用两个数据库进行训练
1. ImageNet中的ILSVRC2012
2. 中国彩色战争片包含像亮剑，血战台儿庄等影片的图片
将自己的数据库链接到本项目的data文件夹下，对齐config中对data root字段即可
示例文件夹结构如下：
```
DeOldify
-- data
---- ILSVRC
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
使用DeOldify原作中的NoGan训练方式
1. 先用ImageNet的数据只用perceptual loss和l1 loss来训练生成器
2. 用生成器的生成的图片，单独训练判别器
3. 最后用普通的Gan同时训练生成器和判别器
4. (实验中)在彩色战争片的数据库上进行finetune
注意，NoGan的三个阶段可以在config中使用Gonly, Donly, GandD字段来控制训练阶段


## 训练指令

```
python ./tools/train.py \
	${CONFIG_PATH} \
	--resume-from=${PATH_OF_CHECKPOINT} \
	--work-dir=${YOUR_EXPERIMENT_FOLDER}
```

NoGan三阶段的执行例子，按顺序执行
```
python ./tools/train.py \
configs/deoldify/deoldify_nogan_Gonly.py \
--work-dir=./work_dirs/Gonly

python ./tools/train.py \
configs/deoldify/deoldify_nogan_Donly.py \
--work-dir=./work_dirs/Donly \
--resume-from=./work_dirs/Gonly/ckpt/Gonly/latest.pth

python ./tools/train.py \
configs/deoldify/deoldify_nogan_GandD.py \
--work-dir=./work_dirs/GandD \
--resume-from=./work_dirs/Donly/ckpt/Donly/latest.pth
```

最后在战争数据集上进行Finetune

```
python ./tools/train.py \
configs/deoldify/deoldify_nogan_GandD_finetune.py \
--work-dir=./work_dirs/GandD_finetune \
--resume-from=./work_dirs/GandD/ckpt/GandD/latest.pth
```

使用Lab色彩空间代替RGB（使用正常数据集）

```
python ./tools/train.py \
configs/deoldify/deoldify_nogan_GandD_Lab.py \
--work-dir=./work_dirs/GandD_Lab
```


预训练模型 DeOldify_ILSVRC_Pretrained：https://openbayes.com/console/OpenBayesAlgo/datasets/TxvrKiejFc6/1/overview

目前各个阶段的step数仍未最佳有比较大的冗余，可在config中自行调整
train和val的上色结果和checkpoint都在work-dir文件夹中

## 测试结果
除了使用自己训练所得的模型权重，
也可加上--deoldify_pretrained 标志来使用DeOldify提供的权重(请同时配合deoldify_pretrained.py的config使用)

### 图片上色
格式:

```
python demo/image_demo.py \
	${IMAGE_PATH} \
	${CONFIG_PATH} \
	${CHECKPOINT_PATH} \
	--out ${RESULT_IMAGE_PATH}
	(optional) --deoldify_pretrained
```
示例:

```
python demo/image_demo.py \
	test.jpg \
	configs/deoldify/deoldify_pretrained.py \
	./work_dirs/OnecycleGandD/ckpt/OnecycleGandD/deoldify_weight_gen.pth \
	--out ./work_dirs/test_results.jpg \
	--deoldify_pretrained
```

### 视频上色
格式:

```
python demo/video_demo.py \
	${VIDEO_PATH} \
	${CONFIG_PATH} \
	${CHECKPOINT_PATH} \
	--out ${RESULT_VIDEO_PATH}
	(optional) --deoldify_pretrained
```
示例:

```
python demo/video_demo.py \
	shangganling_cut.mp4 \
	configs/deoldify/deoldify_nogan_GandD.py \
	./work_dirs/GandD/latest.pth \
	--out ./work_dirs/sgl_result.mp4 \
```
