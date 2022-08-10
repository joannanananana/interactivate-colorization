# Quantexsr
OpenBayes 图像超分

## 简介
Quantexsr PyTorch复现版



## 数据库
本项目共使用一个数据库进行训练
1. DIV2K超分辨率数据集,
将自己的数据库链接到本项目的data文件夹下，对齐config中对data root字段即可.

2. SRAnnotationDataset格式示例文件夹结构如下：

其中生成meta_info.txt文件请用该函数[./tools/generate_meta_info.py](./tools/generate_meta_info.py)
```
Quantexsr
-- data
---- DIV2K
------train
--------dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
--------dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
--------ann_file:    datasets/DIV2K/meta_info_DIV2K800sub_GT.txt
------test
--------dataroot_gt: xxx.jpg
--------dataroot_lq: xxxX4.jpg
--------ann_file:    xxx.txt
---- MyDataset
------train
------test
```
tips:如果想按顺序输出，则使用SRAnnotationDataset这个文件格式，不然用SRFolderDataset格式就行

3. SRFolderDataset格式示例文件夹结构如下：
```
Quantexsr
-- data
---- DIV2K
------train
--------dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
--------dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
------test
--------dataroot_gt: xxx.jpg
--------dataroot_lq: xxxX4.jpg
---- MyDataset
------train
------test
```




## 预训练模型
1. LQ stage (SR model training):(对应于pretrain_model_latest.pth)
2. lpips weight for validation:(对应于lpips文件夹)





## 训练指令

### Train SR model

```
python3 ./tools/train.py configs/quantexsr/Quantexsr.py --work-dir=./work_dirs/Quantexsr
```





## 预训练模型
### Download pretrained model (**provide x4 model now**) from
- pretrain_model_latest: https://openbayes.com/console/OpenBayesAlgo/models/Ia3EMC6xMSA/1/overview.



## 测试结果
格式:
```
python tools/test.py \
	  ${CONFIG_PATH} \
	  ${CHECKPOINT} \
	--save-path   ${RESULT_IMAGE_PATH}
```
示例:

```
python tools/test.py \
	./configs/quantexsr/Quantexsr.py \
	./work_dirs/Quantexsr/FeMaSR_SRX4_model_g.pth \
	--save-path ./results/
```


### 图片超分demo(直接对lq图像进行超分，无gt图像)
格式:

```
python demo/Quantexsr_demo.py \
	--input  ${IMAGE_PATH} \
	--weight ${CHECKPOINT} \
	--out    ${RESULT_IMAGE_PATH}
```
示例:

```
python demo/Quantexsr_demo.py \
	--input ./testset/ \
	--weight ./work_dirs/Quantexsr/FeMaSR_SRX4_model_g.pth \
	--output ./results/
```

#### 或者(只想试试单张图片SISR):
```
python demo/restoration_demo.py \
        ${CONFIG_FILE} \
        ${CHECKPOINT_FILE} \
        ${IMAGE_FILE} \
        ${SAVE_FILE} \
        [--imshow] [--device ${GPU_ID}]
```

如果 `--imshow` 被指定，这个样例也能通过 opencv 展示图片。比如：

```shell
python demo/restoration_demo.py \
        configs/quantexsr/Quantexsr.py \
        work_dirs/Quantexsr/FeMaSR_SRX4_model_g.pth \
        tests/baboon.png \
        demo/demo_out_baboon.png
```
恢复的图片将被存储在 `demo/demo_out_baboon.png`。

注意：如果baboon_x4.png为lq图片，则测试时须将configs/quantexsr/Quantexsr.py中的test_pipeline中的dict(type="Lq_util", key="lq", sf=4)这一句注释掉,若为gt图像，则不需要注释.
```
test_pipeline = [
    dict(type="LoadImageFromFile", io_backend="disk", key="lq", flag="unchanged"),
    dict(type="LoadImageFromFile", io_backend="disk", key="gt", flag="unchanged"),
    #dict(type="Lq_util", key="lq", sf=4),
    dict(type="RescaleToZeroOne", keys=["lq", "gt"]),
    dict(type="Normalize", keys=["lq", "gt"], mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type="Collect", keys=["lq", "gt"], meta_keys=["lq_path", "lq_path"]),
    dict(type="ImageToTensor", keys=["lq", "gt"]),
]
```

##在pipeline中常用hq生成lq的函数
1.Lq_util
```
dict(type="Lq_util", key="lq", sf=4)
```
2.Lq_downsample
```
dict(type="Lq_downsample", key="lq", sf=4)
```
3.Lq_degradation_bsrgan
```
dict(type="Lq_degradation_bsrgan", key="lq", sf=4)
```
4.Lq_degradation_bsrgan_plus
```
dict(type="Lq_degradation_bsrgan_plus", key="lq", sf=4)
```
