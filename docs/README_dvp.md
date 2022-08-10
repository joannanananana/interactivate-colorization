# Deep - video - prior
OpenBayes 图像时域一致性增强

## 简介
deep-video-prior pytorch复现版，用于增强视频上色算法的时域一致性，
用来解决不同相似祯之间的上色不一致问题。


## 使用指令

### Train dvp

直接执行以下命令即可增强视频的时域一致性
```
python scripts/dvp.py --input data/dvp/input \
--processed data/dvp/processed \
--output ./result/ \
--res_name cat

```

--input       未被上色的黑白视频分帧后所在的路径
--prorcessed  上色后的彩色视频分帧后所在的路径
--output      输出结果的第一个目录,输出的路径为
--res_name    增强一致性后的视频名称
