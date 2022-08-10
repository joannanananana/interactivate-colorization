# 训练模型OpenBayes


* Deoldify:视频上色 详细的教程请参考 [README_deoldify.md](docs/README_deoldify.md)

* Quantexsr:图像超分 详细的教程请参考 [README_Quantexsr.md](docs/README_Quantexsr.md)

* dvp:视频一致性增强 详细的教程请参考 [README_dvp.md](docs/README_dvp.md)

* interactive colorization:交互上色 详细的教程请参考 [README_color.md](docs/README_color.md)


### 完整的安装脚本
* mmcv-full https://github.com/open-mmlab/mmcv
	* 安装参考:
	* pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
	* 如果使用其他版本PyTorch或CUDA 请安装对应版本的mmcv，例如PyTorch1.9.0:
	* pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
* Clone本项目
```shell
git clone https://github.com/signcl/obvisionflow.git
cd obvisionflow
pip install -r requirements.txt
python setup.py develop
```
