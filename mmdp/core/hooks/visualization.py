# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
from torchvision.utils import save_image

from mmdp.utils import lab2rgb


@HOOKS.register_module("MMGenVisualizationHook")
class VisualizationHook(Hook):
    """Visualization hook.
    In this hook, we use the official api `save_image` in torchvision to save
    the visualization results.
    Args:
        output_dir (str): The file path to store visualizations.
        res_name_list (str): The list contains the name of results in outputs
            dict. The results in outputs dict must be a torch.Tensor with shape
            (n, c, h, w).
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
        filename_tmpl (str): Format string used to save images. The output file
            name will be formatted as this args. Default: 'iter_{}.png'.
        rerange (bool): Whether to rerange the output value from [-1, 1] to
            [0, 1]. We highly recommend users should preprocess the
            visualization results on their own. Here, we just provide a simple
            interface. Default: True.
        bgr2rgb (bool): Whether to reformat the channel dimension from BGR to
            RGB. The final image we will save is following RGB style.
            Default: True.
        nrow (int): The number of samples in a row. Default: 1.
        padding (int): The number of padding pixels between each samples.
            Default: 4.
    """

    def __init__(
        self,
        output_dir,
        test_output_dir,
        res_name_list,
        use_lab=False,
        interval=-1,
        filename_tmpl="iter_{}.png",
        rerange=True,
        bgr2rgb=True,
        nrow=1,
        padding=4,
    ):
        assert mmcv.is_list_of(res_name_list, str)
        self.output_dir = output_dir
        self.test_output_dir = test_output_dir
        self.res_name_list = res_name_list
        self.use_lab = use_lab
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self.bgr2rgb = bgr2rgb
        self.rerange = rerange
        self.nrow = nrow
        self.padding = padding

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        results = runner.outputs["results"]
        if self.use_lab:
            L = results["image_gray_real"][:, -1:, :, :]
            ab_real = results["image_color_real"]
            ab_fake = results["image_color_fake"]
            results["image_color_real"] = lab2rgb(L, ab_real)
            results["image_color_fake"] = lab2rgb(L, ab_fake)
        inputs = results["image_gray_real"]
        target = results["image_color_real"]

        filename = self.filename_tmpl.format(runner.iter + 1)

        # img_list = [x for k, x in results.items() if k in self.res_name_list]
        # img_list = [results[k] for k in self.res_name_list if k in results]
        img_list = [results["image_color_fake"]]
        # inp_list = [inputs[k] for k in range(inputs.shape[0])]
        inp_list = [inputs]
        tar_list = [target]
        # source_list = img_list + inp_list + tar_list
        source_list = inp_list + tar_list
        img_cat = torch.cat(img_list, dim=3).detach()
        source_cat = torch.cat(source_list, dim=3).detach()
        # img_cat = img_cat * -1
        img_cat = torch.cat([img_cat, source_cat], dim=3)
        if self.rerange:
            img_cat = (img_cat + 1) / 2
        if self.bgr2rgb:
            img_cat = img_cat[:, [2, 1, 0], ...]
        img_cat = img_cat.clamp_(0, 1)

        if not hasattr(self, "_out_dir"):
            self._out_dir = osp.join(runner.work_dir, self.output_dir)
        mmcv.mkdir_or_exist(self._out_dir)
        save_image(
            img_cat,
            osp.join(self._out_dir, filename),
            nrow=self.nrow,
            padding=self.padding,
        )

    @master_only
    def after_val_iter(self, runner):
        """The behavior after each val iteration.

        Args:
            runner (object): The runner.
        """
        # if not self.every_n_iters(runner, self.interval):
        #    return
        results = runner.outputs
        if self.use_lab:
            L = results["img_gray"][:, -1:, :, :]
            ab_real = results["img_color_real"]
            ab_fake = results["img_color_fake"]
            results["img_color_real"] = lab2rgb(L, ab_real)
            results["img_color_fake"] = lab2rgb(L, ab_fake)
        inputs = results["img_gray"]
        target = results["img_color_real"]

        filename = self.filename_tmpl.format(runner.iter + 1)

        # img_list = [x for k, x in results.items() if k in self.res_name_list]
        # img_list = [results[k] for k in self.res_name_list if k in results]
        img_list = [results["img_color_fake"]]
        # inp_list = [inputs[k] for k in range(inputs.shape[0])]
        inp_list = [inputs]
        tar_list = [target]
        # source_list = img_list + inp_list + tar_list
        source_list = inp_list + tar_list
        img_cat = torch.cat(img_list, dim=3).detach()
        source_cat = torch.cat(source_list, dim=3).detach()
        # img_cat = img_cat * -1
        img_cat = torch.cat([img_cat, source_cat], dim=3)
        if self.rerange:
            img_cat = (img_cat + 1) / 2
        if self.bgr2rgb:
            img_cat = img_cat[:, [2, 1, 0], ...]
        img_cat = img_cat.clamp_(0, 1)

        if not hasattr(self, "_test_out_dir"):
            self._test_out_dir = osp.join(runner.work_dir, self.test_output_dir)
        mmcv.mkdir_or_exist(self._test_out_dir)
        print(self._test_out_dir)
        save_image(
            img_cat,
            osp.join(self._test_out_dir, filename),
            nrow=self.nrow,
            padding=self.padding,
        )
