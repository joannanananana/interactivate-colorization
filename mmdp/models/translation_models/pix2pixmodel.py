# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import torch

from mmdp.models import tensor2img

from ..base import BaseModel
from ..builder import MODELS, build_module
from ..util_pix import (
    decode_max_ab,
    decode_mean,
    encode_ab_ind,
    get_colorization_data,
    lab2rgb,
    tensor2im,
)


def L1Loss(in0, in1):
    return torch.sum(torch.abs(in0 - in1), dim=1, keepdim=True)


@MODELS.register_module()
class Pix2Pix(BaseModel):
    """Pix2Pix model for paired image-to-image translation.

    Ref:
    Image-to-Image Translation with Conditional Adversarial Networks
    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator.
        gan_loss (dict): Config for the gan loss.
        pixel_loss (dict): Config for the pixel loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generator
            update.
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
            `direction`: image-to-image translation direction (the model
            training direction): a2b | b2a.
        test_cfg (dict): Config for testing. Default: None.
            You may change the testing of gan by setting:
            `direction`: image-to-image translation direction (the model
            training direction, same as testing direction): a2b | b2a.
            `show_input`: whether to show input real images.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(
        self,
        generator,
        gan_loss,
        l1_loss,
        l2_loss,
        lambda_A,
        pixel_loss=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.lambda_A = lambda_A
        # generator
        self.generator = build_module(generator)
        # losses
        assert gan_loss is not None  # gan loss cannot be None
        self.gan_loss = build_module(gan_loss)
        self.pixel_loss = build_module(pixel_loss) if pixel_loss else None
        self.l1_loss = build_module(l1_loss)
        self.l2_loss = build_module(l2_loss)
        self.criterionCE = torch.nn.CrossEntropyLoss()
        self.disc_steps = (
            1 if self.train_cfg is None else self.train_cfg.get("disc_steps", 1)
        )
        self.disc_init_steps = (
            0 if self.train_cfg is None else self.train_cfg.get("disc_init_steps", 0)
        )
        if self.train_cfg is None:
            self.direction = (
                "a2b"
                if self.test_cfg is None
                else self.test_cfg.get("direction", "a2b")
            )
        else:
            self.direction = self.train_cfg.get("direction", "a2b")
        self.step_counter = 0  # counting training steps

        self.show_input = (
            False if self.test_cfg is None else self.test_cfg.get("show_input", False)
        )
        self.criterionL1 = build_module(l1_loss)
        # support fp16
        self.fp16_enabled = False
        self.init_weights(pretrained)
        self.sample_p = train_cfg.get("sample_p")

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        self.generator.init_weights(pretrained=pretrained)

    def setup(self, img_a, img_b, meta):
        """Perform necessary pre-processing steps.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
        Returns:
            Tensor, Tensor, list[str]: The real images from domain A/B, and \
                the image path as the metadata.
        """
        a2b = self.direction == "a2b"
        real_a = img_a if a2b else img_b
        real_b = img_b if a2b else img_a
        image_path = [v["img_a_path" if a2b else "img_b_path"] for v in meta]

        return real_a, real_b, image_path

    # @auto_fp16(apply_to=('img_a', 'img_b'))  #modified
    def forward_train(self, img_a, hint_b, mask_b):
        """new Args to_do modified input:self.real_A, self.hint_B, self.mask_B
        input for generator."""
        """Forward function for training.
        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
        Returns:
            dict: Dict of forward results for training.
        """
        (self.fake_B_class, self.fake_B_reg) = self.generator(img_a, hint_b, mask_b)
        self.fake_B_dec_max = self.generator.upsample4(
            decode_max_ab(self.fake_B_class, self.train_cfg)
        )
        self.fake_B_distr = self.generator.softmax(self.fake_B_class)
        self.fake_B_dec_mean = self.generator.upsample4(
            decode_mean(self.fake_B_distr, self.train_cfg)
        )
        self.fake_B_entr = self.generator.upsample4(
            -torch.sum(
                self.fake_B_distr * torch.log(self.fake_B_distr + 1.0e-10),
                dim=1,
                keepdim=True,
            )
        )

        results = dict(fake_B_class=self.fake_B_class, fake_B_reg=self.fake_B_reg)
        return results

    def forward_test(
        self,
        img_a,
        hint_b,
        mask_b,
        save_image=False,
        save_path=None,
        iteration=None,
        **kwargs,
    ):
        """Forward function for testing.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            save_image (bool, optional): If True, results will be saved as
                images. Default: False.
            save_path (str, optional): If given a valid str path, the results
                will be saved in this path. Default: None.
            iteration (int, optional): Iteration number. Default: None.
        Returns:
            dict: Dict of forward and evaluation results for testing.
        """
        (fake_B_class, fake_B_reg) = self.generator(img_a, hint_b, mask_b)
        results = dict(real_a=img_a.cpu(), fake_b=fake_B_reg.cpu())

        # save image

        if save_image:
            assert save_path is not None
            # folder_name = osp.splitext(osp.basename(image_path[0]))[0]
            folder_name = ""
            if self.show_input:
                if iteration:
                    save_path = osp.join(
                        save_path,
                        folder_name,
                        f"{folder_name}-{iteration + 1:06d}-ra-fb-rb.png",
                    )
                else:
                    save_path = osp.join(save_path, f"{folder_name}-ra-fb-rb.png")
                output = np.concatenate(
                    [
                        tensor2img(results["real_a"], min_max=(-1, 1)),
                        tensor2img(results["fake_b"], min_max=(-1, 1)),
                        tensor2img(results["real_b"], min_max=(-1, 1)),
                    ],
                    axis=1,
                )
            else:
                if iteration:
                    save_path = osp.join(
                        save_path,
                        folder_name,
                        f"{folder_name}-{iteration + 1:06d}-fb.png",
                    )
                else:
                    save_path = osp.join(save_path, f"{folder_name}-fb.png")

                output = lab2rgb(
                    torch.cat(
                        (
                            results["real_a"].type(torch.cuda.FloatTensor),
                            results["fake_b"].type(torch.cuda.FloatTensor),
                        ),
                        dim=1,
                    ),
                    self.test_cfg,
                )
                output = tensor2im(output)
            flag = mmcv.imwrite(output, save_path)
            results["saved_flag"] = flag

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Dummy input used to compute FLOPs.
        Returns:
            Tensor: Dummy output produced by forwarding the dummy input.
        """
        out = self.generator(img)
        return out

    def forward(self, data_batch, test_mode=False, psnr=False, **kwargs):
        """Forward function."""
        """
        Args:
            img_a (Tensor): Input image.
            hint_b(Tensor): user defined or generated color hint
            mask_b(Tensor):  a binary mask indicating which points are provided by the user
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if psnr:
            data_color = get_colorization_data(
                data_batch, l_norm=100, l_cent=50, ab_norm=110, p=self.sample_p
            )
            try:
                self.real_a = data_color["A"]
                self.real_B = data_color["B"]
                self.hint_B = data_color["hint_B"]
                self.mask_b = data_color["mask_B"]
            except TypeError:
                print("invalid input exists")
        else:
            if not isinstance(data_batch, dict) and not isinstance(data_batch, list):
                return {
                    "fake_img": torch.zeros(1, 3, 1, 1),
                    "noise_batch": torch.zeros(1, 3, 1, 1),
                }  # for train_after_iterations
            z = data_batch["pair"].max() - data_batch["pair"].min()
            img_a = (data_batch["pair"] - data_batch["pair"].min()) / z
            data_color = get_colorization_data(
                img_a, l_norm=100, l_cent=50, ab_norm=110, p=self.sample_p
            )
            try:
                self.real_a = data_color["A"]
                self.real_B = data_color["B"]
                self.hint_B = data_color["hint_B"]
                self.mask_b = data_color["mask_B"]
            except TypeError:
                print("invalid input exists")
        if test_mode:
            return self.forward_test(self.real_a, self.hint_B, self.mask_b, **kwargs)

        return self.forward_train(self.real_a, self.hint_B, self.mask_b)

    def backward_generator(self, outputs):
        """
        marks: compute loss G
               backward()
        """
        """Backward function for the generator.
        Args:
            outputs (dict): Dict of forward results.
        Returns:
            dict: Loss dict.
        """
        losses = dict()
        mask_avg = torch.mean(self.mask_B_nc.type(torch.cuda.FloatTensor)) + 0.000001
        self.loss_G_CE = self.criterionCE(
            self.fake_B_class.type(torch.cuda.FloatTensor),
            self.real_B_enc[:, 0, :, :].type(torch.cuda.LongTensor),
        )
        self.loss_G_entr = torch.mean(self.fake_B_entr.type(torch.cuda.FloatTensor))
        self.loss_G_entr_hint = (
            torch.mean(
                self.fake_B_entr.type(torch.cuda.FloatTensor)
                * self.mask_B_nc.type(torch.cuda.FloatTensor)
            )
            / mask_avg
        )  # entropy of predicted distribution at hint points

        # regression statistics
        self.loss_G_L1_max = 10 * torch.mean(
            L1Loss(
                self.fake_B_dec_max.type(torch.cuda.FloatTensor),
                self.real_B.type(torch.cuda.FloatTensor),
            )
        )
        self.loss_G_L1_mean = 10 * torch.mean(
            L1Loss(
                self.fake_B_dec_mean.type(torch.cuda.FloatTensor),
                self.real_B.type(torch.cuda.FloatTensor),
            )
        )

        self.loss_G_L1_reg = 10 * torch.mean(
            L1Loss(
                self.fake_B_reg.type(torch.cuda.FloatTensor),
                self.real_B.type(torch.cuda.FloatTensor),
            )
        )

        # L1 loss at given points
        self.loss_G_fake_real = (
            10
            * torch.mean(
                L1Loss(
                    self.fake_B_reg * self.mask_B_nc, self.real_B * self.mask_B_nc
                ).type(torch.cuda.FloatTensor)
            )
            / mask_avg
        )
        self.loss_G_fake_hint = (
            10
            * torch.mean(
                L1Loss(
                    self.fake_B_reg * self.mask_B_nc, self.hint_B * self.mask_B_nc
                ).type(torch.cuda.FloatTensor)
            )
            / mask_avg
        )
        self.loss_G_real_hint = (
            10
            * torch.mean(
                L1Loss(self.real_B * self.mask_B_nc, self.hint_B * self.mask_B_nc).type(
                    torch.cuda.FloatTensor
                )
            )
            / mask_avg
        )
        # pixel loss for the generator
        if self.pixel_loss:
            losses["loss_pixel"] = self.pixel_loss(outputs["fake_B_reg"], self.real_B)
        losses["loss_g"] = self.loss_G_CE * self.lambda_A + self.loss_G_L1_reg
        losses["loss_g"].backward()
        loss_g, log_vars_g = self.parse_losses(losses)
        losses["G_CE"] = self.loss_G_CE
        losses["G_entr"] = self.loss_G_entr
        losses["G_entr_hint"] = self.loss_G_entr_hint
        losses["G_L1_max"] = self.loss_G_L1_max
        losses["G_L1_mean"] = self.loss_G_L1_mean
        losses["G_L1_reg"] = self.loss_G_L1_reg
        losses["G_fake_real"] = self.loss_G_fake_hint
        losses["G_fake_hint"] = self.loss_G_fake_hint
        losses["G_real_hint"] = self.loss_G_real_hint
        loss_g, log_vars_g = self.parse_losses(losses)
        return log_vars_g

    def train_step(self, data_batch, optimizer):
        """Training step function.

        called when train_model
        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generator and discriminator.
        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        # outputs = self.forward(data_batch[0])
        outputs = self.forward(data_batch)
        self.mask_B_nc = self.mask_b + 0.5
        self.real_B_enc = encode_ab_ind(self.real_B[:, :, ::4, ::4], self.train_cfg)
        log_vars = dict()
        optimizer["generator"].zero_grad()
        log_vars.update(self.backward_generator(outputs=outputs))
        optimizer["generator"].step()

        self.step_counter += 1

        results = dict(
            log_vars=log_vars,
            num_samples=len(self.real_a),
            results=dict(
                real_a=self.real_a.cpu(),
                fake_b=self.fake_B_reg.cpu(),
                real_b=self.real_B.cpu(),
            ),
        )
        return results

    def val_step(self, data_batch, psnr=False, **kwargs):
        """Validation step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            kwargs (dict): Other arguments.
        Returns:
            dict: Dict of evaluation results for validation.
        """
        # forward generator
        results = self.forward(data_batch, test_mode=True, psnr=True, **kwargs)
        return (results, self.real_B)

    def get_current_visuals(self):
        from collections import OrderedDict

        visual_ret = OrderedDict()
        visual_ret["real"] = lab2rgb(
            torch.cat(
                (
                    self.real_a.type(torch.cuda.FloatTensor),
                    self.real_B.type(torch.cuda.FloatTensor),
                ),
                dim=1,
            ),
            self.test_cfg,
        )
        visual_ret["fake_reg"] = lab2rgb(
            torch.cat(
                (
                    self.real_a.type(torch.cuda.FloatTensor),
                    self.fake_B_reg.type(torch.cuda.FloatTensor),
                ),
                dim=1,
            ),
            self.test_cfg,
        )
        return visual_ret
