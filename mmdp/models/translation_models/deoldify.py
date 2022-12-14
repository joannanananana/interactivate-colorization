# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp

import mmcv
import torch
from mmcv.runner import auto_fp16

from mmdp.models.builder import MODELS, build_module
from mmdp.models.gans.base import BaseModel
from mmdp.models.misc import tensor2img

from ..common import set_requires_grad


@MODELS.register_module()
class DeOldify(BaseModel):
    """DeOldify model for image colorization.

    Ref:
    https://github.com/jantic/DeOldify
    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator.
        gan_loss (dict): Config for the gan loss.
        perceptual_loss (dict): Config for the perceptual loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generator
            update.
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
            You may change the testing of gan by setting:
            `show_input`: whether to show input real images.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(
        self,
        generator,
        discriminator,
        gan_loss,
        use_lab=False,
        nogan_stage="GandD",
        perceptual_loss=None,
        l1_loss=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = build_module(generator)
        # discriminator
        self.discriminator = build_module(discriminator)

        # losses
        assert gan_loss is not None  # gan loss cannot be None
        self.gan_loss = build_module(gan_loss)
        self.use_lab = use_lab
        self.perceptual_loss = (
            build_module(perceptual_loss) if perceptual_loss else None
        )
        self.l1_loss = torch.nn.L1Loss() if l1_loss else None
        self.l1_loss_weight = l1_loss["loss_weight"] if l1_loss else 0.0

        self.disc_steps = (
            1 if self.train_cfg is None else self.train_cfg.get("disc_steps", 1)
        )
        self.disc_init_steps = (
            0 if self.train_cfg is None else self.train_cfg.get("disc_init_steps", 0)
        )
        self.nogan_stage = nogan_stage

        self.step_counter = 0  # counting training steps

        self.show_input = (
            False if self.test_cfg is None else self.test_cfg.get("show_input", False)
        )

        # support fp16
        self.fp16_enabled = False
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        # self.generator.init_weights(pretrained=pretrained)
        # self.discriminator.init_weights(pretrained=pretrained)
        pass

    def setup(self, img_gray, img_color, meta):
        """Perform necessary pre-processing steps.

        Args:
        img_gray (Tensor): Input gray image.
        img_color (Tensor): Input color image.
        meta (list[dict]): Input meta data.

        Returns:
        Tensor, Tensor, list[str]: The gray/color images, and \
         the image path as the metadata.
        """
        image_gray_real = img_gray
        image_color_real = img_color
        image_path = [v["img_gray_path"] for v in meta]

        return image_gray_real, image_color_real, image_path

    @auto_fp16(apply_to=("img_gray",))
    def forward(self, img_gray, img_color=None, meta=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            img_gray (Tensor): Input gray image.
            img_color (Tensor): Input color image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if test_mode:
            return self.forward_test(img_gray, img_color, meta, **kwargs)

        return self.forward_train(img_gray, img_color, meta)

    def forward_train(self, img_gray, img_color, meta):
        """Forward function for training.

        Args:
            img_gray (Tensor): Input gray image.
            img_color (Tensor): Input color image.
            meta (list[dict]): Input meta data.
        Returns:
            dict: Dict of forward results for training.
        """
        # necessary setup
        img_gray_real, img_color_real, _ = self.setup(img_gray, img_color, meta)
        img_color_fake = self.generator(img_gray_real)
        results = dict(
            img_gray_real=img_gray_real,
            img_color_fake=img_color_fake,
            img_color_real=img_color_real,
        )
        return results

    def forward_test(
        self,
        img_gray,
        img_color=None,
        meta=None,
        save_image=False,
        save_path=None,
        iteration=None,
    ):
        """Forward function for testing.

        Args:
            img_gray (Tensor): Input gray image.
            img_color (Tensor): Input color image. Default: None
            meta (list[dict]): Input meta data.
            save_image (bool, optional): If True, results will be saved as
                images. Default: False.
            save_path (str, optional): If given a valid str path, the results
                will be saved in this path. Default: None.
            iteration (int, optional): Iteration number. Default: None.
        Returns:
            dict: Dict of forward and evaluation results for testing.
        """

        img_gray_real, img_color_real = img_gray, img_color

        img_color_fake = self.generator(img_gray_real)

        results = dict(
            img_gray=img_gray_real.cpu(), img_color_fake=img_color_fake.cpu()
        )
        if img_color_real is not None:
            results["img_color_real"] = img_color_real.cpu()

        # save image
        if save_image:
            img_gray_path = meta[0]["img_gray_path"]
            folder_name = osp.splitext(osp.basename(img_gray_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(
                    save_path, folder_name, f"{folder_name}-{iteration + 1:06d}.png"
                )
            elif iteration is None:
                save_path = osp.join(save_path, f"{folder_name}.png")
            else:
                raise ValueError(
                    "iteration should be number or None, " f"but got {type(iteration)}"
                )
            mmcv.imwrite(tensor2img(img_color_fake), save_path)

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

    def backward_discriminator(self, outputs):
        """Backward function for the discriminator.

        Args:
            outputs (dict): Dict of forward results.
        Returns:
            dict: Loss dict.
        """
        # GAN loss for the discriminator
        losses = dict()
        # conditional GAN
        if self.use_lab:
            fake_ab = torch.cat(
                (outputs["img_gray_real"][:, -1:, :, :], outputs["img_color_fake"]), 1
            )
            real_ab = torch.cat(
                (outputs["img_gray_real"][:, -1:, :, :], outputs["img_color_real"]), 1
            )
        else:
            fake_ab = torch.cat(
                (outputs["img_gray_real"], outputs["img_color_fake"]), 1
            )
            real_ab = torch.cat(
                (outputs["img_gray_real"], outputs["img_color_real"]), 1
            )
        fake_pred = self.discriminator(fake_ab.detach().cuda())
        losses["loss_gan_d_fake"] = self.gan_loss(
            fake_pred, target_is_real=False, is_disc=True
        )

        real_pred = self.discriminator(real_ab.cuda())
        losses["loss_gan_d_real"] = self.gan_loss(
            real_pred, target_is_real=True, is_disc=True
        )

        loss_d, log_vars_d = self.parse_losses(losses)
        loss_d *= 0.5
        loss_d.backward()
        return log_vars_d

    def backward_generator(self, outputs):
        """Backward function for the generator.

        Args:
            outputs (dict): Dict of forward results.
        Returns:
            dict: Loss dict.
        """
        if self.use_lab:
            fake_ab = torch.cat(
                (outputs["img_gray_real"][:, -1:, :, :], outputs["img_color_fake"]), 1
            )
            real_ab = torch.cat(
                (outputs["img_gray_real"][:, -1:, :, :], outputs["img_color_real"]), 1
            )
        else:
            fake_ab = torch.cat(
                (outputs["img_gray_real"], outputs["img_color_fake"]), 1
            )
            real_ab = torch.cat(
                (outputs["img_gray_real"], outputs["img_color_real"]), 1
            )
        losses = dict()
        # GAN loss for the generator
        if self.nogan_stage == "GandD":
            fake_pred = self.discriminator(fake_ab.cuda())
            losses["loss_gan_g"] = self.gan_loss(
                fake_pred, target_is_real=True, is_disc=False
            )
        # perceptual loss for the generator
        if self.nogan_stage != "Donly":
            if self.perceptual_loss:
                if self.use_lab:
                    losses["loss_perceptual"], _ = self.perceptual_loss(
                        fake_ab, real_ab
                    )
                else:
                    losses["loss_perceptual"], _ = self.perceptual_loss(
                        outputs["img_color_fake"], outputs["img_color_real"]
                    )
            if self.l1_loss:
                losses["l1_loss"] = self.l1_loss_weight * self.l1_loss(
                    outputs["img_color_fake"], outputs["img_color_real"]
                )

        loss_g, log_vars_g = self.parse_losses(losses)
        loss_g.backward()
        return log_vars_g

    def train_step(self, data_batch, optimizer):
        """Training step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generator and discriminator.
        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        # data
        img_gray = data_batch["img_gray"]
        img_color = data_batch["img_color"]
        meta = data_batch["meta"]

        if self.nogan_stage == "Donly":
            set_requires_grad(self.generator, False)
        if self.nogan_stage == "Gonly":
            set_requires_grad(self.discriminator, False)
        # forward generator
        outputs = self.forward(img_gray, img_color, meta, False)

        log_vars = dict()

        # print("DEDBUG----------------")
        # for k, optim in optimizer.items():
        #     for param_group in optim.param_groups:
        #         print(param_group['lr'])

        if self.nogan_stage != "Gonly":
            # discriminator
            set_requires_grad(self.discriminator, True)
            # optimize
            optimizer["discriminator"].zero_grad()
            log_vars.update(self.backward_discriminator(outputs=outputs))
            optimizer["discriminator"].step()

        if self.nogan_stage != "Donly":
            # generator, no updates to discriminator parameters.
            if (
                self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps
            ):
                set_requires_grad(self.discriminator, False)
                # optimize
                optimizer["generator"].zero_grad()
                log_vars.update(self.backward_generator(outputs=outputs))
                optimizer["generator"].step()

        self.step_counter += 1

        log_vars.pop("loss", None)  # remove the unnecessary 'loss'
        results = dict(
            log_vars=log_vars,
            num_samples=len(outputs["img_gray_real"]),
            results=dict(
                image_gray_real=outputs["img_gray_real"].cpu(),
                image_color_fake=outputs["img_color_fake"].cpu(),
                image_color_real=outputs["img_color_real"].cpu(),
            ),
        )

        return results

    def val_step(self, data_batch, **kwargs):
        """Validation step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            kwargs (dict): Other arguments.
        Returns:
            dict: Dict of evaluation results for validation.
        """
        # data
        img_gray = data_batch["img_gray"]
        img_color = data_batch["img_color"]
        meta = data_batch["meta"]

        # forward generator
        results = self.forward(img_gray, img_color, meta, True, **kwargs)
        return results
