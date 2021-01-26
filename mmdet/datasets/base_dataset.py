from __future__ import division
import json
import logging
import numbers
import os
import os.path as op
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pod.utils.dist_helper import get_world_size
from pycocotools import mask as mask_utils
from torch.nn.modules.utils import _pair
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
)

from ..utils.vis_helper import vis_one_image
from . import transforms as T
from .sampler import IterationBasedBatchSampler

try:
    import accimage
except ImportError:
    accimage = None

logger = logging.getLogger("global")


class BaseDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    ):
        super(BaseDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            self._collate_fn,
            pin_memory,
            drop_last,
        )

    def _collate_fn(self, batch):
        """
        Form a mini-batch from list of data of :meth:`~pod.datasets.base_dataset.BaseDataset.__getitem__`

        Arguments:
            - batch (:obj:`list` of data): type of data depends on output of Dataset.
              For :class:`~pod.datasets.coco_dataset.CocoDataset`,
              :class:`~pod.datasets.voc_dataset.VocDataset` and :class:`~pod.datasets.custom_dataset.CustomDataset`,
              data is a Dictionary.

        Returns:
            - output (:obj:`dict`)

        Output example::

            {
                # (FloatTensor): [B, 3, max_h, max_w], RGB format
                'image': ..,
                # (list of FloatTensor): [B, 5], (resized_h, resize_w, scale_factor, origin_h, origin_w)
                'image_info': ..,
                # (list of FloatTensor): [B] [num_gts, 5] or None
                'gt_bboxes': ..,
                # (list of FloatTensor): [B] [num_igs, 4] or None
                'ig_bboxes': ..,
                #(list of FloatTensor): [B] [num_gts, k, 3] or None
                'gt_keyps': ..,
                #(list of list of list of ndarray): [B] [num_gts] [polygons] or None
                'gt_masks': ..,
                # (FloatTensor) [B, max_num_gts, num_grid_points(9), 3] or None
                'gt_grids': ..,
                # filename of images
                'filenames': ..
            }
        """
        images = [_["image"] for _ in batch]
        image_info = [_["image_info"] for _ in batch]
        filenames = [_["filename"] for _ in batch]

        gt_bboxes = [_.get("gt_bboxes", None) for _ in batch]
        gt_ignores = [_.get("gt_ignores", None) for _ in batch]
        gt_keyps = [_.get("gt_keyps", None) for _ in batch]
        gt_masks = [_.get("gt_masks", None) for _ in batch]
        gt_grids = [_.get("gt_grids", None) for _ in batch]

        alignment = self.dataset.alignment
        padded_images = T.pad_images(images, alignment)
        output = {
            "image": padded_images,
            "image_info": image_info,
            "filenames": filenames,
        }

        output["gt_bboxes"] = gt_bboxes if gt_bboxes[0] is not None else None
        output["gt_ignores"] = gt_ignores if gt_ignores[0] is not None else None
        output["gt_keyps"] = gt_keyps if gt_keyps[0] is not None else None
        output["gt_masks"] = gt_masks if gt_masks[0] is not None else None
        output["gt_grids"] = gt_grids if gt_grids[0] is not None else None

        # Add for OpenImage
        gt_neg_labels = [_.get("neg_labels", None) for _ in batch]
        output["neg_labels"] = gt_neg_labels if gt_neg_labels[0] is not None else None
        return output

    def get_data_size(self):
        if isinstance(self.batch_sampler, IterationBasedBatchSampler):
            return len(self.batch_sampler.batch_sampler)  # training
        return len(self.batch_sampler)


class BaseTransform(object):
    """
    Flip and resize images, gt bboxes, ignore bboxes, gt masks, gt keypoints.

    Arguments:
        scales (list): candicate length of the shorter edge
        max_size (int): maximum length of the longer edge
        flip (bool): if True, flip the gts
    """

    def __init__(self, scales, max_size, flip, flip_p=0.5):
        self.scales = scales
        self.max_size = max_size
        self.flip = flip
        self.flip_p = flip_p

    def __call__(self, input):
        img = input["image"]
        gt_bboxes = input.get("gt_bboxes", None)
        gt_ignores = input.get("gt_ignores", None)
        gt_keyps = input.get("gt_keyps", None)
        gt_masks = input.get("gt_masks", None)
        pairs = input.get("keyp_pairs", None)

        height, width = img.shape[:2]
        if self.flip and T.get_flip_flag(flip_p=self.flip_p):
            img = T.flip_image(img)
            if gt_bboxes is not None:
                gt_bboxes = T.flip_boxes(gt_bboxes, width)
            if gt_ignores is not None:
                gt_ignores = T.flip_boxes(gt_ignores, width)
            if gt_keyps is not None:
                gt_keyps = T.flip_keyps(gt_keyps, pairs, width)
            if gt_masks is not None:
                gt_masks = T.flip_masks(gt_masks, width)
            input["flipped"] = True
        else:
            input["flipped"] = False

        scale_factor = T.get_scale_factor(self.scales, self.max_size, height, width)
        resized_img = T.resize_image(img, scale_factor)
        input["scale_factor"] = scale_factor
        input["image"] = resized_img

        if gt_bboxes is not None:
            input["gt_bboxes"] = T.resize_boxes(gt_bboxes, scale_factor)
        if gt_ignores is not None:
            input["gt_ignores"] = T.resize_boxes(gt_ignores, scale_factor)
        if gt_keyps is not None:
            input["gt_keyps"] = T.resize_keyps(gt_keyps, scale_factor)
        if gt_masks is not None:
            input["gt_masks"] = T.resize_masks(gt_masks, scale_factor)
        return input


class CaffeCocoTransform(object):
    """
    Flip and resize images, gt bboxes, ignore bboxes, gt masks, gt keypoints.

    Arguments:
        scales (list): candicate length of the shorter edge
        max_size (int): maximum length of the longer edge
        flip (bool): if True, flip the gts
    """

    def __init__(self, scales, max_size, flip, flip_p=0.5):
        self.scales = scales
        self.max_size = max_size
        self.flip = flip
        self.flip_p = flip_p

    def __call__(self, input):
        img = input["image"]
        gt_bboxes = input.get("gt_bboxes", None)
        gt_ignores = input.get("gt_ignores", None)
        gt_keyps = input.get("gt_keyps", None)
        gt_masks = input.get("gt_masks", None)
        pairs = input.get("keyp_pairs", None)

        # PIL.Image
        origin_width, origin_height = img.size
        scale_h, scale_w = T.get_scale_factor_v2(
            self.scales, self.max_size, origin_height, origin_width
        )
        scale_factor = (scale_h, scale_w)
        resized_height, resized_width = (
            round(scale_h * origin_height),
            round(scale_w * origin_width),
        )
        img = F.resize(img, (resized_height, resized_width))

        if gt_bboxes is not None:
            gt_bboxes = T.resize_boxes(gt_bboxes, scale_factor)
        if gt_ignores is not None:
            gt_ignores = T.resize_boxes(gt_ignores, scale_factor)
        if gt_keyps is not None:
            gt_keyps = T.resize_keyps(gt_keyps, scale_factor)
        if gt_masks is not None:
            gt_masks = T.resize_masks(gt_masks, scale_factor)

        if self.flip and T.get_flip_flag(flip_p=self.flip_p):
            img = F.hflip(img)
            if gt_bboxes is not None:
                gt_bboxes = T.flip_boxes(gt_bboxes, resized_width)
            if gt_ignores is not None:
                gt_ignores = T.flip_boxes(gt_ignores, resized_width)
            if gt_keyps is not None:
                gt_keyps = T.flip_keyps(gt_keyps, pairs, resized_width)
            if gt_masks is not None:
                gt_masks = T.flip_masks(gt_masks, resized_width)
            flipped = True
        else:
            flipped = False
        img = F.to_tensor(img)
        # to_bgr255
        img = img[[2, 1, 0]] * 255

        return {
            "image": img,
            "gt_bboxes": gt_bboxes,
            "gt_ignores": gt_ignores,
            "gt_keyps": gt_keyps,
            "gt_masks": gt_masks,
            "scale_factor": scale_factor,
            "flipped": flipped,
        }


class BaseDataset(Dataset):
    """
    A dataset should implement interface below:

    1. :meth:`__len__` to get size of the dataset, Required
    2. :meth:`__getitem__` to get a single data, Required
    3. evaluate to get metrics, Required
    4. dump to output results, Required
    5. visualize gts or dts to check annotations, Optional
    """

    def __init__(self):
        """
        """
        super(BaseDataset, self).__init__()

    def __len__(self):
        """
        Returns dataset length
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Get a single image data: from dataset

        Arguments:
            - idx (:obj:`int`): index of image, 0 <= idx < len(self)

        """
        raise NotImplementedError

    def get_image_id(self, path_name):
        raise NotImplementedError

    def _generate_grids(self, gt_boxes):
        """
        Generate grid points by (transformed) box coordinates.
        9 points are generated by default.
        :param gt_bboxes: [N, 5]
        :return: grids: [N, num_grid_points(9), 3]
        """
        num_box_kpt = 9
        N = gt_boxes.shape[0]
        grids = np.zeros([N, num_box_kpt, 2])
        x1 = gt_boxes[:, 0]
        y1 = gt_boxes[:, 1]
        x2 = gt_boxes[:, 2]
        y2 = gt_boxes[:, 3]
        grids[:, :3, 0] = np.tile(x1, (3, 1)).transpose()
        grids[:, 3:6, 0] = np.tile((x1 + x2) / 2, (3, 1)).transpose()
        grids[:, 6:, 0] = np.tile(x2, (3, 1)).transpose()
        grids[:, 0::3, 1] = np.tile(y1, (3, 1)).transpose()
        grids[:, 1::3, 1] = np.tile((y1 + y2) / 2, (3, 1)).transpose()
        grids[:, 2::3, 1] = np.tile(y2, (3, 1)).transpose()
        grids = np.hstack([grids.reshape(-1, 2), np.ones([N * num_box_kpt, 1])])
        grids = grids.reshape(N, num_box_kpt, 3)

        return grids

    def dump(self, writer, output):
        """
        Dump bboxes with format of (img_id, x1, y1, x2, y2, score class)

        Arguments:
            - writer: output stream
            - output (:obj:`dict`): dict with keys: :code:`{'image_info', 'bboxes', 'filenames'}`

        Output example::

            {
                # (FloatTensor): [B, >=3] (resized_h, resized_w, resize_scale)
                'image_info': <tensor>,
                # (FloatTensor) [N, >=7] (batch_idx, x1, y1, x2, y2, score, cls)
                'bboxes': <tensor>,
                # (list of str): [B], image names
                'filenames': []
            }

        """
        filenames = output["filenames"]
        image_info = self.tensor2numpy(output["image_info"])
        bboxes = self.tensor2numpy(output["dt_bboxes"])

        dump_results = []
        for b_ix in range(len(image_info)):
            scale_h, scale_w = _pair(image_info[b_ix][2])
            img_id = self.get_image_id(filenames[b_ix])

            scores = bboxes[:, 5]
            keep_ix = np.where(bboxes[:, 0] == b_ix)[0]
            keep_ix = sorted(keep_ix, key=lambda ix: scores[ix], reverse=True)
            img_bboxes = bboxes[keep_ix]
            img_bboxes[:, 1] /= scale_w
            img_bboxes[:, 2] /= scale_h
            img_bboxes[:, 3] /= scale_w
            img_bboxes[:, 4] /= scale_h

            for bbox in img_bboxes:
                res = {
                    "image_id": img_id,
                    "bbox": bbox[1 : 1 + 4].tolist(),
                    "score": float(bbox[5]),
                    "label": int(bbox[6]),
                }
                dump_results.append(json.dumps(res, ensure_ascii=False))
        writer.write("\n".join(dump_results) + "\n")
        writer.flush()

    def merge(self, prefix):
        """
        Merge results into one file

        Arguments:
            - prefix (:obj:`str`): dir/results.rank
        """
        world_size = get_world_size()
        merged_file = prefix.rsplit(".", 1)[0] + ".all"
        logger.info(f"concat all results into:{merged_file}")
        merged_fd = open(merged_file, "w")
        for rank in range(world_size):
            res_file = prefix + str(rank)
            assert op.exists(res_file), f"No such file or directory: {res_file}"
            with open(res_file, "r") as fin:
                for line_idx, line in enumerate(fin):
                    merged_fd.write(line)
            logger.info(f"merging {res_file} {line_idx+1} results")
        merged_fd.close()
        return merged_file

    def evaluate(self, res_file):
        """
        Arguments:
            - res_file (:obj:`str`): filename
        """
        prefix = res_file.rstrip("0123456789")
        res_file = self.merge(prefix)
        metrics = self.evaluator.eval(res_file) if self.evaluator else {}
        return metrics

    def get_np_images(self, images):
        mean = np.array(self.normalize_fn.mean).reshape(1, 1, 1, 3) * 255
        std = np.array(self.normalize_fn.std).reshape(1, 1, 1, 3) * 255
        images = images.permute(0, 2, 3, 1).contiguous()
        images = images.cpu().float().numpy().copy() * std + mean
        return images.astype(np.uint8)

    def tensor2numpy(self, x):
        if x is None:
            return x
        if torch.is_tensor(x):
            return x.cpu().numpy()
        if isinstance(x, list):
            x = [_.cpu().numpy() if torch.is_tensor(_) else _ for _ in x]
        return x

    def vis_gt(self, vis_cfg, input):
        """
        Arguments:
            - vis_cfg (:obj:`dict`):
            - input (:obj:`dict`): output of model

        """

        def poly_to_mask(polygons, height, width):
            rles = mask_utils.frPyObjects(polygons, height, width)
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
            return mask

        output_dir = vis_cfg.get("output_dir", "vis_gt")
        output_ext = vis_cfg.get("ext", "svg")
        os.makedirs(output_dir, exist_ok=True)
        gt_images = self.get_np_images(input["image"])
        filenames = input["filenames"]
        gt_bboxes = self.tensor2numpy(input.get("gt_bboxes", None))
        gt_keyps = self.tensor2numpy(input.get("gt_keyps", None))
        gt_ignores = self.tensor2numpy(input.get("gt_ignores", None))

        gt_masks = None
        if input.get("gt_masks", None):
            gt_masks = []
            for b_ix, polys in enumerate(input["gt_masks"]):
                height, width = gt_images[b_ix].shape[:2]
                masks = np.array([poly_to_mask(_, height, width) for _ in polys])
                gt_masks.append(masks)

        batch_size = len(filenames)
        for b_ix in range(batch_size):
            image_name = op.splitext(op.basename(filenames[b_ix]))[0]
            image = gt_images[b_ix]
            bboxes = classes = ignores = keyps = masks = None
            if gt_bboxes is not None:
                bboxes = gt_bboxes[b_ix]
                classes = bboxes[:, 4].astype(np.int32)
                scores = np.ones((bboxes.shape[0], 1))
                bboxes = np.hstack([bboxes[:, :4], scores])
            if gt_ignores is not None:
                ignores = gt_ignores[b_ix]
            if gt_keyps is not None:
                keyps = gt_keyps[b_ix].transpose(0, 2, 1)
            if gt_masks is not None:
                masks = np.asfortranarray(gt_masks[b_ix] > 0.5, dtype=np.uint8)
                masks = masks.transpose(1, 2, 0)

            vis_one_image(
                image,
                image_name,
                output_dir,
                bboxes,
                classes,
                ignores,
                masks,
                keyps,
                dataset=self,
                show_class=True,
                box_alpha=0.7,
                kp_thresh=0.0,
                ext=output_ext,
            )

    def vis_dt(self, vis_cfg, input):
        """
        Arguments:
            - vis_cfg (:obj:`dict`):
            - input (:obj:`dict`): output of model

        """
        output_dir = vis_cfg.get("output_dir", "vis_dt")
        output_ext = vis_cfg.get("ext", "svg")
        bbox_thresh = vis_cfg.get("bbox_thresh", 0.3)
        keyp_thresh = vis_cfg.get("keyp_thresh", 0.1)
        mask_thresh = vis_cfg.get("mask_thresh", 0.5)

        os.makedirs(output_dir, exist_ok=True)
        dt_images = self.get_np_images(input["image"])
        filenames = input["filenames"]
        dt_bboxes = self.tensor2numpy(input.get("dt_bboxes", None))
        dt_keyps = self.tensor2numpy(input.get("dt_keyps", None))

        dt_masks = None
        if input.get("dt_masks", None):
            dt_masks = [self.tensor2numpy(m) for m in input["dt_masks"]]

        if dt_bboxes is None:
            return

        batch_size = len(filenames)
        for b_ix in range(batch_size):
            image_name = op.splitext(op.basename(filenames[b_ix]))[0]
            image = dt_images[b_ix]
            image_h, image_w, _ = image.shape
            keep_ix = np.where(dt_bboxes[:, 0] == b_ix)[0]
            bboxes = dt_bboxes[keep_ix]
            if bboxes.size == 0:
                continue
            classes = bboxes[:, 6].astype(np.int32)
            bboxes = bboxes[:, 1:6]
            keyps = masks = None
            if dt_keyps is not None:
                keyps = dt_keyps[keep_ix].transpose(0, 2, 1)
            if dt_masks is not None:
                masks = [dt_masks[ix] for ix in keep_ix]
                masks = np.stack(masks, axis=2)
                masks = cv2.resize(masks, (image_w, image_h))
                masks = (masks > mask_thresh).astype(np.uint8)

            vis_one_image(
                image,
                image_name,
                output_dir,
                bboxes,
                classes,
                None,
                masks,
                keyps,
                dataset=self,
                show_class=True,
                box_alpha=0.7,
                thresh=bbox_thresh,
                kp_thresh=keyp_thresh,
                ext=output_ext,
            )


class RandomColorJitter(object):
    """
    Randomly change the brightness, contrast and saturation of an image.

    Arguments:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.prob = prob

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def get_params(self, brightness, contrast, saturation, hue):
        """
        Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        img_transforms = []

        if brightness is not None and random.random() < self.prob:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_brightness(img, brightness_factor))
            )

        if contrast is not None and random.random() < self.prob:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_contrast(img, contrast_factor))
            )

        if saturation is not None and random.random() < self.prob:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_saturation(img, saturation_factor))
            )

        if hue is not None and random.random() < self.prob:
            hue_factor = random.uniform(hue[0], hue[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_hue(img, hue_factor))
            )

        random.shuffle(img_transforms)
        img_transforms = transforms.Compose(img_transforms)

        return img_transforms

    def __call__(self, img):
        """
        Arguments:
            img (np.array): Input image.
        Returns:
            img (np.array): Color jittered image.
        """
        img = Image.fromarray(np.uint8(img))
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        img = transform(img)
        img = np.asanyarray(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        return format_string

    @classmethod
    def from_params(cls, params):
        brightness = params.get("data_augmentation", 0.1)
        contrast = params.get("contrast", 0.5)
        hue = params.get("hue", 0.07)
        saturation = params.get("saturation", 0.5)
        prob = params.get("prob", 0.25)
        return cls(
            brightness=brightness,
            contrast=contrast,
            hue=hue,
            saturation=saturation,
            prob=prob,
        )
