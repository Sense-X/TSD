import functools
import os
import time

import numpy as np
from tqdm import tqdm

from .custom import CustomDataset
from .registry import DATASETS
from .utils import (
    OpenImagesDetectionChallengeEvaluator,
    get_categories,
    read_dets,
    read_gts,
)


@DATASETS.register_module
class OpenImagesDataset(CustomDataset):
    def load_annotations(self, ann_file):
        print("load annotation begin", flush=True)
        img_infos = []
        with open(ann_file) as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            img_gt = []
            labels = []
            img_name = lines[i].rstrip()
            i += 2
            img_gt_size = int(lines[i])
            i += 1
            for j in range(img_gt_size):
                sp = lines[i + j].split()
                img_gt.append([float(sp[1]), float(sp[2]), float(sp[3]), float(sp[4])])
                labels.append(int(sp[0]))
            i += img_gt_size
            img_infos.append([img_name, np.array(img_gt), np.array(labels)])

        print("load annotation end", flush=True)
        return img_infos

    def prepare_train_img(self, idx):
        info = self.img_infos[idx]
        results = dict(
            img_info=dict(filename=info[0]),
            ann_info=dict(bboxes=info[1], labels=info[2]),
        )
        if self.proposals is not None:
            results["proposals"] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        info = self.img_infos[idx]
        results = dict(img_info=dict(filename=info[0]))
        if self.proposals is not None:
            results["proposals"] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, outputs, metas, file):
        assert len(outputs) == len(metas)
        bbox_list = []
        for bboxes, meta in tqdm(zip(outputs, metas)):
            mt = meta[0].data[0][0]
            h, w = mt["ori_shape"][:2]
            filename = mt["filename"][:-4].split("/")[-1]
            valid_classes = np.where(
                np.array([[bbox.shape[0]] for bbox in bboxes]) != 0
            )[0]
            for valid_class in valid_classes:
                class_bboxes = bboxes[valid_class]
                class_bboxes[:, 0] /= w
                class_bboxes[:, 1] /= h
                class_bboxes[:, 2] /= w
                class_bboxes[:, 3] /= h
                bbox_num = class_bboxes.shape[0]
                for i in range(bbox_num):
                    box = [filename] + list(class_bboxes[i]) + [valid_class + 1]
                    bbox_list.append(box)

        def cmp(x, y):
            if (x[0] < y[0]) or (x[0] == y[0] and x[5] > y[5]):
                return -1
            else:
                return 1

        bbox_list = sorted(bbox_list, key=functools.cmp_to_key(cmp))

        f = open(file, "w+")
        for bbox in bbox_list:
            f.write("{}\n".format(" ".join(map(str, list(bbox)))))
            f.flush()

    def evaluate(self, label_dir, det_file):
        cat_file = os.path.join(label_dir, "cls-label-description.csv")
        categories = get_categories(cat_file)
        evaluator = OpenImagesDetectionChallengeEvaluator(
            categories, group_of_weight=1.0
        )

        gts = read_gts(label_dir)
        st = time.time()
        count = 0
        for im, gt in gts.items():
            evaluator.add_single_ground_truth_image_info(
                image_id=im, groundtruth_dict=gt
            )
        ed = time.time()
        print("\tGts added, using: {:.2f} s, flush=True".format(ed - st))

        dets = read_dets(det_file, label_dir)
        st = time.time()
        count = 0
        for im, det in dets.items():
            evaluator.add_single_detected_image_info(image_id=im, detections_dict=det)
            count += 1
            if (count + 1) % 1000 == 0:
                print(
                    "\t{}/{} done using {:.2f} s".format(
                        count + 1, len(dets), (time.time() - st)
                    ),
                    flush=True,
                )
        ed = time.time()
        print("\tDets evaluated, using: {:.2f} mins".format((ed - st) / 60), flush=True)

        print("\n\taccumulating...", flush=True)
        st = time.time()
        metrics = evaluator.evaluate()
        ed = time.time()
        print("\ttime used: {:.2f} s".format((ed - st)), flush=True)
        print(metrics)
        return metrics
