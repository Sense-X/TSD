import collections
import logging
import os
import time
from abc import ABCMeta, abstractmethod

import numpy as np

from . import metrics, per_image_evaluation


def get_categories(file_name):
    categories = []
    with open(file_name) as f:
        lines = f.readlines()
    for ll in lines:
        vals = ll.strip().split(",")
        categories.append({"id": int(vals[2]), "name": vals[1]})
    return categories


def read_gts(label_dir, gt_need_father=True):
    file_name = os.path.join(label_dir, "challenge-2019-validation-detection-bbox.txt")
    print("\tGet gts: {0} ...".format(file_name))
    if gt_need_father:
        class_label_tree = np.load(
            os.path.join(label_dir, "class_label_tree.np"), allow_pickle=True
        )
    st = time.time()
    gts = {}
    with open(file_name) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        gt_boxes = []
        labels = []
        is_group = []

        # img_name = lines[i].strip() #.rsplit('/',1)[-1].split('.')[0]
        img_name = lines[i].strip()[:-4].split("/")[1]
        i += 1

        neg_clss = [int(x) for x in lines[i].split()]
        i += 1

        img_gt_size = int(lines[i])
        i += 1

        for j in range(img_gt_size):
            vals = lines[i + j].split()
            box = [float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])]
            label = int(vals[0])
            group = int(vals[5])
            if gt_need_father:
                label = np.where(class_label_tree[label])[0].tolist()
                box = [box for _ in label]
                group = [group for _ in label]
            else:
                box = [box]
                label = [label]
                group = [group]
            gt_boxes += box
            labels += label
            is_group += group
        i += img_gt_size

        gts[img_name] = {}
        gts[img_name]["groundtruth_boxes"] = np.array(gt_boxes, dtype=np.float32)
        gts[img_name]["groundtruth_classes"] = np.array(labels, dtype=np.int32)
        gts[img_name]["verified_labels"] = np.array(neg_clss, dtype=np.int32)
        gts[img_name]["groundtruth_group_of"] = np.array(is_group, dtype=np.bool)

    ed = time.time()
    print("\tGts read, using: {:.2f} s".format(ed - st))
    return gts


def read_dets(file_name, label_dir, det_need_father=True, wh_format=False):
    print("\tGet dets: {0} ...".format(file_name))
    if det_need_father:
        class_label_tree = np.load(
            os.path.join(label_dir, "class_label_tree.np"), allow_pickle=True
        )
    st = time.time()
    dets = {}
    with open(file_name) as f:
        lines = f.readlines()
    count = 0
    for ll in lines:
        vals = ll.split()
        img_name = vals[0]
        if img_name not in dets.keys():
            dets[img_name] = {}
            dets[img_name]["detection_boxes"] = []
            dets[img_name]["detection_scores"] = []
            dets[img_name]["detection_classes"] = []

        x1 = float(vals[1])
        y1 = float(vals[2])
        x2 = float(vals[3])
        y2 = float(vals[4])
        if wh_format:
            x2 += x1
            y2 += y1
        box = [x1, y1, x2, y2]
        det_cls = int(vals[6])
        score = float(vals[5])
        if det_need_father:
            det_cls = np.where(class_label_tree[det_cls])[0].tolist()
            box = [box for _ in det_cls]
            score = [score for _ in det_cls]
        else:
            box = [box]
            det_cls = [det_cls]
            score = [score]
        dets[img_name]["detection_boxes"] += box
        dets[img_name]["detection_scores"] += score
        dets[img_name]["detection_classes"] += det_cls

    for k in dets.keys():
        dets[k]["detection_boxes"] = np.array(
            dets[k]["detection_boxes"], dtype=np.float32
        )
        dets[k]["detection_scores"] = np.array(
            dets[k]["detection_scores"], dtype=np.float32
        )
        dets[k]["detection_classes"] = np.array(
            dets[k]["detection_classes"], dtype=np.int32
        )

    ed = time.time()
    print("\tDets read, using: {:.2f} s".format(ed - st))
    return dets


############################################


############################################
class OpenImagesDetectionChallengeEvaluator(object):

    __metaclass__ = ABCMeta

    def __init__(
        self,
        categories,
        matching_iou_threshold=0.5,
        evaluate_corlocs=False,
        metric_prefix=None,
        use_weighted_mean_ap=False,
        evaluate_masks=False,
        group_of_weight=0.0,
    ):
        """Constructor.

        Args:
          categories: A list of dicts, each of which has the following keys -
            'id': (required) an integer id uniquely identifying this category.
            'name': (required) string representing category name e.g., 'cat', 'dog'.
          matching_iou_threshold: IOU threshold to use for matching groundtruth
            boxes to detection boxes.
          evaluate_corlocs: (optional) boolean which determines if corloc scores
            are to be returned or not.
          metric_prefix: (optional) string prefix for metric name; if None, no
            prefix is used.
          use_weighted_mean_ap: (optional) boolean which determines if the mean
            average precision is computed directly from the scores and tp_fp_labels
            of all classes.
          evaluate_masks: If False, evaluation will be performed based on boxes.
            If True, mask evaluation will be performed instead.
          group_of_weight: Weight of group-of boxes.If set to 0, detections of the
            correct class within a group-of box are ignored. If weight is > 0, then
            if at least one detection falls within a group-of box with
            matching_iou_threshold, weight group_of_weight is added to true
            positives. Consequently, if no detection falls within a group-of box,
            weight group_of_weight is added to false negatives.

        Raises:
          ValueError: If the category ids are not 1-indexed.
        """
        self._categories = categories
        self._num_classes = max([cat["id"] for cat in categories])
        if min(cat["id"] for cat in categories) < 1:
            raise ValueError("Classes should be 1-indexed.")
        self._matching_iou_threshold = matching_iou_threshold
        self._use_weighted_mean_ap = use_weighted_mean_ap
        self._label_id_offset = 1
        self._evaluate_masks = evaluate_masks
        self._group_of_weight = group_of_weight
        self._evaluation = ObjectDetectionEvaluation(
            num_groundtruth_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset,
            group_of_weight=self._group_of_weight,
        )
        self._image_ids = set([])
        self._evaluate_corlocs = evaluate_corlocs
        self._metric_prefix = (metric_prefix + "_") if metric_prefix else ""

        self._evaluatable_labels = {}

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.

        Args:
          image_id: A unique string/integer identifier for the image.
          groundtruth_dict: A dictionary containing -
            groundtruth_boxes: float32 numpy array
              of shape [num_boxes, 4] containing `num_boxes` groundtruth boxes of
              the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
            groundtruth_classes: integer numpy array
              of shape [num_boxes] containing 1-indexed groundtruth classes for the
              boxes.
            verified_labels: integer 1D numpy array
              containing all classes for which labels are verified.
            groundtruth_group_of: Optional length
              M numpy boolean array denoting whether a groundtruth box contains a
              group of instances.

        Raises:
          ValueError: On adding groundtruth for an image more than once.
        """
        if image_id in self._image_ids:
            raise ValueError("Image with id {} already added.".format(image_id))
        groundtruth_classes = (
            groundtruth_dict["groundtruth_classes"] - self._label_id_offset
        )
        # If the key is not present in the groundtruth_dict or the array is empty
        # (unless there are no annotations for the groundtruth on this image)
        # use values from the dictionary or insert None otherwise.
        # groundtruth_group_of = groundtruth_dict["groundtruth_group_of"]
        if "groundtruth_group_of" in groundtruth_dict.keys() and (
            groundtruth_dict["groundtruth_group_of"].size
            or not groundtruth_classes.size
        ):
            groundtruth_group_of = groundtruth_dict["groundtruth_group_of"]
        else:
            groundtruth_group_of = None
            if not len(self._image_ids) % 1000:
                logging.warn(
                    "image %s does not have groundtruth group_of flag specified",
                    image_id,
                )

        self._evaluation.add_single_ground_truth_image_info(
            image_id,
            groundtruth_dict["groundtruth_boxes"],
            groundtruth_classes,
            groundtruth_is_difficult_list=None,
            groundtruth_is_group_of_list=groundtruth_group_of,
        )

        self._image_ids.update([image_id])

        groundtruth_classes = (
            groundtruth_dict["groundtruth_classes"] - self._label_id_offset
        )
        # self._evaluatable_labels[image_id] = np.unique(
        #     np.concatenate(((groundtruth_dict.get(
        #         "verified_labels",
        #         np.array([], dtype=int)) - self._label_id_offset),
        #         groundtruth_classes)))
        self._evaluatable_labels[image_id] = np.unique(
            np.concatenate(
                (
                    (np.array([1], dtype=int) - self._label_id_offset),
                    groundtruth_classes,
                )
            )
        )

    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.

        Args:
          image_id: A unique string/integer identifier for the image.
          detections_dict: A dictionary containing -
            detection_boxes: float32 numpy
              array of shape [num_boxes, 4] containing `num_boxes` detection boxes
              of the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
            detection_scores: float32 numpy
              array of shape [num_boxes] containing detection scores for the boxes.
            detection_classes: integer numpy
              array of shape [num_boxes] containing 1-indexed detection classes for
              the boxes.

        Raises:
          ValueError: If detection masks are not in detections dictionary.
        """
        if image_id not in self._image_ids:
            # Since for the correct work of evaluator it is assumed that groundtruth
            # is inserted first we make sure to break the code if is it not the
            # case.
            self._image_ids.update([image_id])
            self._evaluatable_labels[image_id] = np.array([])

        detection_classes = detections_dict["detection_classes"] - self._label_id_offset
        allowed_classes = np.where(
            np.isin(detection_classes, self._evaluatable_labels[image_id])
        )
        detection_classes = detection_classes[allowed_classes]
        detected_boxes = detections_dict["detection_boxes"][allowed_classes]
        detected_scores = detections_dict["detection_scores"][allowed_classes]

        self._evaluation.add_single_detected_image_info(
            image_key=image_id,
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            detected_class_labels=detection_classes,
        )

    def evaluate(self):
        """Compute evaluation result.

        Returns:
          A dictionary of metrics with the following fields -

          1. summary_metrics:
            'Precision/mAP@<matching_iou_threshold>IOU': mean average precision at
            the specified IOU threshold.

          2. per_category_ap: category specific results with keys of the form
            'PerformanceByCategory/mAP@<matching_iou_threshold>IOU/category'.
        """
        # (per_class_ap, mean_ap, _, _, per_class_corloc, mean_corloc) = (
        #     self._evaluation.evaluate())
        (per_class_ap, mean_ap, precision, recall) = self._evaluation.evaluate()
        pascal_metrics = {
            self._metric_prefix
            + "Precision/mAP@{}IOU".format(self._matching_iou_threshold): mean_ap
        }
        mean_recall = np.nanmean([np.nanmean(x) for x in recall])
        name = self._metric_prefix + "Mean Recall@{}IOU".format(
            self._matching_iou_threshold
        )
        pascal_metrics[name] = mean_recall

        # if self._evaluate_corlocs:
        #     pascal_metrics[self._metric_prefix + 'Precision/meanCorLoc@{}IOU'.format(
        #         self._matching_iou_threshold)] = mean_corloc
        category_index = create_category_index(self._categories)
        for idx in range(per_class_ap.size):
            if idx + self._label_id_offset in category_index:
                display_name = (
                    # self._metric_prefix + 'PerformanceByCategory/AP@{}IOU/{}'.format(
                    self._metric_prefix
                    + "AP@{}IOU/{}_{}".format(
                        self._matching_iou_threshold,
                        category_index[idx + self._label_id_offset]["id"],
                        category_index[idx + self._label_id_offset]["name"],
                    )
                )
                pascal_metrics[display_name] = per_class_ap[idx]

                # Optionally add CorLoc metrics.classes
                # if self._evaluate_corlocs:
                #     display_name = (
                #         self._metric_prefix + 'PerformanceByCategory/CorLoc@{}IOU/{}'
                #         .format(self._matching_iou_threshold,
                #                 category_index[idx + self._label_id_offset]['name']))
                #     pascal_metrics[display_name] = per_class_corloc[idx]

        return pascal_metrics

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._evaluation = ObjectDetectionEvaluation(
            num_groundtruth_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset,
        )
        self._image_ids.clear()


# ObjectDetectionEvalMetrics = collections.namedtuple(
#     'ObjectDetectionEvalMetrics', [
#         'average_precisions', 'mean_ap', 'precisions', 'recalls', 'corlocs',
#         'mean_corloc'
#     ])

ObjectDetectionEvalMetrics = collections.namedtuple(
    "ObjectDetectionEvalMetrics",
    ["average_precisions", "mean_ap", "precisions", "recalls"],
)


class ObjectDetectionEvaluation(object):
    """Internal implementation of Pascal object detection metrics."""

    def __init__(
        self,
        num_groundtruth_classes,
        matching_iou_threshold=0.5,
        nms_iou_threshold=1.0,
        nms_max_output_boxes=10000,
        use_weighted_mean_ap=False,
        label_id_offset=0,
        group_of_weight=0.0,
    ):
        if num_groundtruth_classes < 1:
            raise ValueError("Need at least 1 groundtruth class for evaluation.")

        self.per_image_eval = per_image_evaluation.PerImageEvaluation(
            num_groundtruth_classes=num_groundtruth_classes,
            matching_iou_threshold=matching_iou_threshold,
            nms_iou_threshold=nms_iou_threshold,
            nms_max_output_boxes=nms_max_output_boxes,
            group_of_weight=group_of_weight,
        )
        self.group_of_weight = group_of_weight
        self.num_class = num_groundtruth_classes
        self.use_weighted_mean_ap = use_weighted_mean_ap
        self.label_id_offset = label_id_offset

        self.groundtruth_boxes = {}
        self.groundtruth_class_labels = {}
        self.groundtruth_masks = {}
        self.groundtruth_is_difficult_list = {}
        self.groundtruth_is_group_of_list = {}
        # self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=float)
        self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=np.int32)
        self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=np.int32)

        self._initialize_detections()

    def _initialize_detections(self):
        self.detection_keys = set()
        self.scores_per_class = [[] for _ in range(self.num_class)]
        self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
        self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
        self.average_precision_per_class = np.empty(self.num_class, dtype=float)
        self.average_precision_per_class.fill(np.nan)
        self.precisions_per_class = []
        self.recalls_per_class = []
        self.corloc_per_class = np.ones(self.num_class, dtype=float)

    def clear_detections(self):
        self._initialize_detections()

    def add_single_ground_truth_image_info(
        self,
        image_key,
        groundtruth_boxes,
        groundtruth_class_labels,
        groundtruth_is_difficult_list=None,
        groundtruth_is_group_of_list=None,
        groundtruth_masks=None,
    ):
        """Adds groundtruth for a single image to be used for evaluation.

        Args:
          image_key: A unique string/integer identifier for the image.
          groundtruth_boxes: float32 numpy array of shape [num_boxes, 4]
            containing `num_boxes` groundtruth boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
          groundtruth_class_labels: integer numpy array of shape [num_boxes]
            containing 0-indexed groundtruth classes for the boxes.
          groundtruth_is_difficult_list: A length M numpy boolean array denoting
            whether a ground truth box is a difficult instance or not. To support
            the case that no boxes are difficult, it is by default set as None.
          groundtruth_is_group_of_list: A length M numpy boolean array denoting
              whether a ground truth box is a group-of box or not. To support
              the case that no boxes are groups-of, it is by default set as None.
          groundtruth_masks: uint8 numpy array of shape
            [num_boxes, height, width] containing `num_boxes` groundtruth masks.
            The mask values range from 0 to 1.
        """
        if image_key in self.groundtruth_boxes:
            logging.warn(
                "image %s has already been added to the ground truth database.",
                image_key,
            )
            return

        self.groundtruth_boxes[image_key] = groundtruth_boxes
        self.groundtruth_class_labels[image_key] = groundtruth_class_labels
        self.groundtruth_masks[image_key] = groundtruth_masks
        if groundtruth_is_difficult_list is None:
            num_boxes = groundtruth_boxes.shape[0]
            groundtruth_is_difficult_list = np.zeros(num_boxes, dtype=bool)
        self.groundtruth_is_difficult_list[
            image_key
        ] = groundtruth_is_difficult_list.astype(dtype=bool)
        if groundtruth_is_group_of_list is None:
            num_boxes = groundtruth_boxes.shape[0]
            groundtruth_is_group_of_list = np.zeros(num_boxes, dtype=bool)
        self.groundtruth_is_group_of_list[
            image_key
        ] = groundtruth_is_group_of_list.astype(dtype=bool)

        self._update_ground_truth_statistics(
            groundtruth_class_labels,
            groundtruth_is_difficult_list.astype(dtype=bool),
            groundtruth_is_group_of_list.astype(dtype=bool),
        )

    def add_single_detected_image_info(
        self,
        image_key,
        detected_boxes,
        detected_scores,
        detected_class_labels,
        detected_masks=None,
    ):
        """Adds detections for a single image to be used for evaluation.

        Args:
          image_key: A unique string/integer identifier for the image.
          detected_boxes: float32 numpy array of shape [num_boxes, 4]
            containing `num_boxes` detection boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
          detected_scores: float32 numpy array of shape [num_boxes] containing
            detection scores for the boxes.
          detected_class_labels: integer numpy array of shape [num_boxes] containing
            0-indexed detection classes for the boxes.
          detected_masks: np.uint8 numpy array of shape [num_boxes, height, width]
            containing `num_boxes` detection masks with values ranging
            between 0 and 1.

        Raises:
          ValueError: if the number of boxes, scores and class labels differ in
            length.
        """
        if len(detected_boxes) != len(detected_scores) or len(detected_boxes) != len(
            detected_class_labels
        ):
            raise ValueError(
                "detected_boxes, detected_scores and "
                "detected_class_labels should all have same lengths. Got"
                "[%d, %d, %d]" % len(detected_boxes),
                len(detected_scores),
                len(detected_class_labels),
            )

        if image_key in self.detection_keys:
            logging.warn(
                "image %s has already been added to the detection result database",
                image_key,
            )
            return

        self.detection_keys.add(image_key)
        if image_key in self.groundtruth_boxes:
            groundtruth_boxes = self.groundtruth_boxes[image_key]
            groundtruth_class_labels = self.groundtruth_class_labels[image_key]
            # Masks are popped instead of look up. The reason is that we do not want
            # to keep all masks in memory which can cause memory overflow.
            groundtruth_masks = self.groundtruth_masks.pop(image_key)
            groundtruth_is_difficult_list = self.groundtruth_is_difficult_list[
                image_key
            ]
            groundtruth_is_group_of_list = self.groundtruth_is_group_of_list[image_key]
        else:
            import pdb

            pdb.set_trace()  # BREAKPOINT
            groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
            groundtruth_class_labels = np.array([], dtype=int)
            if detected_masks is None:
                groundtruth_masks = None
            else:
                groundtruth_masks = np.empty(shape=[0, 1, 1], dtype=float)
            groundtruth_is_difficult_list = np.array([], dtype=bool)
            groundtruth_is_group_of_list = np.array([], dtype=bool)
        scores, tp_fp_labels, is_class_correctly_detected_in_image = self.per_image_eval.compute_object_detection_metrics(
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            detected_class_labels=detected_class_labels,
            groundtruth_boxes=groundtruth_boxes,
            groundtruth_class_labels=groundtruth_class_labels,
            groundtruth_is_difficult_list=groundtruth_is_difficult_list,
            groundtruth_is_group_of_list=groundtruth_is_group_of_list,
            detected_masks=detected_masks,
            groundtruth_masks=groundtruth_masks,
        )
        # ipdb.set_trace()
        for i in range(self.num_class):
            if scores[i].shape[0] > 0:
                self.scores_per_class[i].append(scores[i])
                self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
        (
            self.num_images_correctly_detected_per_class
        ) += is_class_correctly_detected_in_image

    def _update_ground_truth_statistics(
        self,
        groundtruth_class_labels,
        groundtruth_is_difficult_list,
        groundtruth_is_group_of_list,
    ):
        """Update grouth truth statitistics.

        1. Difficult boxes are ignored when counting the number of ground truth
        instances as done in Pascal VOC devkit.
        2. Difficult boxes are treated as normal boxes when computing CorLoc related
        statitistics.

        Args:
          groundtruth_class_labels: An integer numpy array of length M,
              representing M class labels of object instances in ground truth
          groundtruth_is_difficult_list: A boolean numpy array of length M denoting
              whether a ground truth box is a difficult instance or not
          groundtruth_is_group_of_list: A boolean numpy array of length M denoting
              whether a ground truth box is a group-of box or not
        """
        # for class_index in range(self.num_class):
        #     num_gt_instances = np.sum(groundtruth_class_labels[
        #         ~groundtruth_is_difficult_list
        #         & ~groundtruth_is_group_of_list] == class_index)
        #     num_groupof_gt_instances = self.group_of_weight * np.sum(
        #         groundtruth_class_labels[groundtruth_is_group_of_list] == class_index)
        #     self.num_gt_instances_per_class[
        #         class_index] += num_gt_instances + num_groupof_gt_instances
        #     if np.any(groundtruth_class_labels == class_index):
        #         self.num_gt_imgs_per_class[class_index] += 1

        for ind in groundtruth_class_labels:
            self.num_gt_instances_per_class[ind] += 1

    def evaluate(self):
        """Compute evaluation result.

        Returns:
          A named tuple with the following fields -
            average_precision: float numpy array of average precision for
                each class.
            mean_ap: mean average precision of all classes, float scalar
            precisions: List of precisions, each precision is a float numpy
                array
            recalls: List of recalls, each recall is a float numpy array
            corloc: numpy float array
            mean_corloc: Mean CorLoc score for each class, float scalar
        """
        if (self.num_gt_instances_per_class == 0).any():
            logging.warn(
                "The following classes have no ground truth examples: %s",
                np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0))
                + self.label_id_offset,
            )

        if self.use_weighted_mean_ap:
            all_scores = np.array([], dtype=float)
            all_tp_fp_labels = np.array([], dtype=bool)
        for class_index in range(self.num_class):
            if self.num_gt_instances_per_class[class_index] == 0:
                continue
            if not self.scores_per_class[class_index]:
                scores = np.array([], dtype=float)
                tp_fp_labels = np.array([], dtype=float)
            else:
                scores = np.concatenate(self.scores_per_class[class_index])
                tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
            if self.use_weighted_mean_ap:
                all_scores = np.append(all_scores, scores)
                all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
            precision, recall = metrics.compute_precision_recall(
                scores, tp_fp_labels, self.num_gt_instances_per_class[class_index]
            )
            self.precisions_per_class.append(precision)
            self.recalls_per_class.append(recall)
            average_precision = metrics.compute_average_precision(precision, recall)
            self.average_precision_per_class[class_index] = average_precision

        # self.corloc_per_class = metrics.compute_cor_loc(
        #     self.num_gt_imgs_per_class,
        #     self.num_images_correctly_detected_per_class)

        if self.use_weighted_mean_ap:
            num_gt_instances = np.sum(self.num_gt_instances_per_class)
            precision, recall = metrics.compute_precision_recall(
                all_scores, all_tp_fp_labels, num_gt_instances
            )
            mean_ap = metrics.compute_average_precision(precision, recall)
        else:
            mean_ap = np.nanmean(self.average_precision_per_class)
        # mean_corloc = np.nanmean(self.corloc_per_class)
        # return ObjectDetectionEvalMetrics(
        #     self.average_precision_per_class, mean_ap, self.precisions_per_class,
        #     self.recalls_per_class, self.corloc_per_class, mean_corloc)
        return ObjectDetectionEvalMetrics(
            self.average_precision_per_class,
            mean_ap,
            self.precisions_per_class,
            self.recalls_per_class,
        )


def create_category_index(categories):
    """Creates dictionary of COCO compatible categories keyed by category id.

    Args:
      categories: a list of dicts, each of which has the following keys:
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name
          e.g., 'cat', 'dog', 'pizza'.

    Returns:
      category_index: a dict containing the same entries as categories, but keyed
        by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat["id"]] = cat
    return category_index
