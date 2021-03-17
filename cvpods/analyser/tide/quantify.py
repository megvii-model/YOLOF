# pylint: disable=W0613

import os
from collections import OrderedDict, defaultdict

import numpy as np
from pycocotools import mask as mask_utils

from . import functions as f
from . import plotting as P
from .ap import ClassedAPDataObject
from .data import Data
from .errors.main_errors import (
    BackgroundError,
    BoxError,
    ClassError,
    DuplicateError,
    FalseNegativeError,
    FalsePositiveError,
    MissedError,
    OtherError
)
from .errors.qualifiers import Qualifier


class TIDEExample:
    """ Computes all the data needed to evaluate a set of predictions and gt for a single image. """

    def __init__(
        self,
        preds: list,
        gt: list,
        pos_thresh: float,
        mode: str,
        max_dets: int,
        run_errors: bool = True,
    ):
        self.preds = preds
        self.gt = [x for x in gt if not x["ignore"]]
        self.ignore_regions = [x for x in gt if x["ignore"]]

        self.mode = mode
        self.pos_thresh = pos_thresh
        self.max_dets = max_dets
        self.run_errors = run_errors

        self._run()

    def _run(self):
        preds = self.preds
        gt = self.gt
        ignore = self.ignore_regions
        det_type = "bbox" if self.mode == TIDE.BOX else "mask"
        max_dets = self.max_dets

        if len(preds) == 0:
            raise RuntimeError("Example has no predictions!")

        # Sort descending by score
        preds.sort(key=lambda pred: -pred["score"])
        preds = preds[:max_dets]

        # Update internally so TIDERun can update itself if :max_dets takes effect
        self.preds = preds
        detections = [x[det_type] for x in preds]

        # IoU is [len(detections), len(gt)]
        self.gt_iou = mask_utils.iou(
            detections, [x[det_type] for x in gt], [False] * len(gt)
        )

        # Store whether a prediction / gt got used in their data list
        # Note: this is set to None if ignored, keep that in mind
        for idx, pred in enumerate(preds):
            pred["used"] = False
            pred["_idx"] = idx
            pred["iou"] = 0
        for idx, truth in enumerate(gt):
            truth["used"] = False
            truth["usable"] = False
            truth["_idx"] = idx

        pred_cls = np.array([x["class"] for x in preds])
        gt_cls = np.array([x["class"] for x in gt])

        if len(gt) > 0:
            # A[i,j] is true iff the prediction i is of the same class as gt j
            self.gt_cls_matching = pred_cls[:, None] == gt_cls[None, :]
            self.gt_cls_iou = self.gt_iou * self.gt_cls_matching

            # This will be changed in the matching calculation, so make a copy
            iou_buffer = self.gt_cls_iou.copy()

            for pred_idx, pred_elem in enumerate(preds):
                # Find the max iou ground truth for this prediction
                gt_idx = np.argmax(iou_buffer[pred_idx, :])
                iou = iou_buffer[pred_idx, gt_idx]

                pred_elem["iou"] = np.max(self.gt_cls_iou[pred_idx, :])

                if iou >= self.pos_thresh:
                    gt_elem = gt[gt_idx]

                    pred_elem["used"] = True
                    gt_elem["used"] = True
                    pred_elem["matched_with"] = gt_elem["_id"]
                    gt_elem["matched_with"] = pred_elem["_id"]

                    # Make sure this gt can't be used again
                    iou_buffer[:, gt_idx] = 0

        # Ignore regions annotations allow us to ignore predictions that fall within
        if len(ignore) > 0:
            # Because ignore regions have extra parameters,
            # it's more efficient to use a for loop here
            for ignore_region in ignore:
                if ignore_region["mask"] is None and ignore_region["bbox"] is None:
                    # The region should span the whole image
                    ignore_iou = [1] * len(preds)
                else:
                    if ignore_region[det_type] is None:
                        # There is no det_type annotation for this specific region so skip it
                        continue
                    # Otherwise, compute the crowd IoU between the detections and this region
                    ignore_iou = mask_utils.iou(
                        detections, [ignore_region[det_type]], [True]
                    )

                for pred_idx, pred_elem in enumerate(preds):
                    if (
                        not pred_elem["used"]
                        and (ignore_iou[pred_idx] > self.pos_thresh)
                        and (
                            ignore_region["class"] == pred_elem["class"]
                            or ignore_region["class"] == -1
                        )
                    ):
                        # Set the prediction to be ignored
                        pred_elem["used"] = None

        if len(gt) == 0:
            return

        # Some matrices used just for error calculation
        if self.run_errors:
            self.gt_used = np.array([x["used"] is True for x in gt])[None, :]
            self.gt_unused = ~self.gt_used

            self.gt_unused_iou = self.gt_unused * self.gt_iou
            self.gt_unused_cls = self.gt_unused_iou * self.gt_cls_matching
            self.gt_unused_noncls = self.gt_unused_iou * ~self.gt_cls_matching

            self.gt_noncls_iou = self.gt_iou * ~self.gt_cls_matching

            self.gt_used_iou = self.gt_used * self.gt_iou
            self.gt_used_cls = self.gt_used_iou * self.gt_cls_matching


class TIDERun:
    """ Holds the data for a single run of TIDE. """

    # Temporary variables stored in ground truth that we need to clear after a run
    _temp_vars = ["best_score", "best_id", "used", "matched_with", "_idx", "usable"]

    def __init__(
        self,
        gt: Data,
        preds: Data,
        pos_thresh: float,
        bg_thresh: float,
        mode: str,
        max_dets: int,
        run_errors: bool = True,
    ):
        self.gt = gt
        self.preds = preds

        self.errors = []
        self.error_dict = {_type: [] for _type in TIDE._error_types}
        self.ap_data = ClassedAPDataObject()
        self.qualifiers = {}

        # A list of false negatives per class
        self.false_negatives = {_id: [] for _id in self.gt.classes}

        self.pos_thresh = pos_thresh
        self.bg_thresh = bg_thresh
        self.mode = mode
        self.max_dets = max_dets
        self.run_errors = run_errors

        self._run()

    def _run(self):
        """ And awaaay we go """

        for image in self.gt.images:
            x = self.preds.get(image)
            y = self.gt.get(image)

            # These classes are ignored for the whole image and not in the ground truth, so
            # we can safely just remove these detections from the predictions at the start.
            ignored_classes = self.gt._get_ignored_classes(image)
            x = [pred for pred in x if pred["class"] not in ignored_classes]

            self._eval_image(x, y)

        # Store a fixed version of all the errors for testing purposes
        for error in self.errors:
            error.original = f.nonepack(error.unfix())
            error.fixed = f.nonepack(error.fix())
            error.disabled = False

        self.ap = self.ap_data.get_mAP()

        # Now that we've stored the fixed errors, we can clear the gt info
        self._clear()

    def _clear(self):
        """ Clears the ground truth so that it's ready for another run. """
        for gt in self.gt.annotations:
            for var in self._temp_vars:
                if var in gt:
                    del gt[var]

    def _add_error(self, error):
        self.errors.append(error)
        self.error_dict[type(error)].append(error)

    def _eval_image(self, preds: list, gt: list):  # noqa

        for truth in gt:
            if not truth["ignore"]:
                self.ap_data.add_gt_positives(truth["class"], 1)

        if len(preds) == 0:
            # There are no predictions for this image so add all gt as missed
            for truth in gt:
                if not truth["ignore"]:
                    self.ap_data.push_false_negative(truth["class"], truth["_id"])

                    if self.run_errors:
                        self._add_error(MissedError(truth))
                        self.false_negatives[truth["class"]].append(truth)
            return

        ex = TIDEExample(
            preds, gt, self.pos_thresh, self.mode, self.max_dets, self.run_errors
        )
        preds = ex.preds  # In case the number of predictions was restricted to the max

        for pred_idx, pred in enumerate(preds):

            # None means that the prediction was ignored
            if pred["used"] is not None:
                pred["info"] = {"iou": pred["iou"]}
                if pred["used"]:
                    pred["info"]["matched_with"] = pred["matched_with"]
                self.ap_data.push(
                    pred["class"],
                    pred["_id"],
                    pred["score"],
                    pred["used"],
                    pred["info"],
                )

            # ----- ERROR DETECTION ------ #
            # This prediction is a negative, let's find out why
            if self.run_errors and pred["used"] is False:
                # Test for BackgroundError
                if (
                    len(ex.gt) == 0
                ):  # Note this is ex.gt because it doesn't include ignore annotations
                    # There is no ground truth for this image,
                    # so just mark everything as BackgroundError
                    self._add_error(BackgroundError(pred))
                    continue

                # Test for BoxError
                idx = ex.gt_cls_iou[pred_idx, :].argmax()
                if self.bg_thresh <= ex.gt_cls_iou[pred_idx, idx] <= self.pos_thresh:
                    # This detection would have been positive if it had higher IoU with this GT
                    self._add_error(BoxError(pred, ex.gt[idx], ex))
                    continue

                # Test for ClassError
                idx = ex.gt_noncls_iou[pred_idx, :].argmax()
                if ex.gt_noncls_iou[pred_idx, idx] >= self.pos_thresh:
                    # This detection would have been a positive if it was the correct class
                    self._add_error(ClassError(pred, ex.gt[idx], ex))
                    continue

                # Test for DuplicateError
                idx = ex.gt_used_cls[pred_idx, :].argmax()
                if ex.gt_used_cls[pred_idx, idx] >= self.pos_thresh:
                    # The detection would have been marked positive but the GT was already in use
                    suppressor = self.preds.annotations[ex.gt[idx]["matched_with"]]
                    self._add_error(DuplicateError(pred, suppressor))
                    continue

                # Test for BackgroundError
                idx = ex.gt_iou[pred_idx, :].argmax()
                if ex.gt_iou[pred_idx, idx] <= self.bg_thresh:
                    # This should have been marked as background
                    self._add_error(BackgroundError(pred))
                    continue

                # A base case to catch uncaught errors
                self._add_error(OtherError(pred))

        for truth in gt:
            # If the GT wasn't used in matching, meaning it's some kind of false negative
            if not truth["ignore"] and not truth["used"]:
                self.ap_data.push_false_negative(truth["class"], truth["_id"])

                if self.run_errors:
                    self.false_negatives[truth["class"]].append(truth)

                    # The GT was completely missed, no error can correct it
                    if not truth["usable"]:
                        self._add_error(MissedError(truth))

    def fix_errors(
        self,
        condition=lambda x: False,
        transform=None,
        false_neg_dict: dict = None,
        ap_data: ClassedAPDataObject = None,
        disable_errors: bool = False,
    ) -> ClassedAPDataObject:
        """
        Returns a ClassedAPDataObject where all errors
        given the condition returns True are fixed.
        """
        if ap_data is None:
            ap_data = self.ap_data

        gt_pos = ap_data.get_gt_positives()
        new_ap_data = ClassedAPDataObject()

        # Potentially fix every error case
        for error in self.errors:
            if error.disabled:
                continue

            _id = error.get_id()
            _cls, data_point = error.original

            if condition(error):
                _cls, data_point = error.fixed

                if disable_errors:
                    error.disabled = True

                # Specific for MissingError (or anything else that affects #GT)
                if isinstance(data_point, int):
                    gt_pos[_cls] += data_point
                    data_point = None

            if data_point is not None:
                if transform is not None:
                    data_point = transform(*data_point)
                new_ap_data.push(_cls, _id, *data_point)

        # Add back all the correct ones
        for k in gt_pos.keys():
            for _id, (score, correct, info) in ap_data.objs[k].data_points.items():
                if correct:
                    if transform is not None:
                        score, correct, info = transform(score, correct, info)
                    new_ap_data.push(k, _id, score, correct, info)

        # Add the correct amount of GT positives, and also subtract if necessary
        for k, v in gt_pos.items():
            # In case you want to fix all false negatives without affecting precision
            if false_neg_dict is not None and k in false_neg_dict:
                v -= len(false_neg_dict[k])
            new_ap_data.add_gt_positives(k, v)

        return new_ap_data

    def fix_main_errors(
        self,
        progressive: bool = False,
        error_types: list = None,
        qual: Qualifier = None,
    ) -> dict:
        ap_data = self.ap_data
        last_ap = self.ap

        if qual is None:
            qual = Qualifier("", None)

        if error_types is None:
            error_types = TIDE._error_types

        errors = {}

        for error in error_types:
            _ap_data = self.fix_errors(
                qual._make_error_func(error),
                ap_data=ap_data,
                disable_errors=progressive,
            )

            new_ap = _ap_data.get_mAP()
            # If an error is negative that means it's likely due to binning differences, so just
            # Ignore the negative by setting it to 0.
            errors[error] = max(new_ap - last_ap, 0)

            if progressive:
                last_ap = new_ap
                ap_data = _ap_data

        if progressive:
            for error in self.errors:
                error.disabled = False

        return errors

    def fix_special_errors(self, qual=None) -> dict:
        return {
            FalsePositiveError: self.fix_errors(
                transform=FalsePositiveError.fix
            ).get_mAP()
            - self.ap,
            FalseNegativeError: self.fix_errors(
                false_neg_dict=self.false_negatives
            ).get_mAP()
            - self.ap,
        }

    def count_errors(self, error_types: list = None, qual=None):
        counts = {}

        if error_types is None:
            error_types = TIDE._error_types

        for error in error_types:
            if qual is None:
                counts[error] = len(self.error_dict[error])
            else:
                func = qualifiers.make_qualifier(error, qual)  # noqa
                counts[error] = len([x for x in self.errors if func(x)])

        return counts

    def apply_qualifier(self, qualifier: Qualifier) -> ClassedAPDataObject:
        """
        Applies a qualifier lambda to the AP object for this
        runs and stores the result in self.qualifiers.
        """

        pred_keep = defaultdict(lambda: set())
        gt_keep = defaultdict(lambda: set())

        for pred in self.preds.annotations:
            if qualifier.test(pred):
                pred_keep[pred["class"]].add(pred["_id"])

        for gt in self.gt.annotations:
            if not gt["ignore"] and qualifier.test(gt):
                gt_keep[gt["class"]].add(gt["_id"])

        new_ap_data = self.ap_data.apply_qualifier(pred_keep, gt_keep)
        self.qualifiers[qualifier.name] = new_ap_data.get_mAP()
        return new_ap_data


class TIDE:

    # This is just here to define a consistent order of the error types
    _error_types = [
        ClassError,
        BoxError,
        OtherError,
        DuplicateError,
        BackgroundError,
        MissedError,
    ]
    _special_error_types = [FalsePositiveError, FalseNegativeError]

    # Threshold splits for different challenges
    COCO_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    VOL_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # The modes of evaluation
    BOX = "bbox"
    MASK = "mask"

    def __init__(
        self,
        pos_threshold: float = 0.5,
        background_threshold: float = 0.1,
        mode: str = BOX,
    ):
        self.pos_thresh = pos_threshold
        self.bg_thresh = background_threshold
        self.mode = mode

        self.pos_thresh_int = int(self.pos_thresh * 100)

        self.runs = {}
        self.run_thresholds = {}
        self.run_main_errors = {}
        self.run_special_errors = {}

        self.qualifiers = OrderedDict()

        self.plotter = P.Plotter()

    def evaluate(
        self,
        gt: Data,
        preds: Data,
        pos_threshold: float = None,
        background_threshold: float = None,
        mode: str = None,
        name: str = None,
        use_for_errors: bool = True,
    ) -> TIDERun:
        pos_thresh = self.pos_thresh if pos_threshold is None else pos_threshold
        bg_thresh = (
            self.bg_thresh if background_threshold is None else background_threshold
        )
        mode = self.mode if mode is None else mode
        name = preds.name if name is None else name

        run = TIDERun(
            gt, preds, pos_thresh, bg_thresh, mode, gt.max_dets, use_for_errors
        )

        if use_for_errors:
            self.runs[name] = run

        return run

    def evaluate_range(
        self,
        gt: Data,
        preds: Data,
        thresholds: list = COCO_THRESHOLDS,
        pos_threshold: float = None,
        background_threshold: float = None,
        mode: str = None,
        name: str = None,
    ) -> dict:

        if pos_threshold is None:
            pos_threshold = self.pos_thresh
        if name is None:
            name = preds.name

        self.run_thresholds[name] = []

        for thresh in thresholds:

            run = self.evaluate(
                gt,
                preds,
                pos_threshold=thresh,
                background_threshold=background_threshold,
                mode=mode,
                name=name,
                use_for_errors=(pos_threshold == thresh),
            )

            self.run_thresholds[name].append(run)

    def add_qualifiers(self, *quals):
        """
        Applies any number of Qualifier objects to evaluations that have been run up to now.
        See qualifiers.py for examples.
        """
        raise NotImplementedError("Qualifiers coming soon.")
        # for q in quals:
        # 	for run_name, run in self.runs.items():
        # 		if run_name in self.run_thresholds:
        # 			# If this was a threshold run, apply the qualifier for every run
        # 			for trun in self.run_thresholds[run_name]:
        # 				trun.apply_qualifier(q)
        # 		else:
        # 			# If this had no threshold, just apply it to the main run
        # 			run.apply_qualifier(q)

        # 	self.qualifiers[q.name] = q

    def summarize(self):
        """
        Summarizes the mAP values and errors for all runs in this TIDE object.
        Results are printed to the console.
        """
        main_errors = self.get_main_errors()
        special_errors = self.get_special_errors()

        for run_name, run in self.runs.items():
            print("-- {} --\n".format(run_name))

            # If we evaluated on all thresholds, print them here
            if run_name in self.run_thresholds:
                thresh_runs = self.run_thresholds[run_name]
                aps = [trun.ap for trun in thresh_runs]

                # Print Overall AP for a threshold run
                ap_title = "{} AP @ [{:d}-{:d}]".format(
                    thresh_runs[0].mode,
                    int(thresh_runs[0].pos_thresh * 100),
                    int(thresh_runs[-1].pos_thresh * 100),
                )
                print("{:s}: {:.2f}".format(ap_title, sum(aps) / len(aps)))

                # Print AP for every threshold on a threshold run
                P.print_table(
                    [
                        ["Thresh"]
                        + [str(int(trun.pos_thresh * 100)) for trun in thresh_runs],
                        ["  AP  "]
                        + ["{:6.2f}".format(trun.ap) for trun in thresh_runs],
                    ],
                    title=ap_title,
                )

                # Print qualifiers for a threshold run
                if len(self.qualifiers) > 0:
                    print()
                    # Can someone ban me from using list comprehension? this is unreadable
                    qAPs = [
                        f.mean(
                            [
                                trun.qualifiers[q]
                                for trun in thresh_runs
                                if q in trun.qualifiers
                            ]
                        )
                        for q in self.qualifiers
                    ]

                    P.print_table(
                        [
                            ["Name"] + list(self.qualifiers.keys()),
                            [" AP "] + ["{:6.2f}".format(qAP) for qAP in qAPs],
                        ],
                        title="Qualifiers {}".format(ap_title),
                    )

            # Otherwise, print just the one run we did
            else:
                # Print Overall AP for a regular run
                ap_title = "{} AP @ {:d}".format(run.mode, int(run.pos_thresh * 100))
                print("{}: {:.2f}".format(ap_title, run.ap))

                # Print qualifiers for a regular run
                if len(self.qualifiers) > 0:
                    print()
                    qAPs = [
                        run.qualifiers[q] if q in run.qualifiers else 0
                        for q in self.qualifiers
                    ]
                    P.print_table(
                        [
                            ["Name"] + list(self.qualifiers.keys()),
                            [" AP "] + ["{:6.2f}".format(qAP) for qAP in qAPs],
                        ],
                        title="Qualifiers {}".format(ap_title),
                    )

            print()
            # Print the main errors
            P.print_table(
                [
                    ["Type"] + [err.short_name for err in TIDE._error_types],
                    [" dAP"]
                    + [
                        "{:6.2f}".format(main_errors[run_name][err.short_name])
                        for err in TIDE._error_types
                    ],
                ],
                title="Main Errors",
            )

            print()
            # Print the special errors
            P.print_table(
                [
                    ["Type"] + [err.short_name for err in TIDE._special_error_types],
                    [" dAP"]
                    + [
                        "{:6.2f}".format(special_errors[run_name][err.short_name])
                        for err in TIDE._special_error_types
                    ],
                ],
                title="Special Error",
            )

            print()

    def plot(self, out_dir: str = None):
        """
        Plots a summary model for each run in this TIDE object.
        Images will be outputted to out_dir, which will be created if it doesn't exist.
        """

        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        errors = self.get_all_errors()

        for run_name, run in self.runs.items():
            self.plotter.make_summary_plot(
                out_dir, errors, run_name, run.mode, hbar_names=True
            )

    def get_main_errors(self):
        errors = {}

        for run_name, run in self.runs.items():
            if run_name in self.run_main_errors:
                errors[run_name] = self.run_main_errors[run_name]
            else:
                errors[run_name] = {
                    error.short_name: value
                    for error, value in run.fix_main_errors().items()
                }

        return errors

    def get_special_errors(self):
        errors = {}

        for run_name, run in self.runs.items():
            if run_name in self.run_special_errors:
                errors[run_name] = self.run_special_errors[run_name]
            else:
                errors[run_name] = {
                    error.short_name: value
                    for error, value in run.fix_special_errors().items()
                }

        return errors

    def get_all_errors(self):
        """
        returns {
                'main'   : { run_name: { error_name: float } },
                'special': { run_name: { error_name: float } },
        }
        """
        return {"main": self.get_main_errors(), "special": self.get_special_errors()}
