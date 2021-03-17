from collections import defaultdict


class Data:
    """
    A class to hold ground truth or predictions data in an easy to work with format.
    Note that any time they appear, bounding boxes are [x, y, width, height] and masks
    are either a list of polygons or pycocotools RLEs.

    Also, don't mix ground truth with predictions. Keep them in separate data objects.

    'max_dets' specifies the maximum number of detections the model
    is allowed to output for a given image.
    """

    def __init__(self, name: str, max_dets: int = 100):
        self.name = name
        self.max_dets = max_dets

        self.classes = {}  # Maps class ID to class name
        self.annotations = (
            []
        )  # Maps annotation ids to the corresponding annotation / prediction

        # Maps an image id to an image name and a list of annotation ids
        self.images = defaultdict(lambda: {"name": None, "anns": []})

    def _get_ignored_classes(self, image_id: int) -> set:
        anns = self.get(image_id)

        classes_in_image = set()
        ignored_classes = set()

        for ann in anns:
            if ann["ignore"]:
                if (
                    ann["class"] is not None
                    and ann["bbox"] is None
                    and ann["mask"] is None
                ):
                    ignored_classes.add(ann["class"])
            else:
                classes_in_image.add(ann["class"])

        return ignored_classes.difference(classes_in_image)

    def _make_default_class(self, id: int):
        """ (For internal use) Initializes a class id with a generated name. """

        if id not in self.classes:
            self.classes[id] = "Class " + str(id)

    def _make_default_image(self, id: int):
        if self.images[id]["name"] is None:
            self.images[id]["name"] = "Image " + str(id)

    def _prepare_box(self, box: object):
        return box

    def _prepare_mask(self, mask: object):
        return mask

    def _add(
        self,
        image_id: int,
        class_id: int,
        box: object = None,
        mask: object = None,
        score: float = 1,
        ignore: bool = False,
    ):
        """
        Add a data object to this collection. You should use one of the below functions instead.
        """
        self._make_default_class(class_id)
        self._make_default_image(image_id)
        new_id = len(self.annotations)

        self.annotations.append(
            {
                "_id": new_id,
                "score": score,
                "image": image_id,
                "class": class_id,
                "bbox": self._prepare_box(box),
                "mask": self._prepare_mask(mask),
                "ignore": ignore,
            }
        )

        self.images[image_id]["anns"].append(new_id)

    def add_ground_truth(
        self, image_id: int, class_id: int, box: object = None, mask: object = None
    ):
        """ Add a ground truth. If box or mask is None, this GT will be ignored for that mode. """
        self._add(image_id, class_id, box, mask)

    def add_detection(
        self,
        image_id: int,
        class_id: int,
        score: int,
        box: object = None,
        mask: object = None,
    ):
        """
        Add a predicted detection. If box or mask is None,
        this prediction will be ignored for that mode.
        """
        self._add(image_id, class_id, box, mask, score=score)

    def add_ignore_region(
        self,
        image_id: int,
        class_id: int = None,
        box: object = None,
        mask: object = None,
    ):
        """
        Add a region inside of which background detections should be ignored.
        You can use these to mark a region that has deliberately been left unannotated
        (e.g., if is a huge crowd of people and you
        don't want to annotate every single person in the crowd).

        If class_id is -1, this region will match any class.
        If the box / mask is None, the region will be the entire image.
        """
        self._add(image_id, class_id, box, mask, ignore=True)

    def add_class(self, id: int, name: str):
        """ Register a class name to that class ID. """
        self.classes[id] = name

    def add_image(self, id: int, name: str):
        """ Register an image name/path with an image ID. """
        self.images[id]["name"] = name

    def get(self, image_id: int):
        """ Collects all the annotations / detections for that particular image. """
        return [self.annotations[x] for x in self.images[image_id]["anns"]]
