# pylint: disable=W0613

from .error import BestGTMatch, Error


class ClassError(Error):

    description = (
        "Error caused when a prediction would have been marked positive "
        + "if it had the correct class."
    )
    short_name = "Cls"

    def __init__(self, pred: dict, gt: dict, ex):
        self.pred = pred
        self.gt = gt

        self.match = BestGTMatch(pred, gt) if not self.gt["used"] else None

    def fix(self):
        if self.match is None:
            return None
        return self.gt["class"], self.match.fix()


class BoxError(Error):

    description = (
        "Error caused when a prediction would have "
        "been marked positive if it was localized better."
    )
    short_name = "Loc"

    def __init__(self, pred: dict, gt: dict, ex):
        self.pred = pred
        self.gt = gt

        self.match = BestGTMatch(pred, gt) if not self.gt["used"] else None

    def fix(self):
        if self.match is None:
            return None
        return self.pred["class"], self.match.fix()


class DuplicateError(Error):

    description = (
        "Error caused when a prediction would have been marked positive "
        + "if the GT wasn't already in use by another detection."
    )
    short_name = "Dupe"

    def __init__(self, pred: dict, suppressor: dict):
        self.pred = pred
        self.suppressor = suppressor

    def fix(self):
        return None


class BackgroundError(Error):

    description = (
        "Error caused when this detection should have been"
        "classified as background (IoU < 0.1)."
    )
    short_name = "Bkg"

    def __init__(self, pred: dict):
        self.pred = pred

    def fix(self):
        return None


class OtherError(Error):

    description = "This detection didn't fall into any of the other error categories."
    short_name = "Both"

    def __init__(self, pred: dict):
        self.pred = pred

    def fix(self):
        return None


class MissedError(Error):

    description = (
        "Represents GT missed by the model."
        "Doesn't include GT corrected elsewhere in the model."
    )
    short_name = "Miss"

    def __init__(self, gt: dict):
        self.gt = gt

    def fix(self):
        return self.gt["class"], -1


# These are special errors so no inheritence


class FalsePositiveError:

    description = (
        "Represents the potential AP gained by having perfect precision"
        + " (e.g., by scoring all false positives as conf=0) without affecting recall."
    )
    short_name = "FalsePos"

    @staticmethod
    def fix(score: float, correct: bool, info: dict) -> tuple:
        if correct:
            return 1, True, info
        else:
            return 0, False, info


class FalseNegativeError:

    description = (
        "Represents the potentially AP gained by having perfect recall"
        + " without affecting precision."
    )
    short_name = "FalseNeg"
