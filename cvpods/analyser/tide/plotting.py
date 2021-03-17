import os
from collections import OrderedDict
import pandas as pd
import seaborn as sns

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .datasets import get_tide_path
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


def print_table(rows: list, title: str = None):
    # Get all rows to have the same number of columns
    max_cols = max([len(row) for row in rows])
    for row in rows:
        while len(row) < max_cols:
            row.append("")

    # Compute the text width of each column
    col_widths = [
        max([len(rows[i][col_idx]) for i in range(len(rows))])
        for col_idx in range(len(rows[0]))
    ]

    divider = "--" + ("---".join(["-" * w for w in col_widths])) + "-"
    thick_divider = divider.replace("-", "=")

    if title:
        left_pad = (len(divider) - len(title)) // 2
        print(("{:>%ds}" % (left_pad + len(title))).format(title))

    print(thick_divider)
    for row in rows:
        # Print each row while padding to each column's text width
        print(
            "  "
            + "   ".join(
                [
                    ("{:>%ds}" % col_widths[col_idx]).format(row[col_idx])
                    for col_idx in range(len(row))
                ]
            )
            + "  "
        )
        if row == rows[0]:
            print(divider)
    print(thick_divider)


class Plotter:
    """ Sets up a seaborn environment and holds the functions for plotting our figures. """

    def __init__(self, quality: float = 1):
        # Set mpl DPI in case we want to output to the screen / notebook
        mpl.rcParams["figure.dpi"] = 150

        # Seaborn color palette
        sns.set_palette("muted", 10)
        current_palette = sns.color_palette()

        # Seaborn style
        sns.set(style="whitegrid")

        self.colors_main = OrderedDict(
            {
                ClassError.short_name: current_palette[9],
                BoxError.short_name: current_palette[8],
                OtherError.short_name: current_palette[2],
                DuplicateError.short_name: current_palette[6],
                BackgroundError.short_name: current_palette[4],
                MissedError.short_name: current_palette[3],
            }
        )

        self.colors_special = OrderedDict(
            {
                FalsePositiveError.short_name: current_palette[0],
                FalseNegativeError.short_name: current_palette[1],
            }
        )

        self.tide_path = get_tide_path()

        # For the purposes of comparing across models, we fix the scales on our bar plots.
        # Feel free to change these after initializing if you want to change the scale.
        self.MAX_MAIN_DELTA_AP = 10
        self.MAX_SPECIAL_DELTA_AP = 25

        self.quality = quality

    def _prepare_tmp_dir(self):
        tmp_dir = os.path.join(self.tide_path, "_tmp")

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        for _f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, _f))

        return tmp_dir

    def make_summary_plot(
        self,
        out_dir: str,
        errors: dict,
        model_name: str,
        rec_type: str,
        hbar_names: bool = False,
    ):
        """
        Make a summary plot of the errors for a model, and save it to the figs folder.

        :param out_dir:    The output directory for the summary image. MUST EXIST.
        :param errors:     Dictionary of both main and special errors.
        :param model_name: Name of the model for which to generate the plot.
        :param rec_type:   Recognition type, either TIDE.BOX or TIDE.MASK
        :param hbar_names: Whether or not to include labels for the horizontal bars.
        """

        tmp_dir = self._prepare_tmp_dir()

        high_dpi = int(500 * self.quality)
        low_dpi = int(300 * self.quality)

        # get the data frame
        error_dfs = {
            errtype: pd.DataFrame(
                data={
                    "Error Type": list(errors[errtype][model_name].keys()),
                    "Delta mAP": list(errors[errtype][model_name].values()),
                }
            )
            for errtype in ["main", "special"]
        }

        # pie plot for error type breakdown
        error_types = list(errors["main"][model_name].keys()) + list(
            errors["special"][model_name].keys()
        )
        error_sum = sum([e for e in errors["main"][model_name].values()])
        error_sizes = [e / error_sum for e in errors["main"][model_name].values()] + [
            0,
            0,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(11, 11), dpi=high_dpi)
        patches, outer_text, inner_text = ax.pie(
            error_sizes,
            colors=self.colors_main.values(),
            labels=error_types,
            autopct="%1.1f%%",
            startangle=90,
        )
        for text in outer_text + inner_text:
            text.set_text("")
        for i in range(len(self.colors_main)):
            if error_sizes[i] > 0.05:
                inner_text[i].set_text(list(self.colors_main.keys())[i])
            inner_text[i].set_fontsize(48)
            inner_text[i].set_fontweight("bold")
        ax.axis("equal")
        plt.title(model_name, fontdict={"fontsize": 60, "fontweight": "bold"})
        pie_path = os.path.join(tmp_dir, "{}_{}_pie.png".format(model_name, rec_type))
        plt.savefig(pie_path, bbox_inches="tight", dpi=low_dpi)
        plt.close()

        # horizontal bar plot for main error types
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=high_dpi)
        sns.barplot(
            data=error_dfs["main"],
            x="Delta mAP",
            y="Error Type",
            ax=ax,
            palette=self.colors_main.values(),
        )
        ax.set_xlim(0, self.MAX_MAIN_DELTA_AP)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if not hbar_names:
            ax.set_yticklabels([""] * 6)
        plt.setp(ax.get_xticklabels(), fontsize=28)
        plt.setp(ax.get_yticklabels(), fontsize=36)
        ax.grid(False)
        sns.despine(left=True, bottom=True, right=True)
        hbar_path = os.path.join(tmp_dir, "{}_{}_hbar.png".format(model_name, rec_type))
        plt.savefig(hbar_path, bbox_inches="tight", dpi=low_dpi)
        plt.close()

        # vertical bar plot for special error types
        fig, ax = plt.subplots(1, 1, figsize=(2, 5), dpi=high_dpi)
        sns.barplot(
            data=error_dfs["special"],
            x="Error Type",
            y="Delta mAP",
            ax=ax,
            palette=self.colors_special.values(),
        )
        ax.set_ylim(0, self.MAX_SPECIAL_DELTA_AP)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(["FP", "FN"])
        plt.setp(ax.get_xticklabels(), fontsize=36)
        plt.setp(ax.get_yticklabels(), fontsize=28)
        ax.grid(False)
        sns.despine(left=True, bottom=True, right=True)
        vbar_path = os.path.join(tmp_dir, "{}_{}_vbar.png".format(model_name, rec_type))
        plt.savefig(vbar_path, bbox_inches="tight", dpi=low_dpi)
        plt.close()

        # get each subplot image
        pie_im = cv2.imread(pie_path)
        hbar_im = cv2.imread(hbar_path)
        vbar_im = cv2.imread(vbar_path)

        # pad the hbar image vertically
        hbar_im = np.concatenate(
            [
                np.zeros((vbar_im.shape[0] - hbar_im.shape[0], hbar_im.shape[1], 3))
                + 255,
                hbar_im,
            ],
            axis=0,
        )
        summary_im = np.concatenate([hbar_im, vbar_im], axis=1)

        lpad, rpad = (
            int(np.ceil((pie_im.shape[1] - summary_im.shape[1]) / 2)),
            int(np.floor((pie_im.shape[1] - summary_im.shape[1]) / 2)),
        )
        summary_im = np.concatenate(
            [
                np.zeros((summary_im.shape[0], lpad, 3)) + 255,
                summary_im,
                np.zeros((summary_im.shape[0], rpad, 3)) + 255,
            ],
            axis=1,
        )
        summary_im = np.concatenate([pie_im, summary_im], axis=0)

        if out_dir is None:
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow((summary_im / 255)[:, :, (2, 1, 0)])
            plt.show()
            plt.close()
        else:
            cv2.imwrite(
                os.path.join(out_dir, "{}_{}_summary.png".format(model_name, rec_type)),
                summary_im,
            )
