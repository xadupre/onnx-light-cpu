# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure data-preparation helpers for :mod:`plot_backend_cases_benchmark`.

Kept in a separate module (instead of inline in the example) so the
row-to-plot-data transformation -- the part of the example responsible for
the published chart -- can be unit tested without onnx-light or ONNX
Runtime, which the rest of the example depends on.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

# Each row is (op_type, name, shapes, dtypes, light_time, ort_time, ort_error).
# ``ort_time`` is ``None`` when ONNX Runtime rejected or failed the case; such
# rows still carry ``ort_error`` for the textual/table output but have no
# comparable timing to plot.
_OP_TYPE = 0
_NAME = 1
_LIGHT_TIME = 4
_ORT_TIME = 5
_ORT_ERROR = 6


class NoPlottableCasesError(RuntimeError):
    """Raised when no collected row has a comparable ONNX Runtime timing.

    A chart with zero bars is not a useful (nor a correct) rendering of "no
    comparable case was found": it looks identical to "the build silently
    produced an empty page". Raising here instead turns that situation into a
    build failure with the offending case names/errors attached.
    """


@dataclass
class PlotData:
    """Everything the chart needs, derived once from the plottable rows."""

    plotted_rows: list
    labels: list
    speedups: np.ndarray
    colors: list
    colors_by_op_type: dict


def _short_label(name):
    """Strips the common ``test_cpu_*_benchmark`` case-name affixes."""
    label = name.removeprefix("test_cpu_")
    return label.removesuffix("_benchmark")


def prepare_plot_data(rows):
    """Turns collected benchmark ``rows`` into the values the chart plots.

    Rows whose ``ort_time`` is ``None`` are excluded from the returned plot
    data. Raises :class:`NoPlottableCasesError` -- naming every rejected case
    and its error -- when that leaves nothing to plot, rather than letting
    the caller render an empty chart.
    """
    plotted_rows = [row for row in rows if row[_ORT_TIME] is not None]
    if not plotted_rows:
        if rows:
            details = "; ".join(
                f"{row[_NAME]} ({row[_ORT_ERROR] or 'no error message'})" for row in rows
            )
        else:
            details = "no benchmark case was collected"
        raise NoPlottableCasesError(
            "no benchmark case produced a comparable ONNX Runtime timing; every "
            f"collected case was rejected or failed: {details}"
        )
    unique_op_types = sorted({row[_OP_TYPE] for row in plotted_rows})
    color_map = plt.get_cmap("turbo", len(unique_op_types))
    colors_by_op_type = {
        op_type: color_map(index) for index, op_type in enumerate(unique_op_types)
    }
    labels = [_short_label(row[_NAME]) for row in plotted_rows]
    speedups = np.array([row[_ORT_TIME] / row[_LIGHT_TIME] for row in plotted_rows])
    colors = [colors_by_op_type[row[_OP_TYPE]] for row in plotted_rows]
    return PlotData(
        plotted_rows=plotted_rows,
        labels=labels,
        speedups=speedups,
        colors=colors,
        colors_by_op_type=colors_by_op_type,
    )
