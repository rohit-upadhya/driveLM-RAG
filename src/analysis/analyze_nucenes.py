import os
import re
import json
import random
import matplotlib.pyplot as plt
import numpy as np


class AnalyzeNuscenes:
    def __init__(
        self,
        nuscenes_dataset: list,
    ) -> None:
        self.nuscenes_dataset = nuscenes_dataset
        pass

    def _collect_speed(
        self,
    ):
        speed_list = []
        for item in self.nuscenes_dataset:
            speed = item.get("ego", {}).get("speed_mps", None)
            if speed:
                speed_list.append(speed)
        return speed_list

    def _collect_objects(
        self,
    ):
        object_values = {}
        for item in self.nuscenes_dataset:
            objects = item.get("objects", {})
            for key, value in objects.items():
                if key not in object_values:
                    object_values[key] = [value]
                else:
                    object_values[key].append(value)
        return object_values

    def plot_speed(
        self,
        title: str = "Speed Distribution",
    ):
        speed_list = self._collect_speed()
        speed_list = np.array(speed_list)
        min_ = int(speed_list.min())
        max_ = int(speed_list.max())
        mean_val = np.mean(speed_list)
        median_val = np.median(speed_list)

        plt.figure(figsize=(6, 4))

        plt.hist(
            speed_list,
            bins=range(min_ - 5, max_ + 5),
            alpha=0.7,
        )

        plt.axvline(
            mean_val,
            color="red",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {mean_val:.2f}",
        )
        plt.axvline(
            median_val,
            color="green",
            linestyle="dashed",
            linewidth=1,
            label=f"Median: {median_val:.2f}",
        )

        plt.title(title)
        plt.xlabel("Speed Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def plot_object_distributions(
        self,
        object_values: dict | None = None,
        *,
        title_prefix: str = "Distribution — ",
        xlabel: str = "Count per frame",
        show_empty: bool = False,
    ):
        if object_values is None:
            object_values = self._collect_objects()

        for cls, vals in object_values.items():
            arr = np.array(vals, dtype=float)
            if arr.size == 0 or (np.isnan(arr).all() and not show_empty):
                continue

            min_ = int(arr.min())
            max_ = int(arr.max())
            mean_val = float(np.nanmean(arr)) if arr.size else float("nan")
            median_val = float(np.nanmedian(arr)) if arr.size else float("nan")

            plt.figure(figsize=(6, 4))
            plt.hist(arr[~np.isnan(arr)], bins=range(min_ - 3, max_ + 3), alpha=0.7)
            if not np.isnan(mean_val):
                plt.axvline(
                    mean_val,
                    color="red",
                    linestyle="dashed",
                    linewidth=1,
                    label=f"Mean: {mean_val:.2f}",
                )
            if not np.isnan(median_val):
                plt.axvline(
                    median_val,
                    color="green",
                    linestyle="dashed",
                    linewidth=1,
                    label=f"Median: {median_val:.2f}",
                )

            plt.title(f"{title_prefix}{cls}")
            plt.xlabel(xlabel)
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()
            plt.show()
