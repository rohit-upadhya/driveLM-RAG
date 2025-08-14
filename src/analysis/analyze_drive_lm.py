import os
import re
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DriveLMAnalysis:
    def __init__(
        self,
        train_dict: dict,
    ) -> None:
        self.train_dict = train_dict

    def process_analysis(
        self,
    ):
        all_question_count = {}
        per_scene_question_count = {}
        for k_1, v_1 in self.train_dict.items():
            per_scene_question_count[k_1] = {}
            key_frame = v_1.get("key_frames", {})
            for k_2, v_2 in key_frame.items():
                all_question_count[k_2] = {}
                questions = v_2.get("QA", {})
                for k_3, v_3 in questions.items():
                    if k_3 not in all_question_count[k_2]:
                        all_question_count[k_2][k_3] = len(v_3)
                    if k_3 not in per_scene_question_count[k_1]:
                        per_scene_question_count[k_1][k_3] = len(v_3)
                    else:
                        per_scene_question_count[k_1][k_3] += len(v_3)
        question_types = ["perception", "prediction", "planning", "behavior"]

        for item in question_types:
            for k_all, v_all in all_question_count.items():
                if item not in v_all:
                    print(f"No {item} type question in {k_all}")
                    v_all[item] = 0
            for k_scene, v_scene in per_scene_question_count.items():
                if item not in v_scene:
                    print(f"No {item} type question in {k_scene}")
                    v_scene[item] = 0
        return all_question_count, per_scene_question_count

    def _draw_label(
        self,
        draw,
        xy,
        text,
        font,
    ):
        x1, y1 = xy
        bbox = draw.textbbox((x1, y1), text, font=font)
        pad = 2
        bg = (0, 0, 0)
        fg = (255, 255, 255)
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=bg
        )
        draw.text((x1, y1), text, fill=fg, font=font)

    def _clamp_box(
        self,
        box,
        w,
        h,
    ):
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(0, min(int(round(x2)), w - 1))
        y2 = max(0, min(int(round(y2)), h - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    def _collect_boxes_by_camera(
        self,
        key_object_infos: dict,
        image_paths: dict,
    ):
        cam_images = {}
        for cam, path in image_paths.items():
            try:
                img = Image.open(path).convert("RGB")
                cam_images[cam] = {"image": img, "boxes": []}
            except Exception as e:
                continue

        font = ImageFont.load_default()
        for tag, info in key_object_infos.items():
            meta = self._parse_object_tag(tag=tag)
            if not meta:
                continue
            cam = meta["camera"]
            if cam not in cam_images:
                continue
            bbox = info.get("2d_bbox", None)
            if not bbox or len(bbox) != 4:
                continue

            category = info.get("Category", "") or ""
            status = info.get("Status", "") or ""
            desc = info.get("Visual_description", "") or ""
            label = f"{meta['id']} | {category}" + (f" ({status})" if status else "")

            canvas = cam_images[cam]["image"]
            draw = ImageDraw.Draw(canvas)
            w, h = canvas.size
            x1, y1, x2, y2 = self._clamp_box(bbox, w, h)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            self._draw_label(draw, (x1, max(0, y1 - 18)), label, font)

            cam_images[cam]["boxes"].append(
                {
                    "id": meta["id"],
                    "bbox": [x1, y1, x2, y2],
                    "category": category,
                    "status": status,
                    "desc": desc,
                }
            )

        return cam_images

    # def _detect_tag_in_string(
    #     self,
    #     query: str,
    # ) -> tuple[bool, list[str]]:
    #     pattern = r"<[^>]+>"
    #     matches = re.findall(pattern, query)
    #     parsed_tags = []
    #     for tag in matches:
    #         parsed = self._parse_object_tag(tag)
    #         if parsed:
    #             parsed_tags.append(parsed)
    #     return (len(parsed_tags) > 0, parsed_tags)

    # def _process_questions(
    #     self,
    #     qa: dict,
    #     images: dict,
    # ):
    #     qa_with_tags = []
    #     for question_type, questions in qa.items():
    #         for item in questions:
    #             tags_in_q = self._detect_tag_in_string(item["Q"])
    #             tags_in_a = self._detect_tag_in_string(item["A"])
    #             if tags_in_q[0]:
    #                 for tag in tags_in_q:
    #                 for item in images:
    #                     if images["id"] ==
    #             pass

    def visualize_sample(
        self,
        scene_id: str | None = None,
        key_frame_id: str | None = None,
    ) -> dict | None:
        if scene_id is None:
            scene_id = random.choice(list(self.train_dict.keys()))
        scene_ = self.train_dict.get(scene_id, {})
        key_frames = scene_.get("key_frames", {})

        if not key_frames:
            print(f"No key_frames for scene {scene_id}")
            return

        if key_frame_id is None or key_frame_id not in key_frames:
            key_frame_id = random.choice(list(key_frames.keys()))

        kf = key_frames[key_frame_id]
        key_object_infos = kf.get("key_object_infos", {})
        image_paths = kf.get("image_paths", {})
        qa = kf.get("QA", {})
        cam_images = self._collect_boxes_by_camera(key_object_infos, image_paths)

        return cam_images

    def _parse_object_tag(
        self,
        tag: str,
    ):
        tag = tag.strip("<>")
        parts = tag.split(",")
        if len(parts) < 4:
            return None
        return {
            "id": parts[0],
            "camera": parts[1],
            "cx": float(parts[2]),
            "cy": float(parts[3]),
        }

    def plot_list_distribution(
        self,
        values: list,
        title: str,
        min_: int,
        max_: int,
    ) -> None:
        if not min_:
            min_ = int(values.min())
        if not max_:
            max_ = int(values.max())
        if not values:
            print("Empty list provided. Skipping plot.")
            return

        values = np.array(values)
        mean_val = np.mean(values)
        median_val = np.median(values)
        plt.figure(figsize=(6, 4))
        plt.hist(
            values,
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
        plt.xlabel("Number of Questions")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_total(
        self,
        heading: str,
        lists_dict: dict[str, list],
    ):
        names = list(lists_dict.keys())
        totals = [sum(v) for v in lists_dict.values()]
        means = [np.mean(v) if v else 0 for v in lists_dict.values()]
        medians = [np.median(v) if v else 0 for v in lists_dict.values()]

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        axes[0].bar(names, totals, color="skyblue")
        axes[0].set_title("Total per Question Type")
        axes[0].set_ylabel("Total")
        axes[0].set_xticklabels(names, rotation=45, ha="right")

        axes[1].bar(names, means, color="lightgreen")
        axes[1].set_title("Mean per Question Type")
        axes[1].set_ylabel("Mean")
        axes[1].set_xticklabels(names, rotation=45, ha="right")

        axes[2].bar(names, medians, color="yellow")
        axes[2].set_title("Median per Question Type")
        axes[2].set_ylabel("Median")
        axes[2].set_xticklabels(names, rotation=45, ha="right")

        fig.suptitle(heading, fontsize=14, y=1.05)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def find_min_max_value(
        arrays: list[list],
    ):
        if not arrays or all(not arr for arr in arrays):
            return None, None

        flattened = [val for arr in arrays for val in arr if val is not None]

        if not flattened:
            return None, None

        return min(flattened), max(flattened)


if __name__ == "__main__":

    def read_json(
        file_name: str,
    ) -> dict:
        try:
            with open(file_name, "r") as file:
                data = json.load(file)
            return data
        except json.decoder.JSONDecodeError as e:
            print(f"Error {e} discovered. Issue Parsing json - {file_name}")
            raise

    train_file = "resources/data/drivelm/samples/v1_1_train_nus.json"
    drive_analysis = DriveLMAnalysis(train_dict=read_json(train_file))

    questions, per_scene_questions = drive_analysis.process_analysis()

    with open("resources/output/lm_analysis_qustion.json", "w+") as file:
        json.dump(questions, file, ensure_ascii=False, indent=4)
    with open("resources/output/per_scene_lm_analysis_qustion.json", "w+") as file:
        json.dump(per_scene_questions, file, ensure_ascii=False, indent=4)
