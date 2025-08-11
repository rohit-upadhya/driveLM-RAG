import json
import os
import math


class CreateNuscenesIndex:
    def __init__(
        self,
        root_folder: str = "resources/data/nuscenes-mini/v1.0-mini",
    ) -> None:
        self.root_folder = root_folder

    def _read_json(
        self,
        file_name: str,
    ) -> dict | list:
        try:
            with open(file_name, "r") as file:
                data = json.load(file)
            return data
        except json.decoder.JSONDecodeError as e:
            print(f"Error {e} discovered. Issue Parsing json - {file_name}")
            raise

    def _process_file_name(
        self,
        file: str,
    ) -> str:
        return os.path.join(self.root_folder, f"{file}.json")

    def _process_sample(
        self,
        samples: list,
        sample_data: list,
        ego_pose: list,
        category: list,
        sample_annotation: list,
    ) -> list[dict]:
        sample_data_by_sample = {}
        for sd in sample_data:
            if sd.get("fileformat") == "jpg":
                sample_data_by_sample[sd["sample_token"]] = sd
        final_sample_data = []
        ego_by_token = {e["token"]: e for e in ego_pose}
        sample_by_token = {s["token"]: s for s in samples}
        category_by_token = {c["token"]: c for c in category}
        # sample_annotation_by_token = {sa["token"]: sa for sa in sample_annotation}
        sample_annotation_by_sample = {}
        for item in sample_annotation:
            sample_token = item["sample_token"]
            if sample_token not in sample_annotation_by_sample:
                sample_annotation_by_sample[sample_token] = [item]
            else:
                sample_annotation_by_sample[sample_token].append(item)

        def quat_wxyz_to_xyzw(qwxyz):
            if not qwxyz or len(qwxyz) != 4:
                return [None, None, None, None]
            w, x, y, z = qwxyz
            return [x, y, z, w]

        def calculate_l_2(a, b):
            return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

        for sample in samples:
            sample_data_point = sample_data_by_sample.get(sample["token"], {})
            ego_pose_point = ego_by_token.get(sample_data_point["ego_pose_token"], None)

            translation = (
                ego_pose_point.get("translation", [None, None, None])
                if ego_pose_point
                else [None, None, None]
            )
            rotation_xyzw = (
                quat_wxyz_to_xyzw(ego_pose_point.get("rotation"))
                if ego_pose_point
                else [None, None, None, None]
            )
            speed = None
            prev_token = sample_data_point.get("prev", None)
            sample_token_point = sample_annotation_by_sample.get(sample["token"])
            objects = {}
            if sample_token_point:
                for sample_token_obj in sample_token_point:
                    pass
            if prev_token and prev_token in sample_by_token:
                previous_sample_data = sample_data_by_sample.get(prev_token, None)
                if previous_sample_data:
                    ego_pose_point_previous = ego_by_token.get(
                        previous_sample_data.get("ego_pose_token", None)
                    )
                    if ego_pose_point and ego_pose_point_previous:
                        translation_curr = ego_pose_point.get(
                            "translation", translation
                        )
                        translation_prev = ego_pose_point_previous.get(
                            "translation", translation
                        )
                        timestamp_curr = ego_pose_point.get("timestamp")
                        timestamp_prev = ego_pose_point_previous.get("timestamp")
                        if (
                            None not in translation_curr
                            and None not in translation_prev
                            and timestamp_curr
                            and timestamp_prev
                        ):
                            delta_time = (timestamp_curr - timestamp_prev) / 1e6
                            if delta_time > 0:
                                speed = (
                                    calculate_l_2(translation_curr, translation_prev)
                                    / delta_time
                                )

            final_dict = {
                "sample_token": sample.get("token", None),
                "scene_token": sample.get("scene_token", None),
                "timestamp_us": sample.get("timestamp", None),
                "neighbors": {
                    "prev": sample.get("prev", None),
                    "next": sample.get("next", None),
                },
                "image": {
                    "sensor_channel": sample_data_point["filename"].split("/")[1],
                    "rel_path": os.path.join(
                        self.root_folder, sample_data_point["filename"]
                    ),
                    "ego_pose_token": sample_data_point.get("ego_pose_token", None),
                },
                "ego": {
                    "translation": translation,
                    "rotation_xyzw": rotation_xyzw,
                    "speed_mps": speed,
                },
                "objects": {
                    "car": 4,
                    "truck": 1,
                    "bus": 0,
                    "motorcycle": 0,
                    "bicycle": 1,
                    "pedestrian": 3,
                    "cone": 2,
                    "barrier": 0,
                    "other_vehicle": 0,
                },
            }

            final_sample_data.append(final_dict)
        return final_sample_data

    def create_index(
        self,
    ):
        sample_list = self._read_json(
            file_name=self._process_file_name(file="sample"),
        )
        sample_data_list = self._read_json(
            file_name=self._process_file_name(file="sample_data"),
        )
        ego_pose_list = self._read_json(
            file_name=self._process_file_name(file="ego_pose"),
        )
        sensor_meta_list = self._read_json(
            file_name=self._process_file_name(file="sensor"),
        )
        category_list = self._read_json(
            file_name=self._process_file_name(file="category"),
        )
        scene_list = self._read_json(
            file_name=self._process_file_name(file="scene"),
        )


if __name__ == "__main__":
    create_index_obj = CreateNuscenesIndex()
    create_index_obj.create_index()
    pass
