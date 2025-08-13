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
    ) -> list[dict]:
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
        pass

    def _process_sample(
        self,
        samples: list,
        sample_data: list,
        ego_pose: list,
        category: list,
        sample_annotation: list,
        instance: list,
    ) -> list[dict]:
        sample_data_by_sample = {}
        sample_data_by_token = {}
        for sd in sample_data:
            if sd.get("fileformat") == "jpg":
                sample_data_by_sample[sd["sample_token"]] = sd
                sample_data_by_token[sd["token"]] = sd
        final_sample_data = []
        ego_by_token = {e["token"]: e for e in ego_pose}
        category_by_token = {c["token"]: c for c in category}
        instance_by_token = {i["token"]: i for i in instance}
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
            sample_token_point = sample_annotation_by_sample.get(sample["token"])
            objects = {}
            if sample_token_point:
                for sample_token_obj in sample_token_point:
                    instance_ = instance_by_token.get(
                        sample_token_obj["instance_token"], None
                    )
                    if instance_:
                        category_ = category_by_token.get(
                            instance_["category_token"], None
                        )
                        if category_:
                            category_name = category_.get("name")
                            if category_name not in objects:
                                objects[category_name] = 1
                            else:
                                objects[category_name] += 1
            speed = None
            prev_sd_token = sample_data_point.get("prev")
            if prev_sd_token:
                prev_sd = sample_data_by_token.get(prev_sd_token)
                if prev_sd:
                    ego_pose_prev = ego_by_token.get(prev_sd.get("ego_pose_token"))
                    if ego_pose_point and ego_pose_prev:
                        translation_curr = ego_pose_point.get(
                            "translation", translation
                        )
                        translation_prev = ego_pose_prev.get("translation", translation)
                        t_curr = ego_pose_point.get("timestamp")
                        t_prev = ego_pose_prev.get("timestamp")
                        if (
                            None not in translation_curr
                            and None not in translation_prev
                            and t_curr is not None
                            and t_prev is not None
                        ):
                            dt = (t_curr - t_prev) / 1e6
                            if dt > 0:
                                speed = (
                                    calculate_l_2(translation_curr, translation_prev)
                                    / dt
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
                "objects": objects,
            }

            final_sample_data.append(final_dict)
        return final_sample_data

    def _save_json_files(
        self,
        data: list | dict,
        file_name: str,
    ) -> None:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w+") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )

    def create_index(
        self,
    ) -> list:
        sample_list = self._read_json(
            file_name=self._process_file_name(file="sample"),
        )
        sample_data_list = self._read_json(
            file_name=self._process_file_name(file="sample_data"),
        )
        ego_pose_list = self._read_json(
            file_name=self._process_file_name(file="ego_pose"),
        )
        category_list = self._read_json(
            file_name=self._process_file_name(file="category"),
        )
        sample_annotation_list = self._read_json(
            file_name=self._process_file_name(file="sample_annotation"),
        )
        instance_list = self._read_json(
            file_name=self._process_file_name(file="instance"),
        )
        final_processed_samples = self._process_sample(
            samples=sample_list,
            sample_data=sample_data_list,
            ego_pose=ego_pose_list,
            category=category_list,
            sample_annotation=sample_annotation_list,
            instance=instance_list,
        )
        print(final_processed_samples[0])
        self._save_json_files(
            data=final_processed_samples,
            file_name="resources/output/nusciences_processed.json",
        )
        return final_processed_samples


if __name__ == "__main__":
    create_index_obj = CreateNuscenesIndex()
    create_index_obj.create_index()
    pass
