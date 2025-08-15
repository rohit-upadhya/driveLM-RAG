import json
import yaml
import os


class FileLoader:
    def __init__(
        self,
    ):
        pass

    def load_file(self, file_name: str):
        _, file_extension = os.path.splitext(file_name)
        if "json" in file_extension:
            return self._load_json(file_name=file_name)
        elif "yaml" in file_extension:
            return self._load_yaml(file_name=file_name)
        else:
            print(
                "Unknown file format submitted. Please only submit yaml or json files."
            )
            raise

    def _load_json(
        self,
        file_name: str,
    ) -> dict:
        try:
            with open(file_name, "r") as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print(f"issue opening file {file_name}. Error : {e}")
            raise

    def _load_yaml(
        self,
        file_name: str,
    ):
        try:
            with open(file_name, "r") as file:
                data = yaml.safe_load(file)
                return data
        except yaml.YAMLError as e:
            print(f"issue opening file {file_name}. Error : {e}")
            raise
