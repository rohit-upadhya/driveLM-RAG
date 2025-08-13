import json
import matplotlib.pyplot as plt
import numpy as np


class DriveLMAnalysis:
    def __init__(
        self,
        train_dict: dict,
    ) -> None:
        self.train_dict = train_dict

    def process_analysis(
        self,
    ):
        question_count = {}
        for k_1, v_1 in self.train_dict.items():
            key_frame = v_1.get("key_frames", {})
            for k_2, v_2 in key_frame.items():
                question_count[k_2] = {}
                questions = v_2.get("QA", {})
                for k_3, v_3 in questions.items():
                    if k_3 not in question_count[k_2]:
                        question_count[k_2][k_3] = len(v_3)

        return question_count

    def plot_list_distribution(
        self,
        values: list,
        title: str,
    ) -> tuple:
        if not values:
            print("Empty list provided. Skipping plot.")
            return None, None

        values = np.array(values)
        mean_val = np.mean(values)
        median_val = np.median(values)
        plt.figure(figsize=(6, 4))
        plt.hist(
            values,
            bins=range(int(values.min()) - 5, int(values.max()) + 5),
            alpha=0.7,
            edgecolor="black",
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
        return mean_val, median_val

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

    train_file = "resources/data/drivelm/v1_1_train_nus.json"
    drive_analysis = DriveLMAnalysis(train_dict=read_json(train_file))

    questions = drive_analysis.process_analysis()

    with open("resources/output/lm_analysis_qustion.json", "w+") as file:
        json.dump(questions, file, ensure_ascii=False, indent=4)
