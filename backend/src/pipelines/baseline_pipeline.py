import pandas as pd

from src.data.dataset import QADataset
from src.models.generator import QwenGenerator
from configs.config import NUM_EXAMPLES
from configs.config import RESULT_PATH
from configs.config import DATASET_NAME
from configs.config import DATASET_CONFIG


class BaselinePipeline:

    def __init__(self):

        self.dataset = QADataset(
            DATASET_NAME,
            DATASET_CONFIG
        )
        self.generator = QwenGenerator()

    def run(self):

        validation = self.dataset.get_validation()

        results = []

        num_examples = min(NUM_EXAMPLES, len(validation))

        for i in range(num_examples):

            print("\n" + "=" * 80)
            print(f"Question {i+1}/{num_examples}")
            print("=" * 80)
            print(f"Processing Question {i+1}/{num_examples}")

            sample = validation[i]

            sample = self.dataset.preprocess(sample)

            question = sample["question"]
            ground_truth = sample["answer"]

            prediction = self.generator.generate(question)

            print("Question:")
            print(question)

            print()

            print("Prediction:")
            print(prediction)

            print()

            print("Ground Truth:")
            print(ground_truth)

            print()

            results.append({

                "question": question,

                "prediction": prediction,

                "ground_truth": ground_truth

            })

        df = pd.DataFrame(results)

        df.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8"
        )

        print()

        print("=" * 70)

        print("Saved Successfully")

        print(RESULT_PATH)