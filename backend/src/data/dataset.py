from datasets import load_dataset

class QADataset:

    def __init__(self, name="trivia_qa", config="rc"):
        print(f"Loading dataset: {name}/{config}")
        self.dataset = load_dataset(name, config)

    def preprocess(self, sample):
        return {
            "question": sample["question"],
            "answer": sample["answer"]["value"]
        }

    def get_train(self):
        return self.dataset["train"].map(self.preprocess)

    def get_validation(self):
        return self.dataset["validation"]

    def show_examples(self, n=5):
        train = self.get_train()
        for i in range(min(n, len(train))):
            print("=" * 70)
            print(f"Question: {train[i]['question']}")
            print(f"Answer: {train[i]['answer']}")