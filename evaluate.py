from utils import load_data, extract_gold_answer, extract_pred_answer
import argparse
import jsonlines
import regex as re


def is_correct(dataset_name, gold_answers, pred_answer):
    '''Check if a predicted answer is correct.
    :param dataset_name (str): The name of the dataset.
    :param gold_answers: The gold answer(s).
    :param pred_answer: The predicted answer.

    :return: Whether the prediction is correct (True) or not (False).
    '''

    # saycan has multiple correct plans, so we need to check if the predicted plan is in the list of correct plans
    if dataset_name == "saycan":
        assert type(gold_answers) == list
        assert type(pred_answer) == str
        if pred_answer in ["[error]", "[invalid]"]:
            return False
        else:
            pred_answer = pred_answer.replace("\\n", "\n")
            pred_plan_list = []
            step_count = 0
            steps = re.split(r", |\n", pred_answer.strip())
            for step in steps:
                step_cols = step.split(". ")
                if len(step_cols) != 2:
                    return "[invalid]"
                step_action = step_cols[1]
                if "find(initial)" in step_action:
                    continue
                step_count += 1
                new_step = f"{step_count}. {step_action}"
                pred_plan_list.append(new_step)
            for gold_answer in gold_answers:
                gold_plan_list = gold_answer.strip().split("\n")
                if pred_plan_list == gold_plan_list:
                    return True
        return False

    else:  # all other datasets have a single correct answer
        gold_answer = gold_answers
        return pred_answer == gold_answer


def evaluate_acc(dataset, predictions, dataset_name):
    correct_count, total_count = 0, 0
    correct_list = []
    for example, prediction in zip(dataset, predictions):
        gold_id = int(example["id"])
        if prediction == {}:
            continue
        pred_id = int(prediction["id"])

        try:
            assert gold_id == pred_id
        except:
            raise AssertionError(f"Gold id {gold_id} doesn't match pred id {pred_id}.")

        try:
            gold_answer = extract_gold_answer(dataset_name, example["answer"])
        except SyntaxError as e:
            print("Error: ", e)
            print(gold_id)
            exit(-1)
        pred_answer = extract_pred_answer(dataset_name, prediction["answer"])
        total_count += 1

        try:
            correct = is_correct(dataset_name, gold_answer, pred_answer)
        except Exception as e:
            print("Error: ", e)
            print("Example: ", gold_id)
            print("Question: ", example["question"])
            print("Gold answer: ", gold_answer, type(gold_answer))
            print("Pred answer: ", pred_answer, type(pred_answer))
            print("Completion: ", prediction["completion"])
            print("\n")
            exit(-1)

        if correct:
            correct_count += 1
            correct_list.append(gold_id)
    # print('correct_list: ', correct_list)
    print('correct_count: ', correct_count, 'total_count: ', total_count)
    acc = round(correct_count / total_count * 100, 3)
    return acc


if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--dataset_name", choices=[
        "GSM8K", "ASDiv", "StrategyQA", "MultiArith", "AQUA",
        "SVAMP", "date", "sports", "saycan", "CLUTRR"
    ])
    Parser.add_argument("--file", required=True)
    args = Parser.parse_args()
    dataset_name = args.dataset_name
    pred_frn = args.file
    # load the dataset
    dataset_frn = f"./data/{dataset_name.lower()}/test.jsonl"
    dataset = load_data(dataset_frn)
    with open(pred_frn) as fr:
        reader = jsonlines.Reader(fr)
        predictions = [line for line in reader]

    acc = evaluate_acc(dataset=dataset,
                       predictions=predictions,
                       dataset_name=dataset_name,
                       )
    print(f"file: {pred_frn}")
    print(f"Answer accuracy: {acc}")
