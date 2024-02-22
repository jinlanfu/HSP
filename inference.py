import random
import time

from utils import NO_CODE_STOP_TOKEN, NO_CODE_MAX_TOKEN, extract_gold_answer
import re
from vllm import LLM, SamplingParams
import argparse
import json
import logging

logging.basicConfig(
    format="%(asctime)s <%(name)s>%(levelname)s[%(funcName)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class Model(object):
    def __init__(self):
        args = parse_args()
        self.batch_size = args.batch_size
        self.dataset_name = args.dataset_name  # name of evaluation dataset
        self.limit = args.limit
        self.output_path = args.output_path
        self.hint = args.hint
        self.hint_prompt2 = args.hint_prompt2
        self.hint_prompt3 = args.hint_prompt3
        self.hint_prompt4 = args.hint_prompt4
        self.icl_robust_with_hint = args.icl_robust_with_hint
        self.icl_robust_wo_hint = args.icl_robust_wo_hint
        self.gsm8k_zero_shot = args.gsm8k_zero_shot
        self.n_generate_answer = args.n_generate_answer
        self.temperature = args.temperature
        if self.temperature == 0:
            assert self.n_generate_answer == 1
        self.llm = LLM(model=args.model, tensor_parallel_size=args.tp_degree)
        self.standard_prompt = args.standard_prompt
        self.standard_prompt_hint = args.standard_prompt_hint
        self.least_to_most = args.least_to_most
        self.least_to_most_hint = args.least_to_most_hint
        self.least_to_most_2 = args.least_to_most_2
        self.plan_to_solve = args.plan_to_solve
        self.plan_to_solve_hint = args.plan_to_solve_hint
        # load the prompt and template,(默认使用cot prompt)
        if self.standard_prompt is True:
            file_prefix = 'standard'
        elif self.standard_prompt_hint is True:
            file_prefix = 'standard'
        elif self.least_to_most is True:
            file_prefix = 'LtM'
        elif self.least_to_most_hint is True:
            file_prefix = 'LtM'
        elif self.least_to_most_2 is True:
            file_prefix = 'LtM2'
        elif self.hint is True:
            file_prefix = 'Hint'
        elif self.plan_to_solve is True:
            file_prefix = 'PtS'
        elif self.plan_to_solve_hint is True:
            file_prefix = 'PtS'
        else:
            file_prefix = 'COT'

        template_path = f"prompt/{self.dataset_name.lower()}/{file_prefix}_template.txt"
        if self.hint_prompt2 is True:
            file_suffix = '_2'
        elif self.hint_prompt3 is True:
            file_suffix = '_3'
            template_path = f"prompt/{self.dataset_name.lower()}/{file_prefix}_template_sft.txt"
        elif self.hint_prompt4 is True:
            file_suffix = '_4'
            template_path = f"prompt/{self.dataset_name.lower()}/{file_prefix}_template_sft.txt"
        elif self.standard_prompt_hint is True:
            file_suffix = '_hint'
            template_path = f"prompt/{self.dataset_name.lower()}/{file_prefix}_template_hint.txt"
        elif self.least_to_most_hint is True:
            file_suffix = '_hint'
            template_path = f"prompt/{self.dataset_name.lower()}/{file_prefix}_template_hint.txt"
        elif self.plan_to_solve_hint is True:
            file_suffix = '_hint'
            template_path = f"prompt/{self.dataset_name.lower()}/{file_prefix}_template_hint.txt"
        else:
            file_suffix = ''
        prompt_path = f"prompt/{self.dataset_name.lower()}/{file_prefix}_prompt{file_suffix}.txt"
        with open(prompt_path, 'r', encoding='utf-8') as fr:
            self.prompt = fr.read()
        if self.gsm8k_zero_shot is True:
            template_path = f"prompt/gsm8k/gsm8k_zero_shot.txt"
        with open(template_path, 'r', encoding='utf-8') as fr:
            self.template = fr.read()
        # ICL robust experiment
        if self.icl_robust_with_hint > 0:
            self.icl_robust(need_hint=True, icl_robust_idx=self.icl_robust_with_hint)
        elif self.icl_robust_wo_hint > 0:
            self.icl_robust(need_hint=False, icl_robust_idx=self.icl_robust_wo_hint)

        logger.info(
            'dataset_name: %s, temperature: %s, '
            'n_generate_answer: %s, standard_prompt: %s, '
            'LtM: %s, LtM2: %s, hint: %s, hint_prompt2: %s, '
            'hint_prompt3: %s, hint_prompt4: %s, batch_size: %s, '
            'plan_to_solve: %s, gsm8k_zero_shot: %s, '
            'standard_prompt_hint: %s, least_to_most_hint: %s, '
            'plan_to_solve_hint: %s, '
            'icl_robust_with_hint: %s, icl_robust_wo_hint: %s, ',
            self.dataset_name, self.temperature,
            self.n_generate_answer, self.standard_prompt,
            self.least_to_most, self.least_to_most_2,
            self.hint, self.hint_prompt2,
            self.hint_prompt3, self.hint_prompt4, self.batch_size,
            self.plan_to_solve, self.gsm8k_zero_shot,
            self.standard_prompt_hint, self.least_to_most_hint,
            self.plan_to_solve_hint,
            self.icl_robust_with_hint, self.icl_robust_wo_hint,
        )

    def icl_robust(self, need_hint=True, icl_robust_idx=0):
        self.prompt = ''
        if self.dataset_name == 'GSM8K':
            with open(f'data/{self.dataset_name.lower()}/train.jsonl', 'rb') as f:
                q_a_map = dict()
                for line in f:
                    row = json.loads(line)
                    final_answer = '\nThe answer is: ' + str(extract_gold_answer(self.dataset_name, row['answer']))
                    q_a_map[row['question']] = row['answer'] + final_answer
            with open(f'data/{self.dataset_name.lower()}/icl_robust.jsonl', 'rb') as f:
                for idx, row in enumerate(f):
                    if 8 * icl_robust_idx > idx >= 8 * (icl_robust_idx - 1):
                        row_item = json.loads(row)
                        question = 'Q: ' + row_item['question'].replace('\n', '') + '\n'
                        answer = 'A: ' + q_a_map[row_item['question']].replace('\n\n', '\n')
                        if need_hint is True:
                            hint = row_item['hint'].replace('\n', '') + '\n'
                            self.prompt += question + hint + answer + '\n\n'
                        else:
                            self.prompt += question + answer + '\n\n'
        elif self.dataset_name == 'StrategyQA':
            with open(f'data/{self.dataset_name.lower()}/icl_robust.jsonl', 'rb') as f:
                for idx, row in enumerate(f):
                    if 6 * icl_robust_idx > idx >= 6 * (icl_robust_idx - 1):
                        row_item = json.loads(row)
                        if need_hint is True:
                            self.prompt += row_item['question'] + row_item['hint'] + row_item['answer'] + '\n\n\n'
                        else:
                            self.prompt += row_item['question'] + row_item['answer'] + '\n\n\n'
        self.prompt = self.prompt.strip()

    def handle_all(self):
        data_list = []
        with open(f'data/{self.dataset_name.lower()}/test.jsonl', 'r', encoding='utf-8') as fr:
            for index, row in enumerate(fr, 1):
                data_list.append(json.loads(row))
                if self.limit and index >= self.limit:
                    break
        logger.info('samples count: %s', len(data_list))
        random.seed(1234)
        random.shuffle(data_list)
        output_list = self.predict(data_list)
        output_list.sort(key=lambda x: x['id'])
        with open(f'{self.output_path}', 'w', encoding='utf-8') as fw:
            for output in output_list:
                fw.write(json.dumps(output, ensure_ascii=False) + '\n')

    def predict(self, data_list):
        prompt_list_total = []
        for data in data_list:
            templated_example = self._apply_template(template=self.template, example=data)
            if self.gsm8k_zero_shot is True:
                prompt_and_example = templated_example
            else:
                prompt_and_example = f"{self.prompt}\n\n{templated_example}"
            prompt_list_total.append(prompt_and_example)
        max_tokens = self.get_max_token(self.dataset_name, data_list[0])

        # query the LM to get the completions
        # 批量大小
        output_list = []
        iter_num = len(prompt_list_total) // self.batch_size if len(prompt_list_total) % self.batch_size == 0 else len(
            prompt_list_total) // self.batch_size + 1
        for i_n in range(iter_num):
            logger.info('begin iter_num: %s', i_n + 1)
            prompt_list = prompt_list_total[self.batch_size * i_n:self.batch_size * (i_n + 1)]
            tmp_data_list = data_list[self.batch_size * i_n:self.batch_size * (i_n + 1)]

            completions_list = self._query(prompt_list=prompt_list, max_tokens=max_tokens)
            assert len(tmp_data_list) == len(completions_list) == len(prompt_list)
            for data, completions, prompt_and_example in zip(tmp_data_list, completions_list, prompt_list):
                answer, final_completion = self.derive_answer_from_completions(example=data, completions=completions)
                output = {
                    "id": data['id'],
                    "gold_answer": data.get('answer', '!no-answer!'),
                    "answer": answer,
                    "completion": final_completion,
                    "completions": completions,
                    "question": data["question"],
                    "prompt_and_example": prompt_and_example
                }
                output_list.append(output)
        return output_list

    def _apply_template(self, template: str, example: dict):
        '''Apply the template to a new example.
        @:param template (str): the template to be applied to the example.
        @:param example (str): the example to be converted into the template format.

        @:return (str): the example converted into the template format.
        '''
        # for every [{FIELD}] in the template, replace it with the corresponding value of the key "{field}" in the example dict
        example_in_template = template
        for field in re.findall(r"\[.*?\]", template):
            field_name = field[1:-1]
            field_name = field_name.lower()
            if field_name in example:
                example_in_template = example_in_template.replace(field, str(example[field_name]))
        return example_in_template

    def get_max_token(self, dataset_name, example):
        '''Get the max token for the current dataset.
        @:param dataset_name (str): the name of the dataset
        @:param example (dict): the example dict

        @:return (int): the max token
        '''
        max_token_dict = NO_CODE_MAX_TOKEN

        if dataset_name == "CLUTRR":  # for CLUTRR, the max token depends on the number of steps required (example["k"])
            return max_token_dict[self.dataset_name] * example[
                "k"]  # multiply the max token for each step by the number of steps
        else:  # for other datasets, the max token is static for each dataset
            return max_token_dict[self.dataset_name]

    def _execute(self, example: dict, completion: str):
        '''Execute the code in the model completion.
        @:param example (str): the example
        @:param completion (str): the model completion

        @:return: the answer (the type depends on the dataset)
        '''
        if self.dataset_name == "AQUA":
            if "answer is " not in completion:
                answer = "[invalid]"
            else:
                answer = completion.split("answer is ")[-1].strip("\n().").upper()
        elif self.dataset_name == "GSM8K":
            if 'answer is ' in completion:
                answer = completion.split("answer is ")[-1].strip("\n.")
            elif 'answer is:' in completion:
                answer = completion.split("answer is:")[-1].strip("\n.")
            elif 'Final answer:' in completion:
                answer = completion.split("Final answer:")[-1].strip("\n.").strip()
            else:
                answer = "[invalid]"
        elif self.dataset_name in ["SVAMP", "MultiArith", "ASDiv"]:
            if "answer is " not in completion:
                answer = "[invalid]"
            else:
                answer = completion.split("answer is ")[-1].strip("\n.")
        elif self.dataset_name == "date":
            if "answer is " not in completion:
                answer = "[invalid]"
            else:
                answer = completion.split("answer is ")[-1].strip()
                answer = re.sub(pattern="[\s\.#]", repl="", string=answer)
        elif self.dataset_name == "sports":
            if "answer is " not in completion:
                answer = "[invalid]"
            else:
                answer = completion.split("answer is ")[-1].split()[0].strip(".")
                if answer == "yes":
                    answer = "1"
                elif answer == "no":
                    answer = "0"
                else:
                    answer = "[invalid]"
        elif self.dataset_name == "saycan":
            completion = completion.strip()
            lines = completion.split("\n")
            if len(lines) == 1:
                answer = lines[0].strip()
            else:
                answer_line = [line for line in lines if line.startswith("Plan:")][0]
                answer = answer_line.split("Plan: ")[1].strip()
        elif self.dataset_name == "CLUTRR":
            answer = "[invalid]"
            lines = completion.split("\n")
            lines = [line.strip() for line in lines if line.strip() != ""]
            answer_line = lines[-1]
            # look for patterns like "A is B's xx (relation name)", "A is the xx (relation name) of B"
            patterns = ["(\[?\w+\]?) is (\[?\w+\]?)'s (\w+)",
                        "(\[?\w+\]?) is the (\w+) of (\[?\w+\]?)"]
            relation_position = [3, 2]  # where the relation name is in the matched pattern
            for pattern_id, pattern in enumerate(patterns):
                matched_pattern = re.search(pattern=pattern, string=answer_line)
                if matched_pattern is not None:
                    # extract the relation name
                    relation_name = matched_pattern.group(relation_position[pattern_id])
                    answer = relation_name
                    break
                else:
                    continue
            answer = answer.strip(".")
        elif self.dataset_name == "StrategyQA":
            if "answer is" not in completion:
                answer = "[invalid]"
            else:
                answer = completion.split("answer is ")[-1].split()[0].strip("\n.").lower()
                if answer == "yes":
                    answer = True
                elif answer == "no":
                    answer = False
                else:
                    answer = "[invalid]"
        else:
            for line in completion.split("\n"):
                if line.startswith("Answer: "):
                    answer = line[8:].strip('"')
        return answer

    def derive_answer_from_completions(self, example, completions):
        '''Derive the answer from a list of completions.
        @:param example (dict): the example
        @:param completions (List[str]): the list of completions

        @:return (tuple): answer (type depends on dataset), final_completion (str)
        '''

        # execute the completions to get the answers
        completion_lists = {}  # a dict of lists of completions; each item is {answer: [completions that result in the same answer after execution]}
        for completion in completions:
            try:
                answer = self._execute(example=example, completion=completion)  # execute the completion
            except Exception as e:
                logger.warning(f"Error executing completion: {completion}.\n Error: {e}")
                continue

            if type(answer) == str and "invalid" in answer:
                continue

            answer = self.postprocess_answer(answer)

            # check for answer equivalence
            equivalent_found = False
            for existing_answer in completion_lists.keys():
                if existing_answer == answer:  # if the answer is equivalent to an existing answer
                    completion_lists[existing_answer].append(
                        completion)  # add the completion to list of completions corresponding to the existing answer
                    equivalent_found = True
                    break
            if not equivalent_found:  # if the answer is not equivalent to any existing answer
                completion_lists[answer] = [completion]  # create a new list of completions corresponding to the answer

        # get the top-voted answer as the final answer
        if len(completion_lists) == 0:  # if no valid completion is found
            return "[invalid]", completions[0]

        completion_lists = sorted(completion_lists.items(), key=lambda x: len(x[1]),
                                  reverse=True)  # vote for the majority answer
        final_completion = completion_lists[0][1][0]
        answer = completion_lists[0][0]

        return answer, final_completion

    def postprocess_answer(self, answer):
        '''Postprocess the answer based on the dataset.
        @:param answer: the answer to be postprocessed

        @:return: the postprocessed answer
        '''
        if self.dataset_name in ["GSM8K", "SVAMP", "MultiArith", "ASDiv"]:
            answer = str(answer).strip()
            answer = answer.split("\n")[-1]  # only get the last output
            return answer

        elif self.dataset_name == "AQUA":
            _answer = str(answer).strip()
            answer = _answer[0] if len(_answer) > 0 else _answer
            return answer

        elif self.dataset_name == "date":
            answer = str(answer).strip()
            answer = answer.split("\n")[-1]  # only get the last output
            answer = answer.rstrip("Y")  # strip the trailing "Y"s if it exists
            return answer

        elif self.dataset_name == "sports":
            answer = str(answer).strip()
            answer = answer.split("\n")[-1]  # only get the last output
            return answer

        elif self.dataset_name == "StrategyQA":
            return answer

        elif self.dataset_name == "saycan":
            return answer

        elif self.dataset_name == "CLUTRR":
            answer = str(answer).strip()
            return answer

        else:
            raise NotImplementedError(f"Postprocessing function for dataset {self.dataset_name} is not implemented.")

    def _query(self, prompt_list, max_tokens=500):
        sampling_params = SamplingParams(
            n=self.n_generate_answer,
            temperature=self.temperature,
            top_p=1,
            max_tokens=max_tokens,
            stop=NO_CODE_STOP_TOKEN[self.dataset_name]
        )

        completions = []
        outputs = self.llm.generate(
            prompt_list,
            sampling_params,
            use_tqdm=True
        )

        for output in outputs:
            completions.append([x.text.strip() for x in output.outputs])
        return completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset_name", required=True,
                        choices=[
                            "GSM8K", "ASDiv", "MultiArith",
                            "SVAMP", "AQUA", "date",
                            "StrategyQA", "sports",
                            "saycan", "CLUTRR"
                        ])
    parser.add_argument("--tp_degree", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hint", action="store_true")
    parser.add_argument("--hint_prompt2", action="store_true")
    parser.add_argument("--hint_prompt3", action="store_true")
    parser.add_argument("--hint_prompt4", action="store_true")
    parser.add_argument("--gsm8k_zero_shot", action="store_true")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--n_generate_answer", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--standard_prompt", action="store_true")
    parser.add_argument("--standard_prompt_hint", action="store_true")
    parser.add_argument("--least_to_most", action="store_true")
    parser.add_argument("--least_to_most_hint", action="store_true")
    parser.add_argument("--least_to_most_2", action="store_true")
    parser.add_argument("--plan_to_solve", action="store_true")
    parser.add_argument("--plan_to_solve_hint", action="store_true")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--icl_robust_with_hint", type=int, default=0)
    parser.add_argument("--icl_robust_wo_hint", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    model = Model()
    model.handle_all()
    logger.info('Done!')
