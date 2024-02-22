from utils import NO_CODE_STOP_TOKEN, NO_CODE_MAX_TOKEN
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
        self.dataset_name = args.dataset_name  # name of evaluation dataset
        self.limit = args.limit
        self.output_path = args.output_path
        self.hint1 = args.hint1
        self.hint2 = args.hint2
        self.n_generate_answer = args.n_generate_answer
        self.temperature = args.temperature
        if self.temperature == 0:
            assert self.n_generate_answer == 1
        self.llm = LLM(model=args.model, tensor_parallel_size=args.tp_degree)

        if self.hint1 is True:
            file_suffix = ''
        elif self.hint2 is True:
            file_suffix = '_2'
        else:
            raise ValueError('unsupported hint num!')
        prompt_path = f"prompt/{self.dataset_name.lower()}/Hint_prompt{file_suffix}.txt"
        prompt_noa_path = f"prompt/{self.dataset_name.lower()}/Hint_prompt{file_suffix}_noa.txt"
        template_path = f"prompt/{self.dataset_name.lower()}/Hint_template.txt"
        with open(prompt_path, 'r', encoding='utf-8') as fr:
            self.prompt = fr.read()
        with open(prompt_noa_path, 'r', encoding='utf-8') as fr:
            self.prompt_noa = fr.read()
        with open(template_path, 'r', encoding='utf-8') as fr:
            self.template = fr.read()
        logger.info(
            'dataset_name: %s, temperature: %s, n_generate_answer: %s, hint1: %s, hint2: %s',
            self.dataset_name, self.temperature, self.n_generate_answer, self.hint1, self.hint2
        )

    def handle_all(self):
        with open(f'{self.output_path}', 'w', encoding='utf-8') as fw:
            with open(f'data/{self.dataset_name.lower()}/test.jsonl', 'r', encoding='utf-8') as fr:
                for index, row in enumerate(fr, 1):
                    data = json.loads(row)
                    output = self.predict(data)
                    fw.write(json.dumps(output, ensure_ascii=False) + '\n')
                    if index % 20 == 0:
                        logger.info('handled count: %s', index)
                    if self.limit and index >= self.limit:
                        break

    def predict(self, example_dict: dict):
        question = example_dict["question"]

        # apply the template to the question
        templated_example = self._apply_template(template=self.template, example=example_dict)
        # get the max token for the current dataset
        max_tokens = self.get_max_token(self.dataset_name, example_dict)

        # 第一步生成hint
        # concatenate the few-shot prompt and the example
        prompt_and_example_noa = f"{self.prompt_noa}\n\n{templated_example}"
        # query the LM to get the completions
        completions = self._query(prompt=prompt_and_example_noa, max_tokens=max_tokens)
        _g_hint = completions[0] if completions[0].lower().startswith('hint') else 'Hint: ' + completions[0]
        g_hint = _g_hint.split('\n')[0]

        # 第二步使用question+hint生成solution
        # concatenate the few-shot prompt and the example
        prompt_and_example = f"{self.prompt}\n\n{templated_example}\n{g_hint}"
        # query the LM to get the completions
        completions = self._query(prompt=prompt_and_example, max_tokens=max_tokens)
        answer, final_completion = self.derive_answer_from_completions(example=example_dict, completions=completions)

        output = {
            "id": example_dict['id'],
            "gold_answer": example_dict.get('answer', '!no-answer!'),
            "answer": answer,
            "completion": final_completion,
            "generated_hint": g_hint,
            "generated_hint_raw": _g_hint,
            "completions": completions,
            "question": question,
            "prompt_and_example_noa": prompt_and_example_noa,
            "prompt_and_example": prompt_and_example,
        }
        return output

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
        elif self.dataset_name in ["GSM8K", "SVAMP", "MultiArith", "ASDiv"]:
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

    def _query(self, prompt, max_tokens=1024):
        sampling_params = SamplingParams(
            n=self.n_generate_answer,
            temperature=self.temperature,
            top_p=1,
            max_tokens=max_tokens,
            stop=NO_CODE_STOP_TOKEN[self.dataset_name]
        )

        completions = []
        outputs = self.llm.generate(
            [prompt],
            sampling_params,
            use_tqdm=False
        )

        for output in outputs:
            prompt = output.prompt
            # generated_text = output.outputs[0].text.strip()
            # completions.append(generated_text)
            completions.extend([x.text.strip() for x in output.outputs])
            # logger.debug(f"Prompt: {prompt!r}")
            # logger.debug(f"Generated text: {generated_text!r}")
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
    parser.add_argument("--hint1", action="store_true")
    parser.add_argument("--hint2", action="store_true")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--n_generate_answer", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    model = Model()
    model.handle_all()
