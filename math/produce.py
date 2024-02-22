import os
import random
import numpy as np
import argparse
import logging
import json
from vllm import LLM, SamplingParams

# from utils import smm_obj

random.seed(1234)
np.random.seed(1234)

logging.basicConfig(
    format="%(asctime)s <%(name)s>%(levelname)s[%(funcName)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class Model(object):
    def __init__(self):
        args = parse_args()
        self.limit = args.limit
        self.output_path = args.output_path
        self.choice_prompt = args.choice_prompt
        self.llm = LLM(model=args.model, tensor_parallel_size=args.tp_degree)

        self.standard_prompt = args.standard_prompt
        self.least_to_most = args.least_to_most
        self.least_to_most_2 = args.least_to_most_2

        self.n_generate_answer = args.n_generate_answer
        self.temperature = args.temperature
        if self.temperature == 0:
            assert self.n_generate_answer == 1

        logger.info(
            'standard_prompt: %s, LtM: %s, LtM2: %s, '
            'choice_prompt: %s, limit: %s, n_generate_answer: %s, temperature: %s',
            self.standard_prompt, self.least_to_most,
            self.least_to_most_2, self.choice_prompt, self.limit,
            self.n_generate_answer, self.temperature
        )
        if self.choice_prompt == 'non_hint':
            file = './prompt/COT_prompt.txt'
        elif self.choice_prompt == 'prompt1':
            file = './prompt/Hint_prompt.txt'
        elif self.choice_prompt == 'prompt2':
            file = './prompt/Hint_prompt_2.txt'
        else:
            raise ValueError('unsupported prompt!')
        with open(file, 'r', encoding='utf-8') as f:
            self.prompt = f.read()
        self.type_dir_list = [
            'algebra',
            'counting_and_probability',
            'geometry',
            'intermediate_algebra',
            'number_theory',
            'prealgebra',
            'precalculus',
        ]

    def handle_all(self):
        data_list = []
        for t_d in self.type_dir_list:
            index = 0
            for file in os.listdir(f'./data/test/{t_d}'):
                if file.endswith('.json'):
                    index += 1
                    with open(f'./data/test/{t_d}/{file}', 'r', encoding='utf-8') as fr:
                        data = json.load(fr)
                    data_list.append(data)
                    if self.limit and index >= self.limit:
                        break
        logger.info('samples count: %s', len(data_list))
        output_list = self.predict(data_list)
        with open(f'{self.output_path}', 'w', encoding='utf-8') as fw:
            fw.write(json.dumps(output_list, ensure_ascii=False))

    def predict(self, data_list):
        output_list = []
        prompt_list_total = [f"{self.prompt}\n\nProblem:\n{data['problem']}" for data in data_list]
        # 批量大小
        batch_size = 100
        iter_num = len(prompt_list_total) // batch_size if len(prompt_list_total) % batch_size == 0 else len(
            prompt_list_total) // batch_size + 1
        for i_n in range(iter_num):
            logger.info('begin iter_num: %s', i_n + 1)
            prompt_list = prompt_list_total[batch_size * i_n:batch_size * (i_n + 1)]
            tmp_data_list = data_list[batch_size * i_n:batch_size * (i_n + 1)]
            completions = self._query(prompt_list)
            assert len(prompt_list) == len(completions)
            for idx in range(len(completions)):
                output_list.append({
                    "completion_list": completions[idx],
                    **tmp_data_list[idx],
                    "prompt_and_example": prompt_list[idx],
                })

        return output_list

    def _query(self, prompt_list):
        sampling_params = SamplingParams(
            n=self.n_generate_answer,
            temperature=self.temperature,
            top_p=1,
            max_tokens=4096,
            stop='Problem:'
        )

        completions = []
        outputs = self.llm.generate(
            prompt_list,
            sampling_params,
            use_tqdm=True
        )

        for output in outputs:
            # completions.append(output.outputs[0].text.strip())
            completions.append([x.text.strip() for x in output.outputs])
        return completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    # parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--tp_degree", type=int, default=1)
    parser.add_argument("--choice_prompt", required=True,
                        choices=[
                            "non_hint", "prompt1", "prompt2",
                        ])

    parser.add_argument("--standard_prompt", action="store_true")
    parser.add_argument("--least_to_most", action="store_true")
    parser.add_argument("--least_to_most_2", action="store_true")

    parser.add_argument("--n_generate_answer", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    model = Model()
    model.handle_all()
    logger.info('Done')
