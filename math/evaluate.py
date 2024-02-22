import json
import sys
from collections import Counter
from utils import smm_obj

INVALID_ANSWER = '[invalidanswer]'


def derive_final_answer(completion):
    try:
        search_answer = smm_obj.get_unnormalized_answer(completion)
        if search_answer == INVALID_ANSWER:
            try:
                answer = smm_obj.process_doc(completion)
            except:
                answer = INVALID_ANSWER
        else:
            answer = smm_obj.normalize_tex(search_answer)
    except:
        answer = INVALID_ANSWER
    return answer


print(sys.argv)
file = sys.argv[1]
total = right = gold_answer_invalid = answer_invalid = 0
m_type_dict = dict()
result_list = []
with open(file, 'r', encoding='utf-8') as fr:
    for tmp in json.load(fr):
        # 计算各个type类型的acc
        m_type = tmp['type']
        if m_type not in m_type_dict:
            m_type_dict[m_type] = [0, 0, 0]
        total += 1
        m_type_dict[m_type][0] += 1
        try:
            gold_answer = smm_obj.process_doc(tmp['solution'])
        except:
            gold_answer = INVALID_ANSWER
            gold_answer_invalid += 1

        if 'completion' in tmp:
            final_answer = derive_final_answer(tmp['completion'])
        elif 'completion_list' in tmp:
            answer_list = [derive_final_answer(completion) for completion in tmp['completion_list']]
            for item in Counter(answer_list).most_common(2):
                final_answer = item[0]
                if final_answer != INVALID_ANSWER:
                    break
        else:
            raise ValueError('unsupported key!')

        if final_answer == INVALID_ANSWER:
            answer_invalid += 1

        if gold_answer != INVALID_ANSWER and gold_answer == final_answer:
            tmp['right'] = 1
            right += 1
            m_type_dict[m_type][1] += 1
        else:
            tmp['right'] = 0
        tmp['gold_answer'] = gold_answer
        tmp['answer'] = final_answer
        m_type_dict[m_type][2] = round(m_type_dict[m_type][1] / m_type_dict[m_type][0], 4)
        result_list.append(tmp)

with open(f'eval_{file}', 'w', encoding='utf-8') as fw:
    fw.write(json.dumps(result_list, ensure_ascii=False) + '\n')

print(
    'total: ', total,
    'right: ', right,
    'acc: ', round(right / total, 4),
    'gold_answer_invalid: ', gold_answer_invalid,
    'answer_invalid: ', answer_invalid
)
for k, v in m_type_dict.items():
    t, r, a = v
    print('-' * 60)
    print(k.ljust(25), f'total: {t}'.ljust(10), f'right: {r}'.ljust(10), f'acc: {a}'.ljust(10))
