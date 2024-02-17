import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference import infer_prompt, to_prompt


DEFAULT_SYSTEM_PROMPT = 'あなたは誠実で優秀な日本人のアシスタントです。'

if __name__ == '__main__':
    text = """
    次の[例題]と[解答例]を参考にして、[問題文]の原産地名を列挙して箇条書きにしてください。
    [例題] ブロッコリー(エクアドル)、揚げじゃがいも(じゃがいも(カナダ)、植物油脂)、いか(中国)
    [解答例]
    ```
        - ブロッコリー: エクアドル
        - 揚げじゃがいも: カナダ
        - いか: 中国
    ```
    [問題文] 殻付き海老(インド)、米(日本)、ワイン(フランス)
    """

    model_name = 'elyza/ELYZA-japanese-Llama-2-7b-fast-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')

    if torch.cuda.is_available():
        print('using GPU')
        model = model.to('cuda')
    prompt = to_prompt(
        text,
        tokenizer,
        DEFAULT_SYSTEM_PROMPT,
    )
    res = infer_prompt(model, tokenizer, prompt)
    print(f'input: \n{text}')
    print(f'output: \n{res}')
