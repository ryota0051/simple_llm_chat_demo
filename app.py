import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference import infer_prompt, to_prompt


MODEL_NAME = 'elyza/ELYZA-japanese-CodeLlama-7b-instruct'
DEFAULT_SYSTEM_PROMPT = 'あなたは誠実で優秀な日本人のアシスタントです。'
MAX_NEW_TOKENS = 768


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )
    if torch.cuda.is_available():
        print('using GPU')
        model = model.to('cuda')
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = load_model()
    st.title('simple chat')
    input_txt = st.chat_input('What is up?')
    if input_txt:
        with st.chat_message('user'):
            st.markdown(input_txt)

        with st.chat_message('assistant'):
            prompt = to_prompt(
                input_txt,
                tokenizer,
                DEFAULT_SYSTEM_PROMPT,
            )
            response = infer_prompt(
                model,
                tokenizer,
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            st.markdown(response)
