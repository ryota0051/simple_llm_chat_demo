import torch


def infer_prompt(model, tokenizer, prompt, max_new_tokens=768):
    with torch.no_grad():
        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors='pt'
        )
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
    return output


def to_prompt(text, tokenizer, system_prompt):
    inst_open, inst_close = "[INST]", "[/INST]"
    sys_open, sys_close = "<<SYS>>\n", "\n<</SYS>>\n\n"

    return "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=inst_open,
        system=f"{sys_open}{system_prompt}{sys_close}",
        prompt=text,
        e_inst=inst_close
    )
