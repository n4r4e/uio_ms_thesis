
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# list of model names
model_names = [
    "skt/ko-gpt-trinity-1.2B-v0.5", "EleutherAI/polyglot-ko-1.3b", "ai-forever/mGPT", "facebook/xglm-1.7B", "facebook/xglm-564M", "skt/kogpt2-base-v2"
    "01-ai/Yi-6B", "beomi/Yi-Ko-6B", 
    "facebook/xglm-7.5B", "EleutherAI/polyglot-ko-5.8b", "kakaobrain/kogpt", "facebook/xglm-4.5B",
    "meta-llama/Llama-2-7b-hf", "beomi/llama-2-ko-7b", "beomi/open-llama-2-ko-7b", 
    "meta-llama/Llama-2-13b-hf", "beomi/llama-2-koen-13b", 
    "upstage/SOLAR-10.7B-v1.0", 
    "beomi/OPEN-SOLAR-KO-10.7B", "beomi/SOLAR-KOEN-10.8B",
    "ai-forever/mGPT-13B", "EleutherAI/polyglot-ko-12.8b", "EleutherAI/polyglot-ko-3.8b", 
]

# list of models that need to set 'max_new_tokens' as 256
models_with_256_tokens = ["01-ai/Yi-6B", "meta-llama/Llama-2-7b-hf", "upstage/SOLAR-10.7B-v1.0", "meta-llama/Llama-2-13b-hf"]

# parameter configurations
param_configs = [
    {"do_sample": True, "num_beams": 1, "top_p": 0.95, "temperature": [0.7, 0.8, 0.9, 1.0, 1.1], "repetition_penalty": [1.2, 1.4, 1.6, 1.8, 2.0]},
    {"do_sample": False, "num_beams": 5, "top_p": None, "temperature": [None], "repetition_penalty": [1.2, 1.4, 1.6, 1.8, 2.0]}
]

def load_model_and_tokenizer(model_name):
    hf_token = 'your_hugging_face_token_here' ## your hugging face token here
    common_kwargs = {"device_map": "auto", "torch_dtype": "auto"}
    
    if "meta-llama" in model_name or model_name == 'beomi/llama-2-koen-13b':
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, **common_kwargs)
    elif model_name == 'kakaobrain/kogpt':
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision='KoGPT6B-ryan1.5b-float16', token=hf_token, bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]')
        model = AutoModelForCausalLM.from_pretrained(model_name, revision='KoGPT6B-ryan1.5b-float16', token=hf_token, low_cpu_mem_usage=True, **common_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)

    model.eval().cuda()
    return model, tokenizer

@torch.no_grad()
def generate_text(model, tokenizer, input_ids, attention_mask, temperature, repetition_penalty, config, max_new_tokens):
    return model.generate(
        input_ids, 
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        do_sample=config['do_sample'], 
        num_beams=config['num_beams'], 
        top_p=config['top_p'],
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_length=max_new_tokens
    )

articles_df = pd.read_csv('articles_test.csv') ## 
input_prompts = articles_df['Content'].apply(lambda x: ' '.join(x.split()[:10]))

print(f"Processing Model: {model_names}") ## for debugging

for model_name in model_names:
    model, tokenizer = load_model_and_tokenizer(model_name)
    model_short_name = model_name.split('/')[-1]

    results_df = pd.DataFrame(columns=[
        'model', 'do_sample', 'num_beams', 'top_p', 'temperature', 'repetition_penalty', 'max_new_tokens', 
        'input_prompt', 'num_words', 'num_tokens', 'first_10_tokens', 'generated_text'
    ])

    for input_prompt in input_prompts:
        input_ids = tokenizer.encode(input_prompt, return_tensors="pt").cuda()
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).cuda()

        num_words = len(input_prompt.split())
        num_tokens = len(input_ids[0])
        # first_10_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[:10] # tokens for model
        first_10_tokens = [tokenizer.decode(token_id) for token_id in input_ids[0, :10]] # text for human

        for config in param_configs:
            max_new_tokens = 256 if model_name in models_with_256_tokens else 128
            for temp in config['temperature']:
                for rep_penalty in config['repetition_penalty']:
                    generated_outputs = generate_text(model, tokenizer, input_ids, attention_mask, temp, rep_penalty, config, max_new_tokens)
                    generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
                                    
                    results_df = results_df.append({
                        'model': model_name,
                        'do_sample': config['do_sample'],
                        'num_beams': config['num_beams'],
                        'top_p': config['top_p'],
                        'temperature': temp,
                        'repetition_penalty': rep_penalty,
                        'max_new_tokens': max_new_tokens,
                        'input_prompt': input_prompt,
                        'num_words': num_words,
                        'num_tokens': num_tokens,
                        'first_10_tokens': first_10_tokens,
                        'generated_text': generated_text
                    }, ignore_index=True)

    os.makedirs('results', exist_ok=True)  
    results_df.to_csv(f'results/{model_short_name}.csv', index=False)
