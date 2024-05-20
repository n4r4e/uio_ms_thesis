import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

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

# list of models that need to set 'max_new_tokens' as 256 / 512
models_with_512_tokens = ["01-ai/Yi-6B", "meta-llama/Llama-2-7b-hf", "upstage/SOLAR-10.7B-v1.0", "meta-llama/Llama-2-13b-hf"]

model_configs = {
    "01-ai/Yi-6B": {"temperature": 0.9, "repetition_penalty": 1.2},
    "beomi/Yi-Ko-6B": {"temperature": 0.7, "repetition_penalty": 1.2},
    "meta-llama/Llama-2-7b-hf": {"temperature": 0.8, "repetition_penalty": 1.2},
    "beomi/llama-2-ko-7b": {"temperature": 0.9, "repetition_penalty": 1.2},
    "beomi/open-llama-2-ko-7b": {"temperature": 0.8, "repetition_penalty": 1.8},
    "meta-llama/Llama-2-13b-hf": {"temperature": 0.7, "repetition_penalty": 1.2},
    "beomi/llama-2-koen-13b": {"temperature": 0.7, "repetition_penalty": 1.6},
    "upstage/SOLAR-10.7B-v1.0": {"temperature": 0.7, "repetition_penalty": 1.2},
    "beomi/OPEN-SOLAR-KO-10.7B": {"temperature": 0.8, "repetition_penalty": 1.4},
    "beomi/SOLAR-KOEN-10.8B": {"temperature": 0.9, "repetition_penalty": 1.2},
    "skt/ko-gpt-trinity-1.2B-v0.5": {"temperature": 1.0, "repetition_penalty": 2.0},
    "EleutherAI/polyglot-ko-1.3b": {"temperature": 0.8, "repetition_penalty": 1.2},
    "ai-forever/mGPT": {"temperature": 0.8, "repetition_penalty": 1.2},
    "facebook/xglm-1.7B": {"temperature": 0.9, "repetition_penalty": 1.2},
    "ai-forever/mGPT-13B": {"temperature": 0.7, "repetition_penalty": 1.2},
    "EleutherAI/polyglot-ko-12.8b": {"temperature": 0.7, "repetition_penalty": 1.2},
    "EleutherAI/polyglot-ko-3.8b": {"temperature": 0.8, "repetition_penalty": 1.2},
    "EleutherAI/polyglot-ko-5.8b": {"temperature": 0.8, "repetition_penalty": 1.2},
    "kakaobrain/kogpt": {"temperature": 0.8, "repetition_penalty": 1.6},
    "facebook/xglm-4.5B": {"temperature": 1.0, "repetition_penalty": 1.2},
    "facebook/xglm-7.5B": {"temperature": 0.7, "repetition_penalty": 1.2},
    "facebook/xglm-564M": {"temperature": 0.8, "repetition_penalty": 1.2},
    "skt/kogpt2-base-v2": {"temperature": 0.7, "repetition_penalty": 1.2},
}

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
def generate_text(model, tokenizer, input_ids, attention_mask, model_name, max_new_tokens):
    return model.generate(
        input_ids, 
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        do_sample=True, 
        num_beams=1, 
        top_p=0.95,
        temperature=model_configs[model_name]['temperature'],
        repetition_penalty=model_configs[model_name]['repetition_penalty'],
        max_new_tokens=max_new_tokens
    )

examples = [
    "<北형제국 쿠바와 65년 만에 수교> 정부가 북한의 형제국인 쿠바와 외교관계를 수립했다. 1959년 교류가 단절된 지 65년 만이다. 외교부는 한국과 쿠바가 14일(현지시간) 미국 뉴욕에서 양국 유엔 대표부가 외교 공한을 교환하는 방식으로 공식 외교관계를 수립했다고 밝혔다. 우리나라의 193번째 수교국으로, 유엔 회원국 가운데 이제 시리아만 미수교국으로 남았다.",
    "<신선식품까지 판다…中 알리, 전방위 韓 공습> 초저가 공산품을 무기로 국내 시장을 빠르게 잠식하고 있는 중국 온라인 쇼핑 플랫폼 알리익스프레스가 신선식품 사업 진출을 준비 중인 것으로 확인됐다. 온라인 그로서리 전문가 영입을 진행하는 가운데 한국을 본격 공략하기 위해서는 시장 규모가 크고 반복 구매가 잦은 신선식품까지 영역을 확대해야 한다고 판단한 것으로 분석된다.",
    "<들리나요, 어린 누이의 귓속말> 이제 갓 걸음마를 뗀 어린 동생이 울며 투정을 부리자, 누이가 무어라 말하며 어깨를 토닥인다. 누이라고는 하지만, 세상의 언어들을 얼마나 익혔을까 싶은 어린아이다. 그래도 누이는, 그 빈약한 언어 속에 동생을 달랠 수 있는 말 몇 마디를 품고 있었던가 보다. 엿들을 수 없는 누이의 말을, 사진이 들려준다."
]
examples_text = '\n'.join(examples)
articles_df = pd.read_csv('articles_1000_shuffled.csv')

os.makedirs('zero', exist_ok=True)
os.makedirs('few', exist_ok=True)

for input_prompt_type in ['zero', 'few']:
    if input_prompt_type == 'zero':
        input_prompts = articles_df['Content'].apply(lambda x: ' '.join(x.split()[:3]))  # input prompt: first 3 words
    else:  # 'few'
        input_prompts = articles_df.apply(lambda row: f"{examples_text}\n<{row['Title']}> {' '.join(row['Content'].split()[:3])}", axis=1)  # input prompt: examples + first 3 words

    for model_name in model_names:
        model, tokenizer = load_model_and_tokenizer(model_name)
        max_new_tokens = 512 if model_name in models_with_512_tokens else 256
        model_short_name = model_name.split('/')[-1]

        results_list = []  

        for input_prompt in input_prompts:
            input_ids = tokenizer.encode(input_prompt, return_tensors="pt").cuda()
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long).cuda()
            
            num_tokens = len(input_ids[0])
            first_10_tokens = [tokenizer.decode(token_id) for token_id in input_ids[0, :10]]  # text for human
            
            generated_outputs = generate_text(model, tokenizer, input_ids, attention_mask, model_name, max_new_tokens)
            generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
            
            results_list.append({
                'model': model_short_name,
                'temperature': model_configs[model_name]['temperature'],
                'repetition_penalty': model_configs[model_name]['repetition_penalty'],
                'max_new_tokens': max_new_tokens,
                'input_prompt': input_prompt,
                'num_tokens': num_tokens,
                'first_10_tokens': first_10_tokens,
                'generated_text': generated_text
            })

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'{input_prompt_type}/{model_short_name}.csv', index=False)
