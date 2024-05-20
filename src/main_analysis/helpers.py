from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import unicodedata
import re
import os
import kiwipiepy
import seaborn as sns
import json
import matplotlib.cm as cm
from numpy.linalg import norm
import math
from collections import Counter
from tqdm import tqdm

kiwi = kiwipiepy.Kiwi()

# Set the model paths and corresponding file names
MODEL_PATHS = [
    ("skt/kogpt2-base-v2", "kogpt2-base-v2.csv"),
    ("facebook/xglm-564M", "xglm-564M.csv"),
    ("skt/ko-gpt-trinity-1.2B-v0.5", "ko-gpt-trinity-1.2B-v0.5.csv"),
    ("EleutherAI/polyglot-ko-1.3b", "polyglot-ko-1.3b.csv"),
    ("ai-forever/mGPT", "mGPT.csv"),
    ("facebook/xglm-1.7B", "xglm-1.7B.csv"),
    ("EleutherAI/polyglot-ko-3.8b", "polyglot-ko-3.8b.csv"),
    ("EleutherAI/polyglot-ko-5.8b", "polyglot-ko-5.8b.csv"),
    ("kakaobrain/kogpt", "kogpt.csv"),
    ("facebook/xglm-4.5B", "xglm-4.5B.csv"),
    ("facebook/xglm-7.5B", "xglm-7.5B.csv"),
    ("EleutherAI/polyglot-ko-12.8b", "polyglot-ko-12.8b.csv"),
    ("ai-forever/mGPT-13B", "mGPT-13B.csv"),
    ("01-ai/Yi-6B", "Yi-6B.csv"),
    ("beomi/Yi-Ko-6B", "Yi-Ko-6B.csv"),
    ("meta-llama/Llama-2-7b-hf", "Llama-2-7b-hf.csv"),
    ("beomi/llama-2-ko-7b", "llama-2-ko-7b.csv"),
    ("beomi/open-llama-2-ko-7b", "open-llama-2-ko-7b.csv"),
    ("meta-llama/Llama-2-13b-hf", "Llama-2-13b-hf.csv"),
    ("beomi/llama-2-koen-13b", "llama-2-koen-13b.csv"),
    ("upstage/SOLAR-10.7B-v1.0", "SOLAR-10.7B-v1.0.csv"),
    ("beomi/OPEN-SOLAR-KO-10.7B", "OPEN-SOLAR-KO-10.7B.csv"),
    ("beomi/SOLAR-KOEN-10.8B", "SOLAR-KOEN-10.8B.csv")
]

# Set the order of the models (original names)
MODEL_ORDER = ["kogpt2-base-v2", "ko-gpt-trinity-1.2B-v0.5", "polyglot-ko-1.3b", "polyglot-ko-3.8b", "polyglot-ko-5.8b", "polyglot-ko-12.8b", "kogpt", 
               "xglm-564M", "xglm-1.7B", "xglm-4.5B", "xglm-7.5B", "mGPT", "mGPT-13B",
               "Yi-Ko-6B", "llama-2-ko-7b", "open-llama-2-ko-7b", "llama-2-koen-13b", "OPEN-SOLAR-KO-10.7B", "SOLAR-KOEN-10.8B",
               "Yi-6B", "Llama-2-7b-hf", "Llama-2-13b-hf", "SOLAR-10.7B-v1.0"]

# Set the order of conllu file names
CONLLU_FILE_NAMES = ["corpus_spoken", "corpus_written", "ko_gsd-ud-TB", 
              "kogpt2-base-v2-125M", "ko-gpt-trinity-1.2B-v0.5", "polyglot-ko-1.3b", "polyglot-ko-3.8b", "polyglot-ko-5.8b", "polyglot-ko-12.8b", "kogpt-6B", 
              "xglm-564M", "xglm-1.7B", "xglm-4.5B", "xglm-7.5B", "mGPT-1.3B", "mGPT-13B", 
              "Yi-Ko-6B", "llama-2-ko-7b", "open-llama-2-ko-7b", "llama-2-koen-13b", "OPEN-SOLAR-KO-10.7B", "SOLAR-KOEN-10.8B", 
              "Yi-6B", "Llama-2-7b", "Llama-2-13b", "SOLAR-10.7B-v1.0"]

def remove_pattern(text):
    # Find the pattern: "<...> ... <...> ... <...> ...  <...>"
    pattern = re.compile(r'(<[^>]+>[^<]+){3}<[^>]+>')
    match = pattern.match(text)
    if match:
        return text[match.end():].strip()
    return text  

def remove_pattern_yiko6b(text):
    text_to_remove = " <北형제국 쿠바와 65년 만에 수교> 정부가 북한의 형제국인 쿠바와 외교관계를 수립했다. 1959년 교류가 단절된 지 65년 만이다. 외교부는 한국과 쿠바가 14일(현지시간) 미국 뉴욕에서 양국 유엔 대표부가 외교 공한을 교환하는 방식으로 공식 외교관계를 수립했다고 밝혔다. 우리나라의 193번째 수교국으로, 유엔 회원국 가운데 이제 시리아만 미수교국으로 남았다.\n<신선식품까지 판다...中 알리, 전방위 韓 공습> 초저가 공산품을 무기로 국내 시장을 빠르게 잠식하고 있는 중국 온라인 쇼핑 플랫폼 알리익스프레스가 신선식품 사업 진출을 준비 중인 것으로 확인됐다. 온라인 그로서리 전문가 영입을 진행하는 가운데 한국을 본격 공략하기 위해서는 시장 규모가 크고 반복 구매가 잦은 신선식품까지 영역을 확대해야 한다고 판단한 것으로 분석된다.\n<들리나요, 어린 누이의 귓속말> 이제 갓 걸음마를 뗀 어린 동생이 울며 투정을 부리자, 누이가 무어라 말하며 어깨를 토닥인다. 누이라고는 하지만, 세상의 언어들을 얼마나 익혔을까 싶은 어린아이다. 그래도 누이는, 그 빈약한 언어 속에 동생을 달랠 수 있는 말 몇 마디를 품고 있었던가 보다. 엿들을 수 없는 누이의 말을, 사진이 들려준다.\n"
    cleaned_text = text.replace(text_to_remove, "")
    cleaned_text = re.sub(r'<[^>]*>', '', cleaned_text, count=1).strip()
    return cleaned_text  

def calculate_stats(df, column, func=None):
    if func is not None:
        series = df[column].apply(func)
    else:
        series = df[column]
    avg = series.mean()
    std = series.std()
    min_value = series.min()
    max_value = series.max()
    return avg, std, min_value, max_value

def load_tokenizer(model_name):
    hf_token = 'your_hugging_face_token_here' ## your hugging face token here
    
    if "meta-llama" in model_name or model_name == 'beomi/llama-2-koen-13b':
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    elif model_name == 'kakaobrain/kogpt':
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision='KoGPT6B-ryan1.5b-float16', use_auth_token=hf_token, bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_and_process_data(file_path, has_corpora=False):
    merged_df = pd.read_csv(file_path)
    merged_df = merged_df.set_index('model')

    # Define the desired order of model names
    if has_corpora:
        model_order = ["corpus_spoken", "corpus_written", 
               "kogpt2-base-v2", "ko-gpt-trinity-1.2B-v0.5", "polyglot-ko-1.3b", "polyglot-ko-3.8b", "polyglot-ko-5.8b", "polyglot-ko-12.8b", "kogpt", 
               "xglm-564M", "xglm-1.7B", "xglm-4.5B", "xglm-7.5B", "mGPT", "mGPT-13B",
               "Yi-Ko-6B", "llama-2-ko-7b", "open-llama-2-ko-7b", "llama-2-koen-13b", "OPEN-SOLAR-KO-10.7B", "SOLAR-KOEN-10.8B",
               "Yi-6B", "Llama-2-7b-hf", "Llama-2-13b-hf", "SOLAR-10.7B-v1.0"]

    else:
        model_order = ["kogpt2-base-v2", "ko-gpt-trinity-1.2B-v0.5", "polyglot-ko-1.3b", "polyglot-ko-3.8b", "polyglot-ko-5.8b", "polyglot-ko-12.8b", "kogpt", 
                   "xglm-564M", "xglm-1.7B", "xglm-4.5B", "xglm-7.5B", "mGPT", "mGPT-13B",
                   "Yi-Ko-6B", "llama-2-ko-7b", "open-llama-2-ko-7b", "llama-2-koen-13b", "OPEN-SOLAR-KO-10.7B", "SOLAR-KOEN-10.8B",
                   "Yi-6B", "Llama-2-7b-hf", "Llama-2-13b-hf", "SOLAR-10.7B-v1.0"]
    
    # Check if all models in the desired order exist in the DataFrame
    missing_models = set(model_order) - set(merged_df.index)
    if missing_models:
        raise ValueError(f"The following models are missing in the DataFrame: {', '.join(missing_models)}")
    
    merged_df = merged_df.reindex(model_order)

    # Rename model names
    merged_df = merged_df.rename(index={
        "kogpt2-base-v2": "kogpt2-base-v2-125M",
        "mGPT": "mGPT-1.3B",
        "kogpt": "kogpt-6B",
        "Llama-2-7b-hf": "Llama-2-7b",
        "Llama-2-13b-hf": "Llama-2-13b"
    })               

    # Check if there are any missing values (NaN) in the DataFrame
    if merged_df.isnull().values.any():
        raise ValueError("The DataFrame contains missing values (NaN).")

    return merged_df

def classify_and_save(classify_func, result_folders, file_suffix):
    zero_category_dict = {}
    few_category_dict = {}

    for result_folder in result_folders:
        if result_folder == "zero":
            target_column = "generated_text"
        else:
            target_column = "only_generated_text"

        result_list = []
        category_dict = {}

        for model_addr, file_name in MODEL_PATHS:
            model_name = file_name.replace('.csv', '')
            print(f"{model_addr=}, {file_name=}, {model_name=}")  ## for debugging

            file_path = os.path.join(result_folder, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                if classify_func == classify_chars:
                    for text in df[target_column]:
                        char_stats = classify_chars(text)
                        total = sum(char_stats.values())
                        result = {"model": model_name, "total": total, **char_stats}
                        result_list.append(result)

                        for char, category in ((char, get_char_script(char)) for char in text):
                            category_dict.setdefault(model_name, {}).setdefault(category, []).append(char)
                elif classify_func == classify_tokens:
                    tokenizer = load_tokenizer(model_addr)
                    for text in df[target_column]:
                        result, category_items = classify_tokens(model_name, tokenizer, text)
                        result_list.append(result)

                        for category, items in category_items.items():
                            category_dict.setdefault(model_name, {}).setdefault(category, []).extend(items)

        result_df = pd.DataFrame(result_list)
        columns_order = ["model", "total"] + [col for col in result_df.columns if col not in ["model", "total"]]
        result_df = result_df.reindex(columns=columns_order, fill_value=0)
        result_df = result_df.groupby("model").sum().reset_index()

        if result_folder == "zero":
            zero_df = result_df
            zero_category_dict = category_dict
        else:
            few_df = result_df
            few_category_dict = category_dict

    merged_df = pd.merge(zero_df, few_df, on='model', suffixes=('_no', ''))
    merged_df.to_csv(f'analysis/{file_suffix}_classification.csv', index=False)

    with open(f'{file_suffix}_zero_category.json', 'w') as f:
        json.dump(zero_category_dict, f)
    with open(f'{file_suffix}_few_category.json', 'w') as f:
        json.dump(few_category_dict, f)

    print("\nClassification completed.")


def plot_data(merged_df, y_column, y_label, filename, yerr_column=None, percentage=False):
    plt.figure(figsize=(12, 6))

    # Set bar width
    bar_width = 0.35

    # Define x based on the number of entries
    x = np.arange(len(merged_df))

    # Expand x-axis range
    plt.xlim(-0.5, len(merged_df) - 0.5)

    # Bar graph for y_column_no
    plt.bar(x - bar_width/2, merged_df[y_column + '_no'] * (100 if percentage else 1), yerr=merged_df[yerr_column + '_no'] * (100 if percentage else 1) if yerr_column else None, capsize=2, width=bar_width, label='Zero shot', alpha=0.9, error_kw=dict(elinewidth=1.2, ecolor='cornflowerblue'))

    # Bar graph for y_column
    plt.bar(x + bar_width/2, merged_df[y_column] * (100 if percentage else 1), yerr=merged_df[yerr_column] * (100 if percentage else 1) if yerr_column else None, capsize=2, width=bar_width, label='Few shot', alpha=0.9, error_kw=dict(elinewidth=1.2, ecolor='orange'))

    # Define sections and their positions
    sections = ['Korean Monolingual', 'Multilingual', 'Continually pretrained on Korean', 'Not pretrained on Korean']
    section_colors = ['darkslategray', 'darkred', 'darkcyan', 'darkgreen', 'darkmagenta']
    section_positions = [6.5, 12.5, 18.5, 22.5]

    Korean_Monolingual = ["kogpt2-base-v2-125M", "ko-gpt-trinity-1.2B-v0.5", "polyglot-ko-1.3b", "polyglot-ko-3.8b", "polyglot-ko-5.8b", "polyglot-ko-12.8b", "kogpt-6B"]
    Multilingual = ["xglm-564M", "xglm-1.7B", "xglm-4.5B", "xglm-7.5B", "mGPT-1.3B", "mGPT-13B"]
    Continually_pretrained_on_Korean = ["Yi-Ko-6B", "llama-2-ko-7b", "open-llama-2-ko-7b", "llama-2-koen-13b", "OPEN-SOLAR-KO-10.7B", "SOLAR-KOEN-10.8B"]
    Not_pretrained_on_Korean = ["Yi-6B", "Llama-2-7b", "Llama-2-13b", "SOLAR-10.7B-v1.0"]

    # Calculate mean and standard deviation for each group (Zero-shot and Few-shot)
    group_means_zero = {}
    group_stds_zero = {}
    group_means_few = {}
    group_stds_few = {}
    for group, models in zip(sections, [Korean_Monolingual, Multilingual, Continually_pretrained_on_Korean, Not_pretrained_on_Korean]):
        group_data_zero = merged_df.loc[models, y_column + '_no'] * (100 if percentage else 1)
        group_data_few = merged_df.loc[models, y_column] * (100 if percentage else 1)
        group_means_zero[group] = group_data_zero.mean()
        group_stds_zero[group] = group_data_zero.std()
        group_means_few[group] = group_data_few.mean()
        group_stds_few[group] = group_data_few.std()

    for i, (section, color, end_position) in enumerate(zip(sections, section_colors, section_positions)):
        start_position = -0.5 if i == 0 else section_positions[i-1]
        plt.axvline(end_position, color='black', linestyle='--', linewidth=0.8)
        plt.text(end_position-0.1, plt.ylim()[1] * 0.99, section, color=color, ha='center', va='top', rotation=90, fontsize=11)

        # Display average lines for Zero-shot and Few-shot within the group interval
        plt.axhline(group_means_zero[section], xmin=(start_position + 0.5) / (len(merged_df)), xmax=(end_position + 0.5) / (len(merged_df)), color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
        plt.axhline(group_means_few[section], xmin=(start_position + 0.5) / (len(merged_df)), xmax=(end_position + 0.5) / (len(merged_df)), color='saddlebrown', linestyle='--', linewidth=1.5, alpha=0.7)

        # Display mean and standard deviation for Zero-shot and Few-shot as text
        plt.text(start_position + (end_position - start_position) / 2 - 0.15, group_means_zero[section] + 0.02, f"{group_means_zero[section]:.2f} ({group_stds_zero[section]:.2f})", color='darkblue', ha='right', va='bottom', fontsize=10)
        plt.text(start_position + (end_position - start_position) / 2 + 0.15, group_means_few[section] + 0.02, f"{group_means_few[section]:.2f} ({group_stds_few[section]:.2f})", color='saddlebrown', ha='left', va='bottom', fontsize=10)

        # Add error bars for standard deviation of Zero-shot and Few-shot to the plot
        plt.errorbar(start_position + (end_position - start_position) / 2 - 0.1, group_means_zero[section], yerr=group_stds_zero[section], color='darkblue', fmt='o', capsize=5, capthick=1.5, elinewidth=1.5)
        plt.errorbar(start_position + (end_position - start_position) / 2 + 0.1, group_means_few[section], yerr=group_stds_few[section], color='saddlebrown', fmt='o', capsize=5, capthick=1.5, elinewidth=1.5)

    # Add labels and legend
    plt.ylabel(y_label)
    plt.xlabel('Model')
    plt.xticks(x, merged_df.index, rotation=90)  # Rotate model names if they are long
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)

    # Display the graph
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def get_script(char):
    if char == '�':
        return "Undecodable"
    elif re.match(r'[\u0000-\u001F\u007F-\u009F]', char):
        return "Control"
    elif re.match(r'[^\w\s]', char):
        return "Symbol"
    else:
        try:
            script = unicodedata.name(char).split()[0]
        except ValueError:
            script = "Unknown_Name"
        return script

def get_token_script(token):
    if not token.strip(): # If the token consists only of whitespace characters
        return "Whitespace"
    else:
        scripts = set(get_script(char) for char in token)
        if "Unknown" in scripts:
            return "Unknown_Char"
        elif len(scripts) == 1:
            return scripts.pop()
        else:
            return "Mixed"

def classify_tokens(model_name, tokenizer, text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    result = {"model": model_name, "total": len(tokens)}
    category_tokens = {}  # Dictionary to store tokens by category
    decoded_tokens = []
    for token, token_id in zip(tokens, token_ids):
        if token_id is not None:  
            decoded_token = tokenizer.decode([token_id]).strip()  
            decoded_tokens.append(decoded_token)
            category = get_token_script(decoded_token)
            result[category] = result.get(category, 0) + 1
            category_tokens.setdefault(category, []).append(decoded_token)  # Store tokens by category
    return result, category_tokens

def get_char_script(char):
    if char.isspace():
        return "Whitespace"
    else:
        return get_script(char)

def classify_chars(text):
    result = {}
    for char in text:
        script = get_char_script(char)
        result[script] = result.get(script, 0) + 1
    return result

def plot_stacked_bar(merged_df, columns, total_column, legend_labels, filename, is_zero_shot=False):
    # Add '_no' suffix to the column names if it is a zero-shot setting
    if is_zero_shot:
        columns = [col + '_no' for col in columns]
        total_column += '_no'

    # Calculate the ratio of each column to the total column
    ratio_df = merged_df[columns].div(merged_df[total_column], axis=0)

    # Plot the stacked bar graph
    ax = ratio_df.plot(kind='bar', stacked=True, figsize=(12, 6))

    # Define sections and their positions
    sections = ['Korean Monolingual', 'Multilingual', 'Continued Pretrained on Korean', 'Pretrained not on Korean']
    section_colors = ['red', 'blue', 'darkgreen', 'maroon']
    section_positions = [6.5, 12.5, 18.5, 22.5]
    for i, (section, color, position) in enumerate(zip(sections, section_colors, section_positions)):
        ax.axvline(position, color='black', linestyle='--', linewidth=0.8)
        ax.text(position-0.1, ax.get_ylim()[1] * 0.99, section, color=color, ha='center', va='top', rotation=90, fontsize=10)

    # Add labels 
    ax.set_xlabel('Model')
    ax.set_ylabel('Ratio')
    plt.xticks(rotation=90, ha='right')

    # Add legend
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.45, -0.3), ncol=4)

    # Display the graph
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def split_and_merge_sentences(text):
    # Split the text into sentences
    sentences = kiwi.split_into_sents(text)
    merged_sentences = []
    only_merged_sentences = []
    
    for i in range(len(sentences)):
        sentence_text = sentences[i].text  # get string using Sentence object's text attribute
        # Merge the sentence with the previous sentence if it starts with "며 "
        if i > 0 and sentence_text.startswith("며 "):
            merged_sentence = merged_sentences[-1] + sentence_text 
            merged_sentences[-1] = merged_sentence # update the last element
            only_merged_sentences.append(merged_sentence) # add to the only_merged_sentences list
        else:
            merged_sentences.append(sentence_text) 
    
    return merged_sentences, sentences, only_merged_sentences

def split_and_save_sentences(folder, column, model_name):
    os.makedirs(f"{folder}_txt", exist_ok=True)
    os.makedirs(f"{folder}_txt/org_split_sentences", exist_ok=True)
    os.makedirs(f"{folder}_txt/only_merged_sentences", exist_ok=True)

    # file path for the original split sentences
    org_txt_file = f"{folder}_txt/org_split_sentences/{model_name}_org.txt"
    # file path for the merged sentences
    txt_file = f"{folder}_txt/{model_name}.txt"
    # file path for the merged sentences only
    merged_only_txt_file = f"{folder}_txt/only_merged_sentences/{model_name}_merged_only.txt"
    
    with open( org_txt_file, 'w', encoding='utf-8') as f, open(txt_file, 'w', encoding='utf-8') as merged_f, open(merged_only_txt_file, 'w', encoding='utf-8') as merged_only_f:
        file_path = os.path.join(folder, model_name + ".csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            for text in df[column]:
                merged_sentences, sentences, merged_only_sentences = split_and_merge_sentences(text)
                
                for sentence in sentences:
                    f.write(sentence.text.strip() + "\n") # write the original split sentences
                
                for sentence in merged_sentences:
                    merged_f.write(sentence.strip() + "\n") # write the merged sentences
                
                for sentence in merged_only_sentences:
                    merged_only_f.write(sentence.strip() + "\n") # write the merged sentences only

def calculate_sentence_stats(file_path, total_chars=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    num_sentences = len(sentences)
    avg_length = np.mean([len(sentence) for sentence in sentences])
    std_length = np.std([len(sentence) for sentence in sentences])
    avg_words = np.mean([len(sentence.split()) for sentence in sentences])
    std_words = np.std([len(sentence.split()) for sentence in sentences])
    if total_chars is not None:
        ratio = num_sentences / total_chars
        return num_sentences, ratio, avg_length, std_length, avg_words, std_words, total_chars
    else:
        return num_sentences, avg_length, std_length, avg_words, std_words

def plot_korean_ratio(df, plot_type, filename):
    # set the order and labels of the hue
    hue_order = ['zero_txt', 'few_txt']
    hue_labels = ['Zero-shot', 'Few-shot']

    plt.figure(figsize=(12, 6))

    # plot the data
    if plot_type == 'violin':
        ax = sns.violinplot(x="Model", y="Korean Ratio", hue="Folder", data=df, scale="width", split=True, hue_order=hue_order)
    elif plot_type == 'box':
        ax = sns.boxplot(x="Model", y="Korean Ratio", hue="Folder", data=df, hue_order=hue_order)
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Choose 'violin' or 'box'.")

    plt.xticks(rotation=90)
    plt.ylabel("Korean character ratio per sentence")

    # get the handles and labels from the figure
    handles, labels = ax.get_legend_handles_labels()

    # specify custom labels
    labels = hue_labels

    # re-create legend with correct labels and handles
    plt.legend(handles=handles, labels=labels, title='', loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)

    # add sections and their positions
    sections = ['Korean Monolingual', 'Multilingual', 'Continued Pretrained on Korean', 'Pretrained not on Korean']
    section_colors = ['red', 'blue', 'darkgreen', 'maroon']
    section_positions = [6.5, 12.5, 18.5, 22.5]

    for i, (section, color, position) in enumerate(zip(sections, section_colors, section_positions)):
        plt.axvline(position, color='black', linestyle='--', linewidth=0.8)
        plt.text(position-0.1, plt.ylim()[1] * 0.99, section, color=color, ha='center', va='top', rotation=90, fontsize=10)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def has_korean_character_ratio(sentence, ratio=0.4):
    if len(sentence) == 0:
        return False
    korean_chars = sum(0xAC00 <= ord(char) <= 0xD7A3 for char in sentence)
    return korean_chars / len(sentence) >= ratio

def has_unusual_pattern(sentence):
    # Email addresses
    email_pattern = re.compile(r'\S+@\S+\.\S+')

    # URLs
    url1_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    url2_pattern = re.compile(r'www\S+')
    url3_pattern = re.compile(r'\S+\.(?:com|co\.kr)')

    # news bylines
    news_byline_pattern = re.compile(r'([가-힣]+\s*기자\s*[=/])|(/\s*[가-힣]+\s*기자)|(=\s*[가-힣]+\s*기자)|(-\s*[가-힣]+\s*기자)|(\(\s*[가-힣]+\s*기자\s*\))')

    # consecutive special characters (4 or more)
    special_char_pattern = re.compile(r'[!@$%^&*()_+={}:;\"\'?/\\|`~\-]{4,}') # <>

    # date-format (YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD)
    date_pattern = re.compile(r'\d{1,4}[-/.\s]\d{1,2}[-/.\s]\d{1,4}')

    # time-format (HH:MM:SS, HH:MM)
    time_pattern = re.compile(r'\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?')

    # phone-number pattern (02-123-4567, 031-123-4567, 02-1234-5678, 031-1234-5678)
    phone_pattern = re.compile(r'(\(?\d{2,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4})')

    # html tags (without Korean characters inside to separate headlines)
    html_pattern = re.compile(r'<[^가-힣>]+>') 

    # ellipsis "..."
    ellipsis1_pattern = re.compile(r'\.{3}(?![.\s\"\'])') # ellipsis "..."

    # 'omit' in Korean
    ellipsis2_pattern = re.compile(r'중략') # 'omit' in Korean
    
    symbol_pattern = re.compile(r'[\[\]#=ⓒ©▷▶▲△▼▽◇◆■□◻●○•⊙:→☞�﻿◯◎▨㎫ㅣ|。ㆍ゜『』「」《》〈〉〝〟①②③④⑤❶❷❸❹❺➀➁➂➃➄㈀㈁㈂㈃㈄㈎㈏㈐㈑㈒]') 
    # |<(?!.*?>)|(?<!<.*?)> 

    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]')

    return bool(email_pattern.search(sentence) or
                url1_pattern.search(sentence) or url2_pattern.search(sentence) or url3_pattern.search(sentence) or 
                news_byline_pattern.search(sentence) or
                special_char_pattern.search(sentence) or
                date_pattern.search(sentence) or
                time_pattern.search(sentence) or
                phone_pattern.search(sentence) or
                html_pattern.search(sentence) or
                ellipsis1_pattern.search(sentence) or ellipsis2_pattern.search(sentence) or
                emoji_pattern.search(sentence) or
                symbol_pattern.search(sentence)
                )


def has_headline_pattern(sentence, headline):
    if headline == "all":
        headline_pattern = re.compile(r'<[^>]+>')  # "<x..xx>" in sentence
        if bool(headline_pattern.search(sentence)):
            return True
    elif headline == "part":
        if "<" in sentence or ">" in sentence:  # "<" or ">" in sentence
            return True
    return False

def remove_ending_pattern(sentence):
    # remove the parentheses and its contents if there is a parenthesis at the end of the sentence
    sentence = re.sub(r'\s*\([^)]*\)$', '', sentence.strip())
    # remove the part ending with a Korean consonant or vowel at the end of the sentence
    sentence = re.sub(r'[\u1100-\u1112\u1161-\u1175\u11A8-\u11C2]+$', '', sentence.strip())
    # remove all other symbols and spaces until a character or sentence ending punctuation (.!?) appears from the end of the sentence
    sentence = re.sub(r'[^.!?\w]+$', '', sentence.strip())
    return sentence.strip()

def remove_starting_pattern(sentence):
    # remove the parentheses and its contents if there is a parenthesis at the beginning of the sentence
    sentence = re.sub(r'^\s*\([^)]*\)', '', sentence.strip())
    # remove the number starting and followed by . or ) (must be followed. repetition allowed) with the following space
    sentence = re.sub(r'^\d+[.)]+\s*', '', sentence.strip())
    # remove the part starting with a Korean consonant or vowel at the beginning of the sentence
    sentence = re.sub(r'^[\u1100-\u1112\u1161-\u1175\u11A8-\u11C2]+', '', sentence.strip())
    # remove all other symbols and spaces until a number or character appears from the beginning of the sentence (\w includes numbers)
    sentence = re.sub(r'^[^\w]+', '', sentence.strip())
    return sentence.strip() 

def remove_extra_dots(sentence):
    # remove extra dots at the end of the sentence
    match = re.search(r'[.!?]{2,}$', sentence.strip())
    if match:
        sentence = re.sub(r'[.!?]{2,}$', match.group()[0], sentence.strip())
    return sentence.strip()

def preprocess_sentences(sentences):
    sentences = [remove_ending_pattern(sentence) for sentence in sentences]
    sentences = [remove_starting_pattern(sentence) for sentence in sentences]
    sentences = [remove_extra_dots(sentence) for sentence in sentences]
    return sentences

def complete_korean_sentence(sentence):
    # end with a sentence ending punctuation or common verb ending
    ending_pattern = re.compile(r'.*[.!?]$|.*(?:다|요|죠|네|까|라)$')
    return bool(ending_pattern.match(sentence))

def analyze_patterns(pattern_type, pattern_func, preprocess=False):
    folders = ["zero_txt", "few_txt"]
    for folder in folders:
        result_list = []
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder, filename)
                print(filepath)  ## debug
                with open(filepath, "r", encoding="utf-8") as file:
                    sentences = file.readlines()
                    sentences = [sentence.strip() for sentence in sentences] 
                    
                    if preprocess:
                        sentences = preprocess_sentences(sentences)
                    
                    total_sentences = len(sentences)
                    pattern_sentences = sum(pattern_func(sentence) for sentence in sentences)
                    result_list.append({"model": filename.replace(".txt", ""), 
                                        "total_sentences": total_sentences, 
                                        "pattern_sentences": pattern_sentences, 
                                        "pattern_ratio": pattern_sentences / total_sentences})

        result_df = pd.DataFrame(result_list)

        if folder == "zero_txt":
            zero_df = result_df
        elif folder == "few_txt":
            few_df = result_df

    merged_df = pd.merge(zero_df, few_df, on='model', suffixes=('_no', ''))
    merged_df.to_csv(f"analysis/{pattern_type}_pattern.csv", index=False)
    print("\nAnalysis completed.")


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def load_embeddings(input_folders, base_embedding_file):
    embedding_files = [base_embedding_file]
    # Create full file path by combining folder and file name for distinction
    for folder in input_folders:
        embedding_files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')])

    embeddings = []
    # Read all embedding files and stack them into one matrix
    for file in embedding_files:
        embedding = np.load(file)
        embeddings.append(embedding)
    embeddings = np.vstack(embeddings)
    # return the combined embeddings and the list of embedding files
    return embeddings, embedding_files

def plot_embeddings(embeddings_transformed, embedding_files, output_path):
    # Create subplots for each embedding file
    num_subplots = len(embedding_files)
    num_cols = 6
    num_rows = (num_subplots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 3*num_rows), squeeze=False)

    # Set color map for each embedding file
    cmap = cm.get_cmap('tab20b', len(embedding_files))
    colors = cmap(range(len(embedding_files)))

    # Plot the transformed embeddings for each embedding file
    for idx, (file, ax, color) in enumerate(zip(embedding_files, axes.flatten(), colors)):
        embedding = np.load(file)
        embedding_transformed_single = embeddings_transformed[idx*len(embedding):(idx+1)*len(embedding)]
        ax.scatter(embedding_transformed_single[:, 0], embedding_transformed_single[:, 1], color=color, alpha=0.7)
        # Modify the subplot title according to the file name
        if 'zero' in file:
            title_addition = ' (Zero)'
        elif 'few' in file:
            title_addition = ' (Few)'
        else:
            title_addition = ' (Original)'
        ax.set_title(os.path.basename(file).replace('.npy', '').replace('_1000','')+title_addition)

    # Remove empty subplots
    for idx in range(num_subplots, num_rows*num_cols):
        fig.delaxes(axes.flatten()[idx])

    # Adjust the layout and save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Visualization saved as '{output_path}'")
    plt.show()



def plot_data_with_corpora(merged_df, y_column, y_label, filename, yerr_column=None, percentage=False):
    plt.figure(figsize=(12, 6))

    # Set bar width
    bar_width = 0.35

    # Define x based on the number of entries
    x = np.arange(len(merged_df))

    # Expand x-axis range
    plt.xlim(-0.5, len(merged_df) - 0.5)

    # Bar graph for y_column_no
    plt.bar(x - bar_width/2, merged_df[y_column + '_no'] * (100 if percentage else 1), yerr=merged_df[yerr_column + '_no'] * (100 if percentage else 1) if yerr_column else None, capsize=2, width=bar_width, label='Zero shot', alpha=0.9, error_kw=dict(elinewidth=1.2, ecolor='cornflowerblue'))

    # Bar graph for y_column
    plt.bar(x + bar_width/2, merged_df[y_column] * (100 if percentage else 1), yerr=merged_df[yerr_column] * (100 if percentage else 1) if yerr_column else None, capsize=2, width=bar_width, label='Few shot', alpha=0.9, error_kw=dict(elinewidth=1.2, ecolor='orange'))

    # Define sections and their positions
    sections = ['Korean Corpus', 'Korean Monolingual', 'Multilingual', 'Continually pretrained on Korean', 'Not pretrained on Korean']
    section_colors = ['darkslategray', 'darkred', 'darkcyan', 'darkgreen', 'darkmagenta']
    section_positions = [1.5, 8.5, 14.5, 20.5, 24.5]

    Korean_Corpus_Treebank = ["corpus_spoken", "corpus_written"]
    Korean_Monolingual = ["kogpt2-base-v2-125M", "ko-gpt-trinity-1.2B-v0.5", "polyglot-ko-1.3b", "polyglot-ko-3.8b", "polyglot-ko-5.8b", "polyglot-ko-12.8b", "kogpt-6B"]
    Multilingual = ["xglm-564M", "xglm-1.7B", "xglm-4.5B", "xglm-7.5B", "mGPT-1.3B", "mGPT-13B"]
    Continually_pretrained_on_Korean = ["Yi-Ko-6B", "llama-2-ko-7b", "open-llama-2-ko-7b", "llama-2-koen-13b", "OPEN-SOLAR-KO-10.7B", "SOLAR-KOEN-10.8B"]
    Not_pretrained_on_Korean = ["Yi-6B", "Llama-2-7b", "Llama-2-13b", "SOLAR-10.7B-v1.0"]

    # Calculate mean and standard deviation for each group (Zero-shot and Few-shot)
    group_means_zero = {}
    group_stds_zero = {}
    group_means_few = {}
    group_stds_few = {}
    for group, models in zip(sections, [Korean_Corpus_Treebank, Korean_Monolingual, Multilingual, Continually_pretrained_on_Korean, Not_pretrained_on_Korean]):
        group_data_zero = merged_df.loc[models, y_column + '_no'] * (100 if percentage else 1)
        group_data_few = merged_df.loc[models, y_column] * (100 if percentage else 1)
        group_means_zero[group] = group_data_zero.mean()
        group_stds_zero[group] = group_data_zero.std()
        group_means_few[group] = group_data_few.mean()
        group_stds_few[group] = group_data_few.std()

    for i, (section, color, end_position) in enumerate(zip(sections, section_colors, section_positions)):
        start_position = -0.5 if i == 0 else section_positions[i-1]
        plt.axvline(end_position, color='black', linestyle='--', linewidth=0.8)
        plt.text(end_position-0.1, plt.ylim()[1] * 0.99, section, color=color, ha='center', va='top', rotation=90, fontsize=11)

        # Display average lines for Zero-shot and Few-shot within the group interval
        plt.axhline(group_means_zero[section], xmin=(start_position + 0.5) / (len(merged_df)), xmax=(end_position + 0.5) / (len(merged_df)), color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
        plt.axhline(group_means_few[section], xmin=(start_position + 0.5) / (len(merged_df)), xmax=(end_position + 0.5) / (len(merged_df)), color='saddlebrown', linestyle='--', linewidth=1.5, alpha=0.7)

        # Display mean and standard deviation for Zero-shot and Few-shot as text
        plt.text(start_position + (end_position - start_position) / 2 - 0.15, group_means_zero[section] + 0.02, f"{group_means_zero[section]:.2f} ({group_stds_zero[section]:.2f})", color='darkblue', ha='right', va='bottom', fontsize=10)
        plt.text(start_position + (end_position - start_position) / 2 + 0.15, group_means_few[section] + 0.02, f"{group_means_few[section]:.2f} ({group_stds_few[section]:.2f})", color='saddlebrown', ha='left', va='bottom', fontsize=10)

        # Add error bars for standard deviation of Zero-shot and Few-shot to the plot
        plt.errorbar(start_position + (end_position - start_position) / 2 - 0.1, group_means_zero[section], yerr=group_stds_zero[section], color='darkblue', fmt='o', capsize=5, capthick=1.5, elinewidth=1.5)
        plt.errorbar(start_position + (end_position - start_position) / 2 + 0.1, group_means_few[section], yerr=group_stds_few[section], color='saddlebrown', fmt='o', capsize=5, capthick=1.5, elinewidth=1.5)

    # Add labels and legend
    plt.ylabel(y_label)
    plt.xlabel('Model')
    plt.xticks(x, merged_df.index, rotation=90)  # Rotate model names if they are long
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)

    # Display the graph
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def is_korean(text):
    for char in text:
        if not (0xAC00 <= ord(char) <= 0xD7A3 or 0x1100 <= ord(char) <= 0x11FF or 0x3130 <= ord(char) <= 0x318F):
            return False
    return True

def morpheme_segmentation(text):
    tokens = []
    for token in kiwi.tokenize(text.strip()):
        if ' ' in token.form:
            subtokens = token.form.split()
            for subtoken in subtokens:
                if is_korean(subtoken):
                    tokens.append(subtoken)
        else:
            if is_korean(token.form):
                tokens.append(token.form)
    return tokens

def ttr(tokens):
    return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0

def msttr(tokens, segment_size=200):
    segments = [tokens[i:i+segment_size] for i in range(0, len(tokens), segment_size)]
    return sum(ttr(segment) for segment in segments) / len(segments)

def mattr(tokens, window_size=200):
    scores = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        scores.append(ttr(window))
    return sum(scores) / len(scores)

def rttr(tokens):
    return len(set(tokens)) / math.sqrt(len(tokens))

def cttr(tokens):
    types = len(set(tokens))
    tokens_count = len(tokens)
    return types / math.sqrt(2 * tokens_count)

def mtld(tokens, ttr_threshold=0.72):
    def _mtld(tokens, ttr_threshold, reverse=False):
        if reverse:
            tokens = list(reversed(tokens))
        terms = set()
        word_counter = 0
        factor_count = 0
        for token in tokens:
            word_counter += 1
            terms.add(token)
            ttr = len(terms) / word_counter
            if ttr <= ttr_threshold:
                word_counter = 0
                terms = set()
                factor_count += 1

        if word_counter > 0:
            factor_count += (1 - ttr) / (1 - ttr_threshold)

        if factor_count == 0:
            ttr = len(set(tokens)) / len(tokens)
            if ttr == 1:
                factor_count += 1
            else:
                factor_count += (1 - ttr) / (1 - ttr_threshold)

        return len(tokens) / factor_count

    forward_measure = _mtld(tokens, ttr_threshold, reverse=False)
    reverse_measure = _mtld(tokens, ttr_threshold, reverse=True)
    return (forward_measure + reverse_measure) / 2


def remove_only_punct_X_sentences(conllu_file_path, output_file_path, model_name, removed_sentences_file):
    with open(conllu_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile, \
         open(removed_sentences_file, 'a', encoding='utf-8') as removed_file:
        current_sentence = []
        sentences_to_write = []
        total_sentences = 0
        filtered_sentences = 0
        removed_sentences = []

        for line in infile:
            if line.strip() == "":
                total_sentences += 1
                if current_sentence and not all(len(token.split('\t')) > 3 and token.split('\t')[3] in ["PUNCT", "X"] for token in current_sentence if not token.startswith('#')):
                    sentences_to_write.append("\n".join(current_sentence) + "\n\n")
                    filtered_sentences += 1
                else:
                    removed_sentences.append("\n".join(current_sentence) + "\n\n")
                current_sentence = []
            
            else:
                current_sentence.append(line.strip())
                
        outfile.write("".join(sentences_to_write))
        removed_file.write(f"Model: {model_name}\n")
        removed_file.write("".join(removed_sentences))
        removed_file.write("\n")

    return model_name, total_sentences, filtered_sentences



def plot_ud_distribution_stacked_bar(csv_file, output_file, legend_title):
    # Load data from the CSV file
    df = pd.read_csv(csv_file, index_col='model')
    
    # Create a color map
    num_colors = len(df.columns)
    colors = plt.cm.tab20(range(num_colors))
    
    # Plot the stacked bar graph
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind='bar', stacked=True, ax=ax, color=colors)
    
    # Display sections
    sections = ['Korean Corpus/Treebank', 'Korean Monolingual', 'Multilingual', 'Continually pretrained on Korean', 'Not pretrained on Korean']
    section_colors = ['darkslategray', 'red', 'blue', 'darkgreen', 'maroon']
    section_positions = [2.5, 9.5, 15.5, 21.5, 25.5]
    
    for i, (section, color, position) in enumerate(zip(sections, section_colors, section_positions)):
        ax.axvline(position, color='black', linestyle='--', linewidth=0.8)
        ax.text(position-0.1, ax.get_ylim()[1] * 0.99, section, color=color, ha='center', va='top', rotation=90, fontsize=10)
    
    # Set labels and legend
    ax.set_xlabel('Model')
    ax.set_ylabel('Relative Frequency (%)')
    ax.legend(title=legend_title, bbox_to_anchor=(1, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show(fig)

def get_upos_distribution(folder_path, file_names):
    upos_relative_counts = {}  # dictionary to store relative frequencies
    for file in file_names:
        file_path = os.path.join(folder_path, file + ".conllu")  # add extension
        
        if os.path.isfile(file_path):  # check if the file exists
            with open(file_path, 'r', encoding='utf-8') as f:
                upos_counter = Counter()
                for line in f:
                    if line.startswith('#') or not line.strip(): # skip comments or empty lines
                        continue 
                    parts = line.strip().split('\t')
                    if len(parts) > 3:
                        upos = parts[3] # extract upos tag
                        upos_counter[upos] += 1
                total_tags = sum(upos_counter.values())
                upos_relative_counts[file] = {tag: count / total_tags * 100 for tag, count in upos_counter.items()}
        else:
            print(f"File not found: {file_path}")
    
    # Convert the dictionary to a DataFrame and reorder it in the order of the file list
    df = pd.DataFrame.from_dict(upos_relative_counts, orient='index').fillna(0).reindex(file_names)
    df.index.name = 'model'
    
    return df

def get_xpos_distribution(folder_path, file_names):
    xpos_relative_counts = {}
    for file in file_names:
        file_path = os.path.join(folder_path, file + ".conllu")  
        if os.path.isfile(file_path):  
            with open(file_path, 'r', encoding='utf-8') as f:
                xpos_counter = Counter()
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) > 4:
                        xpos = parts[4] # extract xpos tag
                        xpos_counter[xpos] += 1
                total_count = sum(xpos_counter.values())
                xpos_relative_counts[file] = {tag: (count / total_count) * 100 for tag, count in xpos_counter.items()}

        else:
            print(f"File not found: {file_path}")
    
    df = pd.DataFrame.from_dict(xpos_relative_counts, orient='index').fillna(0).reindex(file_names)
    df.index.name = 'model'
    
    # Add average relative frequency values for each tag
    df.loc['Average'] = df.mean()

    # Select the top 20 tags in descending order of the average value
    top_20_tags = df.loc['Average'].sort_values(ascending=False).head(20).index
    df_top_20 = df[top_20_tags]

    return df_top_20

def get_xpos_split_distribution(folder_path, file_names):
    xpos_relative_counts = {}
    for file in file_names:
        file_path = os.path.join(folder_path, file + ".conllu")  
        if os.path.isfile(file_path):  
            with open(file_path, 'r', encoding='utf-8') as f:
                xpos_counter = Counter()
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) > 4 and parts[4] != '_':
                        tags = parts[4].split('+') # extract and split xpos tags
                        for tag in tags:
                            xpos_counter[tag] += 1
                total_count = sum(xpos_counter.values())
                xpos_relative_counts[file] = {tag: (count / total_count) * 100 for tag, count in xpos_counter.items()}
        else:
            print(f"File not found: {file_path}")
    df = pd.DataFrame.from_dict(xpos_relative_counts, orient='index').fillna(0).reindex(file_names)
    df.index.name = 'model'
    
    df.loc['Average'] = df.mean()
    
    top_20_tags = df.loc['Average'].sort_values(ascending=False).head(20).index
    df_top_20 = df[top_20_tags]

    return df_top_20

def get_deprel_distribution(folder_path, file_names):
    deprel_relative_counts = {}
    for file in file_names:
        file_path = os.path.join(folder_path, file + ".conllu")  
        if os.path.isfile(file_path):  
            with open(file_path, 'r', encoding='utf-8') as f:
                deprel_counter = Counter()
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue  
                    parts = line.strip().split('\t')
                    if len(parts) > 7:
                        deprel = parts[7]  # extract deprel tag
                        deprel_counter[deprel] += 1
                total_count = sum(deprel_counter.values())
                deprel_relative_counts[file] = {tag: (count / total_count) * 100 for tag, count in deprel_counter.items()}
        else:
            print(f"File not found: {file_path}")
    
    df = pd.DataFrame.from_dict(deprel_relative_counts, orient='index').fillna(0).reindex(file_names)
    df.index.name = 'model'
    
    df.loc['Average'] = df.mean()
    
    top_20_tags = df.loc['Average'].sort_values(ascending=False).head(20).index
    df_top_20 = df[top_20_tags]
    
    return df_top_20

def get_dependency_arc_stats(folder_path, file_names):
    stats = []
    for file in file_names:
        file_path = os.path.join(folder_path, file + ".conllu")  
        if os.path.isfile(file_path):  
            with open(file_path, 'r', encoding='utf-8') as f:
                arc_lengths = []
                left_arcs = []
                right_arcs = []
                num_sentences = 0
                for line in f:
                    if line.startswith('#'):
                        if 'sent_id' in line:
                            num_sentences += 1
                        continue
                    if not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) > 6:
                        token_index = int(parts[0])
                        head_index = int(parts[6])
                        if head_index == 0:  # skip root
                            continue
                        arc_length = abs(head_index - token_index) # calculate the arc length
                        arc_lengths.append(arc_length) # add the arc length to the list
                        if token_index < head_index: # check if the token is on the left or right side of the head
                            left_arcs.append(arc_length) # add the arc length to the left arcs
                        elif token_index > head_index: # check if the token is on the left or right side of the head
                            right_arcs.append(arc_length) # add the arc length to the right arcs
            stats.append({
                'percent_left_arcs': len(left_arcs) / len(arc_lengths) * 100 if arc_lengths else 0,
                'percent_right_arcs': len(right_arcs) / len(arc_lengths) * 100 if arc_lengths else 0,
                'average_arc_length': np.mean(arc_lengths) if arc_lengths else 0,
                'average_left_arc_length': np.mean(left_arcs) if left_arcs else 0,
                'average_right_arc_length': np.mean(right_arcs) if right_arcs else 0,
                'std_dev_arc_length': np.std(arc_lengths) if arc_lengths else 0,
                'std_dev_left_arc_length': np.std(left_arcs) if left_arcs else 0,
                'std_dev_right_arc_length': np.std(right_arcs) if right_arcs else 0,
                'num_sentences': num_sentences
            })
        else:
            print(f"File not found: {file_path}")

    df = pd.DataFrame(stats, index=file_names)
    df.index.name = 'model'
    return df

def plot_ud_dependency_arc_stats(csv_file, output_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(16, 10))

    # 4 subplots: 1. Visualize the proportion of left vs right dependency arcs
    plt.subplot(2, 2, 1)
    merged_df = df.set_index('model')
    bar_width = 0.35
    x = np.arange(len(merged_df))
    plt.xlim(-0.5, len(merged_df) - 0.5)
    plt.bar(x - bar_width/2, merged_df['percent_left_arcs'], width=bar_width, label='Zero shot', alpha=0.9)
    plt.bar(x + bar_width/2, merged_df['percent_right_arcs'], width=bar_width, label='Few shot', alpha=0.9)

    sections = ['Korean Corpus/Treebank', 'Korean Monolingual', 'Multilingual', 'Continually pretrained on Korean', 'Not pretrained on Korean']
    section_colors = ['darkslategray', 'darkred', 'darkcyan', 'darkgreen', 'darkmagenta']
    section_positions = [2.5, 9.5, 15.5, 21.5, 25.5]

    Korean_Corpus_Treebank = ["corpus_spoken", "corpus_written", "ko_gsd-ud-TB"]
    Korean_Monolingual = ["kogpt2-base-v2-125M", "ko-gpt-trinity-1.2B-v0.5", "polyglot-ko-1.3b", "polyglot-ko-3.8b", "polyglot-ko-5.8b", "polyglot-ko-12.8b", "kogpt-6B"]
    Multilingual = ["xglm-564M", "xglm-1.7B", "xglm-4.5B", "xglm-7.5B", "mGPT-1.3B", "mGPT-13B"]
    Continually_pretrained_on_Korean = ["Yi-Ko-6B", "llama-2-ko-7b", "open-llama-2-ko-7b", "llama-2-koen-13b", "OPEN-SOLAR-KO-10.7B", "SOLAR-KOEN-10.8B"]
    Not_pretrained_on_Korean = ["Yi-6B", "Llama-2-7b", "Llama-2-13b", "SOLAR-10.7B-v1.0"]

    group_means_zero = {}
    group_stds_zero = {}
    group_means_few = {}
    group_stds_few = {}
    for group, models in zip(sections, [Korean_Corpus_Treebank, Korean_Monolingual, Multilingual, Continually_pretrained_on_Korean, Not_pretrained_on_Korean]):
        group_data_zero = merged_df.loc[models, 'percent_left_arcs'] 
        group_data_few = merged_df.loc[models, 'percent_right_arcs'] 
        group_means_zero[group] = group_data_zero.mean()
        group_stds_zero[group] = group_data_zero.std()
        group_means_few[group] = group_data_few.mean()
        group_stds_few[group] = group_data_few.std()

    for i, (section, color, end_position) in enumerate(zip(sections, section_colors, section_positions)):
        start_position = -0.5 if i == 0 else section_positions[i-1]
        plt.axvline(end_position, color='black', linestyle='--', linewidth=0.8)
        plt.text(end_position-0.1, plt.ylim()[1] * 0.99, section, color=color, ha='center', va='top', rotation=90, fontsize=11)
        plt.axhline(group_means_zero[section], xmin=(start_position + 0.5) / (len(merged_df)), xmax=(end_position + 0.5) / (len(merged_df)), color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
        plt.axhline(group_means_few[section], xmin=(start_position + 0.5) / (len(merged_df)), xmax=(end_position + 0.5) / (len(merged_df)), color='saddlebrown', linestyle='--', linewidth=1.5, alpha=0.7)
        plt.errorbar(start_position + (end_position - start_position) / 2 - 0.1, group_means_zero[section], yerr=group_stds_zero[section], color='darkblue', fmt='o', capsize=5, capthick=1.5, elinewidth=1.5)
        plt.errorbar(start_position + (end_position - start_position) / 2 + 0.1, group_means_few[section], yerr=group_stds_few[section], color='saddlebrown', fmt='o', capsize=5, capthick=1.5, elinewidth=1.5)

    plt.title('Left vs Right Dependency Arcs Proportion')
    plt.xticks(x, merged_df.index, rotation=90, fontsize=11)
    plt.ylabel('Proportion (%)')
    plt.xlabel('Model')

    # 4 subplots: 2. Visualize the average dependency arc length
    plt.subplot(2, 2, 2)
    sns.barplot(x='model', y='average_arc_length', yerr=df['std_dev_arc_length'], data=df)
    for i, (section, color, position) in enumerate(zip(sections, section_colors, section_positions)):
        plt.axvline(position, color='black', linestyle='--', linewidth=0.8)
        plt.text(position-0.1, plt.ylim()[1] * 0.99, section, color=color, ha='center', va='top', rotation=90, fontsize=10)

    plt.title('Average Dependency Arc Length with Std Dev')
    plt.xticks(rotation=90, fontsize=11)
    plt.ylabel('Average Length')
    plt.xlabel('Model')

    # 4 subplots: 3. Visualize the average left dependency arc length
    plt.subplot(2, 2, 3)
    sns.barplot(x='model', y='average_left_arc_length', yerr=df['std_dev_left_arc_length'], data=df)
    for i, (section, color, position) in enumerate(zip(sections, section_colors, section_positions)):
        plt.axvline(position, color='black', linestyle='--', linewidth=0.8)
        plt.text(position-0.1, plt.ylim()[1] * 0.99, section, color=color, ha='center', va='top', rotation=90, fontsize=10)

    plt.title('Average Left Dependency Arc Length with Std Dev')
    plt.xticks(rotation=90, fontsize=11)
    plt.ylabel('Average Length')
    plt.xlabel('Model')

    # 4 subplots: 4. Visualize the average right dependency arc length
    plt.subplot(2, 2, 4)
    sns.barplot(x='model', y='average_right_arc_length', yerr=df['std_dev_right_arc_length'], data=df)
    for i, (section, color, position) in enumerate(zip(sections, section_colors, section_positions)):
        plt.axvline(position, color='black', linestyle='--', linewidth=0.8)
        plt.text(position-0.1, plt.ylim()[1] * 0.99, section, color=color, ha='center', va='top', rotation=90, fontsize=10)

    plt.title('Average Right Dependency Arc Length with Std Dev')
    plt.xticks(rotation=90, fontsize=11)
    plt.ylabel('Average Length')
    plt.xlabel('Model')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def is_num_a(word_info):
    # Check if the word has the form '한' and is a numeric modifier
    return ('한' in word_info.get('form') and word_info.get('deprel') == 'nummod')

def is_plural_marker(word_info):
    # Check if the word has the form '들' and is a noun-deriving suffix
    return '들' in word_info.get('form') and 'xsn' in word_info.get('xpos', '').lower()

def is_det_the(word_info):
    # Check if the word has the form '그' and is a determiner
    return ('그' in word_info.get('form') and word_info.get('upos') == 'DET')

def get_word_info(sentence_lines, index):
    if 0 <= index < len(sentence_lines):
        columns = sentence_lines[index].split('\t')
        if len(columns) >= 8:
            return {
                'upos': columns[3],
                'xpos': columns[4],
                'feats': columns[5],
                'deprel': columns[7]
            }
    return {'upos': '', 'xpos': '', 'feats': '', 'deprel': ''}

def get_word_info_from_columns(columns):
    return {
        'id': columns[0],
        'form': columns[1],
        'lemma': columns[2],
        'upos': columns[3],
        'xpos': columns[4],
        'feats': columns[5],
        'head': columns[6],
        'deprel': columns[7]
    }

def add_feature_example(feature_list, feature_name, folder_name, file_name, sentence_text):
    # Add the sentence with the feature to the list
    feature_list.append({'Feature': feature_name, 'Folder': folder_name, 'File': file_name, 'Sentence': sentence_text})

def get_sentence_text(sentence_lines):
    sentence_text = []
    for line in sentence_lines:
        if not line.startswith("#"):
            parts = line.split('\t')
            if len(parts) > 1:
                sentence_text.append(parts[1])
    return ' '.join(sentence_text)

def translationese_features(file_path, feature_list, folder_name):
    num_a_count = 0
    plural_count = 0
    noun_count = 0
    det_the_count = 0
    passive_count = 0
    omitted_subject_count = 0
    omitted_object_count = 0
    omitted_both_count = 0
    sentence_count = 0
    word_count = 0
    
    obj_verb_order_count = 0
    verb_obj_order_count = 0
    obj_verb_sentence_count = 0
    sentence_text = ""
    sentence_lines = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        if not line.startswith("#") and line:
            sentence_lines.append(line)

        if line.startswith("# sent_id =") or i == len(lines) - 1 or not line:
            if sentence_lines:
                sentence_count += 1
                word_count += len(sentence_lines)
                sentence_text = get_sentence_text(sentence_lines)
                filename = os.path.basename(file_path)
                
                # Check for object-verb or verb-object order
                obj_index = -1
                verb_index = -1
                for j, sent_line in enumerate(sentence_lines):
                    curr_columns = sent_line.split('\t')
                    if len(curr_columns) < 9:
                        continue
                    curr_word_info = get_word_info_from_columns(curr_columns)
                    if curr_word_info.get('deprel') == 'obj':
                        obj_index = j
                    elif curr_word_info.get('upos') == 'VERB':
                        verb_index = j
                
                if obj_index != -1 and verb_index != -1: ## both object and verb exist
                    obj_verb_sentence_count += 1
                    if obj_index < verb_index:
                        obj_verb_order_count += 1
                        add_feature_example(feature_list, 'ObjVerb', folder_name, filename, sentence_text)
                    else:
                        verb_obj_order_count += 1
                        add_feature_example(feature_list, 'VerbObj', folder_name, filename, sentence_text)

                
                for j, sent_line in enumerate(sentence_lines):
                    curr_columns = sent_line.split('\t')
                    if len(curr_columns) < 9:
                        continue
                    
                    curr_word_info = get_word_info_from_columns(curr_columns)

                    if curr_word_info.get('upos') == 'NOUN':
                        noun_count += 1
                    # Check for article/plural translationese features
                    if is_num_a(curr_word_info):
                        add_feature_example(feature_list, 'Numeral', folder_name, filename, sentence_text)
                        num_a_count += 1
                    if is_plural_marker(curr_word_info):
                        add_feature_example(feature_list, 'Plural', folder_name, filename, sentence_text)
                        plural_count += 1
                    if is_det_the(curr_word_info):
                        add_feature_example(feature_list, 'The', folder_name, filename, sentence_text)
                        det_the_count += 1
                
                # Check for passive voice
                has_passive = any('pass' in word_info.split('\t')[7] for word_info in sentence_lines if len(word_info.split('\t')) > 7)
                if has_passive:
                    add_feature_example(feature_list, 'Passive', folder_name, filename, sentence_text)
                    passive_count += 1
                
                # Check for omitted subject or object
                has_subj = any('nsubj' in word_info.split('\t')[7] or 'SBJ' in word_info.split('\t')[7] 
                               for word_info in sentence_lines if len(word_info.split('\t')) > 7)
                has_obj = any('obj' in word_info.split('\t')[7] or 'OBJ' in word_info.split('\t')[7] 
                              for word_info in sentence_lines if len(word_info.split('\t')) > 7)
                if not has_subj:
                    add_feature_example(feature_list, 'NoSubj', folder_name, filename, sentence_text)
                    omitted_subject_count += 1
                if not has_obj:
                    add_feature_example(feature_list, 'NoObj', folder_name, filename, sentence_text)
                    omitted_object_count += 1
                if not has_subj and not has_obj:
                    add_feature_example(feature_list, 'NoSubjObj', folder_name, filename, sentence_text)
                    omitted_both_count += 1

                sentence_lines = []
                sentence_text = ""
 
    return (num_a_count, plural_count, det_the_count, passive_count, 
            omitted_subject_count, omitted_object_count, omitted_both_count,
            obj_verb_order_count, verb_obj_order_count, obj_verb_sentence_count,
            sentence_count, word_count, noun_count)


def plot_translationese_features(merged_df, feature, denominator, output_file, without_spoken_corpus=False):
    if without_spoken_corpus:
        merged_df = merged_df.drop('corpus_spoken')
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    x = np.arange(len(merged_df))

    # x-axis range extension
    plt.xlim(-0.5, len(merged_df) - 0.5)

    plt.bar(x - bar_width/2, merged_df[f'{feature}_no']/merged_df[f'{denominator}_no'] * 100, width=bar_width, label='Zero shot', alpha=0.9)
    plt.bar(x + bar_width/2, merged_df[feature]/merged_df[denominator] * 100, width=bar_width, label='Few shot', alpha=0.9)

    sections = ['Korean Corpus/Treebank', 'Korean Monolingual', 'Multilingual', 'Continually pretrained on Korean', 'Not pretrained on Korean']
    section_colors = ['darkslategray', 'darkred', 'darkcyan', 'darkgreen', 'darkmagenta']
    if without_spoken_corpus:
        section_positions = [1.5, 8.5, 14.5, 20.5, 24.5]
        Korean_Corpus_Treebank = ["corpus_written", "ko_gsd-ud-TB"]
    else:
        section_positions = [2.5, 9.5, 15.5, 21.5, 25.5]
        Korean_Corpus_Treebank = ["corpus_spoken", "corpus_written", "ko_gsd-ud-TB"]
    Korean_Monolingual = ["kogpt2-base-v2-125M", "ko-gpt-trinity-1.2B-v0.5", "polyglot-ko-1.3b", "polyglot-ko-3.8b", "polyglot-ko-5.8b", "polyglot-ko-12.8b", "kogpt-6B"]
    Multilingual = ["xglm-564M", "xglm-1.7B", "xglm-4.5B", "xglm-7.5B", "mGPT-1.3B", "mGPT-13B"]
    Continually_pretrained_on_Korean = ["Yi-Ko-6B", "llama-2-ko-7b", "open-llama-2-ko-7b", "llama-2-koen-13b", "OPEN-SOLAR-KO-10.7B", "SOLAR-KOEN-10.8B"]
    Not_pretrained_on_Korean = ["Yi-6B", "Llama-2-7b", "Llama-2-13b", "SOLAR-10.7B-v1.0"]

    # Calculate the mean and standard deviation of each group (Zero-shot and Few-shot)
    group_means_zero = {}
    group_stds_zero = {}
    group_means_few = {}
    group_stds_few = {}
    for group, models in zip(sections, [Korean_Corpus_Treebank, Korean_Monolingual, Multilingual, Continually_pretrained_on_Korean, Not_pretrained_on_Korean]):
        group_data_zero = merged_df.loc[models, f'{feature}_no'] / merged_df.loc[models, f'{denominator}_no'] * 100
        group_data_few = merged_df.loc[models, feature] / merged_df.loc[models, denominator] * 100
        group_means_zero[group] = group_data_zero.mean()
        group_stds_zero[group] = group_data_zero.std()
        group_means_few[group] = group_data_few.mean()
        group_stds_few[group] = group_data_few.std()

    for i, (section, color, end_position) in enumerate(zip(sections, section_colors, section_positions)):
        start_position = -0.5 if i == 0 else section_positions[i-1]
        plt.axvline(end_position, color='black', linestyle='--', linewidth=0.8)
        plt.text(end_position-0.1, plt.ylim()[1] * 0.99, section, color=color, ha='center', va='top', rotation=90, fontsize=11)

        # Display the average line of Zero-shot and Few-shot within the group
        plt.axhline(group_means_zero[section], xmin=(start_position + 0.5) / (len(merged_df)), xmax=(end_position + 0.5) / (len(merged_df)), color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
        plt.axhline(group_means_few[section], xmin=(start_position + 0.5) / (len(merged_df)), xmax=(end_position + 0.5) / (len(merged_df)), color='saddlebrown', linestyle='--', linewidth=1.5, alpha=0.7)

        # Display the average value and standard deviation of Zero-shot and Few-shot as text
        plt.text(start_position + (end_position - start_position) / 2 - 0.15, group_means_zero[section] + 0.02, f"{group_means_zero[section]:.2f} ({group_stds_zero[section]:.2f})", color='darkblue', ha='right', va='bottom', fontsize=10)
        plt.text(start_position + (end_position - start_position) / 2 + 0.15, group_means_few[section] + 0.02, f"{group_means_few[section]:.2f} ({group_stds_few[section]:.2f})", color='saddlebrown', ha='left', va='bottom', fontsize=10)

        # Display the standard deviation of Zero-shot and Few-shot as error bars on the plot
        plt.errorbar(start_position + (end_position - start_position) / 2 - 0.1, group_means_zero[section], yerr=group_stds_zero[section], color='darkblue', fmt='o', capsize=5, capthick=1.5, elinewidth=1.5)
        plt.errorbar(start_position + (end_position - start_position) / 2 + 0.1, group_means_few[section], yerr=group_stds_few[section], color='saddlebrown', fmt='o', capsize=5, capthick=1.5, elinewidth=1.5)

    plt.ylabel('Frequency (%)')
    plt.xlabel('Model')
    plt.xticks(x, merged_df.index, rotation=90, fontsize=11)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def plot_sentiment_classification(merged_df, output_file):
    plt.figure(figsize=(12, 12))
    bar_width = 0.35
    x = np.arange(len(merged_df))

    sections = ['Korean Corpus/Treebank', 'Korean Monolingual', 'Multilingual', 'Continually pretrained on Korean', 'Not pretrained on Korean']
    section_colors = ['darkslategray', 'darkred', 'darkcyan', 'darkgreen', 'darkmagenta']
    section_positions = [1.5, 8.5, 14.5, 20.5, 24.5]

    Korean_Corpus_Treebank = ["corpus_spoken", "corpus_written"]
    Korean_Monolingual = ["kogpt2-base-v2-125M", "ko-gpt-trinity-1.2B-v0.5", "polyglot-ko-1.3b", "polyglot-ko-3.8b", "polyglot-ko-5.8b", "polyglot-ko-12.8b", "kogpt-6B"]
    Multilingual = ["xglm-564M", "xglm-1.7B", "xglm-4.5B", "xglm-7.5B", "mGPT-1.3B", "mGPT-13B"]
    Continually_pretrained_on_Korean = ["Yi-Ko-6B", "llama-2-ko-7b", "open-llama-2-ko-7b", "llama-2-koen-13b", "OPEN-SOLAR-KO-10.7B", "SOLAR-KOEN-10.8B"]
    Not_pretrained_on_Korean = ["Yi-6B", "Llama-2-7b", "Llama-2-13b", "SOLAR-10.7B-v1.0"]

    for i, sentiment in enumerate(['Neu', 'Pos', 'Neg']):
        plt.subplot(3, 1, i+1)
        plt.xlim(-0.5, len(merged_df) - 0.5)
        plt.ylim(0, 70)
        
        plt.bar(x - bar_width/2, merged_df[f'{sentiment}_no'] * 100, width=bar_width, label='Zero shot', alpha=0.9)
        plt.bar(x + bar_width/2, merged_df[sentiment] * 100, width=bar_width, label='Few shot', alpha=0.9)
        
        group_means_zero = {group: merged_df.loc[models, f'{sentiment}_no'].mean() * 100 for group, models in zip(sections, [Korean_Corpus_Treebank, Korean_Monolingual, Multilingual, Continually_pretrained_on_Korean, Not_pretrained_on_Korean])}
        group_stds_zero = {group: merged_df.loc[models, f'{sentiment}_no'].std() * 100 for group, models in zip(sections, [Korean_Corpus_Treebank, Korean_Monolingual, Multilingual, Continually_pretrained_on_Korean, Not_pretrained_on_Korean])}
        group_means_few = {group: merged_df.loc[models, sentiment].mean() * 100 for group, models in zip(sections, [Korean_Corpus_Treebank, Korean_Monolingual, Multilingual, Continually_pretrained_on_Korean, Not_pretrained_on_Korean])}
        group_stds_few = {group: merged_df.loc[models, sentiment].std() * 100 for group, models in zip(sections, [Korean_Corpus_Treebank, Korean_Monolingual, Multilingual, Continually_pretrained_on_Korean, Not_pretrained_on_Korean])}

        for j, (section, color, end_position) in enumerate(zip(sections, section_colors, section_positions)):
            start_position = -0.5 if j == 0 else section_positions[j-1]
            plt.axvline(end_position, color='black', linestyle='--', linewidth=0.8)
            plt.text(end_position-0.1, plt.ylim()[1] * 0.95, section, color=color, ha='center', va='top', rotation=90, fontsize=11)
            
            plt.axhline(group_means_zero[section], xmin=(start_position + 0.5) / (len(merged_df)), xmax=(end_position + 0.5) / (len(merged_df)), color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
            plt.axhline(group_means_few[section], xmin=(start_position + 0.5) / (len(merged_df)), xmax=(end_position + 0.5) / (len(merged_df)), color='saddlebrown', linestyle='--', linewidth=1.5, alpha=0.7)
            
            plt.text(start_position + (end_position - start_position) / 2 - 0.15, group_means_zero[section] + 0.02, f"{group_means_zero[section]:.2f} ({group_stds_zero[section]:.2f})", color='darkblue', ha='right', va='bottom', fontsize=10)
            plt.text(start_position + (end_position - start_position) / 2 + 0.15, group_means_few[section] + 0.02, f"{group_means_few[section]:.2f} ({group_stds_few[section]:.2f})", color='saddlebrown', ha='left', va='bottom', fontsize=10)
            
            plt.errorbar(start_position + (end_position - start_position) / 2 - 0.1, group_means_zero[section], yerr=group_stds_zero[section], color='darkblue', fmt='o', capsize=5, capthick=1.5, elinewidth=1.5)
            plt.errorbar(start_position + (end_position - start_position) / 2 + 0.1, group_means_few[section], yerr=group_stds_few[section], color='saddlebrown', fmt='o', capsize=5, capthick=1.5, elinewidth=1.5)

        plt.ylabel('Frequency (%)')
        
        if sentiment == 'Neu':
            plt.title('Neutral', fontsize=14)
        elif sentiment == 'Pos':
            plt.title('Positive', fontsize=14)
        else:
            plt.title('Negative', fontsize=14)
        
        if i < 2:
            plt.xticks([], [])
        else:
            plt.xticks(x, merged_df.index, rotation=90, fontsize=11)
            plt.xlabel('Model')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=2, fontsize=12)
            
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()