from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm
import os
import pandas as pd

# # Load model and tokenizer
model_name = "dudcjs2779/sentiment-analysis-with-klue-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sentiment classification function
def classify_sentiment(file_path, model, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    class_counts = {0: 0, 1: 0, 2: 0}

    for line in tqdm(lines, desc=f"Analyzing {os.path.basename(file_path)}"):
        inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        class_counts[predicted_class_id] += 1

    total = sum(class_counts.values())
    relative_counts = {k: v / total for k, v in class_counts.items()}
    
    return class_counts, relative_counts

# Set folder paths and output file paths
folder_paths = ['zero_comp', 'few_comp']
file_names = ["corpus_spoken", "corpus_written", 
              "kogpt2-base-v2", "ko-gpt-trinity-1.2B-v0.5", 
              "polyglot-ko-1.3b", "polyglot-ko-3.8b", "polyglot-ko-5.8b", "polyglot-ko-12.8b", "kogpt", 
              "xglm-564M", "xglm-1.7B", "xglm-4.5B", "xglm-7.5B", "mGPT", "mGPT-13B", 
              "Yi-Ko-6B", "llama-2-ko-7b", "open-llama-2-ko-7b", "llama-2-koen-13b", "OPEN-SOLAR-KO-10.7B", "SOLAR-KOEN-10.8B", 
              "Yi-6B", "Llama-2-7b-hf", "Llama-2-13b-hf", "SOLAR-10.7B-v1.0"]
result_dfs = []

# Get sentiment classification results
for folder_path in folder_paths:
    results = []
    for file in tqdm(file_names, desc=f"Processing files in {folder_path}"):
        file_path = os.path.join(folder_path, file + '.txt')
        if os.path.isfile(file_path):
            absolute_freq, relative_freq = classify_sentiment(file_path, model, tokenizer)
            results.append([file, absolute_freq[0], absolute_freq[1], absolute_freq[2], relative_freq[0], relative_freq[1], relative_freq[2]])
        else:
            print(f"File not found: {file_path}")

    result_df = pd.DataFrame(results, columns=['File', 'Neutral (Absolute)', 'Positive (Absolute)', 'Negative (Absolute)', 'Neutral (Relative)', 'Positive (Relative)', 'Negative (Relative)'])
    result_dfs.append(result_df)

# Merge dataframes
merged_df = pd.merge(result_dfs[0], result_dfs[1], on='model', suffixes=('_no', ''))

# Save to CSV
merged_df.to_csv('ko_analysis/sentiment_classification.csv', index=False)

print("completed.")