"""
To generate texts using language models,
Refer to the code in src/main_analysis/text_generation.py.
"""
#%%
###########################################################################
# Remove input prompts from generated texts for few-shot texts
###########################################################################
import pandas as pd
import os
import re
from helpers import remove_pattern, remove_pattern_yiko6b

folder_path = 'few'
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file in files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    
    if file == 'Yi-Ko-6B.csv':
        df['only_generated_text'] = df['generated_text'].apply(remove_pattern_yiko6b)
    else:
        df['only_generated_text'] = df['generated_text'].apply(remove_pattern)
    
    df.to_csv(file_path, index=False)

print("Processing completed.")





#%%
###########################################################################
########################                        ###########################
########################    Basic statistics    ###########################
########################                        ###########################
###########################################################################

#%%
###########################################################################
# Length of the generated text 
###########################################################################
import pandas as pd
from helpers import load_tokenizer, calculate_stats, MODEL_PATHS

results_folders = ["zero", "few"]
result_dfs = []

for result_folder in results_folders:
    result_list = []
    target_column = "generated_text" if result_folder == "zero" else "only_generated_text"

    for model_name, file_name in MODEL_PATHS:
        print(f"{model_name=}, {file_name=}")  # debug
        file_path = os.path.join(result_folder, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            text_stats = calculate_stats(df, target_column, func=len)
            
            tokenizer = load_tokenizer(model_name)
            token_stats = calculate_stats(df, target_column, func=lambda x: len(tokenizer.encode(x)))

            result_list.append({
                "model": file_name.replace('.csv', ''),
                **dict(zip(['avg_text_length', 'std_text_length', 'min_text_length', 'max_text_length'], text_stats)),
                **dict(zip(['avg_token_count', 'std_token_count', 'min_token_count', 'max_token_count'], token_stats))
            })

    result_df = pd.DataFrame(result_list)
    result_dfs.append(result_df)

merged_df = pd.merge(result_dfs[0], result_dfs[1], on='model', suffixes=('_no', ''))
merged_df.to_csv('analysis/lengths_tokens.csv', index=False)

#%%
###########################################################################
# Visualization : text length
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/lengths_tokens.csv')

# Plot data as a bar plot : text length
# dataframe, target_column, y-axis lable, save_path, yerr_column=None, percentage=False
plot_data(merged_df, 'avg_text_length', 'Average Text Length', 'images/text_length.pdf', yerr_column='std_text_length')

#%%
###########################################################################
# Visualization : token count
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/lengths_tokens.csv')

# Plot data as a bar plot : token count
plot_data(merged_df, 'avg_token_count', 'Average Token Count', 'images/token_count.pdf', yerr_column='std_token_count')

#%%
###########################################################################
# Character classification
###########################################################################
import re
import pandas as pd
import json
from helpers import classify_chars, classify_and_save

# Classify characters and save the results
classify_and_save(classify_chars, ["zero", "few"], "char")

#%%
###########################################################################
# Visualization : character classification in zero-shot (statcked bar plot)
###########################################################################
from helpers import load_and_process_data, plot_stacked_bar

# Load and process data
merged_df = load_and_process_data('analysis/char_classification.csv')

# Plot data as a stacked bar plot : char classification in zero-shot
columns = ['HANGUL', 'LATIN', 'CJK', 'DIGIT', 'Symbol', 'Whitespace', 'Undecodable']
plot_stacked_bar(merged_df, columns, 'total', columns, 'images/char_ratio_zero.pdf', is_zero_shot=True)

#%%
###########################################################################
# Visualization : character classification in few-shot (statcked bar plot)
###########################################################################
from helpers import load_and_process_data, plot_stacked_bar

# Load and process data
merged_df = load_and_process_data('analysis/char_classification.csv')

# Plot data as a stacked bar plot : char classification in few-shot
columns = ['HANGUL', 'LATIN', 'CJK', 'DIGIT', 'Symbol', 'Whitespace', 'Undecodable']
plot_stacked_bar(merged_df, columns, 'total', columns, 'images/char_ratio_few.pdf', is_zero_shot=False)

#%%
###########################################################################
# Token classification
###########################################################################
import pandas as pd
import os
import re
import json
from helpers import classify_tokens, classify_and_save

# Classify tokens and save the results
classify_and_save(classify_tokens, ["zero", "few"], "token")

#%%
###########################################################################
# Visualization : token classification in zero-shot (statcked bar plot)
###########################################################################
from helpers import load_and_process_data, plot_stacked_bar

# Load and process data
merged_df = load_and_process_data('analysis/token_classification.csv')

# Plot data as a stacked bar plot : token classification in zero-shot
columns = ['HANGUL', 'LATIN', 'CJK', 'DIGIT', 'Symbol', 'Mixed', 'Whitespace', 'Undecodable']
plot_stacked_bar(merged_df, columns, 'total', columns, 'images/token_ratio_zero.pdf', is_zero_shot=True)

#%%
###########################################################################
# Visualization : token classification in few-shot (statcked bar plot)
###########################################################################
from helpers import load_and_process_data, plot_stacked_bar

# Load and process data
merged_df = load_and_process_data('analysis/token_classification.csv')

# Plot data as a stacked bar plot : token classification in few-shot
columns = ['HANGUL', 'LATIN', 'CJK', 'DIGIT', 'Symbol', 'Mixed', 'Whitespace', 'Undecodable']
plot_stacked_bar(merged_df, columns, 'total', columns, 'images/token_ratio_few.pdf', is_zero_shot=False)

#%%
###########################################################################
# Split sentences from the generated text
###########################################################################
import os
import pandas as pd
from kiwipiepy import Kiwi
from helpers import split_and_save_sentences, MODEL_PATHS

result_folders = ["zero", "few"]
columns = {"zero": "generated_text", "few": "only_generated_text"}

for folder in result_folders:
    column = columns[folder]
    for _, model_csv in MODEL_PATHS:
        split_and_save_sentences(folder, column, model_csv.replace('.csv', ''))

#%%
###########################################################################
# Calculate the length of sentences (text lenghth, word count)
###########################################################################
#%%
import os
import pandas as pd
import numpy as np
from helpers import calculate_sentence_stats

# Load the char classification data to get the total number of characters
char_classification_df = pd.read_csv("analysis/char_classification.csv", index_col=0)

result_folders = ["zero_txt", "few_txt"]
result_dfs = [] 

for folder in result_folders:
    result_list = []  
    
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            model_name = file_name.replace(".txt", "")
            file_path = os.path.join(folder, file_name)
            
            if folder == "zero_txt":
                total_chars = char_classification_df.loc[model_name, 'total_no']
            else:
                total_chars = char_classification_df.loc[model_name, 'total']
            stats = calculate_sentence_stats(file_path, total_chars)
            result_list.append({"model": model_name, **dict(zip(['num_sentences', 'ratio', 'avg_length', 'std_length', 'avg_words', 'std_words', 'total_chars'], stats))})  
                
    result_df = pd.DataFrame(result_list)
    result_dfs.append(result_df)

# Merge the results of zero-shot and few-shot
merged_df = pd.merge(result_dfs[0], result_dfs[1], on='model', suffixes=('_no', ''))

merged_df.to_csv("analysis/sentence_stats.csv")

#%%
###########################################################################
# Visualization : ratio of the number of sentences to the total length of the text
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/sentence_stats.csv')

# Visualization: number of sentences ratio
plot_data(merged_df, 'ratio', 'Number of sentences / Total length (%)', 'images/sentence_ratio.pdf', percentage=True)

#%%
###########################################################################
# Visualization : sentence length (text length)
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/sentence_stats.csv')

# Visualization: text length in a sentence
plot_data(merged_df, 'avg_length', 'Average Text Length', 'images/sentence_length.pdf', yerr_column='std_length')

#%%
###########################################################################
# Visualization : sentence length (number of words)
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/sentence_stats.csv')

# Visualization: number of words in a sentence
plot_data(merged_df, 'avg_words', 'Average number of words in a sentence', 'images/sentence_num_words.pdf', yerr_column='std_words') ##





#%%
###########################################################################
#######################                            ########################
#######################  Surface-level Evaluation  ########################
#######################                            ########################
###########################################################################


#%%
###########################################################################
# Korean character ratio per sentence
###########################################################################
import os
import pandas as pd
from matplotlib.lines import Line2D
from helpers import MODEL_ORDER

folders = ["zero_txt", "few_txt"]

data = []
for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            
            with open(filepath, "r", encoding="utf-8") as file:
                sentences = file.readlines()
            
            # calculate the ratio of Korean characters per sentence
            korean_ratios = []
            for sentence in sentences:
                sentence = sentence.strip()  # remove newline characters
                # (0xAC00 <= ord(char) <= 0xD7A3: Hangul Syllables, https://www.ssec.wisc.edu/~tomw/java/unicode.html)
                korean_chars = sum(0xAC00 <= ord(char) <= 0xD7A3 for char in sentence)
                korean_ratio = korean_chars / len(sentence) if len(sentence) > 0 else 0
                korean_ratios.append(korean_ratio)
            
            data.extend([(folder, filename.replace(".txt",""), ratio) for ratio in korean_ratios])

df = pd.DataFrame(data, columns=["Folder", "Model", "Korean Ratio"])

# convert the model column to a categorical type
df['Model'] = pd.Categorical(df['Model'], categories=MODEL_ORDER, ordered=True)
# sort the dataframe by the model order
df = df.sort_values('Model')
# replace the model names
df['Model'] = df['Model'].replace({"kogpt2-base-v2": "kogpt2-base-v2-125M", "mGPT": "mGPT-1.3B", "kogpt": "kogpt-6B", "Llama-2-7b-hf": "Llama-2-7b", "Llama-2-13b-hf": "Llama-2-13b"})

df.to_csv("analysis/korean_ratio.csv")
#%%
###########################################################################
# Visualization: Korean character ratio per sentence (violin plot)
###########################################################################
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
from helpers import plot_korean_ratio

# Load the data
df = pd.read_csv("analysis/korean_ratio.csv")

# Violin plot
plot_korean_ratio(df, plot_type='violin', filename='images/korean_ratio_violin.pdf')

#%%
###########################################################################
# Visualization: Korean character ratio per sentence (box plot)
###########################################################################
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from helpers import plot_korean_ratio

# Load the data
df = pd.read_csv("analysis/korean_ratio.csv")

# Box plot
plot_korean_ratio(df, plot_type='box', filename='images/korean_ratio_box.pdf')

#%%
###########################################################################
# Ratio of having unusual patterns in sentence
###########################################################################
import os
import pandas as pd
import re
from helpers import has_unusual_pattern, analyze_patterns

# analyze unusual patterns and save the results
analyze_patterns("unusual", has_unusual_pattern)

#%%
###########################################################################
# Visualization : ratio of having unusual patterns in sentence
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/unusual_pattern.csv')

# Visualization: unusual pattern ratio
plot_data(merged_df, 'pattern_ratio', 'Proportion (%)', 'images/unusual_pattern_ratio.pdf', percentage=True)

#%%
###########################################################################
# Ratio of having headline patterns in sentence
###########################################################################
#%%
from helpers import has_headline_pattern, analyze_patterns

# analyze headline patterns (all, part) and save the results
headlines = ["all", "part"]
for headline in headlines:
    analyze_patterns(f"headline_{headline}", lambda sentence: has_headline_pattern(sentence, headline))

#%%
###########################################################################
# Visualization : ratio of having headline patterns in sentence (all)
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/headline_all_pattern.csv')

# Visualization: headline pattern ratio (all)
plot_data(merged_df, 'pattern_ratio', 'Proportion (%)', 'images/headline_all_pattern_ratio.pdf', percentage=True)

#%%
###########################################################################
# Visualization : ratio of having headline patterns in sentence (part)
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/headline_part_pattern.csv')

# Visualization: headline pattern ratio (part) 
plot_data(merged_df, 'pattern_ratio', 'Proportion (%)', 'images/headline_part_pattern_ratio.pdf', percentage=True)

#%%
###########################################################################
# Ratio of having sentence completion patterns in sentence
###########################################################################
import pandas as pd
from helpers import complete_korean_sentence, analyze_patterns

# analyze sentence completion patterns and save the results
analyze_patterns("sentence_completion", complete_korean_sentence, preprocess=True)

# add 'non_pattern_ratio' columns for the sentence completion pattern
df = pd.read_csv('analysis/sentence_completion_pattern.csv')
df['non_pattern_ratio_no'] = 1 - df['pattern_ratio_no']
df['non_pattern_ratio'] = 1 - df['pattern_ratio']
# reorder the columns
df = df[['model', 'total_sentences_no', 'pattern_sentences_no', 'pattern_ratio_no', 'non_pattern_ratio_no',
         'total_sentences', 'pattern_sentences', 'pattern_ratio', 'non_pattern_ratio']]
# save the results
df.to_csv('analysis/sentence_completion_pattern.csv', index=False)

#%%
###########################################################################
# visualization : ratio of sentence completion patterns in sentence 
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/sentence_completion_pattern.csv')

# Visualization: headline pattern ratio (part) 
plot_data(merged_df, 'non_pattern_ratio', 'Proportion (%)', 'images/sentence_incompletion_ratio.pdf', percentage=True)

#%%
###########################################################################
# Entire filtering and saving of filtered sentences
###########################################################################
import os
import pandas as pd
from helpers import has_korean_character_ratio, has_unusual_pattern, has_headline_pattern, remove_ending_pattern, remove_starting_pattern, remove_extra_dots, preprocess_sentences, complete_korean_sentence

folders = ["zero_txt", "few_txt"]
output_folders = ["zero_comp", "few_comp"]

for folder, output_folder in zip(folders, output_folders):
    os.makedirs(output_folder, exist_ok=True)
    result_list = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                sentences = file.readlines()
                original_sentences = [sentence.strip() for sentence in sentences]
                total_sentences = len(original_sentences)
                
                # Step 1: Extract sentences with a Korean character ratio of 0.4 or higher
                step1_sentences = [sentence for sentence in original_sentences if has_korean_character_ratio(sentence)]
                
                # Step 2: Extract sentences that do not contain unusual patterns
                step2_sentences = [sentence for sentence in step1_sentences if not has_unusual_pattern(sentence)]
                
                # Step 3: Extract sentences that do not contain headline patterns (< or >)
                step3_sentences = [sentence for sentence in step2_sentences if not has_headline_pattern(sentence, headline="part")]
                
                # Step 4: Extract sentences that have a common Korean sentence ending after cleaning the symbols before and after the sentence
                pre_step4_sentences = preprocess_sentences(step3_sentences) # Clean the symbols before and after the sentence
                step4_sentences = [sentence for sentence in pre_step4_sentences if complete_korean_sentence(sentence)] 
                
                # Save the finally filtered sentences to each file
                output_filepath = os.path.join(output_folder, filename)
                with open(output_filepath, "w", encoding="utf-8") as file:
                    file.write("\n".join(step4_sentences))
                
                result_list.append({"model": filename.replace(".txt", ""),
                                    "total_sentences": total_sentences,
                                    "step1_sentences": len(step1_sentences),
                                    "step2_sentences": len(step2_sentences),
                                    "step3_sentences": len(step3_sentences),
                                    "pre_step4_sentences": len(pre_step4_sentences),
                                    "step4_sentences": len(step4_sentences),
                                    "final_ratio": len(step4_sentences) / total_sentences})
    
    result_df = pd.DataFrame(result_list)
    if folder == "zero_txt":
        zero_df = result_df
    elif folder == "few_txt":
        few_df = result_df

merged_df = pd.merge(zero_df, few_df, on='model', suffixes=('_no',''))

merged_df.to_csv("analysis/complete_sentences_filtering_all.csv", index=False)

# %%
###########################################################################
# Visualization : ratio of sentences filtered as complete sentences at the surface-level
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/complete_sentences_filtering_all.csv')

# Visualization: headline pattern ratio (part) 
plot_data(merged_df, 'final_ratio', 'Proportion (%)', 'images/complete_sentences_filtering_all.pdf', percentage=True)





#%%
###########################################################################
########################                           ########################
########################    Semantic Evaluation    ########################
########################                           ########################
###########################################################################

#%%
"""
To create semantic embeddings for the generated texts
Refer to the code in src/main_analysis/semantic_embedding.py.
"""
# %%
###########################################################################
# Calculate the cosine similarity between the original and generated embeddings
###########################################################################
import numpy as np
import pandas as pd
from helpers import cosine_similarity

# load the original embedding
original_embedding = np.load('article_embedding/articles_1000.npy')

# dictionary to store the dataframes for each folder
dfs = {}

for folder in ['zero_embedding', 'few_embedding']:
    models = []
    similarities = []
    
    # process each model
    for file in os.listdir(folder):
        if file.endswith('.npy'):
            model_embedding = np.load(os.path.join(folder, file))
            # calculate the average cosine similarity between the original and model embeddings
            similarity = np.mean([cosine_similarity(original_embedding[i], model_embedding[i]) for i in range(len(original_embedding))])
            models.append(file.replace('.npy', ''))
            similarities.append(similarity)
    
    # create a dataframe for the folder
    df = pd.DataFrame({'model': models, 'similarity': similarities})
    # sort the dataframe by the similarity
    df.sort_values(by='similarity', ascending=False, inplace=True)
    # add the dataframe to the dictionary
    dfs[folder] = df

merged_df = pd.merge(dfs['zero_embedding'], dfs['few_embedding'], on='model', suffixes=('_no', ''))

merged_df.to_csv('analysis/sentence_embedding_similarity.csv', index=False)
#%%
###########################################################################
# visualization : sentence embedding similarity
###########################################################################
from helpers import load_and_process_data, plot_data

# Load and process data
merged_df = load_and_process_data('analysis/sentence_embedding_similarity.csv')

# Visualization: headline pattern ratio (part) 
plot_data(merged_df, 'similarity', 'Cosine similarity score', 'images/sentence_embedding_similarity.pdf')
# %%
###########################################################################
# visualization: sentence embedding distribution
###########################################################################
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from helpers import load_embeddings, plot_embeddings

input_folders = ['zero_embedding', 'few_embedding']
base_embedding_file = 'article_embedding/articles_1000.npy'
embeddings, embedding_files = load_embeddings(input_folders, base_embedding_file)
print(embeddings.shape) ## debug
print(embedding_files) ## debug

# PCA
pca = PCA(n_components=2, random_state=2024)
embeddings_pca = pca.fit_transform(embeddings)
output_path = 'images/pca_visualization_subplots.pdf'
plot_embeddings(embeddings_pca, embedding_files, output_path)

# t-SNE
tsne = TSNE(n_components=2, random_state=2024)
embeddings_tsne = tsne.fit_transform(embeddings)
output_path = 'images/tsne_visualization_subplots.pdf'
plot_embeddings(embeddings_tsne, embedding_files, output_path)

# UMAP
umap = umap.UMAP(n_components=2, random_state=2024)
embeddings_umap = umap.fit_transform(embeddings)
output_path = 'images/umap_visualization_subplots.pdf'
plot_embeddings(embeddings_umap, embedding_files, output_path)

# %%