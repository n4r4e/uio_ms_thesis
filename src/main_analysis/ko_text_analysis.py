#%%
###########################################################################
# Correct spacing in SOLAR-KOEN-10.8B generated texts
###########################################################################
#%%
from kiwipiepy import Kiwi

kiwi = Kiwi()

# Allow up to 2 spaces in morphemes during spacing correction
kiwi.space_tolerance = 2

# Input folder and file name
input_folders = ["zero_comp", "few_comp"]
input_filename = "SOLAR-KOEN-10.8B_org.txt"

# Spacing correction for each input folder
for input_folder in input_folders:
    # Input file path
    input_file = os.path.join(input_folder, input_filename)
    
    # Output file path
    output_file = os.path.join(input_folder, input_filename.replace("_org", ""))
    
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()
    
    # Split by sentence
    sentences = text.split("\n")
    
    # Spacing correction for each sentence
    corrected_sentences = []
    for sentence in sentences:
        corrected_sentence = kiwi.space(sentence, reset_whitespace=False)
        corrected_sentences.append(corrected_sentence)
    
    # Write corrected sentences to output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(corrected_sentences))

print("Spacing correction is complete.")
#%%
"""
To compare with natural language
Add Korean corpora sentences into 'zero_comp' and 'few_comp' folders.
To get Korean sentences from Korean corpora, 
Refer the code in src/main_analysis/corpus_processing.py. 
"""
#%%
###########################################################################
# Calculate sentence length
###########################################################################
#%%
import os
import pandas as pd
from helpers import calculate_sentence_stats

# Input folder path
input_folders = ["zero_comp", "few_comp"]
result_dfs = []

for input_folder in input_folders:
    result_list = []
    # Get a list of .txt files in the folder
    files = [file for file in os.listdir(input_folder) if file.endswith(".txt")]
    
    # For each .txt file
    for file in files:
        print(file) ## fog dubugging
        file_path = os.path.join(input_folder, file)
        
        # Calculate sentence statistics
        stats = calculate_sentence_stats(file_path)
        
        # Add the results to the list
        result_list.append({
            "model": file.replace('.txt', ''),
            **dict(zip(['num_sentences', 'mean_length', 'std_length', 'mean_words', 'std_words'], stats))
        })
    
    result_df = pd.DataFrame(result_list)
    result_dfs.append(result_df)

# Merge dataframes
merged_df = pd.merge(result_dfs[0], result_dfs[1], on='model', suffixes=('_no', ''))
merged_df.to_csv('ko_analysis/sentence_lengths_words.csv', index=False)

#%%
###########################################################################
# visualization : Number of sentences
###########################################################################
from helpers import load_and_process_data, plot_data_with_corpora

# Load and process data
merged_df = load_and_process_data('ko_analysis/sentence_lengths_words.csv', has_corpora=True)

# Plot data as a bar plot : number of sentences
# dataframe, target_column, y-axis lable, save_path, yerr_column=None, percentage=False
plot_data_with_corpora(merged_df, 'num_sentences', 'Number of sentences', 'ko_images/num_sentence.pdf')

#%%
###########################################################################
# visualization : Average text length per sentence
###########################################################################
from helpers import load_and_process_data, plot_data_with_corpora

# Load and process data
merged_df = load_and_process_data('ko_analysis/sentence_lengths_words.csv', has_corpora=True)

# Plot data as a bar plot : text length
plot_data_with_corpora(merged_df, 'mean_length', 'Average sentence length', 'ko_images/sentence_length.pdf', yerr_column='std_length')

#%%
###########################################################################
# visualization : Average number of words per sentence
###########################################################################
from helpers import load_and_process_data, plot_data_with_corpora

# Load and process data
merged_df = load_and_process_data('ko_analysis/sentence_lengths_words.csv')

# Plot data as a bar plot : number of words
plot_data_with_corpora(merged_df, 'mean_words', 'Average number of words per sentence', 'ko_images/sentence_words.pdf', yerr_column='std_words')





#%%
###########################################################################
#########################                        ##########################
#########################   Lexical Evaluation   ##########################
#########################                        ##########################
###########################################################################

#%%
###########################################################################
# Lexical diversity original
###########################################################################
import os
import pandas as pd
from tqdm import tqdm
from helpers import morpheme_segmentation, ttr, msttr, mattr, rttr, cttr, mtld

input_folders = ["zero_comp", "few_comp"]
result_dfs = []

# Create a subfolder for unique tokens
os.makedirs('unique_tokens', exist_ok=True)

for folder in input_folders:
    results = []
    
    for filename in tqdm(os.listdir(folder), desc=f'Processing {folder}'):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                tokens = morpheme_segmentation(text)
                unique_tokens = set(tokens)
                
                result = {
                    'model': filename.replace(".txt",""),
                    'Total Tokens': len(tokens),
                    'Unique Tokens': len(unique_tokens),
                    'TTR': ttr(tokens),
                    'MSTTR': msttr(tokens),
                    'MATTR': mattr(tokens),
                    'RTTR': rttr(tokens),
                    'CTTR': cttr(tokens),
                    'MTLD': mtld(tokens),
                }
                results.append(result)
                
                # Save unique tokens to a file in the subfolder
                unique_tokens_filename = f"{os.path.basename(folder)}_{os.path.splitext(filename)[0]}_unique_tokens.txt"
                unique_tokens_path = os.path.join('ko_analysis/lexical_unique_tokens', unique_tokens_filename)
                with open(unique_tokens_path, 'w', encoding='utf-8') as unique_tokens_file:
                    unique_tokens_file.write('\n'.join(unique_tokens))

    result_df = pd.DataFrame(results)
    result_dfs.append(result_df)

# Merge dataframes
merged_df = pd.merge(result_dfs[0], result_dfs[1], on='model', suffixes=('_no', ''))

# Save merged dataframe
merged_df.to_csv('ko_analysis/lexical_diversity_no_dict.csv', index=False)
#%%

###########################################################################
# Lexical diversity only considering tokens included in the dictionary morpheme set
###########################################################################
#%%
"""
To create a morpheme set from a Korean dictionary,
Refer to the code in src/main_analysis/dictionary_processing.py
"""
#%%
import os
import pandas as pd
from tqdm import tqdm
from helpers import morpheme_segmentation, ttr, msttr, mattr, rttr, cttr, mtld

dictionary_tokens = set(pd.read_csv('20240501/urimalsam_morpheme_set.csv', squeeze=True))
len(dictionary_tokens)
#%%
input_folders = ["zero_comp", "few_comp"]
result_dfs = []

# Create subfolders for unique tokens and excluded tokens
os.makedirs('ko_analysis/lexical_unique_tokens_with_dict', exist_ok=True)
os.makedirs('ko_analysis/lexical_excluded_tokens_with_dict', exist_ok=True)

for folder in input_folders:
    results = []
    
    for filename in tqdm(os.listdir(folder), desc=f'Processing {folder}'):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                tokens = morpheme_segmentation(text)
                
                # Filter tokens based on dictionary
                filtered_tokens = [token for token in tokens if token in dictionary_tokens]
                excluded_tokens = set(tokens) - set(filtered_tokens)
                
                unique_tokens = set(filtered_tokens)
                
                result = {
                    'model': filename.replace(".txt",""),
                    'Total Tokens': len(filtered_tokens),
                    'Unique Tokens': len(unique_tokens),
                    'TTR': ttr(filtered_tokens),
                    'MSTTR': msttr(filtered_tokens),
                    'MATTR': mattr(filtered_tokens),
                    'RTTR': rttr(filtered_tokens),
                    'CTTR': cttr(filtered_tokens),
                    'MTLD': mtld(filtered_tokens),
                }
                results.append(result)
                
                # Save unique tokens to a file in the subfolder
                unique_tokens_filename = f"{os.path.basename(folder)}_{os.path.splitext(filename)[0]}_unique_tokens.txt"
                unique_tokens_path = os.path.join('ko_analysis/lexical_unique_tokens_with_dict', unique_tokens_filename)
                with open(unique_tokens_path, 'w', encoding='utf-8') as unique_tokens_file:
                    unique_tokens_file.write('\n'.join(unique_tokens))
                
                # Save excluded tokens to a file in the subfolder
                excluded_tokens_filename = f"{os.path.basename(folder)}_{os.path.splitext(filename)[0]}_excluded_tokens.txt"
                excluded_tokens_path = os.path.join('ko_analysis/lexical_excluded_tokens_with_dict', excluded_tokens_filename)
                with open(excluded_tokens_path, 'w', encoding='utf-8') as excluded_tokens_file:
                    excluded_tokens_file.write('\n'.join(excluded_tokens))

    result_df = pd.DataFrame(results)
    result_dfs.append(result_df)

# Merge dataframes
merged_df = pd.merge(result_dfs[0], result_dfs[1], on='model', suffixes=('_no', ''))

# Save merged dataframe
merged_df.to_csv('ko_analysis/lexical_diversity_dict.csv', index=False)
#%%

###########################################################################
# Visualization : Lexical diversity original
###########################################################################
from helpers import load_and_process_data, plot_data_with_corpora

# Load and process data
merged_df = load_and_process_data('ko_analysis/lexical_diversity_no_dict.csv', has_corpora=True)

metrics = ['TTR', 'MSTTR', 'MATTR', 'RTTR', 'CTTR', 'MTLD']
for metric in metrics:
    plot_data_with_corpora(merged_df, metric, metric, f'ko_images/{metric}_no_dict.pdf')

#%%
###########################################################################
# Visualization : Lexical diversity of tokens included in the dictionary morpheme set
###########################################################################
from helpers import load_and_process_data, plot_data_with_corpora

# Load and process data
merged_df = load_and_process_data('ko_analysis/lexical_diversity_dict.csv', has_corpora=True)

metrics = ['TTR', 'MSTTR', 'MATTR', 'RTTR', 'CTTR', 'MTLD']
for metric in metrics:
    plot_data_with_corpora(merged_df, metric, metric, f'ko_images/{metric}_dict.pdf')





#%%
###########################################################################
#######################                          ##########################
#######################   Syntactic Evaluation   ##########################
#######################                          ##########################
###########################################################################


#%%
"""
To get conllu format files for corpora and generated sentences,
Use https://lindat.mff.cuni.cz/services/udpipe/ (Use Korean-GSD model)
Save the conllu format files for corpora and generated sentences in the 'conllus_org' folders.
"""
#%%
###########################################################################
# Remove sentences consisting only of 'punct' and 'X' UD tags from UD result conllu files
###########################################################################
#%%
import os
import pandas as pd
from helpers import remove_only_punct_X_sentences

folder_names = ['zero', 'few']

for folder_name in folder_names:

    # Folder to save the results
    input_folder = f'{folder_name}_conllus_org'
    output_folder = f'{folder_name}_conllus'
    removed_sentences_file = f'ko_analysis/{folder_name}_conllus_org_removed_sentences.txt'

    os.makedirs(output_folder, exist_ok=True)

    # Initialize the file to save the removed sentences
    with open(removed_sentences_file, 'w', encoding='utf-8') as file:
        pass

    results = []

    for conllu_file in os.listdir(input_folder):
        conllu_file_path = os.path.join(input_folder, conllu_file)
        output_file_path = os.path.join(output_folder, conllu_file)
        model_name = os.path.splitext(conllu_file)[0]

        model_name, total_sentences, filtered_sentences = remove_only_punct_X_sentences(conllu_file_path, output_file_path, model_name, removed_sentences_file)
        results.append({'Model': model_name, 'Original Sentences': total_sentences, 'Filtered Sentences': filtered_sentences})

    df = pd.DataFrame(results)
    df.to_csv(f'ko_analysis/{folder_name}_filtering_only_PUNCT_X.csv', index=False)

#%%
"""
To compare with natural language and UD annotation model training data,
Add the ko_gsd-ud-train.conllu file from the Korean-GSD treebank to the 'zero_conllus' and 'few_conllus' folders.
(https://github.com/UniversalDependencies/UD_Korean-GSD/tree/master) 
"""
#%%
###########################################################################
# UPOS tag distribution
###########################################################################
from helpers import get_upos_distribution, CONLLU_FILE_NAMES

folder_paths = ['zero_conllus','few_conllus']

for folder_path in folder_paths:
    df = get_upos_distribution(folder_path, CONLLU_FILE_NAMES)
    df.to_csv(f'ko_analysis/{folder_path}_upos_distribution.csv')

print("Completed.")

#%%
###########################################################################
# Visualization : UPOS tag distribution
###########################################################################
from helpers import plot_ud_distribution_stacked_bar

for folder in ['zero_conllus', 'few_conllus']:
    plot_ud_distribution_stacked_bar(f'ko_analysis/{folder}_upos_distribution.csv', f'ko_images/{folder}_upos_distribution.pdf', 'UPOS Tags')

# %%
###########################################################################
# XPOS tag distribution. (Keep the tags connected by "+"" as they are. Print only the top 20 tags on average)
###########################################################################
from helpers import get_xpos_distribution, CONLLU_FILE_NAMES

folder_paths = ['zero_conllus','few_conllus']

for folder_path in folder_paths:
    df = get_xpos_distribution(folder_path, CONLLU_FILE_NAMES)
    df.to_csv(f'ko_analysis/{folder_path}_xpos_org_distribution_top20.csv')

print("Completed.")

#%%
###########################################################################
# Visualization : XPOS tag (original) distribution
###########################################################################
from helpers import plot_ud_distribution_stacked_bar

for folder in ['zero_conllus', 'few_conllus']:
    plot_ud_distribution_stacked_bar(f'ko_analysis/{folder}_xpos_org_distribution_top20.csv', f'ko_images/{folder}_xpos_distribution_top20.pdf', 'XPOS Tags')

#%%
###########################################################################
# XPOS tag distribution. (Separate tags connected by "+". Print only the top 20 tags on average) 
###########################################################################
from helpers import get_xpos_split_distribution, CONLLU_FILE_NAMES

folder_paths = ['zero_conllus','few_conllus']

for folder_path in folder_paths:
    df = get_xpos_split_distribution(folder_path, CONLLU_FILE_NAMES)
    df.to_csv(f'ko_analysis/{folder_path}_xpos_split_distribution_top20.csv')
    
print("Completed.")

#%%
###########################################################################
# Visualization : XPOS tag (separated) distribution
###########################################################################
from helpers import plot_ud_distribution_stacked_bar

for folder in ['zero_conllus', 'few_conllus']:
    plot_ud_distribution_stacked_bar(f'ko_analysis/{folder}_xpos_split_distribution_top20.csv', f'ko_images/{folder}_xpos_split_distribution_top20.pdf', 'XPOS Tags')

#%%
###########################################################################
# deprel tag distribution.
###########################################################################
from helpers import get_deprel_distribution, CONLLU_FILE_NAMES

folder_paths = ['zero_conllus','few_conllus']

for folder_path in folder_paths:
    df = get_deprel_distribution(folder_path, CONLLU_FILE_NAMES)
    df.to_csv(f'ko_analysis/{folder_path}_deprel_distribution.csv')
    
print("Completed.")

#%%
###########################################################################
# Visualization : deprel tag distribution
###########################################################################
from helpers import plot_ud_distribution_stacked_bar
for folder in ['zero_conllus', 'few_conllus']:
    plot_ud_distribution_stacked_bar(f'ko_analysis/{folder}_deprel_distribution.csv', f'ko_images/{folder}_deprel_distribution.pdf', 'Deprel Tags')

#%%
###########################################################################
# dependency_arc direction and lengths
###########################################################################
from helpers import get_dependency_arc_stats, CONLLU_FILE_NAMES

folder_paths = ['zero_conllus','few_conllus']

for folder_path in folder_paths:
    df = get_dependency_arc_stats(folder_path, CONLLU_FILE_NAMES)
    df.to_csv(f'ko_analysis/{folder_path}_dependency_arc_stats.csv') 

print("Completed.")

#%%
###########################################################################
# Visualization : dependency arc direction and lengths
###########################################################################
from helpers import plot_ud_dependency_arc_stats

for folder in ['zero_conllus', 'few_conllus']:
    plot_ud_dependency_arc_stats(f'ko_analysis/{folder}_dependency_arc_stats.csv', f'ko_images/{folder}_dependency_arc_stats.pdf')




#%%
###########################################################################
######################                            #########################
######################   English translationese   #########################
######################                            #########################
###########################################################################


#%%
######################
# Detect translationese features and save the corresponding sentences
######################
import pandas as pd
import os
from helpers import translationese_features, CONLLU_FILE_NAMES

folder_paths = ['zero_conllus','few_conllus']

for folder_path in folder_paths:
    feature_list = []
    summary_results = []
    for file_name in CONLLU_FILE_NAMES:
        file_path = os.path.join(folder_path, file_name + ".conllu")
        if os.path.isfile(file_path):
            print(file_path)
            counts = translationese_features(file_path, feature_list, folder_path)
            summary_results.append((file_name,) + counts)
        else:
            print(f"File not found: {file_path}")

    df_features = pd.DataFrame(feature_list, columns=['Feature', 'Folder', 'File', 'Sentence'])
    df_summary = pd.DataFrame(summary_results, columns=['model', 'Numeral', 'Plural', 'Det', 'Passive', 
                                                        'NoSubj', 'NoObj', 'NoSubjObj', 'ObjVerb', 'VerbObj', 'ObjVerbSentence', 
                                                        'Sentence', 'Word', 'Noun'])

    df_features.to_csv(f'ko_analysis/{folder_path}_translationese_feature_sentences.csv', index=False)
    df_summary.to_csv(f'ko_analysis/{folder_path}_translationese_feature_detections.csv', index=False)

    if folder_path.startswith('zero'):
        zero_df = df_summary
    else:
        few_df = df_summary

merged_df = pd.merge(zero_df, few_df, on='model', suffixes=('_no', ''))
merged_df.to_csv('ko_analysis/translationese_feature_detections.csv', index=False)

#%%
###########################################################################
# Visualization: Translationese features 
###########################################################################
import pandas as pd
from helpers import plot_translationese_features

# Load and process data
merged_df = pd.read_csv('ko_analysis/translationese_feature_detections.csv').set_index('model')
merged_df
#%%
# proportion of the word '한' (as a translationese of article 'a/an') to the nouns in sentences
plot_translationese_features(merged_df, 'Numeral', 'Noun', 'ko_images/A_Noun_proportion.pdf')

#%%
# proportion of the word '그' (as a translationese of article 'the') to the nouns in sentences
plot_translationese_features(merged_df, 'Det', 'Noun', 'ko_images/Det_Noun_proportion.pdf')

#%%
# proportion of the word '그' (as a translationese of article 'the') to the nouns in sentences, without spoken corpus
plot_translationese_features(merged_df, 'Det', 'Noun', 'ko_images/Det_Noun_proportion.pdf', without_spoken_corpus=True)

#%%
# proportion of the word '들' (as a translationese of plurals) to the nouns in sentences
plot_translationese_features(merged_df, 'Plural', 'Noun', 'ko_images/Plural_Noun_proportion.pdf')

#%%
# proportion of the sentences with passive subjects to total sentences
plot_translationese_features(merged_df, 'Passive', 'Sentence', 'ko_images/Passive_Sentence_proportion.pdf')

#%%
# proportion of the sentences without subjects to total sentences
plot_translationese_features(merged_df, 'NoSubj', 'Sentence', 'ko_images/NoSubj_Sentence_proportion.pdf')

#%%
# proportion of the sentences without objects to total sentences
plot_translationese_features(merged_df, 'NoObj', 'Sentence', 'ko_images/NoObj_Sentence_proportion.pdf')

#%%
# proportion of the sentences without both subjects and objects to total sentences
plot_translationese_features(merged_df, 'NoSubjObj', 'Sentence', 'ko_images/NoSubjObj_Sentence_proportion.pdf')

#%%
# proportion of the sentences with object-verb order to sentences with both object and verb
plot_translationese_features(merged_df, 'ObjVerb', 'ObjVerbSentence', 'ko_images/ObjVerb_ObjVerbSentence_proportion.pdf')

#%%
# proportion of the sentences with verb-object order to sentences with both object and verb
plot_translationese_features(merged_df, 'VerbObj', 'ObjVerbSentence', 'ko_images/VerbObj_ObjVerbSentence_proportion.pdf')





#%%
###########################################################################
########################                         ##########################
########################   Semantic evaluation   ##########################
########################                         ##########################
###########################################################################


#%%
"""
For sentiment classification of corpora and generated sentences,
Refer to the code in src/main_analysis/sentiment_classification.py
"""
###########################################################################
# visualization: sentiment classification distribution
###########################################################################
import pandas as pd
from helpers import load_and_process_data, plot_sentiment_classification

# Load and process data
merged_df = load_and_process_data('ko_analysis/sentiment_classification.csv', has_corpora=True)

plot_sentiment_classification(merged_df, 'ko_images/sentiment_distribution.pdf')
# %%
