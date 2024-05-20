#%%
###########################################################################
# Extract a list of entries from the Korean Dictionary 'Urimalsaem'
###########################################################################
#%%
import re
import os
from tqdm import tqdm
import pandas as pd

# get the list of entries from the dictionary
def process_words_from_dictionary_xls(folder_path):
    words_list = []

    for filename in tqdm(os.listdir(folder_path), desc="Processing files"):
        if filename.endswith(".xls"):
            # Load the dictionary spreadsheet
            file_path = os.path.join(folder_path, filename)
            data = pd.read_excel(file_path)

            # Extract only the contents of the '어휘' column and add them to the list.
            for word in data['어휘']:
                # Set words that can be written together in their combined form (remove '-' and '^')
                processed_word = word.replace('-', '').replace('^', '')
                words_list.append(processed_word)

    # Remove duplicate morpheme forms
    words_list = list(set(words_list))

    # Convert list to DataFrame
    words_df = pd.DataFrame(words_list, columns=['Word'])

    # Save to CSV
    output_path = os.path.join(folder_path, 'urimalsam_words.csv')
    words_df.to_csv(output_path, index=False)

    return words_df


"""
Download dictionary files from the Urimalsaem(우리말샘) website (https://opendict.korean.go.kr/main) (e.g., '20240501')
"""
# get the list of entries from the dictionary
folder_path = '20240501'
processed_words_df = process_words_from_dictionary_xls(folder_path)
print(processed_words_df.head())

#%%
###########################################################################
# Create a set of morphemes from the dictionary words
###########################################################################
from tqdm import tqdm
from helpers import morpheme_segmentation

# Load dictionary words and tokenize them
dictionary_df = pd.read_csv('20240501/urimalsam_words.csv')
dictionary_words = dictionary_df['Word'].tolist()
dictionary_tokens = set()

for word in tqdm(dictionary_words, desc='Tokenizing dictionary words'):
    tokens = morpheme_segmentation(word)
    dictionary_tokens.update(tokens)

dictionary_df = pd.DataFrame({'Token': list(dictionary_tokens)})
dictionary_df.to_csv('20240501/urimalsam_morpheme_set.csv', index=False)

# %%
