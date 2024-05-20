# %%
###########################################################################
# Save 'sentence' column data from corpus csv files to a text file
###########################################################################
import pandas as pd
import os

# Folder paths containing CSV files of the corpora
folder_paths = ['corpus/spoken', 'corpus/written']

# Set the paths to save the results
txt_file_paths = ['corpus/corpus_spoken_org.txt', 'corpus/corpus_written_org.txt']

# Zip folder_paths and txt_file_paths and iterate
for folder_path, txt_file_path in zip(folder_paths, txt_file_paths):
    
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        # Iterate over all .csv files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                csv_file_path = os.path.join(folder_path, filename)
                
                df = pd.read_csv(csv_file_path)
                
                # Save the data in the 'sentence' column to a text file
                for sentence in df['sentence'].unique():
                    txt_file.write(sentence + '\n')
    
    print(f"All sentences from {folder_path} have been saved to {txt_file_path}")

#%%
###########################################################################
# Split the corpus text file into sentences and save them
###########################################################################
#%%
from kiwipiepy import Kiwi
from helpers import split_and_merge_sentences

input_file_paths = ['corpus/corpus_spoken_org.txt', 'corpus/corpus_written_org.txt']
output_file_paths = ['corpus/corpus_spoken_sentences.txt', 'corpus/corpus_written_sentences.txt']

kiwi = Kiwi()

def split_and_save_corpus_sentences(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        content = " ".join(line.strip() for line in lines)  # Combine all lines into one string
        
        # Sentence separation
        merged_sentences, _, _ = split_and_merge_sentences(content)
        
    # Save the separated sentences to a text file
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for sentence in merged_sentences:
            outfile.write(sentence + '\n')
            
# split and save the sentences of the corpora
for input_file_path, output_file_path in zip(input_file_paths, output_file_paths):
    split_and_save_corpus_sentences(input_file_path, output_file_path)



#%%
###########################################################################
# Apply the same filtering process used for generated sentences to the corpus sentences and save them
###########################################################################
import os
import pandas as pd
from helpers import has_korean_character_ratio, has_unusual_pattern, has_headline_pattern, remove_ending_pattern, remove_starting_pattern, remove_extra_dots, preprocess_sentences, complete_korean_sentence

# folders = ["zero_txt", "few_txt"]
# output_folders = ["zero_comp", "few_comp"]
folders = ["corpus"]
output_folders = ["corpus_comp"]

for folder, output_folder in zip(folders, output_folders):
    os.makedirs(output_folder, exist_ok=True)
    result_list = []
    for filename in os.listdir(folder):
        if filename.endswith("sentences.txt"):
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
result_df.to_csv(f"{output_folder}_filtering_all.csv", index=False)

#%%
###########################################################################
# Save the first 10000 sentences of the spoken and written corpora
###########################################################################
def save_first_n_lines(input_file, output_file, n):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i >= n:
                break
            outfile.write(line)

# Save the first 10000 lines of the corpus_spoken.txt file
input_file = 'corpus_comp/corpus_spoken_sentences.txt'
output_file = 'corpus_comp/corpus_spoken.txt'
save_first_n_lines(input_file, output_file, 10000)

# Save the first 10000 lines of the corpus_written.txt file
input_file = 'corpus_comp/corpus_written_sentences.txt'
output_file = 'corpus_comp/corpus_written.txt'
save_first_n_lines(input_file, output_file, 10000)