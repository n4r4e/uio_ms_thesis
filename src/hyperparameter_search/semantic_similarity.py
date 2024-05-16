import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm  
import pandas as pd
import os

# load data
generated_text_df = pd.read_csv('results/all.csv')
org_text_df = pd.read_csv('articles_test.csv')

# load a semantic similarity model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
tokenizer = model.tokenizer

# create a dictionary with the start of each Content as the key and the full Content as the value
content_dict = {row['Content'].split()[0]: row['Content'] for _, row in org_text_df.iterrows()}

# process all rows in generated_text_df
for index, row in tqdm(generated_text_df.iterrows(), total=generated_text_df.shape[0]):
    input_prompt = row['input_prompt']
    first_word = input_prompt.split()[0]  # get the first word of the input prompt

    # find matching Content in content_dict using the first word
    if first_word in content_dict:
        matched_content = content_dict[first_word]

        # tokenization and truncation
        generated_text_tokens = tokenizer.tokenize(row['generated_text'])[:model.get_max_seq_length()]
        matched_content_tokens = tokenizer.tokenize(matched_content)[:model.get_max_seq_length()]

        # tokens to string
        truncated_generated_text = tokenizer.convert_tokens_to_string(generated_text_tokens)
        truncated_matched_content = tokenizer.convert_tokens_to_string(matched_content_tokens)

        # vectorization
        prompt_vector = model.encode(truncated_matched_content, convert_to_tensor=True).unsqueeze(0)
        generated_text_vector = model.encode(truncated_generated_text, convert_to_tensor=True).unsqueeze(0)

        # calculate cosine similarity
        similarity_score = util.cos_sim(prompt_vector, generated_text_vector).item()

        # add results
        generated_text_df.at[index, 'similarity_score'] = similarity_score

# save to csv
generated_text_df.to_csv('all_with_similarity_scores.csv', index=False)