import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from helpers import split_and_merge_sentences

# Set input and output folder paths
input_folders = ['zero', 'few', 'dataset']
output_folders = ['zero_embedding', 'few_embedding', 'article_embedding']

# Load sentence embedding model
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# Create Kiwi object
kiwi = Kiwi()

# Process each folder
for input_folder, output_folder in zip(input_folders, output_folders):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            print(f'{filename=}')
            # Read CSV file
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)
            # Select target column
            if input_folder == 'zero':
                target_column = 'generated_text'
            elif input_folder == 'few':
                target_column = 'only_generated_text'
            elif input_folder == 'dataset':
                df['combined_text'] = df['Title'] + '. ' + df['Content']
                target_column = 'combined_text'

            # Split sentences and vectorize
            sentence_embeddings = []
            for text in df[target_column]:
                sentences, _ = split_and_merge_sentences(text)
                embeddings = model.encode(sentences)
                mean_embedding = np.mean(embeddings, axis=0)
                sentence_embeddings.append(mean_embedding)

            # Set output file paths
            output_path = os.path.join(output_folder, filename.replace('.csv', '.npy'))

            # Save embeddings
            np.save(output_path, np.array(sentence_embeddings))
            
