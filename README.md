# Investigating linguistic bias in large language models: A case study of Korean text generation

This repository contains the full text of my UiO Master's thesis "Investigating linguistic bias in large language models: A case study of Korean text generation" and the code used in the research.

## Thesis

The full text of the thesis is available in this repository as a PDF file. This version has been updated from the one uploaded to the UiO DUO Research Archive, with corrections of typos, cross-references, and other minor improvements.

## Code Structure

The code is organized into different sections and scripts for various purposes. Please note that in some parts of the code, the `%##` marker indicates that the code was executed in cell units.

## Web Scraping

To provide input prompts for text generation that are not included in the pre-training data of the models, recent news articles and columns are collected.

- `scraping.py`: Headlines and the first parts of the body text (approximately 120 characters) are collected from the news and column pages of the news section of the Naver portal in Korea.
  
## Dataset

Due to copyright issues, the dataset collected through web scraping is not directly shared in this repository. Instead, the `articles_1000.csv` file containing the URLs and sources of the dataset is provided, which can be used to reproduce the experiments.

## Hyperparameter Search

The optimal hyperparameter configurations for each language model to generate high-quality results are explored.

- `text_generation.py`: Text generation is performed for each model using 30 configurations. The dataset used for generation is also not directly shared here; instead, the `articles_test.csv` file containing URLs and sources is provided.
- `semantic_similarity.py`: The semantic similarity with the original text is calculated.
- `analysis.py`: The results are examined, and the top 5 configurations for each model are saved.

## Main Analysis

The main analysis involves generating text using language models and performing analysis on the generated texts as well as the Korean sentences filtered from the generated text.

- `text_generation.py`: Texts are generated using language models in zero-shot and few-shot settings with Korean input prompts.
- `text_analysis.py`: Analysis and visualization are performed on the texts generated by the language models.
- `ko_text_analysis.py`: Analysis and visualization are performed on the Korean sentences filtered from the texts generated by the language models.
- `corpus_processing.py`: Korean sentences from Korean corpora (spoken and written) CSV files are separated and stored.
- `dictionary_processing.py`: A headword list is extracted from Korean dictionary (Urimalsaem) XLS files, and a morpheme set is obtained by morphologically segmenting them.
- `semantic_embedding.py`: Sentence embeddings are obtained from the original articles and the generated texts.
- `sentiment_classification.py`: Sentiment classification is performed on Korean sentences generated by the language models and those from Korean corpora.
- `helpers.py`: Various functions and constants used in the main analysis are stored.
