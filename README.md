# Investigating linguistic bias in large language models: A case study of Korean text generation

This repository contains the code used in the research for my UiO Master's thesis "Investigating linguistic bias in large language models: A case study of Korean text generation".

## Code Structure

The code is organized into different sections and scripts for various purposes. Please note that in some parts of the code, the `%##` marker indicates that the code was executed in cell units.

## Web Scraping

- To provide input prompts for text generation that are not included in the pre-training data of the models, recent news articles and columns are collected.
- `scraping.py`: Headlines and the first parts of the body text (approximately 120 characters) are collected from the news and column pages of the news section of the Naver portal in Korea.
  
## Dataset

- Due to copyright issues, the dataset collected through web scraping is not directly shared in this repository. Instead, the `articles_1000.csv` file containing the URLs and sources of the dataset is provided, which can be used to reproduce the experiments.

## Hyperparameter Search

- The optimal hyperparameter configurations for each language model to generate high-quality results are explored.
- `text_generation.py`: Text generation is performed for each model using 30 configurations. The dataset used for generation is also not directly shared here; instead, the `articles_test.csv` file containing URLs and sources is provided.
- `semantic_similarity.py`: The semantic similarity with the original text is calculated.
- `analysis.py`: The results are examined, and the top 5 configurations for each model are saved.
