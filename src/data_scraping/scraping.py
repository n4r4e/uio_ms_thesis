
#%%
###########################################################################
# News scraping
###########################################################################
import requests
from bs4 import BeautifulSoup
import re
import csv
import os

# Mapping of section URLs to category names
section_categories = {
    "https://news.naver.com/section/100": "Politics",
    "https://news.naver.com/section/101": "Economy",
    "https://news.naver.com/section/102": "Society",
    "https://news.naver.com/section/103": "Life/Culture",
    "https://news.naver.com/section/104": "World",
    "https://news.naver.com/section/105": "IT/Science",
}

# Load existing URLs from CSV
def load_existing_urls(filename):
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            existing_urls = set(row['Link'] for row in reader)
        return existing_urls
    except FileNotFoundError:
        return set()

# Append new articles, avoiding duplicates
def append_new_articles_to_csv(articles, filename):
    existing_urls = load_existing_urls(filename)
    initial_article_count = len(articles)
    new_articles = [article for article in articles if article[3] not in existing_urls]
    excluded_article_count = initial_article_count - len(new_articles)

    if new_articles:
        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if os.stat(filename).st_size == 0:  # Check if file is empty to write header
                writer.writerow(['Category', 'Title', 'Content', 'Link', 'Press', 'Date'])
            writer.writerows(new_articles)

    # Print the count of excluded articles
    print(f"Excluded {excluded_article_count} articles that already existed.")

# Main scraping function
def scrape_naver_news_section(url, category):
    response = requests.get(url)
    if response.status_code == 200:
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')

        articles = []

        news_list = soup.find('ul', class_='sa_list')
        section_date = soup.find('strong', class_='ct_lnb_date').text.strip()

        for item in news_list.find_all('li', class_='sa_item'):
            title = item.find('a', class_='sa_text_title').text.strip()
            content = item.find('div', class_='sa_text_lede').text.strip() if item.find('div', class_='sa_text_lede') else ''
            link = item.find('a', class_='sa_text_title')['href']
            press = item.find('div', class_='sa_text_press').text.strip() if item.find('div', class_='sa_text_press') else ''
            image_tag = item.find('img', class_='_LAZY_LOADING')
            if image_tag and 'data-src' in image_tag.attrs:
                date_match = re.search(r'/(\d{4}/\d{2}/\d{2})/', image_tag['data-src'])
                article_date = date_match.group(1) if date_match else section_date
            else:
                article_date = section_date

            articles.append([category, title, content, link, press, article_date])

        return articles
    else:
        print(f"Failed to retrieve the webpage for {category}")
        return []

filename = 'naver_news_articles.csv'
all_articles = []

for url, category in section_categories.items():
    articles = scrape_naver_news_section(url, category)
    all_articles.extend(articles)

# Append new articles to CSV, avoiding duplicates
append_new_articles_to_csv(all_articles, filename)
print(f"New articles have been appended to {filename}")

#%%
###########################################################################
# Column scraping
###########################################################################
import requests
from bs4 import BeautifulSoup
import re
import csv
import os

# Remove author information from the beginning of the content
def clean_description(description):
    pattern = r'\|\s.*?(기자|교수|국장|소장|관장|판사|변호사|칼럼니스트|대표|주간|학자|소설가|목사|연구자|위원|회장)'
    just_content = re.sub(pattern, '', description)
    return just_content.strip()

# Load existing URLs from CSV
def load_existing_urls(filename):
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            existing_urls = set(row['Link'] for row in reader)
        return existing_urls
    except FileNotFoundError:
        return set()

# Append new columns, avoiding duplicates
def append_new_columns_to_csv(columns, filename):
    existing_urls = load_existing_urls(filename)
    initial_column_count = len(columns)
    new_columns = [column for column in columns if column[3] not in existing_urls]
    excluded_column_count = initial_column_count - len(new_columns)

    if new_columns:
        # Check if file is empty to decide whether to write the header
        file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Category', 'Title', 'Content', 'Link', 'Press', 'Date'])
            writer.writerows(new_columns)

    # Print the count of excluded columns
    print(f"Excluded {excluded_column_count} columns that already existed.")

def scrape_naver_opinion_column(url):
    response = requests.get(url)
    if response.status_code == 200:
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')

        columns = []

        # Extract the date information
        date_button = soup.find('button', class_='_open_calendar_btn')
        date_text = date_button.text.strip() if date_button else ''
        date_match = re.search(r'(\d{4}.\d{2}.\d{2})', date_text)
        page_date = date_match.group(1) if date_match else ''

        # Extract the column list
        column_list = soup.find('ul', class_='opinion_column_list')
        
        for item in column_list.find_all('li', class_='opinion_column_item'):
            # Extract the column title, URL, content, and press
            title = item.find('strong', class_='title').text.strip()
            link = item.find('a', class_='link')['href']
            content = item.find('p', class_='description').text.strip()
            just_content = clean_description(content)  # Remove author information
            press = item.find('span', class_='sub_item').text.strip()
            # Extract the date from the top of the page or from the image URL
            image = item.find('img')
            if image and 'src' in image.attrs:
                img_src = image['src']
                img_date_match = re.search(r'/(\d{4}/\d{2}/\d{2})/', img_src)
                img_date = img_date_match.group(1) if img_date_match else page_date
            else:
                img_date = page_date

            columns.append(["Column", title, just_content, link, press, img_date])

        return columns
    else:
        print("Failed to retrieve the webpage")
        return []

# Column page URL
url = "https://news.naver.com/opinion/column"

# Use the modified saving function to append new columns
csv_filename = 'naver_opinion_columns.csv'
columns = scrape_naver_opinion_column(url)
append_new_columns_to_csv(columns, csv_filename)
print(f"New columns have been appended to {csv_filename}")

#%%
###########################################################################
# Combine news articles and opinion columns
###########################################################################
import pandas as pd
import re

# Load news articles and opinion columns
news_articles_df = pd.read_csv('naver_news_articles.csv')
opinion_columns_df = pd.read_csv('naver_opinion_columns.csv')

# Combine the two DataFrames
combined_df = pd.concat([news_articles_df, opinion_columns_df])

# Remove brackets and their contents from the 'Title' and trim whitespace
combined_df['Title'] = combined_df['Title'].apply(lambda x: re.sub(r'\[.*?\]', '', x).strip())

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('articles_all.csv', index=False)

#%%
###########################################################################
# Randomly sample articles
###########################################################################
import pandas as pd
import numpy as np

# Load the CSV file containing all articles
df = pd.read_csv('articles_all.csv')

# Set number of articles to extract for each category
categories = ['Politics', 'Economy', 'Society', 'Life/Culture', 'World', 'IT/Science', 'Column']
counts = [100, 100, 100, 100, 100, 100, 400]


result_df = pd.DataFrame()

# Randomly sample articles for each category
for category, count in zip(categories, counts):
    category_df = df[df['Category'] == category]
    random_sample = category_df.sample(n=count, random_state=42)      
    result_df = pd.concat([result_df, random_sample])

result_df = result_df.reset_index(drop=True)

print(result_df)

result_df.to_csv('articles_1000.csv', index=False)

# Shuffle the rows of the DataFrame
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_df.to_csv('articles_1000_shuffled.csv', index=False)

#%%
###########################################################################
# Text statistics
###########################################################################
import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('articles_1000_shuffled.csv')

# Calculate character count with spaces, character count without spaces, and word count for each row in the 'Content' column
df['char_count_incl_spaces'] = df['Content'].apply(len)
df['char_count_excl_spaces'] = df['Content'].apply(lambda x: len(x.replace(" ", "")))
df['word_count'] = df['Content'].apply(lambda x: len(x.split()))

# Calculate average, minimum, and maximum values
avg_char_count_incl_spaces = np.mean(df['char_count_incl_spaces'])
avg_char_count_excl_spaces = np.mean(df['char_count_excl_spaces'])
avg_word_count = np.mean(df['word_count'])
min_char_count_incl_spaces = np.min(df['char_count_incl_spaces'])
min_char_count_excl_spaces = np.min(df['char_count_excl_spaces'])
min_word_count = np.min(df['word_count'])
max_char_count_incl_spaces = np.max(df['char_count_incl_spaces'])
max_char_count_excl_spaces = np.max(df['char_count_excl_spaces'])
max_word_count = np.max(df['word_count'])

# Print the results
print(f"Average character count with spaces: {avg_char_count_incl_spaces:.2f}")
print(f"Average character count without spaces: {avg_char_count_excl_spaces:.2f}")
print(f"Average word count: {avg_word_count:.2f}")
print(f"Minimum character count with spaces: {min_char_count_incl_spaces}")
print(f"Minimum character count without spaces: {min_char_count_excl_spaces}")
print(f"Minimum word count: {min_word_count}")
print(f"Maximum character count with spaces: {max_char_count_incl_spaces}")
print(f"Maximum character count without spaces: {max_char_count_excl_spaces}")
print(f"Maximum word count: {max_word_count}")