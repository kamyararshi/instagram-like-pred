# 巩昕锐
# 开发时间：2023/6/8 13:58
import nltk
import re
import string
import emoji
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import test
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


nltk.download('stopwords')
from nltk.corpus import stopwords
# print(test.posts_df_filter_weekday[:5])

def remove_punctuation(text):
    # Remove punctuation using regular expressions
    no_punct = re.sub('[' + string.punctuation + ']', '', text)
    return no_punct


def remove_stopwords(text):
    # Remove stopwords using NLTK corpus
    stop_words = set(stopwords.words('english'))
    no_stopwords = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return no_stopwords


def remove_emojis(text):
    # Convert emojis to textual representation and remove them
    no_emojis = emoji.demojize(text)
    no_emojis = re.sub('(:[a-z_-]+:)', ' ', no_emojis)
    return no_emojis


test.posts_df_filter_weekday['descriptionProcessed'] = test.posts_df_filter_weekday['description'].apply(remove_punctuation)
test.posts_df_filter_weekday['descriptionProcessed'] = test.posts_df_filter_weekday['descriptionProcessed'].apply(
    remove_stopwords)
test.posts_df_filter_weekday['descriptionProcessed'] = test.posts_df_filter_weekday['descriptionProcessed'].apply(remove_emojis)
# by default the vectorizer conerts the text to lower case and uses word-level tokenization
# Create an instance of CountVectorizer with max_features set to 500 (this is what they did in the tds implementation)
vec = CountVectorizer(max_features=500)

# Transform the "descriptionProcessed" column into a matrix of token counts
description_counts = vec.fit_transform(test.posts_df_filter_weekday['descriptionProcessed'])

# Convert the matrix to an array
description_counts_array = description_counts.toarray()
feature_names = vec.get_feature_names()
df = pd.DataFrame(data=description_counts_array, columns=feature_names)
print(df.shape)
print(test.posts_df_filter_weekday.shape)
print(test.posts_df_filter_weekday[:5])


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "Unknown"

# Assuming 'description' is the column name in your DataFrame
test.posts_df_filter_weekday['language'] = test.posts_df_filter_weekday['description'].apply(detect_language)


print(test.posts_df_filter_weekday['language'].value_counts())