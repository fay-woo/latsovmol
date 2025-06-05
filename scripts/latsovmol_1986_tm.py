# -*- coding: utf-8 -*-
"""
latsovmol_1986_tm
"""

!pip install bertopic
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from google.colab import drive
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
from wordcloud import WordCloud
from IPython.display import display, HTML
import shutil
import random
import requests
from io import StringIO

github_url = 'https://raw.githubusercontent.com/fay-woo/latsovmol/main/latsovmol_processed.csv'

response = requests.get(github_url)
response.raise_for_status()

df = pd.read_csv(StringIO(response.text))

texts_by_year = df.groupby('year')['processed_text'].apply(lambda x: ' '.join(x.dropna())).reset_index(name='full_text')

np.random.seed(42)
random.seed(42)

target_year = 1986
full_text = texts_by_year[texts_by_year['year'] == target_year]['full_text'].tolist()[0]
texts_year = [full_text[i:i+250] for i in range(0, len(full_text), 250)]
texts_year = [t for t in texts_year if len(t) > 50]

custom_stopwords = [ "это", "этот", "весь", "наш", "свой", "который", "год", "например", "очень", "много", "такой", "день" ]

model = BERTopic(
    language="russian",
    min_topic_size=25,
    nr_topics=11,
    n_gram_range=(1, 3),
    vectorizer_model=CountVectorizer(
        stop_words=custom_stopwords,
        ngram_range=(1, 3),
        min_df=2
    ),
    umap_model=UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        n_components=10
    ),
    hdbscan_model=HDBSCAN(
        min_cluster_size=35,
        min_samples=10,
        cluster_selection_epsilon=0.1,
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=1
    ),
    calculate_probabilities=False,
    verbose=True
)

topics, _ = model.fit_transform(texts_year)

topic_info = model.get_topic_info()
valid_topics = topic_info[topic_info['Topic'] != -1]

print("\n тем", len(valid_topics))
display(valid_topics[['Topic', 'Count', 'Name']].sort_values('Count', ascending=False))

model.visualize_barchart(
    top_n_topics=min(11, len(valid_topics)),
    title=f"Темы {target_year}"
)

topic_info = model.get_topic_info()

for index, row in topic_info.iterrows():
    topic_num = row['Topic']
    words = [word for word, _ in model.get_topic(topic_num)[:5]]
    print(f"Тема {topic_num}: {', '.join(words)}")

custom_names = {
    0: "Семья",
    1: "Советско-американские отношениия",
    2: "Культура",
    3: "Экономика",
    4: "Национальная политика",
    5: "Образование",
    6: "Путешествия",
    7: "Космос",
    8: "Инфраструктура",
    9: "Архитектура",
    }
model.set_topic_labels(custom_names)

os.makedirs('bertopic_graphs', exist_ok=True)

topic_info = model.get_topic_info()
valid_topics = topic_info[topic_info['Topic'] != -1]
num_topics = len(valid_topics)

df_counts = valid_topics.sort_values('Count', ascending=False)
fig0 = px.bar(
    df_counts,
    x='CustomName',
    y='Count',
    title=f'Текстовые фрагменты в темах {target_year} года',
    labels={'CustomName': 'Тема', 'Count': 'Фрагментов'},
    color='Count',
    color_continuous_scale='Magma'
)
fig0.update_layout(xaxis_tickangle=-45, showlegend=False)
fig0.show()
fig0.write_html("bertopic_graphs/topic_counts.html")
display(HTML('<a href="bertopic_graphs/topic_counts.html" download>topic_counts</a>'))

fig1 = model.visualize_barchart(
    top_n_topics=num_topics,
    n_words=10,
    title=f"Топ слов по темам в выпусках Советской молодежи {target_year} года",
    width=300,
    height=500,
    custom_labels=True
)
fig1.show()
fig1.write_html("bertopic_graphs/top_words.html")

for topic_num in valid_topics['Topic']:
    words_weights = dict(model.get_topic(topic_num))
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='magma'
    ).generate_from_frequencies(words_weights)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f"Тема {topic_num}: {model.custom_labels_[topic_num + 1]}", fontsize=14)
    plt.axis('off')

    wc_path = f"bertopic_graphs/wordcloud_topic_{topic_num}.png"
    plt.savefig(wc_path, bbox_inches='tight', dpi=200)
    display(HTML(f'<a href="{wc_path}" download>WordCloud{topic_num}</a>'))
    plt.show()

fig2 = model.visualize_hierarchy(
    top_n_topics=min(10, len(topic_info)),
    title=f"Иерархия тем в Советской молодежи {target_year} ",
    custom_labels=True
)
fig2.show()
fig2.write_html("bertopic_graphs/hierarchy.html")

if len(valid_topics) >= 2:
    fig3 = model.visualize_heatmap(
        topics=valid_topics['Topic'],
        n_clusters=min(5, len(valid_topics)-1),
        title=f"Heatmap тем в выпусках Советской молодежи {target_year} года",
        custom_labels=True
    )
    fig3.show()
    fig3.write_html("bertopic_graphs/heatmap.html")

fig4 = model.visualize_documents(
    texts_year,
    topics=topics,
    hide_annotations=False,
    title=f"Распределение текста в выпусках {target_year} года",
    custom_labels=True
)
fig4.show()
fig4.write_html("bertopic_graphs/document_distribution.html")

!zip -r bertopic_graphs.zip bertopic_graphs/
display(HTML('<a href="bertopic_graphs.zip" download>(ZIP)</a>'))