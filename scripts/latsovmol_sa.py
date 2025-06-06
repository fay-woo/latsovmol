# -*- coding: utf-8 -*-
"""
latsovmol_sa
"""

!pip install transformers pandas tqdm plotly matplotlib wordcloud
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline
import torch
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords

gh = "https://raw.githubusercontent.com/fay-woo/latsovmol/main/latsovmol_processed.csv"
df = pd.read_csv(gh)

# sentiment_analyzer = pipeline("text-classification", model="seara/rubert-tiny2-russian-sentiment", truncation=True, max_length=512)
sentiment_analyzer = pipeline("text-classification", model="blanchefort/rubert-base-cased-sentiment", truncation=True, max_length=512)

def analyze_text(text):
    try:
        res = sentiment_analyzer(str(text)[:4000])[0]
        return res['label'], round(res['score'], 2)
    except:
        return 'NEUTRAL', 0.5

sampled_df = df.groupby('year', group_keys=False)\.apply(lambda x: x.sample(n=33, random_state=42))\.reset_index(drop=True)

tqdm.pandas(desc="Пргресс")
sampled_df[['sentiment', 'confidence']] = sampled_df['original_text'].progress_apply(lambda x: pd.Series(analyze_text(x)))

sentiment_dist = pd.crosstab(index=sampled_df['year'], columns=sampled_df['sentiment'], normalize='index').round(3) * 100

confidence_avg = sampled_df.groupby(['year', 'sentiment'])['confidence'].mean().unstack()

fig = px.bar(sentiment_dist.reset_index(), x='year', y=['POSITIVE', 'NEUTRAL', 'NEGATIVE'],
    title='сентимент по годам %', labels={'value': 'процент текста', 'variable': 'сентимент'},
             barmode='group', color_discrete_sequence=['#4CAF50', '#FFC107', '#F44336'], height=500)

fig.update_layout(
    xaxis_title="Год",
    yaxis_title="Процент статей",
    legend_title="",
    hovermode="x unified"
)

for year in sentiment_dist.index:
    for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        fig.add_annotation(
            x=year,
            y=sentiment_dist.loc[year, sentiment],
            text=f"{confidence_avg.loc[year, sentiment]:.2f}",
            showarrow=False,
            yshift=10
        )

fig.show()

pio.write_html(fig, "sentiment_by_year.html")