import streamlit as st
st.title('Netflix Recommendation System')


ttl = st.text_input('Input movie name')
import scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
import requests
import csv
import pandas as pd
import io
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Download the CSV file from GitHub
url = 'https://raw.githubusercontent.com/ykwong1223/Principle-DS-new/main/Netflix.csv'
response = requests.get(url)
content = response.content.decode('utf-8')
csvreader = csv.reader(content.splitlines())

raw = pd.read_csv(io.StringIO(content))
raw=pd.DataFrame(raw)
raw.drop_duplicates(keep='last', inplace=True)
raw["name"]=raw["title"]
for i in ["country","date_added","rating"]:
  raw[i].fillna(raw[i].mode().iloc[0], inplace=True)

for i in ["director","cast"]:
  raw[i].fillna("", inplace=True)

def data_clean(k):
  return str.lower(k.replace(" ",""))

raw["combine"]=raw["title"].apply(data_clean)+" "+raw["cast"].apply(data_clean)+" "+raw["director"].apply(data_clean)+" "+raw["description"].apply(data_clean)
raw["name"]=raw["name"].apply(data_clean)

count = CountVectorizer().fit_transform(raw['combine'])
cosine_similar_score = cosine_similarity(count, count)
def cosine_similar(nm):
    nm=data_clean(nm)
    if nm in raw["name"].values:
      indx = raw.index[raw['name'] == nm].tolist()
      score=list(enumerate(cosine_similar_score[indx[0]]))
      raw["score"]=[m[1] for m in score]
      subset=raw[["title","score"]]
      subset=subset.sort_values(by='score',ascending=False)
    else:
      subset=" Please enter a right name!!!"
    return st.tex(subset[1:])


st.button('Recommend', on_click=recommend(ttl, mthd))
