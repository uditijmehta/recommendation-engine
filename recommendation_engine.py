
# coding: utf-8

# In[64]:


get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import warnings; warnings.simplefilter('ignore')


# In[65]:


md = pd. read_csv('C:/Users/Pavan Maru/Documents/the-movies-dataset/movies_metadata.csv')
md.head()


# In[66]:


md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[67]:


vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
C


# In[68]:


m = vote_counts.quantile(0.95)
m


# In[69]:


md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[70]:


qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape


# In[71]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[72]:


qualified['wr'] = qualified.apply(weighted_rating, axis=1)


# In[73]:


qualified = qualified.sort_values('wr', ascending=False).head(250)


# ### Top Movies

# In[22]:


qualified.head(15)


# In[23]:


s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)


# In[24]:


def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


# In[25]:


build_chart('Romance').head(15)


# In[26]:


links_small = pd.read_csv('C:/Users/Pavan Maru/Documents/the-movies-dataset/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[27]:


md = md.drop([19730, 29503, 35587])


# In[28]:


#Check EDA Notebook for how and why I got these indices.
md['id'] = md['id'].astype('int')


# In[29]:


smd = md[md['id'].isin(links_small)]
smd.shape


# In[30]:


smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')


# In[31]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])


# In[32]:


tfidf_matrix.shape


# In[33]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[34]:


cosine_sim[0]


# In[35]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[36]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[37]:


get_recommendations('The Godfather').head(10)


# In[38]:


#Metadata based recommender
credits = pd.read_csv('C:/Users/Pavan Maru/Documents/the-movies-dataset/credits.csv')
keywords = pd.read_csv('C:/Users/Pavan Maru/Documents/the-movies-dataset/keywords.csv')


# In[39]:


keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')


# In[40]:


md.shape


# In[41]:


md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')


# In[42]:


smd = md[md['id'].isin(links_small)]
smd.shape


# In[43]:


smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))


# In[44]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[45]:


smd['director'] = smd['crew'].apply(get_director)


# In[46]:


smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)


# In[47]:


smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[48]:


smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[49]:


smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x, x])


# In[50]:


s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'


# In[51]:


s = s.value_counts()
s[:5]


# In[52]:


s = s[s > 1]


# In[53]:


stemmer = SnowballStemmer('english')
stemmer.stem('dogs')


# In[54]:


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


# In[55]:


smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[56]:


smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))


# In[57]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])


# In[58]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[59]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[63]:


get_recommendations('Batman Begins').head(10)

