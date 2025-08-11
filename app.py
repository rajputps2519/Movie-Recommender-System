#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 


# In[2]:


movies= pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


movies.merge(credits, on='title')


# In[6]:


movies.shape


# In[7]:


credits.shape


# In[8]:


movies=movies.merge(credits, on='title')


# In[9]:


movies.head()


# In[10]:


#genres
# id
#keywords
#title
#overview
#cast
#crew
movies[['movie_id','title','genres','keywords','cast','crew']]


# In[11]:


movies=movies[['movie_id','title','genres','keywords','cast','crew']]


# In[12]:


movies.head()


# In[13]:


movies.isnull().sum()


# In[14]:


movies.duplicated().sum()


# In[15]:


movies.iloc[0].genres


# In[16]:


import ast


# In[17]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

        


# In[18]:


movies['genres']=movies['genres'].apply(convert)


# In[19]:


movies.head()


# In[20]:


movies['keywords']=movies['keywords'].apply(convert)


# In[21]:


movies.head()


# In[22]:


movies['cast'][0]


# In[23]:


import ast

def convert3(obj):
    L = []
    for idx, i in enumerate(ast.literal_eval(obj)):
        if idx < 3:
            L.append(i['name'])
        else:
            break
    return L



# In[24]:


movies['cast']=movies['cast'].apply(convert3)


# In[25]:


movies.head()


# In[26]:


import ast

def fetch_director(obj):
    L = []
    for idx, i in enumerate(ast.literal_eval(obj)):
        if i['job']=='Director':
            L.append(i['name'])
        else:
            break
    return L


# In[27]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[28]:


movies.head()


# In[29]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[30]:


movies.head()


# In[31]:


movies['tag']= movies['genres']+ movies['keywords']+ movies['cast']+ movies['crew']


# In[32]:


movies.head()


# In[33]:


new_df=movies[['movie_id','title','tag']]


# In[34]:


new_df['tag']=new_df['tag'].apply(lambda x:" ".join(x))


# In[35]:


new_df.head()


# In[36]:


# get_ipython().system('pip install nltk')


# In[37]:


import nltk


# In[38]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[39]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[40]:


new_df['tag']=new_df['tag'].apply(stem)


# In[41]:


new_df['tag'][0]


# In[42]:


new_df['tag']=new_df['tag'].apply(lambda x:x.lower()) 


# In[43]:


new_df.head()


# In[44]:


new_df['tag'][0]


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000,stop_words='english')


# In[46]:


vector=cv.fit_transform(new_df['tag']).toarray()


# In[47]:


vector[0]


# In[48]:


cv.get_feature_names_out()


# In[49]:


from sklearn.metrics.pairwise import cosine_similarity


# In[50]:


similarity=cosine_similarity(vector)


# In[51]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[52]:


def recommend(movie): 
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        
    return


# In[53]:


recommend('Batman Begins')


# In[54]:


import pickle


# In[55]:


pickle.dump(new_df,open('movie.pkl','wb'))


# In[56]:


# get_ipython().system('pip install streamlit')


# In[58]:





# In[ ]:




