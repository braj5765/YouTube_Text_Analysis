#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[3]:


comments=pd.read_csv(r"C:\Users\brajk\Desktop\Data Analytics Projects\Project 1- Youtube text data analysis/GBcomments.csv",error_bad_lines=False)


# In[4]:


comments.head()


# In[5]:


#!pip install textblob


# # Perform Sentiment analysis on Youtube comments

# In[8]:


from textblob import TextBlob


# In[9]:


TextBlob('Its more accurate to call it the M+ (1000) be...').sentiment.polarity


# In[10]:


## check for all the missing values in the data
comments.isna().sum()


# In[11]:


## As small no of NA are there we will drop them 
comments.dropna(inplace=True)


# In[12]:


##We will collect polarity of all the comments 
polarity=[]
for i in comments['comment_text']:
    polarity.append(TextBlob(i).sentiment.polarity)


# In[13]:


#Add this polarity data to the dataframe
comments['polarity']=polarity


# In[14]:


comments.head(10)


# ##### Perform Exploratory Data Analysis(EDA) on positive sentiments

# In[17]:


## collect all the rows with positive sentiments
comments_positive=comments[comments['polarity']==1]


# In[21]:


comments_positive.shape


# In[22]:


comments_positive.head()


# In[23]:


## we will use worldcloud library for the analysis
from wordcloud import WordCloud,STOPWORDS


# In[25]:


## to get a unique set of stopwords
stopwords=set(STOPWORDS)


# In[26]:


## now we will join all the comments into a single one
total_comments=" ".join(comments_positive['comment_text'])


# In[29]:


## now we will setup our basic wordcloud structure and store it for total_comments
wordcloud=WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_comments)


# In[30]:


## now we will plot this
plt.figure(figsize=(15,5)) #define the plot figure size
plt.imshow(wordcloud)  # define what we want to show in the figure
plt.axis('off') # to keep axis off


# ##### Perform Exploratory Data Analysis(EDA) on negative sentiments

# In[31]:


## collect all the rows with negative sentiments
comments_negative=comments[comments['polarity']==-1]


# In[34]:


total_comments2=" ".join(comments_negative['comment_text'])
wordcloud2=WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_comments2)


# In[35]:


## now we will plot this
plt.figure(figsize=(15,5)) #define the plot figure size
plt.imshow(wordcloud2)  # define what we want to show in the figure
plt.axis('off') # to keep axis off


# # Analyze Treanding Tags and Views of Youtube

# #### Analysing Tags

# In[36]:


videos=pd.read_csv(r'C:\Users\brajk\Desktop\Data Analytics Projects\Project 1- Youtube text data analysis/USvideos.csv',error_bad_lines=False)


# In[37]:


videos.head()


# In[39]:


##we will store all tags in a single tag by joining them
tags_complete=' '.join(videos['tags'])


# In[40]:


tags_complete


# In[41]:


import re


# In[45]:


## now we will remove any type of speacial character present in tags
tags=re.sub('[^a-zA-Z]',' ',tags_complete)


# In[47]:


## now we will remove extra spacings available in between
tags=re.sub(' +'," ",tags)


# In[49]:


##now we will make a wordcloud for this data
wordcloud3=WordCloud(width=1000,height=500,stopwords=set(STOPWORDS)).generate(tags)


# In[51]:


#Now we will plot
plt.figure(figsize=(15,5))
plt.imshow(wordcloud3)
plt.axis('off')


# #### Analysing Likes And Dislikes

# In[52]:


#we will plot a regression plot between views and likes
sns.regplot(data=videos,x='views',y='likes')
plt.title('Regression plot between VIEWS and LIKES')


# In[53]:


#we will plot a regression plot between views and dislikes
sns.regplot(data=videos,x='views',y='dislikes')
plt.title('Regression plot between VIEWS and DISLIKES')


# #### Correlation between views, likes and dislikes

# In[56]:


## we will create a new dataframe to store just views, likes and dislikes
df_corr=videos[['views','likes','dislikes']]
df_corr.corr()


# In[57]:


##we will plot a heatmap for this data
sns.heatmap(df_corr.corr(),annot=True) #annot makes the values visible on plot


# # Emoji Analysis

# In[58]:


comments.head()


# In[59]:


import emoji


# In[82]:


## we will extract the emoji using the unicode and store it collectively
str=''
for i in comments['comment_text']:
    list=[c for c in i if c in emoji.UNICODE_EMOJI_ALIAS_ENGLISH]
    for ele in list:
        str=str+ele


# In[83]:


len(str)


# In[84]:


str


# In[85]:


##frequency count of all distinct emojis
result={}
for i in set(str):
    result[i]=str.count(i)


# In[86]:


result


# In[89]:


##sort all this data and store it
final={}
for key,value in sorted(result.items(), key=lambda item:item[1]):
    final[key]=value


# In[91]:


##now we will unzip this dictionary and storeit in form of a list using '[* dict_name.key_or_value]'
keys=[*final.keys()]
values=[*final.values()]


# In[94]:


#now we will make a dataframe for these two
df=pd.DataFrame({'chars':keys[-20:],'num':values[-20:]})
df


# In[96]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[97]:


#Now we will plot this data
trace=go.Bar(x=df['chars'],y=df['num'])
iplot([trace])


# In[ ]:




