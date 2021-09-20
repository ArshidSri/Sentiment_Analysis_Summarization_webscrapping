#!/usr/bin/env python
# coding: utf-8

# # Stock Sentiment Analysis and Summarization via Web scrapping 
# **CHECK OUT**<br>
# <a href="https://github.com/ArshidSri/Sentiment_Analysis_Summarization_webscrapping"> source_code_github </a>(Jupyter Notebook and Python file)<br>
# <a href="https://jovian.ai/arshidsriraam7/sentiment-analysis-finviz"> source_code_jovian </a>(Jupyter Notebook)

# ### Inspiration for the project:
# Stock Market is a huge gamble for some because they are not informed with proper data to make the right decisions. People take a lot of time in deciding which Cafe they would walk into but not spend enough time on the stock they invest. It is because people have less time but this is when AI comes to the rescue. Abstractive summarization and webscrapping seems to aid us more to gather the required information to make the right decisions.
# <br>
# Big thumbs up to <a href="https://www.jovian.ai">Jovian</a> team for introducing the concept of sentiment analysis in their free course assignments which paved the way for the current pipelines of the project. 
# <br>
# Thanks to <a href="https://www.youtube.com/c/nicholasrenotte">Nicholas Reonette</a> for his work on NLP code which is highly customizable for any NLP project which is the base for the pipeline 2 of my project.

# ##### Structure of the project:
# 1. Install and Import Dependencies
# 2. Summarization models
# - Type1: Summarization Model ------------>(Basic newspaper3k)
# - Type2 Summarization Model ------------>(Financial Summarization Pegasus model)
# 3. News and Sentiment Pipeline 1: Finiviz website
# - 3a_1 Web Scrapping from finviz website using the ticker (Output: CSV file)
# - 3a_2 Web Scrapping from finviz website using the ticker_list (Output: CSV file)  
# - 3a_3  View the stock as a Data_frame and perform sentiment analysis
# - 3a_4 Cleaning the data in the data frame
# - 3a_5 Sentiment Analysis
# - 3a_6 Scraping Articles
# 4. News and Sentiment Pipeline 2: Stock News from Google & any Stock NEWS  website 
# - 4a_1 Search for Stock News using Google and Yahoo Finance and strip out unwanted URLs
# - 4a_2 Searching and Webscrapping final URLs
# - 4a_3 Summarizing
# - 4a_4 Adding Sentiment Analysis (Using transformers)
# - 4a_5 Export to CSV 

# ##### Module references:
# 1. Webscraping modules:
# - <a href="https://pypi.org/project/requests/">Requests Module</a>
# - <a href="https://pypi.org/project/beautifulsoup4/">Beautiful soup</a>
# 2. Standard modules:
# - <a href ="https://pandas.pydata.org/docs/getting_started/install.html">Pandas module</a>
# - <a href="https://numpy.org/install/">Numpy module</a>
# - <a href="https://pypi.org/project/matplotlib/">Matplotlib</a>
# 3. Sentiment analyser modules:
# - <a href = "https://www.nltk.org/">NLTK</a>
# - <a href='https://textblob.readthedocs.io/en/dev/'>Textblob</a>
# - <a href="https://huggingface.co/transformers/">Transformers</a>
# 4. Article summarization:
# - <a href="https://newspaper.readthedocs.io/en/latest/Newspaper3k">Newspaper3k</a>
# - <a href="https://huggingface.co/human-centered-summarization/financial-summarization-pegasus">financial-summarization-pegasus</a>

# # 1. Install and Import Dependencies

# In[ ]:


# Requests Module
get_ipython().system('pip install requests --upgrade --quiet')


# In[ ]:


# Parse and read html
get_ipython().system('pip install beautifulsoup4 --upgrade --quiet  ')


# In[ ]:


# handling and Manipulating Tabular Data
get_ipython().system('pip install pandas --upgrade --quiet')
# Numerical operations
get_ipython().system('pip install numpy')
# visualizer
get_ipython().system('pip install matplotlib')


# In[ ]:


# Sentiment analyzer
get_ipython().system('pip install -U textblob --upgrade --quiet')


# In[ ]:


# NLTK Module
# Using natural language tool kit to import stop words
# Sentiment analyzer
get_ipython().system('pip install nltk --upgrade  --quiet ')


# In[ ]:


# nltk downloads for the Project
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# In[ ]:


# Basic aricle summarization
get_ipython().system('pip install newspaper3k --upgrade --quiet')


# In[ ]:


# Deep learning article summarization
get_ipython().system('pip install transformers --upgrade --quiet')


# In[1]:


# Import the necessary libraries ------> requests
import requests
from urllib.request import urlopen,Request


# In[2]:


# Import the necessary libraries ------> BeautifulSoup
from bs4 import BeautifulSoup


# In[3]:


# Import the necessary libraries ------> Pandas,numpy,matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot


# In[4]:


# Import the necessary libraries ------> nltk Module, sentiment analyser nltk vader
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[5]:


# Import the necessary libraries ------> sentiment analyser textblob
from textblob import TextBlob
from textblob import Word


# In[6]:


# Import the necessary libraries ------> basic aricle summarization 
from newspaper import Article


# In[7]:


# Import the necessary libraries ------> Deep Learning Summarization
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
# Import the necessary libraries ------> sentiment analyser Pegasus
from transformers import pipeline


# In[8]:


# Strip out unwanted URLs
import re 


# In[9]:


# create CSVStock Sentiment Analysis and Summarization via Web scrapping
import csv


# # 2. Summarization models

# The summarization models abstracts the given text to logical and concise summarization.</br>
# 
# Example:<a href="https://huggingface.co/human-centered-summarization/financial-summarization-pegasus">Source:Huggingface Financial-summarization-pegasus</a> </br>
# 
# **Input**</br>
# 
# National Commercial Bank (NCB), Saudi Arabia’s largest lender by assets, agreed to buy rival Samba Financial Group for 15 dollars billion in the biggest banking takeover this year.NCB will pay 28.45 riyals (dollars7.58) for each Samba share, according to a statement on Sunday, valuing it at about 55.7 billion riyals. NCB will offer 0.739 new shares for each Samba share, at the lower end of the 0.736-0.787 ratio the banks set when they signed an initial framework agreement in June.The offer is a 3.5 percentage premium to Samba’s Oct. 8 closing price of 27.50 riyals and about 24 percentage higher than the level the shares traded at before the talks were made public. Bloomberg News first reported the merger discussions.The new bank will have total assets of more than 220 billion, creating the Gulf region third-largest lender. The entity’s 46 billion market capitalization nearly matches that of Qatar National Bank QPSC, which is still the Middle East’s biggest lender with about 268 billion of assets. </br>
# 
# **Output**</br>
# 
# NCB to pay 28.45 riyals for each Samba share. Deal will create Gulf region’s third-largest lender

# # Type1: Summarization Model (Basic newspaper3k)

# All the three function takes inputs such as URLs, df(url) and file(url) and scrapes the articles.

# In[103]:


"""     Get the Article....
#author=[],article_date= []
#author.append(article.authors),article_date.append(article.publish_date),df and file and scrapes the Urls for summaries.
"""    
def newspaper3k_summary_from_df(df,column_url_name="URL",output_file_name='summaries'):
    url_df= df[column_url_name]
    article_summary=[]
    title=[]
    counter= 0
    for url in url_df:
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            title.append(article.title)
            article_summary.append(article.summary)
            counter+=1
            print(counter)
        except Exception as e:
            title.append(e)
            article_summary.append(e)
            counter+=1
            print(counter)
    data = {'Article_title':title,"Article_summary": article_summary}#"Date_Time":article_date,"Author": author,
    summary_df = pd.DataFrame.from_dict(data)
    summary_df["URL"] =url_df
    summary_df.to_csv(output_file_name+".csv")
    print(output_file_name+".csv is created")
    return summary_df
def newspaper3k_summary_from_csvfile(file_name,output_file_name='summaries'):
    df = pd.read_csv(file_name).drop(["Unnamed: 0"],axis = 1)
    url_df= df["URL"]
    article_summary=[]
    title=[]
    counter= 0
    for url in url_df:
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            title.append(article.title)
            article_summary.append(article.summary)
            counter+=1
            print(counter)
        except Exception as e:
            title.append(e)
            article_summary.append(e)
            counter+=1
            print(counter)    
    data = {'Article_title':title,"Article_summary": article_summary}
    summary_df = pd.DataFrame.from_dict(data)
    summary_df["URL"] =url_df
    summary_df.to_csv(output_file_name+".csv")
    print(output_file_name+".csv is created")
    return summary_df


# # Type2 Summarization Model (Financial Summarization Pegasus model)

# In[ ]:


#model_setup
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)


# In[ ]:


def pegasus_summarize(articles):
    try:
        summaries = []
        for article in articles:
            input_ids = tokenizer.encode(article, return_tensors='pt')
            output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            summaries.append(summary)
    except Exception as e:
            summaries.append(e)
    return summaries


# ## 3A News and Sentiment Pipeline 1: Finiviz website

# ### 3a_1     Web Scrapping from finviz website using the ticker (Output: CSV file)

# 1. function: **finviz_parser_data(ticker)**: This function is defined to scrape from the <a href="https://finviz.com/quote.ashx?t={}">Finviz website</a> using the requests library. Then the downloaded content should have a status code of 200 or greater.</br>
# The BeautifulSoup class is used to parse the html response and returned as soup. It is to be noted that soup is bs4.BeautifulSoup.</br>
# </br>
# 2. function: **correct_time_formatting(time_data)**: This function helps to rectify the bad date & time format acquired from the finviz website to standardized format. 
# - Before the function execution: </br>
# 	
# 0	Sep-20-21 07:53AM  
# 1	06:48AM  
# 2	06:46AM  
# 3	12:01AM  
# 4	Sep-19-21 06:45AM  
# 5	Sep-18-21 05:50PM  
# 6	10:34AM  
# </br>
# - After the function execution: </br>
# 	
# 0	Sep-20-21 07:53AM  
# 1	Sep-20-21 06:48AM  
# 2	Sep-20-21 06:46AM  
# 3	Sep-20-21 12:01AM  
# 4	Sep-19-21 06:45AM  
# 5	Sep-18-21 05:50PM  
# 6	Sep-18-21 10:34AM  
# 
# 3. function: **finviz_create_write_data(soup,file_name="MSFT")**: The soup is passed as position argument and the file_name is passed as keyword argument hence the file_name is customizable. </br>
# Example: finviz_create_write_data(soup,file_name="Amazon") </br>
# The function basically scrapes the Url,time, News Reporter and News headline. </br>
# It creates a dataframe using Pandas and writes it to a CSV and also returns the dataframe. </br>

# In[28]:



def finviz_parser_data(ticker):
    
    url = 'https://finviz.com/quote.ashx?t={}'.format(ticker)
    # sending request for getting the html code of the Url
    try:
        request = Request(url=url,headers={'user-agent':'my-app'})
        response = urlopen(request)

        #parsing the HTML with BeautifulSoup
        soup = BeautifulSoup(response,'html')
        return soup
    except Exception as e:
        print(e)
    
def correct_time_formatting(time_data):
    date = []
    time=[]
    for z in time_data:
        a = z.split(" ")
        if len(a) == 2:
            date.append(a[0])
            time.append(a[1])
        else:
            date.append("r")
            time.append(a[0])
    l=0
    r=1
    lister=[]
    #print(l,r)
    while r<len(date):
        if len(date[r]) ==9:
            lister.append(date[l:r])
            #print(l,r)
            l=r
            #print(l,r)
        elif r== len(date)-1:                      
                r=len(date)    
                #print(l,r)
                lister.append(date[l:r])
        r+=1
    n =0
    while n <len(lister):

        lister[n] =[lister[n][0] for x in lister[n] if x=='r' or x==lister[n][0] ]
        n+=1
    final_time= []
    y =0
    while y<len(lister):
        final_time+=lister[y]
        y+=1    
    count = 0
    time_correct =[]
    while count<len(final_time):
        time_correct.append((final_time[count]+" "+time[count]))
        count+=1
    return time_correct

def finviz_create_write_data(soup,file_name="MSFT"):   
    try:
        news_reporter_title = [row.text for row in soup.find_all(class_ ='news-link-right') if row is not None]
        #news_reporter_title
        news_reported = [row.text for row in soup.find_all(class_ ='news-link-left') if row is not None]
        #news_reported
        news_url = [row.find('a',href=True)["href"] for row in soup.find_all(class_ ='news-link-left') if row is not None]
        '''
        solution 2:
        atags = [row.find('a') for row in soup.find_all(class_ ='news-link-left') if row is not None]
        news_url = [link['href'] for link in atags]
        '''
        date_data = [row.text for row in soup.find_all('td', attrs ={"width":"130",'align':'right'}) if row is not None]
        time = correct_time_formatting(date_data)
    except Exception as e:
        print(e)
    data = { "Time":time,'News Reporter': news_reporter_title,"News Headline": news_reported, "URL": news_url }
    finviz_news_df = pd.DataFrame.from_dict(data)
    finviz_news_df.to_csv(file_name + '_finviz_stock.csv')
    print(file_name + "_finviz_stock.csv is created" )
    return finviz_news_df


# In[29]:


soup = finviz_parser_data("TSLA")
finviz_create_write_data(soup,file_name="Tesla")


# ### 3a_2 Web Scrapping from finviz website using the ticker_list (Output: CSV file)  
# - finviz_url = 'https://finviz.com/quote.ashx?t='

# 1.function:**create_csv_ticker_list(ticker_list)**: This function automates the process for a ticker_list containing multiple stocks.

# In[ ]:


ticker_list = ['WOOF','MSFT',"GOOG",'FB',"AMZN"]


# In[34]:


def create_csv_ticker_list(ticker_list):
    try:
        for ticker in ticker_list:
            soup = finviz_parser_data(ticker)
            finviz_create_write_data(soup,file_name=ticker)
    except Exception as e:
        print(e)


# In[33]:


create_csv_ticker_list(ticker_list)


# ### 3a_3  View the stock as a Data_frame and perform sentiment analysis

# 1.**def finviz_view_pandas_dataframe(ticker)**:Sometimes an analyst needs to perform calculations on the dataframe from a paricular stock this function aids in the process of analysis.</br>
# 
# **For an example let us take Google stock and analyse**

# In[39]:


def finviz_view_pandas_dataframe(ticker):
    url = 'https://finviz.com/quote.ashx?t={}'.format(ticker)
    # sending request for getting the html code of the Url
    try:
        request = Request(url=url,headers={'user-agent':'my-app'})
        response = urlopen(request)
        news_reporter_title = [row.text for row in soup.find_all(class_ ='news-link-right') if row is not None]
        news_reported = [row.text for row in soup.find_all(class_ ='news-link-left') if row is not None]
        news_url = [row.find('a',href=True)["href"] for row in soup.find_all(class_ ='news-link-left') if row is not None]
        date_data = [row.text for row in soup.find_all('td', attrs ={"width":"130",'align':'right'}) if row is not None]
        time = correct_time_formatting(date_data)
    except Exception as e:
        print(e)
    data = { "Time":time,'News Reporter': news_reporter_title,"News Headline": news_reported, "URL": news_url }
    finviz_news_df = pd.DataFrame.from_dict(data)
    return finviz_news_df


# In[44]:


google_stock = finviz_view_pandas_dataframe('GOOG')
google_stock


# In[48]:


google_stock["Time_pdformat"]= pd.to_datetime(google_stock['Time'],infer_datetime_format=True)
google_stock


# ### 3a_4 Cleaning the data in the data frame

# 1. function:**clean_data(df,column_filter ='News Headline',other_column='Time')**:The sentiment analyzer that we use if effecient like transformers or lower effecient analyzer works much better when the text is cleaned like lower casing, removing punctuation marks, removing stop words and lemmatizing the text. 
# 2. function:**(Optional)find_unnecessary_stop_words(df,count) & cleaning_secondary(df,apply_column = "lemmatizated"):**
# The other stop words has to be finded by manual search and these functions aid the process.

# In[49]:


from nltk.corpus import stopwords
stop_words = stopwords.words("english")
stop_words


# In[50]:


def clean_data(df,column_filter ='News Headline',other_column='Time_pdformat'):
    try:
        new_df = df.filter([column_filter,other_column])
        new_df['lower_case_headlines'] = new_df[column_filter].apply(lambda x: " ".join(word.lower() for word in x.split()))
        new_df['punctuation_remove'] = new_df['lower_case_headlines'].str.replace("[^\w\s]","",regex = True)
        new_df["stop_words_removed"] = new_df['punctuation_remove'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
        new_df['lemmatizated'] = new_df["stop_words_removed"].apply(lambda x: ' '.join(Word(word).lemmatize() for word in x.split()))
        return new_df
    except Exception as e:
        print(e)
# To find other unnecessary stop word -------->Optional function
def find_unnecessary_stop_words(df,count):
    try:
        series = pd.Series(''.join(df["lemmatizated"]).split()).value_counts()[:count]
        return series
    except Exception as e:
        print(e)
def cleaning_secondary(df,apply_column = "lemmatizated"):
    try:
        df['final_sentiment_cleaned'] =df[apply_column].apply(lambda x: " ".join(word for word in x.split() if word not in other_stop_words ))
        return df
    except Exception as e:
        print(e)


# In[52]:


cleaned_df = clean_data(google_stock,column_filter ='News Headline',other_column='Time_pdformat') #other_column is generally time field in df
cleaned_df


# In[53]:


series = find_unnecessary_stop_words(cleaned_df,30)
series


# In[55]:


# manual analysis
other_stop_words = ['ev','pickup',"stock",'china']


# In[56]:


cleaned_final = cleaning_secondary(cleaned_df)
cleaned_final


# ### 3a_5 Sentiment Analysis

# #### From Dataframe
# 1.function: **sentiment_analyzer(df,column_applied_df = "final_sentiment_cleaned",other_column="Time_pdformat")**:Basically the function uses sentiment analyzers like nltk vader and textblob with df as input.

# In[71]:


def sentiment_analyzer(df,column_applied_df = "final_sentiment_cleaned",other_column="Time_pdformat"):
    try:
        
        analyzer = SentimentIntensityAnalyzer()
        df['nltk_subjective'] = df[column_applied_df].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        df['nltk_positive'] = df[column_applied_df].apply(lambda x: analyzer.polarity_scores(x)['pos'])
        df['nltk_neutral'] = df[column_applied_df].apply(lambda x: analyzer.polarity_scores(x)['neu'])
        df['nltk_negative'] = df[column_applied_df].apply(lambda x: analyzer.polarity_scores(x)['neg'])
        df['textblob_polarity'] = df[column_applied_df].apply(lambda x: TextBlob(x).sentiment[0])
        df['textblob_subjective'] = df[column_applied_df].apply(lambda x: TextBlob(x).sentiment[1])
        #'nltk_positive','nltk_neutral','nltk_negative',
        new_df = df.filter([other_column,'News Headline',column_applied_df,'nltk_subjective','textblob_polarity','textblob_subjective'])
        return new_df
    except Exception as e:
        print(e)
        #(Optional)
        #sentiment = pipeline('sentiment-analysis')
        #df['transformers_label']= df[column_applied_df].apply(lambda x: sentiment(x)['label'])
        #df['transformers_score']= df[column_applied_df].apply(lambda x: sentiment(x)['score'])


# In[72]:


sentiment = sentiment_analyzer(cleaned_final,column_applied_df = "final_sentiment_cleaned") #other_column is generally time field in df
sentiment


# In[76]:


sentiment_df= sentiment.sort_values(by=['nltk_subjective','textblob_polarity',"textblob_subjective"],ascending=[True,True,True],na_position='first')
sentiment_df


# In[79]:


sentiment_df.to_csv('sentiment_google.csv')
print("file created")


# #### From file

# In[80]:


def sentiment_analyzer_from_file(file_name):
    import pandas as pd
    try:
        df = pd.read_csv(file_name).drop(["Unnamed: 0"],axis = 1)
        clean_df = clean_data(df,column_filter ='News Headline')
        sentiment_df = sentiment_analyzer(clean_df,column_applied_df = "lemmatizated")
        return sentiment_df
    except Exception as e:
        print(e)


# In[81]:


sentiment_analyzer_from_file('WOOF_finviz_stock.csv')


# ### 3a_6 Scraping Articles

# ##### From Urls

# In[96]:


#google_stock = finviz_view_pandas_dataframe('GOOG') previously executed
Url= google_stock["URL"]
Url


# ##### From df

# In[100]:


newspaper3k_summary_from_df(google_stock,output_file_name='google_summaries')


# ##### From file

# In[104]:


newspaper3k_summary_from_csvfile("MSFT_finviz_stock.csv",output_file_name='MSFT_6_summaries')


# ## 4B News and Sentiment Pipeline 2: Stock News from Google & any Stock NEWS  website 

# ### 4b_1 Search for Stock News using Google and Yahoo Finance and strip out unwanted URLs
# 1. **def google_search_stocknews(ticker,num=100,site="yahoo+finance")**: This function takes the "ticker" as positional argument and "num" is the number of pages to search and "site" can be any reliable site.
# 2. **def strip_unwanted_urls(urls)**: As the name suggests it removes the unclean urls from the list and filters the urls which meets the standard.

# In[82]:


tickers_2 = ['MSFT','TSLA', 'BTC']


# In[84]:


def google_search_stocknews(ticker,num=100,site="yahoo+finance"):
    try:
        search_url = "https://www.google.com/search?q={}+{}&tbm=nws&num={}".format(site,ticker,num)
        # url_analysis: https://www.google.com/search?q={query}&tbm=nws&num=100
        # &tbm=nws: google new Tab, &num={} example: 100 and will return the top 100 results, query:site +ticker
        r = requests.get(search_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        atags = soup.find_all('a')
        hrefs = [link['href'] for link in atags]
        return hrefs 
    except Exception as e:
        print(e)


# In[85]:


#site is a Keyword argument and its default Value is yahoo+finance
#other examples of site can be livemint,money+control,hindustan+times
example_for_site = google_search_stocknews("AAPL",50,site='hindustan+times') 
example_for_site


# In[88]:


# storing the URLs in a dictionary
raw_urls_to_dictionary = {ticker:google_search_stocknews(ticker,5) for ticker in tickers_2}
raw_urls_to_dictionary


# In[89]:


def strip_unwanted_urls(urls):
    try:
        # list of x that we dont want in our urls
        strip_list = ['maps','policies', 'preferences', 'accounts', 'support']
        value = []
        # LOOPING through URLs in oone ticker at a time
        for url in urls: 
            if 'https://' in url and not any(strip_word in url for strip_word in strip_list):
                result = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                value.append(result)
        return list(set(value))
        '''
        list(set(value)) It removes the duplicate Values
        Solution 2:output =[]
        [output.append(x) for x in value if x not in output]
        '''
    except Exception as e:
        print(e)


# In[90]:


(strip_unwanted_urls(raw_urls_to_dictionary['TSLA'])) 
len(strip_unwanted_urls(raw_urls_to_dictionary['TSLA']))


# In[92]:


final_urls_list = {ticker:strip_unwanted_urls(raw_urls_to_dictionary[ticker]) for ticker in raw_urls_to_dictionary.keys()}
#final_urls_list = {ticker:strip_unwanted_urls(raw_urls_to_dictionary[ticker]) for ticker in ticker_2}
len(final_urls_list['MSFT'])
final_urls_list


# ### 4b_2 Searching and Webscrapping final URLs
# 1. **def scrape_articles(URLs):** The function scrapes the Url for text and limits it to 350 words by parsing it.

# In[105]:


def scrape_articles(URLs):
    try:
        ARTICLES = []
        counter =0
        for url in URLs: 
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = [paragraph.text for paragraph in paragraphs]
            words = ' '.join(text).split(' ')[:350]
            ARTICLE = ' '.join(words)
            ARTICLES.append(ARTICLE)
            print(counter)
            counter+=1
        return ARTICLES
    except Exception as e:
        print(e)


# In[106]:


articles = {ticker:scrape_articles(final_urls_list[ticker]) for ticker in final_urls_list.keys()}
#articles = {ticker:scrape_articles(final_urls_list[ticker]) for ticker in tickers_2}     
articles


# In[107]:


articles["TSLA"]


# ### 4b_3 Summarizing

# ##### Type2: Summarization Model ------------> Pegasus Model

# In[ ]:


summaries = {ticker:pegasus_summarize(articles[ticker]) for ticker in tickers_2}
summaries


# ###  4b_4 Adding Sentiment Analysis (Using transformers)

# In[ ]:


sentiment = pipeline('sentiment-analysis')


# In[ ]:


scores = {ticker:sentiment(summaries[ticker]) for ticker in tickers_2}
scores


# ### 4b_5 Export to CSV

# In[36]:


def create_csv(summaries, scores, final_urls_list):
    try:
        output = []
        for ticker in tickers_2:
            for counter in range(len(summaries[ticker])):
                output_this = [
                    ticker,
                    summaries[ticker][counter],
                    scores[ticker][counter]['label'],
                    scores[ticker][counter]['score'],
                    final_urls_list_list[ticker][counter]
                ]
                output.append(output_this)
        output.insert(0, ['Ticker', 'Summary', 'Label', 'Confidence', 'URL'])
        with open('ticket_summaries.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerows(final_output)
        return output
    except Exception as e:
        print(e)


# In[ ]:





# In[ ]:





# In[1]:


import jovian
project = "Sentiment_analysis_finviz"
jovian.commit(project=project)


# In[ ]:




