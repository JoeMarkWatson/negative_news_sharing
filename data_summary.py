import pandas as pd
import datetime
import math
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import numpy as np
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



def manip_df(df_mdd, df_tweets, df_fb):
    """manipulate df.s inc merging and creating vars employed in analysis"""

    df = pd.merge(
        df_mdd[['text_id', 'mean_valence_vader_words', 'my_word_count', 'mean_my_word_len', 'mean_my_words_in_sen',
                'vader_words_count', 'mean_valence_vader_words_title', 'date', 'UK_cons_count', 'UK_lab_count',
                'US_rep_count', 'US_dem_count', 'US_lib_id_count', 'US_con_id_count']], df_tweets)

    df = pd.merge(df, df_fb)

    df = df[df['my_word_count'] > 99]

    df.dropna(subset=['mean_valence_vader_words'], how='all', inplace=True)

    df['year'] = list(map(lambda x: '20' + x[:2], df['date']))
    df['month'] = list(map(lambda x: x[3:5], df['date']))
    df['day_of_week'] = list(
        map(lambda x: datetime.date(int('20' + x[:2]), int(x[3:5]), int(x[-2:])).weekday(), df['date']))  # 0 is Mon

    df['n_tweetsPlusRTs'] = df['n_tweets'] + df['n_RTs']

    df['neg_art'] = np.where(df['mean_valence_vader_words'] < 0, 1, 0)  # better for story and means that exact same neg
    # cutoff can be used for tweet-focused 'why' analysis
    df['pos_art'] = df['neg_art']*(-1)  # to give more intuitive in-group interaction findings
    df['neg_art_r'] = np.where(df['mean_valence_vader_words'] < df['mean_valence_vader_words'].mean(), 1,
                               0)  # mean-based neg_art cutoff used for robustness check

    df['neg_title'] = np.where(df['mean_valence_vader_words_title'] < 0, 1, 0)

    df['n_tweetsPlusPosts'] = (df['n_tweets'] + df['fb_shares'])

    df['US_rep_con_count'] = df['US_rep_count'] + df['US_con_id_count']
    df['US_dem_lib_count'] = df['US_dem_count'] + df['US_lib_id_count']
    df['UK_cons_con_count'] = df['UK_cons_count'] + df['US_con_id_count']
    df['UK_lab_lib_count'] = df['UK_lab_count'] + df['US_lib_id_count']

    df['maj_rep'] = df['US_rep_con_count'] > df['US_dem_lib_count']
    df['maj_dem'] = df['US_rep_con_count'] < df['US_dem_lib_count']
    df['maj_con'] = df['UK_cons_con_count'] > df['UK_lab_lib_count']
    df['maj_lab'] = df['UK_cons_con_count'] < df['UK_lab_lib_count']

    return df

# for descriptives
def art_df_descriptives(manip_dfs_list_df):

    data_name = 'News articles'
    manip_dfs_list_df['n_RTs'] = manip_dfs_list_df['n_RTs'].replace(np.nan, 0)

    num_of_docs = int(len(manip_dfs_list_df))
    mean_words_per_doc = np.round(manip_dfs_list_df['my_word_count'].mean(), 3)  # same as how words counted for tweets, but
    mean_v_words_per_doc = np.round(manip_dfs_list_df['vader_words_count'].mean(), 3)
    mean_doc_senti = np.round(manip_dfs_list_df['mean_valence_vader_words'].mean(), 3)
    prop_of_docs_neg = np.round(manip_dfs_list_df['neg_art'].mean(), 3)
    mean_tweets_cont_art_link = np.round(manip_dfs_list_df['n_tweets'].mean(), 3)
    mean_tweets_RTs_cont_art_link = np.round(manip_dfs_list_df['n_tweetsPlusRTs'].mean(), 3)
    mean_posts_cont_art_link = np.round(manip_dfs_list_df['fb_shares'].mean(), 3)

    mean_retweets = np.nan
    mean_replies = np.nan
    mean_likes = np.nan

    return [data_name, num_of_docs, mean_words_per_doc, mean_v_words_per_doc, mean_doc_senti, prop_of_docs_neg,
            mean_posts_cont_art_link, mean_tweets_cont_art_link, mean_tweets_RTs_cont_art_link, mean_retweets,
            mean_replies, mean_likes]

def art_df_descriptives_agg(manip_dfs_list_df, data_name='All'):

    data_name = data_name
    manip_dfs_list_df['n_RTs'] = manip_dfs_list_df['n_RTs'].replace(np.nan, 0)

    num_of_docs = int(len(manip_dfs_list_df))
    mean_words_per_doc = np.round(manip_dfs_list_df['my_word_count'].mean(), 3)  # same as how words counted for tweets, but
    mean_v_words_per_doc = np.round(manip_dfs_list_df['vader_words_count'].mean(), 3)
    mean_doc_senti = np.round(manip_dfs_list_df['mean_valence_vader_words'].mean(), 3)
    prop_of_docs_neg = np.round(manip_dfs_list_df['neg_art'].mean(), 3)

    Mean_tweets_posts_cont_art_link = np.round(manip_dfs_list_df['n_tweetsPlusPosts'].mean(), 3)
    Mean_tweets_posts_c_std = np.round(manip_dfs_list_df['n_tweetsPlusPosts'].std(), 3)

    mean_posts_cont_art_link = np.round(manip_dfs_list_df['fb_shares'].mean(), 3)
    mean_posts_c_std = np.round(manip_dfs_list_df['fb_shares'].std(), 3)

    mean_tweets_cont_art_link = np.round(manip_dfs_list_df['n_tweets'].mean(), 3)
    mean_tweets_c_std = np.round(manip_dfs_list_df['n_tweets'].std(), 3)

    mean_tweets_RTs_cont_art_link = np.round(manip_dfs_list_df['n_tweetsPlusRTs'].mean(), 3)
    mean_tweets_RTs_c_std = np.round(manip_dfs_list_df['n_tweetsPlusRTs'].std(), 3)

    return [data_name, num_of_docs, mean_words_per_doc, mean_v_words_per_doc, mean_doc_senti, prop_of_docs_neg,
            Mean_tweets_posts_cont_art_link, Mean_tweets_posts_c_std,
            mean_posts_cont_art_link, mean_posts_c_std,
            mean_tweets_cont_art_link, mean_tweets_c_std,
            mean_tweets_RTs_cont_art_link, mean_tweets_RTs_c_std]

def post_df_descriptives(posts_df, data_name='Tweets VADER'):

    if data_name == 'Facebook all':
        num_of_docs = posts_df['fb_shares'].sum()
        (mean_words_per_doc, mean_v_words_per_doc, mean_doc_senti, prop_of_docs_neg, mean_retweets, mean_replies,
         mean_likes) = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        if data_name == 'Tweets VADER':
            posts_df = posts_df[posts_df['vader_words_count_tweet'] > 0]
        num_of_docs = int(len(posts_df))
        mean_words_per_doc = np.round(posts_df['my_word_count_tweet'].mean(), 3)
        mean_v_words_per_doc = np.round(posts_df['vader_words_count_tweet'].mean(), 3)
        mean_doc_senti = np.round(posts_df['mean_valence_vader_words_tweet'].mean(), 3)
        below_zero_count = posts_df['mean_valence_vader_words_tweet'].lt(0).sum()
        total_valid_values = posts_df['mean_valence_vader_words_tweet'].notna().sum()
        prop_of_docs_neg = np.round(below_zero_count / total_valid_values, 3)
        mean_retweets = np.round(posts_df['n_RTs'].mean(), 3)
        mean_replies = np.round(posts_df['n_replies'].mean(), 3)
        mean_likes = np.round(posts_df['n_likes'].mean(), 3)
        mean_doc_senti = mean_doc_senti if (data_name == 'Tweets VADER') else np.nan
        prop_of_docs_neg = prop_of_docs_neg if (data_name == 'Tweets VADER') else np.nan  # funny to provide
        # mean_doc_senti and prop_of_docs_neg for all_t_descs as some of these tweets have no vader words

    return [data_name, num_of_docs, mean_words_per_doc, mean_v_words_per_doc, mean_doc_senti, prop_of_docs_neg,
            np.nan, np.nan, np.nan, mean_retweets, mean_replies, mean_likes]

def add_dummies(df, cols=['year', 'day_of_week', 'month', 'dominant_topic'],
                drop_vars=['year_2019', 'day_of_week_0', 'month_01', 'dominant_topic_0']):
    """make dummies, store their col names and mix them into df"""
    dummy_df = pd.get_dummies(
        df[cols].astype(str))
    dummy_df = dummy_df.drop(columns=drop_vars)
    mddsld = pd.concat([df.reset_index(), dummy_df.reset_index()], axis=1, join='inner')
    mddsld = mddsld.drop(columns=['index'])
    return mddsld


# load data

df_tweetsDM = pd.read_csv('data/twitter_shares_DM.csv')  # Daily Mail
df_i_tweetsDM = pd.read_csv('data/twitter_i_shares_DM.csv')
df_fb_sharesDM = pd.read_csv('data/fb_shares_DM.csv')
df_mddDM = pd.read_csv('data/news_DM.csv')

df_tweetsG = pd.read_csv('data/twitter_shares_G.csv')  # Guardian
df_i_tweetsG = pd.read_csv('data/twitter_i_shares_G.csv')
df_fb_sharesG = pd.read_csv('data/fb_shares_G.csv')
df_mddG = pd.read_csv('data/news_G.csv')

df_tweetsNYP = pd.read_csv('data/twitter_shares_NYP.csv')  # New York Post
df_i_tweetsNYP = pd.read_csv('data/twitter_i_shares_NYP.csv')
df_fb_sharesNYP = pd.read_csv('data/fb_shares_NYP.csv')
df_mddNYP = pd.read_csv('data/news_NYP.csv')

df_tweetsNYT = pd.read_csv('data/twitter_shares_NYT.csv')  # New York Times
df_i_tweetsNYT = pd.read_csv('data/twitter_i_shares_NYT.csv')
df_fb_sharesNYT = pd.read_csv('data/fb_shares_NYT.csv')
df_mddNYT = pd.read_csv('data/news_NYT.csv')


# Apply functions

dfs_i_list = [df_i_tweetsDM, df_i_tweetsG, df_i_tweetsNYP, df_i_tweetsNYT]
dfs_list = [[df_tweetsDM, df_fb_sharesDM, df_mddDM], [df_tweetsG, df_fb_sharesG, df_mddG],
            [df_tweetsNYP, df_fb_sharesNYP, df_mddNYP], [df_tweetsNYT, df_fb_sharesNYT, df_mddNYT]]
manip_dfs_list = [manip_df(df_mdd=dfs[2], df_tweets=dfs[0], df_fb=dfs[1]) for dfs in dfs_list]

# generate descriptives for main article text

data_overviews_list = []
for i in range(4):
    news_source = 'Daily Mail'
    if i == 1:
        news_source = 'Guardian'
    if i == 2:
        news_source = 'New York Post'
    if i == 3:
        news_source = 'New York Times'

    art_descs = art_df_descriptives(manip_dfs_list[i])
    all_fb_descs = post_df_descriptives(manip_dfs_list[i], data_name='Facebook all')
    all_t_descs = post_df_descriptives(dfs_i_list[i], data_name='Tweets all')
    vad_t_descs = post_df_descriptives(dfs_i_list[i], data_name='Tweets VADER')
    data_overview = pd.DataFrame(zip(art_descs, all_fb_descs, all_t_descs, vad_t_descs)).T
    data_overview.columns = ['data_name', 'num_of_docs', 'mean_words_per_doc', 'mean_v_words_per_doc', 'mean_doc_senti',
                             'prop_of_docs_neg', 'mean_FbPosts_cont_art_link', 'mean_tweets_cont_art_link', 'mean_tweets_RTs_cont_art_link',
                             'mean_retweets', 'mean_replies', 'mean_likes']
    data_overview['news_source'] = news_source
    data_overview = data_overview[['news_source', 'data_name', 'num_of_docs', 'mean_words_per_doc', 'mean_v_words_per_doc', 'mean_doc_senti',
                             'prop_of_docs_neg', 'mean_FbPosts_cont_art_link', 'mean_tweets_cont_art_link', 'mean_tweets_RTs_cont_art_link',
                             'mean_retweets', 'mean_replies', 'mean_likes']]
    data_overviews_list.append(data_overview)

data_overviews_df = pd.concat(data_overviews_list, axis=0, ignore_index=True)

#data_overviews_df.to_csv('data/data_summary_file.csv', index=False)


# generate descriptives for aggregated dataframes

paper_names = ['DM', 'G', 'NYP', 'NYT']

# create maj_out_or_in column, following data_modelling
for i, df in enumerate(manip_dfs_list):
    df['paper'] = paper_names[i]
    df['right_lean'] = 1 if paper_names[i] in ['DM', 'NYP'] else 0
all_manip_dfs = pd.concat(manip_dfs_list, axis=0, ignore_index=True)
all_manip_dfs = add_dummies(df=all_manip_dfs, cols=['paper'], drop_vars=['paper_DM'])  # paper dummies for agg models

for i, df in enumerate(manip_dfs_list):
    df['paper'] = paper_names[i]
    df['right_lean'] = 1 if paper_names[i] in ['DM', 'NYP'] else 0
all_manip_dfs = pd.concat(manip_dfs_list, axis=0, ignore_index=True)
all_manip_dfs = add_dummies(df=all_manip_dfs, cols=['paper'], drop_vars=['paper_DM'])  # paper dummies for agg models

all_manip_dfs['maj_out_group'] = np.where(
    ((all_manip_dfs['paper'] == 'G') & (all_manip_dfs['maj_con'] == 1)) |
    ((all_manip_dfs['paper'] == 'DM') & (all_manip_dfs['maj_lab'] == 1)) |
    ((all_manip_dfs['paper'] == 'NYT') & (all_manip_dfs['maj_rep'] == 1)) |
    ((all_manip_dfs['paper'] == 'NYP') & (all_manip_dfs['maj_dem'] == 1)),
    1, 0)
all_manip_dfs['maj_in_group'] = np.where(
    ((all_manip_dfs['paper'] == 'G') & (all_manip_dfs['maj_lab'] == 1)) |
    ((all_manip_dfs['paper'] == 'DM') & (all_manip_dfs['maj_con'] == 1)) |
    ((all_manip_dfs['paper'] == 'NYT') & (all_manip_dfs['maj_dem'] == 1)) |
    ((all_manip_dfs['paper'] == 'NYP') & (all_manip_dfs['maj_rep'] == 1)),
    1, 0)
all_manip_dfs['maj_out_or_in'] = np.where(
    ((all_manip_dfs['maj_in_group']) | (all_manip_dfs['maj_out_group'])),
    1, 0)

# create subset based on maj_out_or_in column
all_SRQ_dfs = all_manip_dfs[all_manip_dfs['maj_out_or_in'] == 1]

# create aggregated df descriptives
art_descs_all = art_df_descriptives_agg(all_manip_dfs)
art_descs_polit = art_df_descriptives_agg(all_SRQ_dfs, data_name='Political articles')

agg_overview = pd.DataFrame({'All': art_descs_all, 'Political articles': art_descs_polit}).T
agg_overview.columns = ['news_article_subet', 'num_of_docs', 'mean_words_per_doc', 'mean_v_words_per_doc', 'mean_doc_senti',
                             'prop_of_docs_neg', 'mean_tweets_FbPosts_cont_art_link', 'mtfb_std',
                             'mean_FbPosts_cont_art_link', 'mfb_std',
                             'mean_tweets_cont_art_link', 'mt_std', 'mean_tweets_RTs_cont_art_link', 'mtrt_std']
