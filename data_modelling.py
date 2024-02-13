import pandas as pd
import datetime
import re
import matplotlib.pylab as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.formula.api as smf
import nltk
import math
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from joblib import Parallel, delayed  # for parallel processing


# def functions

def add_dummies(df, cols=['year', 'day_of_week', 'month', 'dominant_topic'],
                drop_vars=['year_2019', 'day_of_week_0', 'month_01', 'dominant_topic_0']):
    """make dummies, store their col names and mix them into mddsl"""
    dummy_df = pd.get_dummies(
        df[cols].astype(str))
    dummy_df = dummy_df.drop(columns=drop_vars)
    mddsld = pd.concat([df.reset_index(), dummy_df.reset_index()], axis=1, join='inner')
    mddsld = mddsld.drop(columns=['index'])
    return mddsld


def manip_df(df_mdd, df_tweets, df_fb, topics):
    """manipulate df.s inc merging and creating vars employed in analysis"""

    df = pd.merge(
        df_mdd[['text_id', 'mean_valence_vader_words', 'my_word_count', 'mean_my_word_len', 'mean_my_words_in_sen',
                'vader_words_count', 'mean_valence_vader_words_title', 'date', 'UK_cons_count', 'UK_lab_count',
                'US_rep_count', 'US_dem_count', 'US_lib_id_count', 'US_con_id_count']], df_tweets)

    df = pd.merge(df, df_fb)

    df = df[df['my_word_count'] > 99]
    # mdds = mdds[mdds['vader_words_count'] > 9]  # not adding this, as weakens comparison vs title neg

    df = pd.merge(df, topics[['text_id', 'dominant_topic']])  # add topic column
    df.dropna(subset=['mean_valence_vader_words'], how='all', inplace=True)

    df['year'] = list(map(lambda x: '20' + x[:2], df['date']))
    df['month'] = list(map(lambda x: x[3:5], df['date']))
    df['day_of_week'] = list(
        map(lambda x: datetime.date(int('20' + x[:2]), int(x[3:5]), int(x[-2:])).weekday(), df['date']))  # 0 is Mon

    df = add_dummies(df)

    df['my_word_count_log'] = np.log(df['my_word_count'] + 1)
    df['mean_my_word_len_log'] = np.log(df['mean_my_word_len'] + 1)
    df['mean_my_words_in_sen_log'] = np.log(df['mean_my_words_in_sen'] + 1)
    df['n_tweets_log'] = np.log(df['n_tweets'] + 1)

    df['n_tweetsPlusRTs_log'] = np.log((df['n_tweets'] + df['n_RTs']) + 1)
    df['n_replies_log'] = np.log(df['n_replies'] + 1)
    df['n_RTs_log'] = np.log(df['n_RTs'] + 1)
    df['n_likes_log'] = np.log(df['n_likes'] + 1)

    df['fb_shares_log'] = np.log(df['fb_shares'] + 1)

    df['n_tweetsPlusPosts_log'] = np.log((df['n_tweets'] + df['fb_shares']) + 1)

    df['any_pol_pers_term_US'] = df['US_rep_count'] + df['US_dem_count'] + df['US_lib_id_count'] + df['US_con_id_count'] > 0
    df['any_pol_pers_term_UK'] = df['UK_cons_count'] + df['UK_lab_count'] + df['US_lib_id_count'] + df['US_con_id_count'] > 0

    df['any_US_pol'] = df['US_rep_count'] + df['US_dem_count'] > 0
    df['any_UK_pol'] = df['UK_cons_count'] + df['UK_lab_count'] > 0

    df['US_rep_con_count'] = df['US_rep_count'] + df['US_con_id_count']
    df['US_dem_lib_count'] = df['US_dem_count'] + df['US_lib_id_count']
    df['UK_cons_con_count'] = df['UK_cons_count'] + df['US_con_id_count']
    df['UK_lab_lib_count'] = df['UK_lab_count'] + df['US_lib_id_count']

    df['maj_rep'] = df['US_rep_con_count'] > df['US_dem_lib_count']
    df['maj_dem'] = df['US_rep_con_count'] < df['US_dem_lib_count']
    df['maj_con'] = df['UK_cons_con_count'] > df['UK_lab_lib_count']
    df['maj_lab'] = df['UK_cons_con_count'] < df['UK_lab_lib_count']

    boolean_columns = ['any_pol_pers_term_US', 'any_pol_pers_term_UK', 'any_US_pol', 'maj_rep', 'maj_dem', 'any_UK_pol',
                       'maj_con', 'maj_lab']
    df[boolean_columns] = df[boolean_columns].astype(int)  # convert to facil output from later inf analysis funs

    df['neg_art'] = np.where(df['mean_valence_vader_words'] < 0, 1, 0)  # better for story and means that exact same neg
    # cutoff can be used for tweet-focused 'why' analysis
    df['pos_art'] = df['neg_art']*(-1)  # to give more intuitive in-group interaction findings
    df['neg_art_r'] = np.where(df['mean_valence_vader_words'] < df['mean_valence_vader_words'].mean(), 1,
                               0)  # mean-based neg_art cutoff used for robustness check

    df['neg_title'] = np.where(df['mean_valence_vader_words_title'] < 0, 1, 0)

    return df


def manip_i_df(df_i_tweets, manipd_df):
    df = pd.merge(df_i_tweets, manipd_df[['text_id', 'neg_art', 'neg_title', 'my_word_count_log', 'mean_my_word_len_log',
     'mean_my_words_in_sen_log', 'year_2020', 'year_2021', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3',
     'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
     'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12']], on='text_id')  # left/inner join

    df['char_length_tweet_log'] = np.log(df['char_length_tweet'])
    df['n_RTs_log'] = np.log(df['n_RTs'] + 1)
    df['n_replies_log'] = np.log(df['n_replies'] + 1)
    df['n_likes_log'] = np.log(df['n_likes'] + 1)

    return df


def print_plot_descs(df):
    """prints some mean values, makes some tweet dist plots, and runs an OLS model with no controls.
    OLS coeff could have some positive bias, hence the use of controls in inf models."""

    print("Number of rows in df:", len(df))
    print("Proportion of news articles that are negative:", df['neg_art'].mean())
    print("Total number of tweets concerning all news articles:", df['n_tweets'].sum())
    print("Mean number of tweets per news article:", df['n_tweets'].mean())

    print("Total number of FB shares concerning all news articles:", df['fb_shares'].sum())
    print("Mean number of FB shares per news article:", df['fb_shares'].mean())

    plt.hist(df['n_tweets_log'], bins=20)
    plt.show()

    plt.hist(df['fb_shares_log'], bins=20)
    plt.show()

    print("tweets OLS:")
    print(smf.ols("n_tweets_log ~ neg_art", data=df).fit().summary().tables[1])

    print("FB shares OLS:")
    print(smf.ols("fb_shares_log ~ neg_art", data=df).fit().summary().tables[1])


def run_standard_reg(Y, T, X, df, paper_name):
    """run a standard multivariate regression on df and print the output"""
    model_NT_1 = smf.ols(f"{Y}~{T}+{'+'.join(X)}", data=df).fit()
    print(model_NT_1.summary().tables[1])

    T_splits = T.split("+")

    point_estT1, lower_conf_intT1, upper_conf_intT1 = model_NT_1.params[T_splits[0]], \
        model_NT_1.conf_int(alpha=0.05).loc[T_splits[0]][0], \
        model_NT_1.conf_int(alpha=0.05).loc[T_splits[0]][1]

    if len(T_splits) > 1:
        point_estT2, lower_conf_intT2, upper_conf_intT2 = model_NT_1.params[T_splits[1]], \
            model_NT_1.conf_int(alpha=0.05).loc[T_splits[1]][0], \
            model_NT_1.conf_int(alpha=0.05).loc[T_splits[1]][1]

        if len(T_splits) == 2:
            point_estI, lower_conf_intI, upper_conf_intI = ['NA'] * 3

        else:
            point_estI, lower_conf_intI, upper_conf_intI = model_NT_1.params[re.sub("\\*", ":", T_splits[2])], \
                model_NT_1.conf_int(alpha=0.05).loc[re.sub("\\*", ":", T_splits[2])][0], \
                model_NT_1.conf_int(alpha=0.05).loc[re.sub("\\*", ":", T_splits[2])][1]

    else:
        point_estT2, lower_conf_intT2, upper_conf_intT2, \
            point_estI, lower_conf_intI, upper_conf_intI = ['NA'] * 6

    controls = 'core plus topic' if 'dominant_topic' in X else 'core'

    return [paper_name, 'multiple regression', Y, T, controls, len(df), point_estT1,
            lower_conf_intT1, upper_conf_intT1, point_estT2, lower_conf_intT2, upper_conf_intT2,
            point_estI, lower_conf_intI, upper_conf_intI]


def run_ps(df, X, T, Y):
    """compute the IPTW estimator within run_prop_score_model"""
    ps = LogisticRegression(penalty=None, max_iter=10000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]  # estimate the propensity score
    weight = (df[T] - ps) / (ps * (1 - ps))  # define the weights
    return np.mean(weight * df[Y])  # compute the ATE


def run_prop_score_model(df, X, T, Y, paper_name, bss=1000, ran_seed=0):
    """calculate mean bootstrapped ATE for PS and plots bootstrap distribution.
    Use bss of 1000 for when writing up results"""

    np.random.seed(ran_seed)
    ates = Parallel(n_jobs=-1)(delayed(run_ps)(df.sample(frac=1, replace=True), X, T, Y)
                              for _ in range(bss))
    ates = np.array(ates)
    print(f"ATE: {ates.mean()}")
    print(f"95% C.I.: {(np.percentile(ates, 2.5), np.percentile(ates, 97.5))}")

    sns.distplot(ates, bins=17, kde=False)  # added bins=17 to match earlier plots
    plt.vlines(np.percentile(ates, 2.5), 0, 35, linestyles="dotted")
    plt.vlines(np.percentile(ates, 97.5), 0, 35, linestyles="dotted", label="95% CI")
    plt.title("ATE Bootstrap Distribution: PS")
    plt.legend()
    plt.show()

    controls = 'core plus topic' if 'dominant_topic' in X else 'core'

    return [paper_name, 'propensity score', Y, T, controls, len(df), ates.mean(), np.percentile(ates, 2.5),
            np.percentile(ates, 97.5)] + ['NA']*6


def doubly_robust(df, X, T, Y):
    """compute the doubly robust est within run_dr_model"""
    ps = LogisticRegression(penalty=None, max_iter=10000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )


def run_dr_model(df, X, T, Y, paper_name, bss=1000, ran_seed=0):
    """calculate mean bootstrapped ATE for DR and plots bootstrap distribution.
        Use bss of 1000 for when writing up results"""
    np.random.seed(ran_seed)
    # run 1000 bootstrap samples
    ates = Parallel(n_jobs=-1)(delayed(doubly_robust)(df.sample(frac=1, replace=True), X, T, Y)
                              for _ in range(bss))
    ates = np.array(ates)
    print(f"ATE: {ates.mean()}")
    print(f"ATE 95% CI:", (np.percentile(ates, 2.5), np.percentile(ates, 97.5)))

    sns.distplot(ates, bins=17, kde=False)  # added bins=17 to match earlier plots
    plt.vlines(np.percentile(ates, 2.5), 0, 35, linestyles="dotted")
    plt.vlines(np.percentile(ates, 97.5), 0, 35, linestyles="dotted", label="95% CI")
    #plt.title("ATE Bootstrap Distribution: DR")
    plt.legend()
    plt.show()

    controls = 'core plus topic' if 'dominant_topic' in X else 'core'

    return [paper_name, 'doubly robust', Y, T, controls, len(df), ates.mean(), np.percentile(ates, 2.5),
            np.percentile(ates, 97.5)] + ['NA']*6


def make_desc_plot(manip_dfs_list):
    df1 = manip_dfs_list[0][['text_id', 'n_tweets', 'n_RTs', 'n_tweets_log', 'fb_shares', 'fb_shares_log']]  # DM
    df2 = manip_dfs_list[1][['text_id', 'n_tweets', 'n_RTs', 'n_tweets_log', 'fb_shares', 'fb_shares_log']]  # G
    df3 = manip_dfs_list[2][['text_id', 'n_tweets', 'n_RTs', 'n_tweets_log', 'fb_shares', 'fb_shares_log']]  # NYP
    df4 = manip_dfs_list[3][['text_id', 'n_tweets', 'n_RTs', 'n_tweets_log', 'fb_shares', 'fb_shares_log']]  # NYT

    dfs_all = pd.concat([df1, df2, df3, df4])

    jitter_amount = 1
    jittered_n_tweets_log = dfs_all['n_tweets_log'] + np.random.uniform(-jitter_amount, jitter_amount, len(dfs_all))
    jittered_fb_shares_log = dfs_all['fb_shares_log'] + np.random.uniform(-jitter_amount, jitter_amount, len(dfs_all))

    # Create the scatter plot with jitter
    plt.scatter(jittered_n_tweets_log, jittered_fb_shares_log, color='darkgrey', alpha=0.2)
    plt.xlabel('Log of original tweets per article')
    plt.ylabel('Log of Facebook shares per article')

    corr_coeff = np.corrcoef(dfs_all['n_tweets_log'], dfs_all['fb_shares_log'])[0, 1]

    # Add a correlation line
    z = np.polyfit(dfs_all['n_tweets_log'], dfs_all['fb_shares_log'], 1)
    p = np.poly1d(z)
    plt.plot(dfs_all['n_tweets_log'], p(dfs_all['n_tweets_log']), label=f'Correlation: {corr_coeff:.2f}', color='black')

    #plt.legend()
    plt.show()


# Load data

topics = pd.read_csv('article_topics.csv')  # topics, all arts all papers

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


paper_names = ['DM', 'G', 'NYP', 'NYT']
out_groups = ['maj_lab', 'maj_con', 'maj_dem', 'maj_rep']
in_groups = ['maj_con', 'maj_lab', 'maj_rep', 'maj_dem']
topic_nums = range(len(topics['dominant_topic'].unique()))


dfs_list = [[df_tweetsDM, df_fb_sharesDM, df_mddDM], [df_tweetsG, df_fb_sharesG, df_mddG],
            [df_tweetsNYP, df_fb_sharesNYP, df_mddNYP], [df_tweetsNYT, df_fb_sharesNYT, df_mddNYT]]

manip_dfs_list = [manip_df(df_mdd=dfs[2], df_tweets=dfs[0], df_fb=dfs[1], topics=topics) for dfs in dfs_list]  # Create nec var.s and manip data
for i, df in enumerate(manip_dfs_list):
    df['paper'] = paper_names[i]
    df['right_lean'] = 1 if paper_names[i] in ['DM', 'NYP'] else 0
all_manip_dfs = pd.concat(manip_dfs_list, axis=0, ignore_index=True)
all_manip_dfs = add_dummies(df=all_manip_dfs, cols=['paper'], drop_vars=['paper_DM'])  # paper dummies for agg models

# Create 'maj_out_group' and 'maj_in_group' column
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

manip_i_dfs_list = []
for i, df_i_tweets, manipd_df in zip([0, 1, 2, 3], [df_i_tweetsDM, df_i_tweetsG, df_i_tweetsNYP, df_i_tweetsNYT], manip_dfs_list):
    manipd_i_df = manip_i_df(df_i_tweets, manipd_df)
    manipd_i_df['paper'] = paper_names[i]
    manip_i_dfs_list.append(manipd_i_df)
all_manipd_i_dfs = pd.concat(manip_i_dfs_list, axis=0, ignore_index=True)
all_manipd_i_dfs = add_dummies(df=all_manipd_i_dfs, cols=['paper'], drop_vars=['paper_DM'])  # paper dummies for agg models


# # Apply functions

desc_plotting = False
if desc_plotting:
    make_desc_plot(manip_dfs_list)


X = ['my_word_count_log', 'mean_my_word_len_log', 'mean_my_words_in_sen_log', 'year_2020', 'year_2021', 'day_of_week_1',
     'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 'month_02', 'month_03',
     'month_04', 'month_05', 'month_06', 'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12']
T = 'neg_art'
Y = 'n_tweets_log'


analyses_dict = {}

# main models - Twitter
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'TwDR_core{paper_name}'] = run_dr_model(Y=Y, T=T, X=X, df=manip_dfs_list[i], paper_name=paper_name)
# main models - FB
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'FbDR_core{paper_name}'] = run_dr_model(Y='fb_shares_log', T=T, X=X, df=manip_dfs_list[i], paper_name=paper_name)
# main model - All papers, OGTweets plus Fb posts
analyses_dict['TwPlusFbDR_coreAll'] = run_dr_model(Y='n_tweetsPlusPosts_log', T=T, X=X+['paper_G', 'paper_NYP', 'paper_NYT'], df=all_manip_dfs, paper_name='All')
# main model - All papers, OGTweets only
analyses_dict['TwDR_coreAll'] = run_dr_model(Y=Y, T=T, X=X+['paper_G', 'paper_NYP', 'paper_NYT'], df=all_manip_dfs, paper_name='All')
# main model - All papers, FB posts only
analyses_dict['FbDR_coreAll'] = run_dr_model(Y='fb_shares_log', T=T, X=X+['paper_G', 'paper_NYP', 'paper_NYT'], df=all_manip_dfs, paper_name='All')
# main model - All right-leaning papers, Tw and FB posts
analyses_dict['FbTwDR_coreAllRight'] = run_dr_model(Y='n_tweetsPlusPosts_log', T=T, X=X+['paper_NYP'], df=all_manip_dfs[all_manip_dfs['paper'].isin(['DM', 'NYP'])], paper_name='All_right')
# main model - All left-leaning papers, Tw and FB posts
analyses_dict['FbTwDR_coreAllLeft'] = run_dr_model(Y='n_tweetsPlusPosts_log', T=T, X=X+['paper_NYT'], df=all_manip_dfs[all_manip_dfs['paper'].isin(['G', 'NYT'])], paper_name='All_left')
# main model - All right-leaning papers, FB posts only
analyses_dict['FbDR_coreAllRight'] = run_dr_model(Y='fb_shares_log', T=T, X=X+['paper_NYP'], df=all_manip_dfs[all_manip_dfs['paper'].isin(['DM', 'NYP'])], paper_name='All_right')
# main model - All left-leaning papers, FB posts only
analyses_dict['FbDR_coreAllLeft'] = run_dr_model(Y='fb_shares_log', T=T, X=X+['paper_NYT'], df=all_manip_dfs[all_manip_dfs['paper'].isin(['G', 'NYT'])], paper_name='All_left')
# main model - All right-leaning papers, Tw posts only
analyses_dict['TwDR_coreAllRight'] = run_dr_model(Y=Y, T=T, X=X+['paper_NYP'], df=all_manip_dfs[all_manip_dfs['paper'].isin(['DM', 'NYP'])], paper_name='All_right')
# main model - All left-leaning papers, Tw posts only
analyses_dict['TwDR_coreAllLeft'] = run_dr_model(Y=Y, T=T, X=X+['paper_NYT'], df=all_manip_dfs[all_manip_dfs['paper'].isin(['G', 'NYT'])], paper_name='All_left')

# robustness checks, using alt neg art defin - Twitter
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'TwDR_core_rob{paper_name}'] = run_dr_model(Y=Y, T='neg_art_r', X=X, df=manip_dfs_list[i],
                                                               paper_name=paper_name)
# robustness checks, using alt neg art defin - FB
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'FbDR_core_rob{paper_name}'] = run_dr_model(Y='fb_shares_log', T='neg_art_r', X=X, df=manip_dfs_list[i],
                                                               paper_name=paper_name)

# PS 'robustness' check models - Twitter
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'Tw_core_PSrob{paper_name}'] = run_prop_score_model(Y=Y, T=T, X=X, df=manip_dfs_list[i],
                                                               paper_name=paper_name)
# PS 'robustness' check models - FB
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'Fb_core_PSrob{paper_name}'] = run_prop_score_model(Y='fb_shares_log', T=T, X=X, df=manip_dfs_list[i],
                                                               paper_name=paper_name)

# MR 'robustness' check models - Twitter
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'Tw_core_MRrob{paper_name}'] = run_standard_reg(Y=Y, T=T, X=X, df=manip_dfs_list[i],
                                                               paper_name=paper_name)
# MR 'robustness' check models - FB
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'Fb_core_MRrob{paper_name}'] = run_standard_reg(Y='fb_shares_log', T=T, X=X, df=manip_dfs_list[i],
                                                               paper_name=paper_name)

# core models with topic controls 'robustness' check models - Twitter
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'TwDR_core_topic_cont{paper_name}'] = run_dr_model(Y=Y, T=T,
                                                                          X=X + ['dominant_topic_1'] + ['dominant_topic_2'] + ['dominant_topic_3'] + ['dominant_topic_4'],
                                                                          df=manip_dfs_list[i],
                                                               paper_name=paper_name)
# core models with topic controls 'robustness' check models - Facebook
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'FbDR_core_topic_cont{paper_name}'] = run_dr_model(Y='fb_shares_log', T=T,
                                                                          X=X + ['dominant_topic_1'] + ['dominant_topic_2'] + ['dominant_topic_3'] + ['dominant_topic_4'],
                                                                          df=manip_dfs_list[i],
                                                               paper_name=paper_name)

# dividing by topic - Twitter
for i, paper_name in enumerate(paper_names):
    for j in topic_nums:
        by_topic_df = manip_dfs_list[i][manip_dfs_list[i]['dominant_topic'] == j]
        analyses_dict[f'TwMR_core_by_topic{j}{paper_name}'] = run_standard_reg(Y=Y, T=T, X=X, df=by_topic_df,
                                                                       paper_name=paper_name)
# dividing by topic - Facebook
for i, paper_name in enumerate(paper_names):
    for j in topic_nums:
        by_topic_df = manip_dfs_list[i][manip_dfs_list[i]['dominant_topic'] == j]
        analyses_dict[f'FbMR_core_by_topic{j}{paper_name}'] = run_standard_reg(Y='fb_shares_log', T=T, X=X, df=by_topic_df,
                                                                       paper_name=paper_name)

# MR model on polit subset - Twitter
for i, paper_name in enumerate(paper_names):
    any_nation = 'any_pol_pers_term_UK' if paper_name in ['DM', 'G'] else 'any_pol_pers_term_US'
    cols_of_interest = [out_groups[i], in_groups[i]]
    subset_df = manip_dfs_list[i][
        (manip_dfs_list[i][cols_of_interest[0]] == 1) | (manip_dfs_list[i][cols_of_interest[1]] == 1)]
    analyses_dict[f'TwMR_politSubset{paper_name}'] = run_standard_reg(Y=Y, T='neg_art', X=X,
                                                                  df=subset_df,
                                                                  paper_name=paper_name)
# MR model on polit subset - Facebook
for i, paper_name in enumerate(paper_names):
    any_nation = 'any_pol_pers_term_UK' if paper_name in ['DM', 'G'] else 'any_pol_pers_term_US'
    cols_of_interest = [out_groups[i], in_groups[i]]
    subset_df = manip_dfs_list[i][
        (manip_dfs_list[i][cols_of_interest[0]] == 1) | (manip_dfs_list[i][cols_of_interest[1]] == 1)]
    analyses_dict[f'FbMR_politSubset{paper_name}'] = run_standard_reg(Y='fb_shares_log', T='neg_art', X=X,
                                                                  df=subset_df,
                                                                  paper_name=paper_name)

# MR out-group model on polit subset - Twitter
for i, paper_name in enumerate(paper_names):
    any_nation = 'any_pol_pers_term_UK' if paper_name in ['DM', 'G'] else 'any_pol_pers_term_US'
    cols_of_interest = [out_groups[i], in_groups[i]]
    subset_df = manip_dfs_list[i][
        (manip_dfs_list[i][cols_of_interest[0]] == 1) | (manip_dfs_list[i][cols_of_interest[1]] == 1)]
    analyses_dict[f'TwMR_out_group{paper_name}'] = run_standard_reg(Y=Y, T=out_groups[i], X=X,
                                                                  df=subset_df,
                                                                  paper_name=paper_name)
# MR out-group model on polit subset - Facebook
for i, paper_name in enumerate(paper_names):
    any_nation = 'any_pol_pers_term_UK' if paper_name in ['DM', 'G'] else 'any_pol_pers_term_US'
    cols_of_interest = [out_groups[i], in_groups[i]]
    subset_df = manip_dfs_list[i][
        (manip_dfs_list[i][cols_of_interest[0]] == 1) | (manip_dfs_list[i][cols_of_interest[1]] == 1)]
    analyses_dict[f'FbMR_out_group{paper_name}'] = run_standard_reg(Y='fb_shares_log', T=out_groups[i], X=X,
                                                                  df=subset_df,
                                                                  paper_name=paper_name)

# out-group neg art interaction, out vs in group - Twitter
for i, paper_name in enumerate(paper_names):
    T = 'neg_art+' + out_groups[i] + '+neg_art*' + out_groups[i]
    cols_of_interest = [out_groups[i], in_groups[i]]
    subset_df = manip_dfs_list[i][(manip_dfs_list[i][cols_of_interest[0]] == 1) | (manip_dfs_list[i][cols_of_interest[1]] == 1)]
    analyses_dict[f'TwMR_neg_outVin_int{paper_name}'] = run_standard_reg(Y=Y, T=T, X=X,
                                                                  df=subset_df,
                                                                  paper_name=paper_name)
# out-group neg art interaction, out vs in group - Facebook
for i, paper_name in enumerate(paper_names):
    T = 'neg_art+' + out_groups[i] + '+neg_art*' + out_groups[i]
    cols_of_interest = [out_groups[i], in_groups[i]]
    subset_df = manip_dfs_list[i][(manip_dfs_list[i][cols_of_interest[0]] == 1) | (manip_dfs_list[i][cols_of_interest[1]] == 1)]
    analyses_dict[f'FbMR_neg_outVin_int{paper_name}'] = run_standard_reg(Y='fb_shares_log', T=T, X=X,
                                                                  df=subset_df,
                                                                  paper_name=paper_name)
# out-group neg art interaction, out vs in group - Twitter and Fb. All papers.
analyses_dict['TwPlusFbMR_neg_outVin_intAll'] = run_standard_reg(Y='n_tweetsPlusPosts_log',
                                                                 T='neg_art+maj_out_group+neg_art*maj_out_group',
                                                                 X=X + ['paper_G', 'paper_NYP', 'paper_NYT'],
                                                                 df=all_manip_dfs[all_manip_dfs['maj_out_or_in']==1],
                                                                 paper_name='All')

# predicting RTs of tweets re neg arts - Twitter only
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'TwMR_LogRTs{paper_name}'] = run_standard_reg(Y='n_RTs_log', T='neg_art', X=X+['char_length_tweet_log'],
                                                                 df=manip_i_dfs_list[i], paper_name=paper_name)
# predicting RTs of tweets re neg arts - Twitter only, all papers
analyses_dict[f'TwMR_LogRTsAll'] = run_standard_reg(Y='n_RTs_log', T='neg_art',
                                                             X=X+['char_length_tweet_log', 'paper_G', 'paper_NYP', 'paper_NYT'],
                                                             df=all_manipd_i_dfs, paper_name='All')

# predicting Likes of tweets re neg arts - Twitter only
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'TwMR_LogLikes{paper_name}'] = run_standard_reg(Y='n_likes_log', T='neg_art', X=X+['char_length_tweet_log'],
                                                                 df=manip_i_dfs_list[i], paper_name=paper_name)
# predicting Likes of tweets re neg arts - Twitter only, all papers
analyses_dict[f'TwMR_LogLikesAll'] = run_standard_reg(Y='n_likes_log', T='neg_art',
                                                             X=X+['char_length_tweet_log', 'paper_G', 'paper_NYP', 'paper_NYT'],
                                                             df=all_manipd_i_dfs, paper_name='All')

# predicting Replies of tweets re neg arts - Twitter only
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'TwMR_LogReplies{paper_name}'] = run_standard_reg(Y='n_replies_log', T='neg_art', X=X+['char_length_tweet_log'],
                                                                 df=manip_i_dfs_list[i], paper_name=paper_name)
# predicting Replies of tweets re neg arts - Twitter only, all papers
analyses_dict[f'TwMR_LogRepliesAll'] = run_standard_reg(Y='n_replies_log', T='neg_art',
                                                             X=X+['char_length_tweet_log', 'paper_G', 'paper_NYP', 'paper_NYT'],
                                                             df=all_manipd_i_dfs, paper_name='All')

# core but predicting tweets+RTs - Twitter only
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'TwDR_core_pred_tweetsRTs{paper_name}'] = run_dr_model(Y='n_tweetsPlusRTs_log', T='neg_art', X=X, df=manip_dfs_list[i], paper_name=paper_name)
# core but predicting tweets+RTs - Twitter only, All papers
analyses_dict['TwDR_core_pred_tweetsRTsAll'] = run_dr_model(Y='n_tweetsPlusRTs_log',
                                                            T='neg_art',
                                                            X=X+['paper_G', 'paper_NYP', 'paper_NYT'],
                                                            df=all_manip_dfs, paper_name='All')


all_results = pd.DataFrame(analyses_dict).T
all_results.columns = ['paper', 'analysis_method', 'y_variable', 'key_predictors', 'controls', 'sample_size',
                       'point_est1', 'lower_est1', 'upper_est1', 'point_est2', 'lower_est2', 'upper_est2', 'point_est3',
                       'lower_est3', 'upper_est3']

# Convert cols to numeric and round them
all_results_r = all_results.copy()
est_columns = [col for col in all_results_r.columns if '_est' in col]
all_results_r[est_columns] = all_results_r[est_columns].apply(pd.to_numeric, errors='coerce')
numeric_columns = all_results_r.select_dtypes(include='number').columns
all_results_r[numeric_columns] = all_results_r[numeric_columns].round(3)
all_results_r['sample_size'] = pd.to_numeric(all_results_r['sample_size'], errors='coerce')

#all_results.to_csv('data/all_results.csv')  # not rounded
#all_results_r.to_csv('data/all_results_rounded.csv')  # rounded



# NOTES

# To convert ATE coeff (e.g., 0.5) to % increase, following conversion formula:
(math.exp(0.5) - 1) * 100
# to get % increase, following the conversion formula, for a point on an interaction plot, sum coeff 1 (e.g. 0.2), coeff 2 (e.g. 0.3) and coeff 3 (e.g. 0.4):
(math.exp(0.2+0.3+0.4) - 1) * 100
