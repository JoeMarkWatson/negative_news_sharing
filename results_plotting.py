import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import math


## IMPORT DATA

results = pd.read_csv('data/all_results.csv', index_col=0)
results['error_diff1'] = results['point_est1'] - results['lower_est1']
results['error_diff2'] = results['point_est2'] - results['lower_est2']
results['error_diff3'] = results['point_est3'] - results['lower_est3']

# Add new rows with only nan, to keep Twitter results in consistent postn across subplots when no Fb
row_names = ['FbDR_core_pred_tweetsRTsDM', 'FbDR_core_pred_tweetsRTsG', 'FbDR_core_pred_tweetsRTsNYP', 'FbDR_core_pred_tweetsRTsNYT']
new_rows = pd.DataFrame(index=row_names, columns=results.columns)
new_rows[:] = -100
results = pd.concat([results, new_rows])
results['error_diff1'] = results['error_diff1'].apply(lambda x: 0.1 if x == -100 else x)


def gather_results(row_names_core, social_prefix):
    row_names_c = social_prefix + row_names_core
    row_name1 = row_names_c + "DM"
    row_name2 = row_names_c + "G"
    row_name3 = row_names_c + "NYP"
    row_name4 = row_names_c + "NYT"
    row_names_all = [row_name1, row_name2, row_name3, row_name4]
    return row_names_all


## PRIMARY RESULTS

cmap = ListedColormap(['#0368FD', '#0915BA', '#D11702', '#393939'])

def plotplot1(ax, row_names_core='DR_core', social_prefix=['Tw', 'Fb'], xlabel='', title='', title_fontsize=8,
              legend=True, percentage_conv=True, show_xlabels=False):

    if xlabel != None:
        xlabel = 'Change in log(+1) Social Posts'
        if percentage_conv == True:
            xlabel = 'Increase in social shares for negative articles (%)'

    if social_prefix==['Tw', 'Fb']:
        row_names_Fb = gather_results(row_names_core, 'Fb')
        row_names_Tw = gather_results(row_names_core, 'Tw')
        row_names_all = row_names_Fb + row_names_Tw
    if social_prefix == 'Tw':
        row_names_all = gather_results(row_names_core, 'Tw')

    bars = results.loc[row_names_all, 'point_est1']  # X location of points
    lower_error = results.loc[row_names_all, 'lower_est1']  # Lower error
    upper_error = results.loc[row_names_all, 'upper_est1']  # Upper error

    if percentage_conv == True:
        bars = [(math.exp(x)-1)*100 if x > 0 else -(math.exp(-1*x)-1)*100 for x in bars]
        lower_error = [(math.exp(x)-1)*100 if x > 0 else -(math.exp(-1*x)-1)*100 for x in lower_error]
        upper_error = [(math.exp(x)-1)*100 if x > 0 else -(math.exp(-1*x)-1)*100 for x in upper_error]

    ys = np.append(np.arange(0, 4), np.arange(4, 4 * 2) + 1)  # Y location of points
    fmts = ['o'] * 4 + ['x'] * 4  # Point shapes

    papers = ['Daily Mail', 'Guardian', 'New York Post', 'New York Times'] * 2
    colors = cmap.colors * 2  # Colors for papers
    if not ax:
        fig, ax = plt.subplots(figsize=(7, 5))

    for y, fmt, bar, color, paper in zip(ys, fmts, bars, colors, papers):
        ax.plot(bar, y, fmt, color=color, label=paper)  # , alpha=0.7
    for y, lower, upper, color, paper in zip(ys, lower_error, upper_error, colors, papers):
        ax.hlines(y, xmin=lower, xmax=upper, colors=color)
    #for y, upper, bar, color, paper in zip(ys, upper_error, bars, colors, papers):
    #    ax.hlines(y, xmin=bar, xmax=upper, colors=color)

    leg_loc = 'upper right' if row_names_core == 'MR_core' else 'upper left'
    if legend:
        legend = ax.legend(loc=leg_loc)
    ax.yaxis.set_tick_params(labelbottom=False)

    ax.set_xlabel(xlabel, fontsize=title_fontsize, loc='left')  # delete loc='left' if shorter xlabel
    ax.tick_params(axis='x', labelsize=8)

    if show_xlabels==True:
        ax.xaxis.set_tick_params(labelbottom=True)  # over-rides the part of sharex='col' that gets rid of
    # plot markers for upper subplots

    ax.axvline(x=0, color='grey', linestyle='--', alpha=0.5)  # Plot y-axis
    # ax.set_xlim(0, None)
    ax.set_yticks([])
    ax.set_title(title, fontsize=title_fontsize)
    # fig.tight_layout()
    # return fig



def add_legend(ax, ncols_papers=2, ncols_socials=2, loc='upper right', bbox_to_anchor=(0.99, 0.8), fontsize=10, leg_title_fontsize=10, titles=True):
    # Create proxy artists for the newspapers with the desired colors
    DailyMail_patch = mpatches.Patch(color=cmap(0), label='DailyMail')
    Guardian_patch = mpatches.Patch(color=cmap(1), label='Guardian')
    NYPost_patch = mpatches.Patch(color=cmap(2), label='NYPost')
    NYTimes_patch = mpatches.Patch(color=cmap(3), label='NYTimes')
    # Create proxy artists for social media with grey color and markers
    Facebook_line = mlines.Line2D([], [], color='black', marker='o', markersize=7, label='Facebook', linestyle='None')
    Twitter_line = mlines.Line2D([], [], color='black', marker='x', markersize=7, label='Twitter', linestyle='None')

    newspapers = [DailyMail_patch, Guardian_patch, NYPost_patch, NYTimes_patch]
    socials = [Twitter_line, Facebook_line]

    news_title = "Newspaper" if titles else None
    soc_title = "Social Network" if titles else None

    newspaper_legend = ax.legend(handles=newspapers, loc=loc, ncol=ncols_papers, fontsize=fontsize, title=news_title, title_fontsize=leg_title_fontsize, frameon=False)
    # Add the legend manually to the current Axes.
    ax.add_artist(newspaper_legend)

    ax.legend(handles=socials, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncols_socials, fontsize=fontsize, title=soc_title, title_fontsize=leg_title_fontsize, frameon=False)
    # ax.legend(handles=legend_items, ncol=ncol, loc=loc, fontsize=fontsize)


## CORE AND ALT MODELS RESULTS FOR TW and FB - main text
fig, ax = plt.subplots(1, 3, figsize=(8, 4), sharex='col')  # sharex='col'

plotplot1(ax=ax[0], legend=False, title='A: Core Model')
plotplot1(row_names_core='DR_core_topic_cont', ax=ax[1], title='B: Topic controls', xlabel=None, legend=False)
plotplot1(row_names_core='DR_core_pred_tweetsRTs', ax=ax[2], title='C: Tweets and Retweets', xlabel=None, legend=False)

add_legend(ax=ax[0], bbox_to_anchor=(0.99, 0.84), fontsize=6, titles=False)

ax[0].set_xlim(-14, 172)
ax[1].set_xlim(-14, 172)
ax[2].set_xlim(-14, 172)
fig.tight_layout()
plt.show();

## ALT MODELS RESULTS FOR TW and FB - appendix 1
fig, ax = plt.subplots(1, 3, figsize=(8, 4), sharex='col')  # sharex='col'
plotplot1(row_names_core='_core_MRrob', ax=ax[0], title='A: Multiple Regression', legend=False)
plotplot1(row_names_core='_core_PSrob', ax=ax[1], title='B:  Propensity Score Modelling', xlabel=None, legend=False)
plotplot1(row_names_core='DR_core_rob', ax=ax[2], title='C: Alternate Treatment', legend=False, xlabel=None)
add_legend(ax=ax[0], bbox_to_anchor=(0.99, 0.84), fontsize=6, titles=False)
ax[0].set_xlim(-14, 172)
ax[1].set_xlim(-14, 172)
ax[2].set_xlim(-14, 172)
fig.tight_layout()
plt.show();



## RESULTS BY TOPIC

fig, ax = plt.subplots(2, 3, figsize=(9.6, 8), sharex='col')

plotplot1(row_names_core='MR_core_by_topic0', ax=ax[0, 0], title='A: Politics', xlabel=None, legend=False)
plotplot1(row_names_core='MR_core_by_topic1', ax=ax[0, 1], title='B: Family and Home', xlabel=None, legend=False)
plotplot1(row_names_core='MR_core_by_topic2', ax=ax[0, 2], title='C: Sport', xlabel=None, legend=False, show_xlabels=True)
plotplot1(row_names_core='MR_core_by_topic3', ax=ax[1, 0], title='D: Local', legend=False)
plotplot1(row_names_core='MR_core_by_topic4', ax=ax[1, 1], title='E: Global', xlabel=None, legend=False)

add_legend(ax=ax[0, 0], loc='upper left', bbox_to_anchor=(0.005, 0.75), fontsize=6, leg_title_fontsize=8)

ax[1, 2].axis('off')
ax[0, 0].set_xlim(-195, 195)
ax[0, 1].set_xlim(-195, 195)
ax[0, 2].set_xlim(-195, 195)
fig.tight_layout()
plt.show();


def plotInteraction(ax, social='Tw', row_names_core='MR_neg_outVin_int', paper='DM',
               xlabel = 'Increase in social shares for negative articles (%)',
               title='', title_fontsize=10, legend=True,
               point_no=1,
               percentage_conv=True):

    # gather rows
    row_names_all = social + row_names_core + paper

    neg_effects = results.loc[row_names_all, 'point_est1']
    out_effects = results.loc[row_names_all, 'point_est2']
    int_effects = results.loc[row_names_all, 'point_est3']

    Line1_points = [neg_effects, neg_effects + out_effects + int_effects]
    Line2_points = [neg_effects * 0, neg_effects * 0 + out_effects + int_effects * 0]
    if percentage_conv==True:
        Line1_points = [(math.exp(x) - 1) * 100 if x > 0 else -(math.exp(-1 * x) - 1) * 100 for x in Line1_points]
        Line2_points = [(math.exp(x) - 1) * 100 if x > 0 else -(math.exp(-1 * x) - 1) * 100 for x in Line2_points]

    # Define labels for the lines
    Line1_label = "Negative\nArticle"
    Line2_label = "Positive\nArticle"

    # Define the X-axis labels
    x_labels = ["In\nGroup", "Out\nGroup"]

    color='blue' if social=='Fb' else 'black'
    marker = 'o' if social=='Fb' else 'x'

    if percentage_conv == True:
        ylabel = 'Increase in Facebook Posts (%)' if social=='Fb' else 'Increase in Tweets (%)'
    else:
        ylabel = 'Change in log(+1) Facebook Posts' if social == 'Fb' else 'Change in log(+1) Tweets'

    set_ylim = (-20, 150) if social=='Fb' else (-20, 100)

    plt.figure(figsize=(10, 6))
    ax.plot(x_labels, Line1_points, marker=marker, label=Line1_label, color=color)
    ax.plot(x_labels, Line2_points, marker=marker, label=Line2_label, color=color, linestyle=':')

    #ax.xaxis.set_tick_params(labelbottom=True)  # over-rides the part of sharex='col' that gets rid of
    # plot markers for upper subplots

    ax.set_ylim(set_ylim)

    # Set labels and title
    if (title == 'A') | (title == 'E'):
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left', fontsize=9)
    else:
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=10, loc='left')

fig, ax = plt.subplots(2, 4, figsize=(10, 8), sharex='col')

plotInteraction(social='Fb', row_names_core='MR_neg_outVin_int', ax=ax[0, 0], title='A', xlabel=None, legend=False)
plotInteraction(social='Fb', row_names_core='MR_neg_outVin_int', ax=ax[0, 1], paper='G', title='B', xlabel=None, legend=False)
plotInteraction(social='Fb', row_names_core='MR_neg_outVin_int', ax=ax[0, 2], paper='NYP', title='C', xlabel=None, legend=False)
plotInteraction(social='Fb', row_names_core='MR_neg_outVin_int', ax=ax[0, 3], paper='NYT', title='D', xlabel=None, legend=False)
plotInteraction(row_names_core='MR_neg_outVin_int', ax=ax[1, 0], title='E', xlabel=None, legend=False)
plotInteraction(row_names_core='MR_neg_outVin_int', ax=ax[1, 1], paper='G', title='F', xlabel=None, legend=False)
plotInteraction(row_names_core='MR_neg_outVin_int', ax=ax[1, 2], paper='NYP', title='G', xlabel=None, legend=False)
plotInteraction(row_names_core='MR_neg_outVin_int', ax=ax[1, 3], paper='NYT', title='H', xlabel=None, legend=False)

# ax[0, 1].set_xlim(-110, 110)
fig.tight_layout()
plt.show();
