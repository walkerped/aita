import sys
import os
from dotenv import load_dotenv, find_dotenv
import praw
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime, timezone
import numpy as np
from random import (choice, seed)

sys.path.insert(0,'./config')
from config_predict import *
from judgesComments import judges_comments
sys.path.insert(0,'./src/utils')
from utils import *
import pytz


# set df with running predictions that have not been resolved
current_sheet_df = read_csv_ifexist(current_sheet_path)

################# use praw to pull AITA posts, and load into df #################

# load variables from .env as environment variables
# including reddit and twitter credentials
load_dotenv(dotenv_path)

if not quiet and not current_sheet_df.empty:
    print(
        f'loading sheet from {current_sheet_path}'
    )
    print('top rows from current sheet or active (unresolved) predictions:')
    print(current_sheet_df.head())

# Authenticate
reddit = praw.Reddit(
    client_id=os.environ.get("CLIENT_ID"),
    client_secret=os.environ.get("CLIENT_SECRET"),
    user_agent=os.environ.get("USER_AGENT")
)

# Get the subreddit
subreddit = reddit.subreddit('AmITheAsshole')

# Fetch new posts
new_posts = subreddit.new(limit=n_posts)

raw_post_df = pd.DataFrame(
    [ vars(post) for post in subreddit.new(limit=n_posts)]
    )

vars_to_keep = ['title','selftext','link_flair_text','created_utc','url']

index_flair = raw_post_df[raw_post_df['link_flair_text'].notnull()].index

abbrev_post_df = raw_post_df[vars_to_keep].drop(index_flair)

if current_sheet_df.empty:
    filtered_post_df = abbrev_post_df
else:
    filtered_post_df = abbrev_post_df[
        ~abbrev_post_df['title'].isin(current_sheet_df['title'])
        ]

filtered_post_df.loc[:, 'post_date'] = filtered_post_df['created_utc'].apply(
    lambda x: datetime.fromtimestamp(x).astimezone(pytz.utc).strftime('%Y-%m-%d')
)

filtered_post_df.loc[:, 'titles_and_texts'] = (filtered_post_df['title']
                                          + filtered_post_df['selftext'])

filtered_post_df.loc[:, 'date_created'] = datetime.today().strftime('%Y-%m-%d')

post_df = filtered_post_df.drop(['created_utc','selftext'], axis=1)

if not quiet:
    print(f'Top lines of df with posts with no flair yet pulled by praw:')
    print(post_df.head())

################# make prediction for each post #################

device = assign_torch_device()

to_predict_dataset = aita_prep_data(post_df['titles_and_texts'].tolist())

pred = aita_predict(model_path,to_predict_dataset, device)

################# create df with post & prediction info #################
logits_list = pred.predictions.reshape(-1)
bin_pred_list = (pred.predictions > 0.5).astype(np.int32).reshape(-1)

pred_df = post_df[['title', 'post_date', 'url', 'date_created']]

pred_df.loc[:, 'logits'] = logits_list

pred_df.loc[:, 'pred'] = bin_pred_list

if not quiet:
    print('Top lines of df with prediction info:')
    print(pred_df.head())


# get 10% most extreme pos and neg predictions, respectively
pred_percentile_df = subset_extremes(pred_df, 'pred', 'logits', quiet=quiet)

if not quiet:
    print(pred_percentile_df.sort_values('logits'))

################# select posts to tweet and tweet them w prediction #################
selected_preds_df = pred_percentile_df.sample(n=n_tweets)

client = tweepy_auth()

intro_string = 'Judge bot has heard the case of'
ruling_string = 'Judge bot rules:'

label_name_dict = {0:'NTA', 1:'YTA'}

seed(a=None, version=2)

tweet_id_list = []
# create and post tweet
for index, row in selected_preds_df.iterrows():

    title_trunc = f"\"{smart_truncate(row['title'])}\""

    comment_string = choice(judges_comments[label_name_dict[row['pred']].lower()])

    full_tweet_string = (
        f"{intro_string} {title_trunc}. {ruling_string} "
        f"{label_name_dict[row['pred']]}. {comment_string}  {row['url']}"
    )

    # post to twitter
    tweet_id = client.create_tweet(text=full_tweet_string)
    tweet_id_list.append(tweet_id[0]['id'])

selected_preds_df["tweet_id"] = tweet_id_list

if current_sheet_df.empty:
    new_pred_df = selected_preds_df
else:
    new_pred_df = update_df(current_sheet_df,selected_preds_df)

new_pred_df.reset_index(drop=True).to_csv(path_or_buf=current_sheet_path)

if not quiet:
    print('Top lines of new spreadsheet of active predictions:')
    print(new_pred_df.head())