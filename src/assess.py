import os
import sys
from dotenv import load_dotenv
sys.path.insert(0,'./config')
from config_assess import *
sys.path.insert(0,'./src/utils')
from utils import *
from assess_helpers import *
from judgesComments import judges_comments
import pandas as pd
import praw
from datetime import datetime
from random import (choice, seed)
import shutil

# set user defined vars

# paths
sheet_path = os.path.join(main_path
        ,'data/app_tracking/prediction_sheets/')
unresolved_preds_path = os.path.join(sheet_path,'unresolved_predictions.csv')

# read in unresolved_preds_path as a pandas df
# set df with running predictions that have not been resolved
if os.path.exists(unresolved_preds_path):
  unresolved_preds_df = pd.read_csv(unresolved_preds_path, index_col = 0)
else:
  print("No unresolved prediction spreadsheet. Exiting.")
  quit()

# load variables from .env as environment variables
# including reddit and twitter credentials
load_dotenv(dotenv_path)

# Authenticate
reddit = praw.Reddit(
    client_id=os.environ.get("CLIENT_ID"),
    client_secret=os.environ.get("CLIENT_SECRET"),
    user_agent=os.environ.get("USER_AGENT")
)

# get the subreddit
aita_subreddit = reddit.subreddit('AmITheAsshole')

# search praw for each unresolved prediction post
resolved_titles = []
resolved_outcomes = []
for index, row in unresolved_preds_df.iterrows():

  # search the subreddit for the title
  results = aita_subreddit.search(row['title'], limit=1)

  for post in results:
    outcome_str = post.link_flair_text
    post_title = post.title

    if row['title'] == post_title:
      resolved_titles.append(row['title'])
      resolved_outcomes.append(outcome_str)

resolved_df = pd.DataFrame.from_dict(
    {'title':resolved_titles,'outcome_str':resolved_outcomes}
)

resolved_df['outcome'] = resolved_df['outcome_str'].apply(code_outcome)

if resolved_df['outcome'].isnull().all():
  print('None of the pending prediction posts have been assigned a valid flair. Exiting.')
  exit()

# merge resolved_df and unresolved_preds_df on title
merged_preds_df = pd.merge(unresolved_preds_df, resolved_df, on='title', how='left')
print('\nunresolved\n',unresolved_preds_df,'\nresolved\n',resolved_df,'\nmerged\n',merged_preds_df
# ,merged_preds_df['outcome_x'],merged_preds_df['outcome_y']
)

#create two new dfs from merged_preds_df one named new_resolved_df with a valid (not NaN) value for outcome, and one new_unresolved_preds_df where outcome is nan

new_resolved_df = merged_preds_df[pd.notnull(merged_preds_df['outcome'])].reset_index()
new_unresolved_preds_df = merged_preds_df[pd.isnull(
  merged_preds_df['outcome'])].drop(
    ['outcome','outcome_str'], axis=1
    ).reset_index()

if not quiet:
  print('Top rows of df tracking resolved predictions: ')
  print(new_resolved_df.head())
  print('Top rows of df tracking unresolved predictions:')
  print(new_unresolved_preds_df)

label_name_dict = {0:'NTA', 1:'YTA', 2:'Everybody sucks', 3:'Not enough info'}

seed(a=None, version=2)

client = tweepy_auth()

if not quiet:
  print('Tweet strings: ')
# def assign_flavor_string:
for index, row in new_resolved_df.iterrows():

  title_trunc = f"\"{smart_truncate(row['title'])}\""

  judge_ruling = label_name_dict[row['pred']]
  reddit_ruling = label_name_dict[row['outcome']]

  if row['outcome']>1:
    comment_string = choice(judges_comments['mistrial'])
    judge_acc = f"Judge bot was not trained to rule on {row['outcome_str']} cases - let's call it a mistrial."
  elif row['pred']==row['outcome']:
    comment_string = choice(judges_comments['correct'])
    judge_acc = 'I was correct!'
  else:
    comment_string = choice(judges_comments['incorrect'])
    judge_acc = 'I was incorrect!'

  full_string = (
      f"PREDICTION UPDATE: "
      f"In the case of {title_trunc}. Judge bot ruled {judge_ruling}. "
      f"Reddit ruled {reddit_ruling}. {judge_acc} {comment_string} "
      f"https://twitter.com/AITA_judgebot/status/{row["tweet_id"]}"
  )

  if not quiet:
    print(full_string)

  if not dry_run:
    # post to twitter
    client.create_tweet(text=full_string)


today = datetime.today().strftime('%Y-%m-%d-%H-%M')
new_resolved_sheet_archive = (f'data/app_tracking/prediction_sheets/resolved_predictions/new_resolved_df_{today}.csv')

#write new_resolved_df to the aita_new directory
if not dry_run:
  new_resolved_df_csv = os.path.join(main_path,new_resolved_sheet_archive)
  today = datetime.today().strftime('%Y-%m-%d-%H-%M')
  new_resolved_df.to_csv(new_resolved_df_csv,index=False)
  if new_resolved_df is not None:
    copy_and_replace(new_resolved_df_csv,unresolved_preds_path)
  if not quiet:
    print('Top lines of new resolved dfs')
    print(new_resolved_df.head())


