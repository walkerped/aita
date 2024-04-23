import os
import sys
from dotenv import load_dotenv
sys.path.insert(0,'./config')
from config_monitor import *
sys.path.insert(0,'./src/utils')
from utils import *
from datetime import datetime
import praw
import pandas as pd
import pytz
import numpy as np
import matplotlib.pyplot as plt

# create monitor_sheet_path folder if it does not already exist
# Check if the directory exists
if not os.path.exists(monitor_sheet_path):
    # Create the directory
    os.makedirs(monitor_sheet_path)

# Check if the directory exists
if not os.path.exists(monitor_archive_path):
    # Create the directory
    os.makedirs(monitor_archive_path)

# set strings that indicate a YTA or NTA outcome
YTA_strings = ['ASSHOLE','YTA',"YOU'RE THE ASSHOLE"]
NTA_strings = ['NOT THE A-HOLE','NOT THE ASSHOLE','NTA']

# get today's date
today = datetime.today().strftime('%Y-%m-%d')

# load variables from .env as environment variables
# including reddit and twitter credentials
load_dotenv(dotenv_path)

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


# Clear lists and counts
submission_ids = []
texts = []
flairs = []
dates = []
titles = []
YTA_count = 0
NTA_count = 0
post_count = 0
# Extract post data and store in lists
for post in new_posts:

  post_count += 1

  # only get posts with a flair, so we are making prospective
  # predictions
  if not post.link_flair_text:
    continue
  # if not post.link_flair_text.upper() in NTA_strings + YTA_strings:
  #   continue

  # remove posts that are unusually short or long, as these may not be
  # actual AITA posts, or may have been deleted or edited by OP
  if 200 >= len(post.selftext) >= 6500:
    continue

  # count the number of YTA and NTA's, just to output to terminal
  if post.link_flair_text.upper() in YTA_strings:
    YTA_count += 1
  elif post.link_flair_text.upper() in NTA_strings:
    NTA_count += 1

  # get and format post date
  post_date = datetime.fromtimestamp(post.created_utc).astimezone(pytz.utc).strftime('%Y-%m-%d')

  # add post data to lists
  submission_ids.append(post.id)
  titles.append(post.title)
  texts.append(post.selftext)
  flairs.append(post.link_flair_text)
  dates.append(post_date)

# print counts
print(f'Got {YTA_count} YTAs and {NTA_count} NTAs after going through ')
print(f'{post_count} posts.')

# create df with combined_text and flairs and dates as columns
post_df = pd.DataFrame({
    'submission_ids': submission_ids,
    'titles': titles,
    'texts': texts,
    'outcome_str': flairs,
    'dates': dates,
})

# create combined text column, for passing to tokenizer
post_df['combined_text'] = post_df[['titles','texts']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# create a coded outcome column
post_df['outcome'] = post_df['outcome_str'].apply(code_outcome)

# make directory to store data, if it doesn't exist yet
os.makedirs(raw_data_path, exist_ok=True)

# get unique dates in post_df
post_dates = post_df['dates'].unique()

# get list of filenames in raw_data_path
filenames = os.listdir(raw_data_path)

# get those filenames that match dates in post_df
matching_files = []
for date in post_dates:
  # set path to output file for date
  date_csv = f'{raw_data_path}/{date}_post.csv'
  # file is in directory, add to matching files list
  if f'{date}_post.csv' in filenames:
    matching_files.append(date_csv)
# if there are any matching files, concat them
if matching_files:
  old_posts_df = pd.concat((pd.read_csv(f) for f in matching_files)
                , ignore_index=True)
  # filter out posts that are already in old_posts_df
  filtered_post_df = post_df[
  ~post_df['submission_ids'].isin(old_posts_df['submission_ids'])
  ]
# if there are no old
else:
  filtered_post_df = post_df

if filtered_post_df.empty:
  print("There are no new posts with valid flairs that aren't already saved in the csv sheets. Exiting.")
  exit()

# ################# make prediction for each post #################

device = assign_torch_device()

to_predict_dataset = aita_prep_data((filtered_post_df['combined_text']).tolist())

pred = aita_predict(model_path,to_predict_dataset, device)

logits_list = pred.predictions.reshape(-1)
bin_pred_list = (pred.predictions > 0.5).astype(np.int32).reshape(-1)

# add logits_list and bin_pred_list to balanced_posts_df
preds_df = filtered_post_df

preds_df['logits'] = logits_list
preds_df['predictions'] = bin_pred_list

#using prob_df, generate a new column, accurate based on predictions
# and label_ids
preds_df['accurate'] = (
    preds_df['predictions'] == preds_df['outcome']).astype(int)

print(preds_df)

# get unique dates in preds_df
preds_dates = preds_df['dates'].unique()

# for each date in preds dates, add posts for that date
# to appropriate csv
for date in preds_dates:
  date_csv = f'{raw_data_path}/{date}_post.csv'
  new_date_df = preds_df.loc[preds_df['dates'] == date]
  if os.path.exists(date_csv):
    old_date_df = pd.read_csv(date_csv, index_col=0)
    full_date_df = pd.concat([old_date_df, new_date_df])
  else:
    full_date_df = new_date_df.reset_index(drop=True)
  if not dry_run:
    full_date_df.reset_index(drop=True).to_csv(date_csv)
  

exit()
# # for each value in post_df["outcome"] randomly select rows with that value
# '''
# for each value in df["colname"], randomly select N rows and subset to those rows
# '''
# def random_n_by_cat(df,df_colname,n):
#   return (df.groupby(df_colname)
#               .apply(lambda x: x.sample(n=n))
#               .reset_index(drop=True))

# balanced_posts_df = random_n_by_cat(post_df,"outcome",YTA_count)

# print(balanced_posts_df)
  

# acc_binom = binomial_stats(preds_df['accurate'])

# YTA_acc_binom = binomial_stats(preds_df[preds_df['outcome'] == 1]['accurate'])

# NTA_acc_binom = binomial_stats(preds_df[preds_df['outcome'] == 0]['accurate'])

# # create binom df
# new_monitor_dict = {
#     'date':today
#     , 'outcome_type':['Total','YTA','NTA']
#     , 'success':[acc_binom.success,YTA_acc_binom.success,NTA_acc_binom.success]
#     , 'n':[acc_binom.n,YTA_acc_binom.n,NTA_acc_binom.n]
#     , 'rate':[acc_binom.rate,YTA_acc_binom.rate,NTA_acc_binom.rate]
#     , 'se':[acc_binom.se,YTA_acc_binom.se,NTA_acc_binom.se]
#     , 'min_post_date':[
#         post_df["dates"].min(),post_df["dates"].min(),post_df["dates"].min()]
#     , 'max_post_date':[
#         post_df["dates"].max(),post_df["dates"].max(),post_df["dates"].max()]
# }

# new_monitor_df = pd.DataFrame.from_dict(new_monitor_dict)

# for index, row in new_monitor_df.iterrows():
#   print(
#       f'{row["outcome_type"]} Accuracy Rate: {row["rate"]:.1%}'
#       f' SE: {row["se"]:.1%}'
#       )

# #read in current_monitor_df from csv

# current_monitor_csv = os.path.join(monitor_sheet_path, "current_monitor.csv")
# if os.path.exists(current_monitor_csv):

#   current_monitor_df = pd.read_csv(current_monitor_csv)

#   if overwrite_day == True and today in current_monitor_df['date'].values:

#     current_monitor_df = current_monitor_df[current_monitor_df['date'] != today]

#   if not today in current_monitor_df['date'].values:

#     new_current_monitor_df = pd.concat([current_monitor_df,new_monitor_df])

#     # create an archive copy
#     new_current_monitor_df.to_csv(os.path.join(
#       monitor_archive_path, f'monitor_{today}.csv'), index=False)

#     # save as current
#     new_current_monitor_df.to_csv(current_monitor_csv, index=False)

#   else:

#     print(f"{current_monitor_csv} already contains entries with today's date")
#     print(f"sheet will not be updated, current predictions will not be archived")


# else:
#   new_current_monitor_df = new_monitor_df

#   # create an archive copy
#   new_current_monitor_df.to_csv(os.path.join(
#       monitor_archive_path, f'monitor_{today}.csv'), index=False)

#   # save as current
#   new_current_monitor_df.to_csv(current_monitor_csv, index=False)

# # create plot

# # Filter data
# total_data = new_current_monitor_df[new_current_monitor_df['outcome_type'] == 'Total']

# # Calculate the mean proportion
# mean_proportion = total_data['rate'].mean()

# # Calculate the standard deviation of the proportion
# std_dev_proportion = total_data['rate'].std()

# # Calculate the control limits (assuming normal approximation)
# UCL = mean_proportion + 3 * std_dev_proportion
# LCL = mean_proportion - 3 * std_dev_proportion

# # Create plot
# plt.figure(figsize=(8, 6))

# # Plot data with error bars
# plt.errorbar(total_data['date'], total_data['rate'], yerr=total_data['se'], capsize=5, color='black', zorder=1)

# # Plot the control limits
# plt.axhline(y=mean_proportion, color='#B7410E', linestyle='--', label='Mean', zorder=5, alpha=.75)
# plt.axhline(y=UCL, color='gray', linestyle='--', label='Control Limits', zorder=2)
# plt.axhline(y=LCL, color='gray', linestyle='--', label='_Hidden label', zorder=2)

# # Plot data points
# plt.plot(total_data['date'], total_data['rate'], 'o', color='teal', markeredgecolor='black', zorder=4, markersize=14)

# # Customize ticks
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# # Add labels and title
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('Accuracy Rate', fontsize=14)
# plt.title('Judgebot Accuracy Over Time', fontsize=14)
# plt.legend()

# # Show plot
# plt.tight_layout()
# plt.show()