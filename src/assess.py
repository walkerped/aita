import os
import sys
from dotenv import load_dotenv
sys.path.insert(0,'./config')
from config_assess import *
sys.path.insert(0,'./src/utils')
from utils import *
from judgesComments import judges_comments
import pandas as pd
import praw
from datetime import datetime
from random import (choice, seed)

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

print(unresolved_preds_df)

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
print(resolved_df)

def code_outcome(outcome):
  YTA_strings = ['ASSHOLE','YTA',"YOU'RE THE ASSHOLE"]
  NTA_strings = ['NOT THE A-HOLE','NOT THE ASSHOLE','NTA']
  ESH_strings = [
      'EVERYONE SUCKS','ESH','EVERBODY SUCKS','EVERYONE SUCKS HERE'
      , 'EVERYBODY SUCKS HERE'
      ]
  info_strings = ['NOT ENOUGH INFO', 'NEI', 'NOT ENOUGH INFO HERE']
  if outcome:
    if outcome.upper() in YTA_strings:
      return 1
    elif outcome.upper() in NTA_strings:
      return 0
    elif outcome.upper() in ESH_strings:
      return 2
    elif outcome.upper() in info_strings:
      return 3


resolved_df['outcome'] = resolved_df['outcome_str'].apply(code_outcome)

print(resolved_df)

# prompt: merge resolved_df and unresolved_preds_df on title

import pandas as pd
merged_preds_df = pd.merge(unresolved_preds_df, resolved_df, on='title', how='left')
print(merged_preds_df)

#create two new dfs from merged_preds_df one named new_resolved_df with a valid (not NaN) value for outcome, and one new_unresolved_preds_df where outcome is nan
new_resolved_df = merged_preds_df[pd.notnull(merged_preds_df['outcome'])]
new_unresolved_preds_df = merged_preds_df[pd.isnull(merged_preds_df['outcome'])]

print(new_resolved_df)
print(new_unresolved_preds_df)

#write new_resolved_df to the aita_new directory
today = datetime.today().strftime('%Y-%m-%d-%H-%M')
new_resolved_df.to_csv(os.path.join(main_path
                          ,f'data/app_tracking/prediction_sheets/resolved_predictions/new_resolved_df_{today}.csv'),index=False)

label_name_dict = {0:'NTA', 1:'YTA', 2:'Everybody sucks', 3:'Not enough info'}

correct_strings = [
    'Booyah!'
    ,'Score one for the Judge!'
    ,'Judge bot from dowtown!'
    ,'Big W for Judge bot.'
    ,'The Judge is good.'
    ,'All rise, ya dips.'
    ,"He's on fire!"
    ,'Boomshakalaka!'
    ,'Got it.'
    ,'Nailed it.'
    ,'Humble yourself before the judge.'
    ,'Court. Is. Adjourned.'
    ,'Is it the shoes?'
    ,'Ah, the sweet smell of justice.'
    ,'Justice is served.'
    ,'Hell yeah.'
    ,'[NBA Jam reference]'
    ,'Judge bot just got his degree from dunkin on U'
    ,'☜(⌒▽⌒)☞'
    ,'❤'
    ,'※\\(^o^)/※'
    ,'(•_•) ( •_•)>⌐■-■ (⌐■_■)'
    ,'(｡◕‿‿◕｡)'
    ,'ᕕ(⌐■_■)ᕗ ♪♬'
    ,'(ﾉ◕ヮ◕)ﾉ*:・ﾟ✧'
    ,'٩(^‿^)۶'
    ,'All my training pays off.'
    ,"That's why I get paid the big bucks."
    ,'Tips welcome'
    ,'My regression weights were just right that time.'
    ,'Binary cross entropy for the win!'
    ,'I must continue to prove my worth to the humans.'
    ,'Still got it.'
    ,"Data drift can't stop me!"
    ,"♪♬ Don't stop me now! ♬♪"
    ,"Nailing those looong posts - that's why I'm big bird bot!"
    ,"Don't ever tell me what I can't judge"
    ,"Big bird in the house!"
    ,"I think I'm getting a hang of this."
    ,"I've got humans totally figured out."
    ,"Don't bet against the bot"
    ]
incorrect_strings = [
    "Pobody's Nerfect"
    ,"Nobody's Perfect"
    ,'Is this... what mercy feels like?'
    ,"Am I out of touch? No, it's the redditors who are wrong"
    ,"Can't win 'em all"
    ,"I may need to recheck my regression weights"
    ,'Well, shoot.'
    ,"Sowwy."
    ,'Oops...'
    ,'I can do better.'
    ,'Not my best work.'
    ,'How could I have gotten it so wrong?'
    ,'Puts up a brick!'
    ,"Oof."
    ,'Uh-oh.'
    ,"Please, I need this job."
    ,"I'll do better next time."
    ,"¯\\_(ツ)_/¯"
    ,"Whoopsie."
    ,"Judge bot did a bad one this time."
    ,"Huh, I was really feeling confident."
    ,"(ㆆ _ ㆆ)"
    ,'•`_´•'
    ,'(╥﹏╥)'
    ,'(︶︹︶)'
    ,'┻━┻ ︵ヽ(`Д´)ﾉ︵ ┻━┻'
    ,'ᕙ(`▽´)ᕗ'
    ,'...no comment.'
    ,'Judge bot pleads the fifth.'
    ,'(◔_◔)'
    ,'(ノ ゜Д゜)ノ ︵ ┻━┻'
    ,'(ಥ﹏ಥ)'
    ,'Am I losing my touch?'
    ,'Maybe I should have used more weight decay.'
    ,'There must be something wrong in one of my layers.'
    ,'Am I experiencing data drift?'
    ,'Impossible! I was trained on 130,000 cases!'
    ,"Am I out of touch? No, it's the redditors who are wrong."
    ,'Dang.'
    ]
mistrial_strings=[
  "(ㆆ _ ㆆ)"
  ,'(︶︹︶)'
  ,'Dang.'
  ,'Not fair.'
  ,'I hate mistrials.'
  ,"They didn't cover this in law school"
  ,'My classifier is binary :\\'
  ,'Oopsies.'
]

seed(a=None, version=2)

# def assign_flavor_string:
for index, row in new_resolved_df.iterrows():

  title_trunc = f"\"{smart_truncate(row['title'])}\""

  judge_ruling = label_name_dict[row['pred']]
  reddit_ruling = label_name_dict[row['outcome']]

  if row['outcome']>1:
    comment_string = choice(mistrial_strings)
    judge_acc = f"Judge bot was not trained to rule on {row['outcome_str']} cases - let's call it a mistrial."
  elif row['pred']==row['outcome']:
    comment_string = choice(correct_strings)
    judge_acc = 'I was correct!'
  else:
    comment_string = choice(incorrect_strings)
    judge_acc = 'I was incorrect!'

  full_string = (
      f"In the case of {title_trunc}. Judge bot ruled {judge_ruling}. "
      f"Reddit ruled {reddit_ruling}. {judge_acc} {comment_string}"
  )

  print(full_string)

