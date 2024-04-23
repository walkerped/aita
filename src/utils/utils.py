# read in csv if it exists
def read_csv_ifexist(csv_path):
    import pandas as pd
    import os
    # read in current pred sheet
    if os.path.exists(csv_path):
        sheet_df = pd.read_csv(csv_path, index_col = 0)
    else:
        sheet_df = pd.DataFrame()
    return sheet_df

# assign GPU as device if available, otherwise CPU
def assign_torch_device(quiet=False):
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU
        if not quiet:
            print("GPU available. Using GPU.")
    else:
        device = torch.device("cpu")  # CPU
        if not quiet:
            print("GPU is not available. Using CPU.")
    return device

# tokenize data and pack into tensor
def aita_prep_data(post_list):
    import sentencepiece
    from torch.utils.data import TensorDataset
    from transformers import BigBirdTokenizer
    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

    post_tokens = tokenizer(post_list, padding=True
                            , truncation=True, return_tensors="pt")

    return TensorDataset(post_tokens['input_ids']
                                    , post_tokens['attention_mask'])

def aita_predict(model_path, data_to_predict, device):

    from transformers import (BigBirdForSequenceClassification, Trainer
                            , TrainingArguments)
    import torch
    from accelerate import DataLoaderConfiguration, Accelerator

    def data_collector(features):
        batch = {}
        batch['input_ids'] = torch.stack([f[0] for f in features])
        batch['attention_mask'] = torch.stack([f[1] for f in features])
        return batch

    accelorator = Accelerator()

    model_loaded = BigBirdForSequenceClassification.from_pretrained(
        model_path).to(device)

    training_args = TrainingArguments("test_trainer",report_to='none')

    # Initialize Trainer
    trainer = Trainer(
        model=model_loaded,
        args=training_args,
        data_collator=data_collector,
    )

    return trainer.predict(data_to_predict)

# subset data to most extreme percentiles on thresholding column
def subset_extremes(in_df, case_pred_col_name, thresholding_col_name
                    , percentile_thresh=.1, quiet=False):
  threshs = []
  for case_type in [0,1]:

    if case_type == 0:
      perc_thresh = percentile_thresh
    elif case_type == 1:
      perc_thresh = 1-percentile_thresh

    threshs.append(in_df[(in_df[case_pred_col_name] == case_type)][
        thresholding_col_name].quantile(q=perc_thresh))


  df_subset = in_df.loc[
      (in_df[thresholding_col_name] < threshs[0])
        | (in_df[thresholding_col_name] > threshs[1])
      ]
  if not quiet:
      print(
          f'{thresholding_col_name} threshold for percentile of '
          f'{percentile_thresh} is {threshs[0]:.3} where {case_pred_col_name} '
          f'= 0 and {threshs[1]:.3} where {case_pred_col_name} = 1'
          )

  return df_subset

# authenticate twitter
def tweepy_auth():
  import tweepy
  import os

  consumer_key = os.environ.get("CONSUMER_KEY")
  consumer_secret = os.environ.get("CONSUMER_SECRET")
  access_token = os.environ.get("ACCESS_TOKEN")
  access_token_secret = os.environ.get("ACCESS_TOKEN_SECRET")

  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  return tweepy.Client(consumer_key=consumer_key,
                    consumer_secret=consumer_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret)

# truncate string to 100 characters, stripping at whole word
# and adding ellipsis to ending
def smart_truncate(content, length=100, suffix='...'):
    content_strip = content.strip()
    if len(content_strip) <= length:
        return content_strip
    else:
        return content_strip[:length].rsplit(' ', 1)[0]+suffix

# concat two dfs, dropping duplicates
def update_df(existing_df,new_def):
  import pandas as pd

  concat_df = pd.concat([existing_df,new_def])
  return concat_df.drop_duplicates(ignore_index=True)

# function to recode reddit post flairs into mutually exclusive outcomes
def code_outcome(outcome):
  YTA_strings = ['ASSHOLE','YTA',"YOU'RE THE ASSHOLE"]
  NTA_strings = ['NOT THE A-HOLE','NOT THE ASSHOLE','NTA']
  ESH_strings = [
      'EVERYONE SUCKS','ESH','EVERBODY SUCKS','EVERYONE SUCKS HERE'
      , 'EVERYBODY SUCKS HERE'
      ]
  info_strings = ['NOT ENOUGH INFO', 'NEI', 'NOT ENOUGH INFO HERE']
  NAH_strings = ['NO A-HOLES HERE','NAH','NO A-HOLES']
  if outcome:
    if outcome.upper() in YTA_strings:
      return float(1)
    elif outcome.upper() in NTA_strings:
      return float(0)
    elif outcome.upper() in ESH_strings:
      return float(2)
    elif outcome.upper() in info_strings:
      return float(3)
    elif outcome.upper() in NAH_strings:
        return float(4)
    
# copy and replace function
def copy_and_replace(source_path, destination_path):
    import os
    import shutil
    if os.path.exists(destination_path) and os.path.exists(source_path):
        os.remove(destination_path)
    shutil.copy2(source_path, destination_path)

# binomial stats class
class binomial_stats:
  def __init__(self,df_col):

    import numpy as np

    self.df_col = df_col

    self._success = self.df_col.sum()

    self._n = len(df_col)

    self._rate = self._success/self._n

    self._sd_prop = np.sqrt(self._rate*(1-self._rate))

    self._sd_n = np.sqrt(self._rate*(1-self._rate)*self._n)

    self._se = np.sqrt(self._rate*(1-self._rate)/self._n)

  @property
  def success(self):
      return self._success

  @property
  def n(self):
      return self._n

  @property
  def rate(self):
      return self._rate

  @property
  def sd_prop(self):
      return self._sd_prop

  @property
  def sd_n(self):
      return self._sd_n

  @property
  def se(self):
      return self._se
  
import os
from datetime import datetime, timedelta

def files_last_n_days(directory, n, date_format="%Y-%m-%d", exclude_today=False):
    """
    Find files with dates from the last n days in the filename.

    Parameters:
    directory (str): The directory to search for files.
    n (int): The number of days to look back.

    Returns:
    list: A list of file paths that match the criteria.
    """

    if exclude_today == True:
        date_range = range(1,n)
    else:
       date_range = range(n)

    today = datetime.now()
    last_n_days = [today - timedelta(days=i) for i in date_range]

    matching_files = []
    for filename in os.listdir(directory):
        for day in last_n_days:
            date_str = day.strftime(date_format)
            if date_str in filename:
                matching_files.append(os.path.join(directory, filename))
                break

    return matching_files