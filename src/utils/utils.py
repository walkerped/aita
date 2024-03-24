# read in csv if it exists
def read_csv_ifexist(csv_path):
    import pandas as pd
    import os
    # read in current pred sheet
    if os.path.exists(csv_path):
        sheet_df = pd.read_csv(csv_path, index_col = 0)
    else:
        sheet_df = None
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

