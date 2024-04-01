import os

# quiet?
quiet = True

# paths
main_path = '.'

# path containing model
model_path = os.path.join(main_path,'models','fullModel')

# path containing sheets that track predictions made
sheet_path = os.path.join(main_path,'data','app_tracking','prediction_sheets')
current_sheet_path = os.path.join(sheet_path,'unresolved_predictions.csv')

# path to env file with credentials
dotenv_path = os.path.join(main_path,'.env')

# set number of reddit posts to pull, before selecting those to tweet out
n_posts = 500

# set number of predictions to turn into tweets
n_tweets = 2
