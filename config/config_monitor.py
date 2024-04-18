import os

# dry run?
dry_run = 0

# quiet?
quiet = False

overwrite_day = True

# number of posts to pull
n_posts = 500

# paths
main_path = '.'

# path containing model
model_path = os.path.join(main_path,'models','fullModel')

# path to env file with credentials
dotenv_path = os.path.join(main_path,'.env')

# tracking sheets
monitor_sheet_path = os.path.join(main_path,'data/app_tracking/monitor_sheets')
raw_data_path = os.path.join(monitor_sheet_path,'raw_daily')
monitor_archive_path = os.path.join(monitor_sheet_path,'archive')