import os

# dry run?
dry_run = 0

# quiet?
quiet = False

# paths
main_path = '.'

# path containing model
model_path = os.path.join(main_path,'models','fullModel')

# path to sheets
sheet_path = os.path.join(main_path,'data/app_tracking/prediction_sheets')
unresolved_preds_path = os.path.join(sheet_path,'unresolved_predictions.csv')
archive_sheet_path = os.path.join(sheet_path,'resolved_predictions')

# path to env file with credentials
dotenv_path = os.path.join(main_path,'.env')
