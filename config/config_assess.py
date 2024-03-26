import os

# quiet?
quiet = True

# paths
main_path = '.'

# path containing model
model_path = os.path.join(main_path,'models','fullModel')

# path to sheets
sheet_path = os.path.join(main_path,'prediction_sheets')
unresolved_preds_path = os.path.join(sheet_path,'unresolved_predictions.csv')

# path to env file with credentials
dotenv_path = os.path.join(main_path,'.env')
