# Directories and Files

acoustic/ - Contains the .csv files for each acoustic feature set
linguistic/ - Contains the .csv files for each linguistic feature set


IMPORTANT - Download the mpnet_768 model from here, and store inside linguistic/:

https://drive.google.com/file/d/1ZFhJPi3BB-5MF85dBxPRargbaK16H2D5/view?usp=drive_link

models/ - Contains the .pkl files for each model
labels/ - Contains the .csv file for valence and arousal labels
testing.py - Python script to evaluate each model and store its statistics in stats.txt

stats.txt - File showing the R2 Score, Explained Variance Score, Root Mean Squared Error and Max Error for each model, broken up by valence and arousal.
