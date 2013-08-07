kaggle-yelp
===========

#### How to use it
You'll need Python 2.7.x on a 64bit system (just for text feature extractions).
The following step will guide you on how to get your predictions.

#### Convert the data
Place the copies of all the json data into the 'Data' folder, then run convert.py

#### Data Preparation
All the three python script will pre-process the data (some details inside)
and save it in a csv. Make sure you've created a folder named 'DataProcessed' before
running the scripts. Then run in the following order:

1) preprocess_business.py

2) preprocess_review.py (this will take some time)

3) preprocess_review_testdata.py

#### Train & Predict
Running model.py will load the data, and train a Gradient Boosting Regressor model
and make some predictions. Training the model will take approximately 30 minutes
on an average speed computer. The submission should give you an private leaderboard
of 0.46049 (21st standing)

