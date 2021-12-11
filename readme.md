Required Pacakages:
tensorflow
keras
pyyaml h5py
sklearn
pandas
numpy



Required Downloading Data:
https://datasets.imdbws.com/ 

Steps to run our project:
1. Install packages
2. Download files from above link
3. run filter_data.py

Run either 
1. python AdultQuestion.py or 
2. Go to the movieTrends folder and run "python3 fileTrendFilter.py". This will create 2 datasets, one with imputed nan values and one without nan values
3. run "python3 filmTrend.py". There will be a prompt that will ask "Train with imputed NaN values(Y/N)?" where the user can train on either imputed datasets or datasets without nan values.
4. To test predictions made by the model, run "python3 filmTrendPredict.py". Similar to step 2, a prompt will ask whether the prediction will be done on the model trained with the imputed dataset or the model trained with the nanless dataset.
5. Follow the input prompts as followed
