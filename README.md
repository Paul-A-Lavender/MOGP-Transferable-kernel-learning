# Guides for using this repository
This should generally explains the functions of each member in this repository in alphabetic order, starting from each folder, and then each file at root directory.
## /data
This directory contains all the data being used in this project. There are datasets stored in the folders, and those not divided further into folders. Those in the folders contains the major subset that is directly used in other parts, i.e. the direct dependents of this projects. For those not further classified, they are either data from Davin Jacob's work or exploratory attempts that may or may not have been used to generate those direct dependents.

If direct dependents in this section are corrupted or lost for reasons, re-run the script "Data_Preprocessing.ipynb" should re-generate them.

***Some further details for 4-params sigmoid related datasets:*** this is a dataset try to convert the data into sigmoid functions, and to fit this funciton instead of concentration levels in order to obtain full curve response. This was later abandoned(but should be probably usable with at most minor adjusments), as the prediction of the model is far from ideal and the upper and lower bound given by the model is not very intuitive and therefore hardly informative.

## /logs
This research involves massive amount of time spend on training models with different hyperparamaters to compare comprehensively their performance. Therefore, the logging mechanism is introduced for 2 reasons:

***1. Training 1 specific model does not take incredibly long, but time consumption becomes gradually unacceptable with the increase of hyperparameters combinations.*** Therefore, instead of storing the entire model as object, it is considered more convenient and storage-wise safer to store only the hyperparameters combination and corresponding performance. 

***2. There would be a need to query on performance, grouping by kernel and model classes.*** Therefore a data structure with easy access to these are prioritized.

For these reasons, the logging mechanism is introduced. This is done via the logger provided by Python, which also enables later query on performance or grouping by kernel or model classes. The script "Extract_Model_Info.ipynb" is a query function implemeted on regular expression matching. For it to work the structure of the logging can't be changed casually.
