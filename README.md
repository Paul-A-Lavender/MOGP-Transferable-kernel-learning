# Guides for using this repository
This should generally explain the functions of each member in this repository in alphabetic order, starting from each folder, and then each file at the root directory.
## /data
This directory contains all the data being used in this project. There are datasets stored in the folders, and those not divided further into folders. Those in the folders contains the major subset that is directly used in other parts, i.e. the direct dependents of this projects. For those not further classified, they are either data from Davin Jacob's work or exploratory attempts that may or may not have been used to generate those direct dependents.

If direct dependents in this section are corrupted or lost for reasons, re-run the script "Data_Preprocessing.ipynb" should re-generate them.

***Some further details for 4-params sigmoid-related datasets:*** This dataset tries to convert the data into sigmoid functions and to fit this function instead of concentration levels to obtain a full curve response. This was later abandoned(but should probably be usable with at most minor adjustments), as the model's prediction is far from ideal and the upper and lower bound given by the model is not very intuitive and therefore hardly informative.

## /logs
This research involves a massive amount of time spent on training models with different hyperparameters to compare comprehensively their performance. Therefore, the logging mechanism is introduced for 2 reasons:

***1. Training 1 specific model does not take incredibly long, but time consumption becomes gradually unacceptable with the increase of hyperparameter combinations.*** Therefore, instead of storing the entire model as an object, it is considered more convenient and storage-wise safer to store only the hyperparameters combination and corresponding performance. 

***2. There would be a need to query on performance, grouping by kernel and model classes.*** Therefore a data structure with easy access to these is prioritized.

For these reasons, the logging mechanism is introduced. This is done via the logger provided by Python, which also enables later queries on performance or grouping by kernel or model classes. The script "Extract_Model_Info.ipynb" is a query function implemented on regular expression matching, therefore changing the format of logging casually may render it unfunctional.

Logs are by default generated at the root directory, with errors and results logged separately. The usual procedure in this research is to first confirm the batch of result is valid, and then move it to the logs folder. You may instead generate the log into /logs folder directly by editing the path of the file handler in "Playground.ipynb".

## /paper
This directory contains the report of this research. You may check this for further technical details.

## /ref
This directory should contain the code repositories this research depends on. There used to be multiple repositories, yet after iterations only 1 remains, being Divin Jacob's code, used as the raw dataset.

## /utils
This directory contains most of the implementations in this research. Models and kernels are modulized such that they can be easily imported and used elsewhere. HelperFunctions are for the convenience of training and testing.

## Data_Preprocessing.ipynb
This script is responsible for processing data from Divin Jacob's code and generating the datasets needed in this experiment. No modification to this part is expected to be necessary, yet explorations are welcomed if interested.

## Extract_Model_Info.ipynb
A script that extracts performance and parameter information from generated log files. Using this would involve adapting certain code blocks to needs. More details are in the comment of this file.

## Multi_Domain_Toy_Dataset.ipynb & Single_Domain_Toy_Dataset.ipynb
Both file works on toy datasets for very preliminary presentation on a most simple dataset, that verifies the transfer learning ability of a model.

Minor adjustments are needed for the module in /utils to work on these 2 files, so as you find it these 2 files provide isolated implementation-independent of the modules in /utils. This avoids adding unnecessary redundancy to /utils modules and saves some trouble when one tweaks these implementations for exploration.

## Playground.ipynb
This is where cross-validation and grid search could be carried out. In this script, both of the job is set up with the necessary variables configured. For both of the jobs, progress reports are stored in "result.log"(for those who end up normally) and "error.log"(when an error disrupts the job).

