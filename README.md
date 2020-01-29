# TCD-ML-Comp.-2019-20---Rec-Alg-Click-Pred-Group-
Predict whether a set of delivered recommendations will be clicked or not sourced from three different companies who need ML.

Group:
Jiachun Wang 
Tanmay Bagla
Himanshu Gupta

Results:
https://www.kaggle.com/c/tcd-ml-comp-201920-rec-alg-click-pred-group/leaderboard
![Image of Kaggle](https://octodex.github.com/images/?)
Rank Top 1st for first batch of data - kaggle public leader board 
Rank Top 2nd for entire data set - kaggle private leader board
Final score:91%

With other courseworks, competitions and a final exam, I utimately achieved Distincation for ML course.

# My Volts
@Author Jiachuan Wang - Team 29
### Data Analysis ###
The following code is used to generate a report
import pandas_profiling
train_myVolts.profile_report()

Auto Analysis:
app_version has constant value "\N"
ctr is highly correlated with clicks (ρ = 0.9968227792)
document_language_provided has constant value "\N
first_author_id has constant value "\N"
num_pubs_by_first_author has constant value "\N
number_of_authors has constant value "\N"
organization_id has constant value "4"
recommendation_set_id is highly correlated with df_index (ρ = 0.9999760388)
set_clicked is highly correlated with ctr (ρ = 0.9372770115)
user_java_version has constant value "\N"	Rejected
user_os has constant value "\N"	Rejected
user_os_version has constant value "\N"	Rejected
user_timezone has constant value "\N"	Rejected
year_published has constant value "\N"

Detailed analysis:
abstract_char_count- fill na mean
abstract_detected_language- cat, fill na
abstract_word_count - fill na mean
algorithm_class- cat. most of them are content based filtering algorithm
app_lang- remove. only en or null
application_type - remove only 145 values are 0. A lot others are e commence
cbf_parser- cat, fill na
clicks- remove. we are using set_clicked
country_by_ip - cat. lot of countries
ctr - remove
dff_index - remove. We are using recomend_set_id
document_language_provided - remove. 
first_author_id- remove. 
hour_request_received- num. 0-24 h
item_type- cat. need to fill /N
local_hour_of_request- remove.  don't know how to use it.
local_time_of_request- remove. don't know how to use it.
num_pubs_by_first_author - remove. constant value
number_of_authors - remove. Constant value
number_of_recs_in_set- remove. Don't know how to use it.
organization_id- remove. constant value.
query_char_count- cat. convert to numberic
query_detected_language- cat. fill na /N
query_document_id - remove. We use query_identifier
query_identifier - cat. fill na /N
query_word_count - convert to number. number of words
rec_processing_time - num. remove the row if it is not the difference
The duration in seconds it took the server to calculate recommendations. This should be equal to the difference of request_received and response_delivered.
recommendation_algorithm_id_used - already num
recommendation_set_id - our x1. we should remove for training.
request_received- cat convert to num
request_received- cat convert to num
search_abstract-cat binary
search_keywords-cat binary
search_title-cat binary
session_id - remove
set_clicked - what we are going to predict
time_recs_displayed- first filtering strategy - remove the records that has value "/N" and set_clicked =0
time_recs_recieved - remove as duplicate as time_recs_displayed
time_recs_viewed- second filtering strategy - remove value /N if the user can't see it of course there is no click.
timezone_by_ip- a lot of /N remove.
user_id- remove. not many users use this.
user_java_version-remove.constant value.
user_os- remove. constant value.
user_os_version -remove. constant value.
user_timezone- remove.constant value
year_published -remove constant value.

### Data Standarization, Encoding and Impution ###
Give my credits to my teammates Himansu/Tanmay, target encoding, filna, (-1,1) standarization, SimpleImputermean and SimpleImputermedian works well in the initial stage. 

In the later stage, I figured out a smart approach to improve it further with inspiration from a notebook I researched in the Titanic competition posted by Masum Rumi. In the competition, he fillna by fare price. I studied the entire data set and concluded that we can fillna for some categorical data columns based on the mean value of query_char_count.
In my volts dataset, country_by_ip still adopt the normal fillna as
a. It is difficult to define the gap
b. There aren't much data missing for this column

I shared the solution with the team as soon as I verified it by submission. It is great to know that it also works for Jaref.
Reference: 
1.https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic/notebook

### Feature Engineering###
Tried creating new feature engineering columns
a. Number of chars of query_identifier
b. Category of char number of query_identifer
c. Create goups of query_identifier
However, they doesn't seem enhance our score in public leader board.

### Model selection ###
Tried RandomForestClassifer, XGBoostClassifier, SVM and logistic regression. XGBoostClassifier is better in the accuracy score and confusion matrix.
Nail down logistic regression eventually.


