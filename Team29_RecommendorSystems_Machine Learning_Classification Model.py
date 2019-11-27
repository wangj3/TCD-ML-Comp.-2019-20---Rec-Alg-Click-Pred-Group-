#######Seperated the data into three train sets(Blog, JabRef, and My Volts) using Organization ID and running different models.

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
################----------------------------###################################
####MY-VOLTS########################

train = pd.read_csv('tcdml1920-rec-click-pred--training.csv')
test = pd.read_csv("tcdml1920-rec-click-pred--test.csv")

# Myvolts dataset
train_myVolts = train[train['organization_id'] == 4]
train_jabRef = train[train['organization_id'] == 1]
train_theBlog = train[train['organization_id'] == 8]
train_theBlog

test_myVolts = test[test['organization_id'] == 4]
test_jabRef = test[test['organization_id'] == 1]

test_theBlog = test[test['organization_id'] == 8]

train_myVolts[(train_myVolts['query_char_count']=='\\N') & (train_myVolts['set_clicked']==1)].describe()

y_col = ['set_clicked']
# rec_processing_time if it is too large the user may close the page and leave. This only removes 113 records.
train_myVolts = train_myVolts[train_myVolts['rec_processing_time']<9.2]

# import pandas_profiling 
# train_myVolts[feature_cols].profile_report()
feature_cols_to_drop =  ['app_lang','app_version','application_type','clicks',
                         'ctr','document_language_provided','first_author_id','local_hour_of_request'
                        ,'local_time_of_request','num_pubs_by_first_author','number_of_authors', 'organization_id'
                        ,'query_document_id','rec_processing_time','recommendation_set_id','session_id'
                        ,'time_recs_displayed','time_recs_recieved','time_recs_viewed','timezone_by_ip','user_id'
                        ,'user_java_version','user_os','user_os_version','user_timezone','year_published'
                        ,'response_delivered','number_of_recs_in_set']
print('columns',train_myVolts.columns)
feature_cols=train_myVolts.columns.drop('set_clicked').drop(feature_cols_to_drop)


print('Values with \\N Train',train_myVolts[feature_cols].isnull().sum())
print(len(train_myVolts[feature_cols].columns)) 
train_myVolts[train_myVolts[feature_cols]=='\\N']= np.nan
print('Values with NANs Train',train_myVolts[feature_cols].isnull().sum())

simpleimputermedian=SimpleImputer(strategy='median')
simpleimputermean=SimpleImputer(strategy='mean')

train_myVolts['query_detected_language']=train_myVolts['query_detected_language'].fillna('missing')
train_myVolts['abstract_detected_language']=train_myVolts['abstract_detected_language'].fillna('missing')

train_myVolts['query_char_count']=simpleimputermean.fit_transform(train_myVolts['query_char_count'].values.reshape(-1,1))
train_myVolts['query_word_count'] = simpleimputermedian.fit_transform(train_myVolts['query_word_count'].values.reshape(-1,1))
train_myVolts['abstract_word_count'] = simpleimputermedian.fit_transform(train_myVolts['abstract_word_count'].values.reshape(-1,1))
train_myVolts['abstract_char_count'] = simpleimputermedian.fit_transform(train_myVolts['abstract_char_count'].values.reshape(-1,1))

missing_value = train_myVolts[(train_myVolts['hour_request_received']== 23.0) & (train_myVolts['number_of_recs_in_set']==7)].query_char_count.mean()
print('missing value',missing_value)
train_myVolts[train_myVolts['recommendation_set_id']==311406.0].query_char_count =missing_value


test_myVolts['query_char_count']=simpleimputermean.fit_transform(test_myVolts['query_char_count'].values.reshape(-1,1))

all_data_myVolts = pd.concat([train_myVolts,test_myVolts],ignore_index=True)
train_myVolts.query_identifier.head()

# IMPORTANT Predict missing value based on other columns!!
all_data_myVolts['query_char_count'].describe()
all_data_myVolts.groupby("item_type")['query_char_count'].mean().sort_values()

def item_type_estimator(i):
    """Grouping item_type feature by query_char_count """
    a = 0
    if i<59:
        a = "TVs & monitors"
    elif i>=59 and i<62:
        a = "Networking"
    elif i>=62 and i<64:
        a = "Home entertainment"
    elif i>=64 and i<66:
        a = "Photo & frames"
    elif i>= 66 and i<67.3:
        a = "Music making & pedals"
    elif i>= 67.3 and i<68:
        a = "Everything else"
    elif i>=68 and i<68.8:
        a = 'DAB & audio'
    elif i>=68.8 and i<71.5:
        a = 'DVD players'
    elif i>=71.5 and i<77.5:
        a = 'Gaming & toys'
    else:
        a = "Hard drives & NAS"
    return a

all_data_myVolts.groupby("cbf_parser")['query_char_count'].mean().sort_values()

def cbf_parser_estimator(i):
    """Grouping item_type feature by query_char_count"""
    a = 0
    if i<69:
        a = "mlt_QP"
    elif i>=69 and i<70:
        a = "edismax_QP"
    else:
        a = "standard_QP"
    return a

# train_myVolts['item_type']=train_myVolts['item_type'].fillna('missing')
# train_myVolts['cbf_parser']=train_myVolts['cbf_parser'].fillna('missing')

##applying cabin estimator function. 
train_myVolts_Null_item_type= train_myVolts[train_myVolts['item_type'].isnull()]
train_myVolts_Not_Null_item_type= train_myVolts[train_myVolts['item_type'].notnull()]
train_myVolts_Null_cbf_parser= train_myVolts[train_myVolts['cbf_parser'].isnull()]
train_myVolts_Not_Null_cbf_parser= train_myVolts[train_myVolts['cbf_parser'].notnull()]

train_myVolts_Null_item_type['item_type']=train_myVolts_Null_item_type.query_char_count.apply(lambda x: item_type_estimator(x))
train_myVolts_Null_cbf_parser['cbf_parser']=train_myVolts_Null_cbf_parser.query_char_count.apply(lambda x: cbf_parser_estimator(x))


train_data_1=pd.concat([train_myVolts_Null_item_type, train_myVolts_Not_Null_item_type], axis=0)
train_data_2=pd.concat([train_myVolts_Null_cbf_parser, train_myVolts_Not_Null_cbf_parser], axis=0)

train_myVolts['item_type']=train_data_1['item_type']
train_myVolts['cbf_parser']=train_data_2['cbf_parser']
train_myVolts['country_by_ip']=train_myVolts['country_by_ip'].fillna('missing')
print('Values with NANs Train',train_myVolts[feature_cols].isnull().sum())

y = train_myVolts.set_clicked
X = train_myVolts[feature_cols]
from category_encoders import TargetEncoder
t1 = TargetEncoder()
t1.fit(X, y)
X = t1.transform(X)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1234)

##check
X_train.to_csv('X_train4.csv',index=False)
y_train.to_csv('y_train4.csv',index=False)


from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
logreg1 =LogisticRegression ()
# logreg1 = RandomForestClassifier(n_estimators=500)
# for random forest is 0.9920 but logistic is 0.9922
logreg2 = XGBClassifier()

# fit model
logreg1.fit(X_train, y_train)
logreg2.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class1 = logreg1.predict(X_test)  #pred1
y_pred_class2 = logreg2.predict(X_test)  #pred2
#np.sqrt(metrics.mean_squared_error(y_test, y_pred_class))

# calculate accuracy
from sklearn import metrics
print('Accuracy Score1',metrics.accuracy_score(y_test, y_pred_class1))
print('Accuracy Score2',metrics.accuracy_score(y_test, y_pred_class2))
# this produces a 2x2 numpy array (matrix)
print('Confusion Matrix1',metrics.confusion_matrix(y_test, y_pred_class1))
print('Confusion Matrix2',metrics.confusion_matrix(y_test, y_pred_class2))


print('Values with \\N Test',test_myVolts.isnull().sum())
test_myVolts[test_myVolts=='\\N']= np.nan
print('Values with NANs Train',train_myVolts.isnull().sum())

test_myVolts['query_detected_language']=test_myVolts['query_detected_language'].fillna('missing')
test_myVolts['abstract_detected_language']=test_myVolts['abstract_detected_language'].fillna('missing')

#test_myVolts['item_type']=test_myVolts['item_type'].fillna('missing')
#test_myVolts['cbf_parser']=test_myVolts['cbf_parser'].fillna('missing')

##applying cabin estimator function. 
test_myVolts_Null_item_type= test_myVolts[test_myVolts['item_type'].isnull()]
test_myVolts_Not_Null_item_type= test_myVolts[test_myVolts['item_type'].notnull()]
test_myVolts_Null_cbf_parser= test_myVolts[test_myVolts['cbf_parser'].isnull()]
test_myVolts_Not_Null_cbf_parser= test_myVolts[test_myVolts['cbf_parser'].notnull()]


test_myVolts_Null_item_type['item_type']=test_myVolts_Null_item_type.query_char_count.apply(lambda x: item_type_estimator(x))
test_myVolts_Null_cbf_parser['cbf_parser']=test_myVolts_Null_cbf_parser.query_char_count.apply(lambda x: cbf_parser_estimator(x))
test_data_1=pd.concat([test_myVolts_Null_item_type, test_myVolts_Not_Null_item_type], axis=0)
test_data_2=pd.concat([test_myVolts_Null_cbf_parser, test_myVolts_Not_Null_cbf_parser], axis=0)

test_myVolts['item_type']=test_data_1['item_type']
test_myVolts['cbf_parser']=test_data_2['cbf_parser']


test_myVolts['country_by_ip']=test_myVolts['country_by_ip'].fillna('missing')

test_myVolts['query_char_count']=simpleimputermean.fit_transform(test_myVolts['query_char_count'].values.reshape(-1,1))
test_myVolts['query_word_count'] = simpleimputermedian.fit_transform(test_myVolts['query_word_count'].values.reshape(-1,1))

test_myVolts['abstract_word_count'] = simpleimputermedian.fit_transform(test_myVolts['abstract_word_count'].values.reshape(-1,1))
test_myVolts['abstract_char_count'] = simpleimputermedian.fit_transform(test_myVolts['abstract_char_count'].values.reshape(-1,1))


# split X and y into training and testing sets
################################################
E = test_myVolts[feature_cols]
E = t1.transform(E)
B1 = logreg1.predict(E)
B2 = logreg2.predict(E)

B= B1
# stacked_predictions = np.column_stack((y_pred_class1,y_pred_class2))
# stacked_test_predictions = np.column_stack((B1,B2))

# meta_model = LogisticRegression()
# meta_model.fit(stacked_predictions,y_test)
# B = meta_model.predict(stacked_test_predictions) #Final_prediction
E.head()

df2=pd.DataFrame()
df2['recommendation_set_id'] = test_myVolts['recommendation_set_id']
df2['set_clicked'] = B

df2.to_csv('ML_Assignment2_My_Volts_20191116.csv',index=False)

df2[df2['set_clicked']==1].sum()


###############-----------------------------###################################
##################JABREF#######################################################

#####Reading Trainf and Test Data
train_df = pd.read_csv('tcdml1920-rec-click-pred--training.csv')
test_df = pd.read_csv("tcdml1920-rec-click-pred--test.csv")

###filtering JabRef
train_df_jabRef = train_df[train_df['organization_id'] == 1]
test_df_jabRef = test_df[test_df['organization_id'] == 1]

##replacing \N values with NAN

train_df_jabRef = train_df_jabRef.replace("\\N",np.nan)

test_df_jabRef = test_df_jabRef.replace("\\N",np.nan)
##########Removing columns with more than 50 percent NA values

pct_null = train_df_jabRef.isnull().sum() / len(train_df_jabRef)
missing_features = pct_null[pct_null > 0.50].index
train_df_jabRef.drop(missing_features, axis=1, inplace=True)

pct_null = test_df_jabRef.isnull().sum() / len(test_df_jabRef)
missing_features = pct_null[pct_null > 0.50].index
test_df_jabRef.drop(missing_features, axis=1, inplace=True)

######Removing more columns from test data
test_df_jabRef = test_df_jabRef.drop(['time_recs_recieved','time_recs_displayed','time_recs_viewed'],axis=1)
test_df_jabRef.insert(27, "set_clicked",np.nan) 

####################
train_df_jabRef.describe()
train_df_jabRef.info()

##Finding Correlation
print(train_df_jabRef.corr(method ='pearson'))

######Removing more Train and Test Columns

train_df_jabRef = train_df_jabRef.drop(['item_type','application_type','query_identifier','request_received','response_delivered','rec_processing_time','timezone_by_ip','local_time_of_request','number_of_recs_in_set'],axis=1)
test_df_jabRef = test_df_jabRef.drop(['item_type','application_type','query_identifier','request_received','response_delivered','rec_processing_time','timezone_by_ip','local_time_of_request','number_of_recs_in_set'],axis=1)

######Fillling categorical NA values Train Data
train_df_jabRef["query_detected_language"] = train_df_jabRef["query_detected_language"].fillna(method='ffill')
train_df_jabRef["app_lang"] = train_df_jabRef["app_lang"].fillna(method='ffill')
train_df_jabRef["country_by_ip"] = train_df_jabRef["country_by_ip"].fillna(method='ffill')

######Fillling categorical NA values Test Data
test_df_jabRef["query_detected_language"] = test_df_jabRef["query_detected_language"].fillna(method='ffill')
test_df_jabRef["app_lang"] = test_df_jabRef["app_lang"].fillna(method='ffill')
test_df_jabRef["country_by_ip"] = test_df_jabRef["country_by_ip"].fillna(method='ffill')

######Fillling numerical NA values Train Data
train_df_jabRef["local_hour_of_request"] = train_df_jabRef["local_hour_of_request"].fillna(train_df_jabRef["local_hour_of_request"].median())
train_df_jabRef["recommendation_algorithm_id_used"] = train_df_jabRef["recommendation_algorithm_id_used"].fillna(train_df_jabRef["recommendation_algorithm_id_used"].median())
train_df_jabRef["app_version"] = train_df_jabRef['app_version'].replace('*unknown*','NA')
train_df_jabRef["app_version"] = train_df_jabRef['app_version'].replace(np.nan,'NA')

######Fillling numerical NA values Test Data
test_df_jabRef["local_hour_of_request"] = test_df_jabRef["local_hour_of_request"].fillna(test_df_jabRef["local_hour_of_request"].median())
test_df_jabRef["recommendation_algorithm_id_used"] = test_df_jabRef["recommendation_algorithm_id_used"].fillna(test_df_jabRef["recommendation_algorithm_id_used"].median())
test_df_jabRef["app_version"] = test_df_jabRef['app_version'].replace('*unknown*','NA')
test_df_jabRef["app_version"] = test_df_jabRef['app_version'].replace(np.nan,'NA')

simpleimputermean=SimpleImputer(strategy='mean')
test_df_jabRef['query_char_count']=simpleimputermean.fit_transform(test_df_jabRef['query_char_count'].values.reshape(-1,1))

train_df_jabRef['query_char_count']=simpleimputermean.fit_transform(train_df_jabRef['query_char_count'].values.reshape(-1,1))

all_data = pd.concat([train_df_jabRef,test_df_jabRef],ignore_index=True)

predict1 = all_data.copy()
predict1 = predict1.drop(['ctr'],axis=1)

predict1.groupby("cbf_parser")['query_char_count'].mean().sort_values()
predict1_cbfparser_null= predict1[predict1['cbf_parser'].isnull()]

def cbf_parser_estimator_jab(i):
    """Grouping item_type feature by the first letter"""
    a = 0
    if i<62:
        a = "mlt_QP"
    elif i>=62 and i<66.5:
        a = "edismax_QP"
    else:
        a = "standard_QP"
    return a

predict1_cbfparser_null['cbf_parser']=predict1_cbfparser_null.query_char_count.apply(lambda x: cbf_parser_estimator_jab(x))
predict1_cbfparser_Not_null= predict1[predict1['cbf_parser'].notnull()]
x = pd.concat([predict1_cbfparser_Not_null, predict1_cbfparser_null], axis=0)
predict1['cbf_parser'] = x['cbf_parser']

#####Categorical Encoding
predict1 = predict1.replace('unknown','NA')
predict1['algorithm_class'] = predict1['algorithm_class'].replace(np.nan,'NA')
predict1['cbf_parser'] = predict1['cbf_parser'].replace(np.nan,'NA')

predict2 = predict1.copy()
predict2 = pd.get_dummies(predict2, columns=['query_detected_language','cbf_parser','algorithm_class', 'app_version', 'app_lang', 'country_by_ip'])

from sklearn.preprocessing import LabelBinarizer
lb_style = LabelBinarizer()
predict2["search_title"] = lb_style.fit_transform(predict2["search_title"])
predict2["search_keywords"] = lb_style.fit_transform(predict2["search_keywords"])
predict2["search_abstract"] = lb_style.fit_transform(predict2['search_abstract'])

Test_JabRef_df = predict2[predict2['clicks'] == 'nA']
train_jabRef_df = predict2[predict2['clicks'] != 'nA']

train_jabRef_df = train_jabRef_df.drop(['clicks'],axis=1)
Test_JabRef_df = Test_JabRef_df.drop(['clicks'],axis=1)

######Model Predict
X = np.array(train_jabRef_df.drop('set_clicked',axis=1))
y = np.array(train_jabRef_df['set_clicked'])
from sklearn.model_selection import KFold, train_test_split
kfold = KFold(n_splits=100, shuffle=True, random_state=42)
scores = []
for train_index, test_index in kfold.split(train_jabRef_df):   
    X_train, X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy",max_depth=32)
clf.fit(X_train, y_train)
# make class predictions for the testing set
y_pred_class_co = clf.predict(X_test)
#np.sqrt(metrics.mean_squared_error(y_test, y_pred_class))
# calculate accuracy
from sklearn import metrics
print('Accuracy Score',metrics.accuracy_score(y_test, y_pred_class_co))

##########Implementing Model on Test Data
Test_JabRef_df.drop('set_clicked',axis=1,inplace=True)
y_pred_read = clf.predict(Test_JabRef_df)

dfX=pd.DataFrame()
dfX['recommendation_set_id'] = Test_JabRef_df['recommendation_set_id']
dfX['set_clicked'] = y_pred_read

dfX.to_csv('JabRefPrediction.csv',index=False)

#predict_tests = pd.read_csv("tcdml1920-rec-click-pred--test.csv")
#predict_tests['set_clicked'] = pd.DataFrame(y_pred_read).iloc[:,-1]
#pd.DataFrame(y_pred_read).to_csv("JabRefPrediction.csv", index = False)

################----------------------------###################################
#######BLOG######################

from category_encoders import TargetEncoder
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#train = pd.read_csv("tcdml1920-rec-click-pred--training.csv")
#test = pd.read_csv("tcdml1920-rec-click-pred--test.csv")

train_theBlog = train[train['organization_id'] == 8]

print(len(train_theBlog.columns))
train_theBlog = train_theBlog.drop(["user_id", "session_id", "document_language_provided","year_published","number_of_authors","first_author_id","num_pubs_by_first_author","app_version","app_lang","user_os","user_os_version","user_java_version","user_timezone"], axis=1)
train_theBlog = train_theBlog.drop(["response_delivered", "rec_processing_time", "number_of_recs_in_set", "time_recs_recieved", "time_recs_displayed", "time_recs_viewed", "clicks","ctr"], axis=1)
print('Values with \\N Train',train_theBlog.isnull().sum())

print(len(train_theBlog.columns))

 
train_theBlog[train_theBlog=='\\N']= np.nan

print('Values with NANs Train',train_theBlog.isnull().sum())

from sklearn.impute import SimpleImputer
simpleimputermedian=SimpleImputer(strategy='median')

simpleimputermean=SimpleImputer(strategy='mean')

simpleimputermode=SimpleImputer(strategy='most_frequent')

###############################################################################
train_theBlog['query_detected_language']=train_theBlog['query_detected_language'].fillna('missing')
train_theBlog['abstract_detected_language']=train_theBlog['abstract_detected_language'].fillna('missing')

train_theBlog['item_type']=train_theBlog['item_type'].fillna('missing')
train_theBlog['cbf_parser']=train_theBlog['cbf_parser'].fillna('missing')


train_theBlog['query_char_count']=simpleimputermean.fit_transform(train_theBlog['query_char_count'].values.reshape(-1,1))
train_theBlog['query_document_id'] = simpleimputermedian.fit_transform(train_theBlog['query_document_id'].values.reshape(-1,1))
train_theBlog['abstract_word_count'] = simpleimputermode.fit_transform(train_theBlog['abstract_word_count'].values.reshape(-1,1))
train_theBlog['abstract_char_count'] = simpleimputermode.fit_transform(train_theBlog['abstract_char_count'].values.reshape(-1,1))
train_theBlog['timezone_by_ip'] = simpleimputermedian.fit_transform(train_theBlog['timezone_by_ip'].values.reshape(-1,1))

y = train_theBlog.set_clicked

train_theBlog = train_theBlog.drop('set_clicked',axis=1)
# define X and y
feature_cols = train_theBlog.columns

# X is a matrix, hence we use [] to access the features we want in feature_cols
X = train_theBlog[feature_cols]

################################################
t1 = TargetEncoder()
t1.fit(X, y)
X = t1.transform(X)
################################################
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=1234)

#from sklearn.linear_model import LogisticRegression

# instantiate model
#logreg = LogisticRegression()
logreg = XGBClassifier()

# fit model
logreg.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)
#np.sqrt(metrics.mean_squared_error(y_test, y_pred_class))

# calculate accuracy
from sklearn import metrics
print('Accuracy Score',metrics.accuracy_score(y_test, y_pred_class))

# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
print('Confusion Matrix',metrics.confusion_matrix(y_test, y_pred_class))

#Predicting on Testing Data
test_theBlog = test[test['organization_id'] == 8]
test_theBlog = test_theBlog.drop(["user_id", "session_id", "document_language_provided","year_published","number_of_authors","first_author_id","num_pubs_by_first_author","app_version","app_lang","user_os","user_os_version","user_java_version","user_timezone"], axis=1)
test_theBlog = test_theBlog.drop(["response_delivered", "rec_processing_time", "number_of_recs_in_set", "time_recs_recieved", "time_recs_displayed", "time_recs_viewed", "clicks","ctr"], axis=1)

print('Values with \\N Test',test_theBlog.isnull().sum())
test_theBlog[test_theBlog=='\\N']= np.nan
print('Values with NANs Train',train_theBlog.isnull().sum())
###############################################################################
test_theBlog['query_detected_language']=test_theBlog['query_detected_language'].fillna('missing')
test_theBlog['abstract_detected_language']=test_theBlog['abstract_detected_language'].fillna('missing')

test_theBlog['item_type']=test_theBlog['item_type'].fillna('missing')
test_theBlog['cbf_parser']=test_theBlog['cbf_parser'].fillna('missing')


test_theBlog['query_char_count']=simpleimputermean.fit_transform(test_theBlog['query_char_count'].values.reshape(-1,1))
test_theBlog['query_document_id'] = simpleimputermedian.fit_transform(test_theBlog['query_document_id'].values.reshape(-1,1))
test_theBlog['abstract_word_count'] = simpleimputermode.fit_transform(test_theBlog['abstract_word_count'].values.reshape(-1,1))
test_theBlog['abstract_char_count'] = simpleimputermode.fit_transform(test_theBlog['abstract_char_count'].values.reshape(-1,1))
test_theBlog['timezone_by_ip'] = simpleimputermedian.fit_transform(test_theBlog['timezone_by_ip'].values.reshape(-1,1))

test_theBlog = test_theBlog.drop('set_clicked',axis=1)
test_theBlog_backup = test_theBlog
# define X and y
feature_cols = test_theBlog.columns
print("feature_cols",feature_cols)



# X is a matrix, hence we use [] to access the features we want in feature_cols
test_theBlog = test_theBlog[feature_cols]

# split X and y into training and testing sets
################################################
E = test_theBlog
#E=t1.transform(E)
E = t1.transform(E)

B = logreg.predict(E)

df2=pd.DataFrame()
df2['recommendation_set_id'] = test_theBlog_backup['recommendation_set_id']
df2['set_clicked'] = B

df2.to_csv('ML_Assignment2_Recommender_SystemXGboost.csv',index=False)

################----------------------------###################################
pred_myVolts = pd.read_csv('ML_Assignment2_My_Volts_20191116.csv')
pred_jabRef = pd.read_csv('JabRefPrediction.csv')
pred_theBlog = pd.read_csv('ML_Assignment2_Recommender_SystemXGboost.csv')

pred = pd.concat([pred_myVolts,pred_jabRef, pred_theBlog],ignore_index=True)
pred=pred.sort_values(by=['recommendation_set_id'])
pred.to_csv("tcdml1920-rec-click-pred--submission file.csv",index=False,float_format='%.f')
'done'
