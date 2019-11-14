import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import make_scorer
import datetime
from itertools import chain
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


def sniff(df):
  '''
  Function to check all columns in dataframe for missing values
  and see how what type each column is storing the data as
  '''
  with pd.option_context("display.max_colwidth", 20):
      info = pd.DataFrame()
      info['sample'] = df.iloc[0]
      info['data type'] = df.dtypes
      info['percent missing'] = df.isnull().sum()*100/len(df)
      return info.sort_values('data type')

def add_null_for_blanks(df, columns):
  '''
  Takes in a dataframe and a list of column names then creates a 'isnull'
  column that comprises of a boolean if the value has a length zero
  for each row.
  '''
  for col in columns:
    df[f'{col}_isnull'] = df[col].map(lambda x: True if len(x) == 0 else False)


def add_hour_dow_columns(df, columns):
  ''' 
  Takes in a dataframe and a list of column names
  then creates an hour and week column for each date
  column.
  '''
  for col in columns:
        df[f'{col}_dow'] = df[col].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).weekday())
        df[f'{col}_hour'] = df[col].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).hour)

def add_null_col (df, columns):
  '''
  Creates a new column 'col_isnull' to the dataframe to indicate if the values
  in each row in the indicated columns had a null or a ''.
  '''
  for col in columns:
    df[col] = df[col].apply(lambda x: np.nan if x == '' else x)
    df[f'{col}_isnull'] = df[col].isnull()
    df.drop(labels=[col],axis=1,inplace=True)

def categorical_conversion (df, columns):
  '''
  Using pandas to set all categorical features to type category. Ex. All email domains 
  will be set to numbers, and the +1 is to set nan values from -1 to 0. So 0's can be
  trained as a category for the models
  '''
  for col in columns:
    df[col] = df[col].astype('category').cat.as_ordered()
    df[col] = df[col].cat.codes + 1






if __name__ == "__main__":

# Used to check progress of our dataframe (e.g. what columns still need to be cleaned)

# qq=[drop_features,bl_features,null_bool,
# cont_features, date_features, cat_features]

# qq=list(chain(*qq))
# dealwith=[]
# for col in df.columns:
#     if col not in qq:
#         print(col)
#         dealwith.append(col)
# sniff(df[dealwith])
# Jeffrey Epstein didn't kill himself lol

  df=pd.read_json('data/data.json')
  columns=['acct_type', 'approx_payout_date', 'body_length', 'channels', 'country',
       'currency', 'delivery_method', 'description', 'email_domain',
       'event_created', 'event_end', 'event_published', 'event_start',
       'fb_published', 'gts', 'has_analytics', 'has_header', 'has_logo',
       'listed', 'name', 'name_length', 'num_order', 'num_payouts',
       'object_id', 'org_desc', 'org_facebook', 'org_name', 'org_twitter',
       'payee_name', 'payout_type', 'previous_payouts', 'sale_duration',
       'sale_duration2', 'show_map', 'ticket_types', 'user_age',
       'user_created', 'user_type', 'venue_address', 'venue_country',
       'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state']

  target='acct_type'

  # df.acct_type.unique()
  # (['fraudster_event', 'premium', 'spammer_warn', 'fraudster',
  #       'spammer_limited', 'spammer_noinvite', 'locked', 'tos_lock',
  #       'tos_warn', 'fraudster_att', 'spammer_web', 'spammer'])

  df['Fraud'] = df['acct_type'].str.contains('fraud', regex=True)

  cat_features = ['venue_country', 'country', 'payout_type', 'venue_state',
  'email_domain', 'listed', 'currency', 'user_type', 'delivery_method',
  'has_header']

  #These are in Unix(seconds)
  date_features = ['event_created', 'event_end', 'event_published',
  'event_start', 'approx_payout_date','user_created',]

  cont_features = ['user_age', 'num_payouts', 'org_desc_length','body_length',
  'name_length','channels','num_order','gts','sale_duration2', 'sale_duration',
  'num_previous_payouts']

  #boolean features

  bl_features= ['has_header', 'has_analytics', 'has_logo', 'show_map',
  'fb_published']

  #create separate columns marking a row as True for nan values in these features
  
  null_bool=['venue_name', 'org_facebook','org_twitter']

# Deal with nan values, create separate columns indicating True
# if the nan value was replaced by the median of the feature column

  df['sale_duration_isnull'] = df['sale_duration'].isnull()
  df['sale_duration'] = df['sale_duration'].fillna(df['sale_duration'].median())
  df['event_published'] = df['event_published'].fillna(df['event_published'].median())


#Aggregating the amount sold by each row. Extracted from ticket_type.
#Created a new column
  df['qsold']=df.ticket_types.apply(lambda x: sum([b['quantity_sold'] for b in x]))

#Finding the number of payouts by row. Creating a new column.
  df['num_previous_payouts'] = df.previous_payouts.map(lambda x: len(x))

# Using functions from above, read their docstring if more info is needed
  null_blank = ['payee_name', 'org_name', 'previous_payouts']
  add_null_for_blanks(df, null_blank)
  add_hour_dow_columns(df, date_features)
  categorical_conversion(df, cat_features)
  add_null_col(df, null_bool)
  

  #Organization Description Length
  df['org_desc_length'] = df.org_desc.map(lambda x: len(x))
  
  #Name length
  df['name_length'] = df.name.map(lambda x: len(x))

# Features to drop before training
  drop_features= ['object_id', 'venue_latitude', 'venue_longitude','acct_type',
  'description', 'venue_address','name','org_desc', 'ticket_types', 'previous_payouts',
  'org_name','payee_name']

# Dropping date columns as we already extracted day of week and hour the columns
  for col in date_features:
    drop_features.append(col)

  targets=df.pop('Fraud')
  df.drop(labels=drop_features,axis=1,inplace=True)

  X_train,X_test, y_train, y_test = train_test_split(df,targets)

  rf=RandomForestClassifier()

  # Gridsearch for random forest and gradient boosting below

  # paramgrid={'max_depth':[None, 100, 50, 20, 10],
  #     'min_samples_leaf':[1, 2, 4 ],
  #     'min_samples_split': [2,5,10],
  #     'n_estimators': [100,150,200],
  #     'warm_start':[True,False],
  #     'max_features': ['sqrt',5,10,20]}

  # Setting roc_auc_score as our scoring metric

  auc_scorer = make_scorer(roc_auc_score)

  # paramgrid={'max_depth':[ 50,40,60],
  #     'min_samples_leaf':[1, 2, 4 ],
  #     'min_samples_split': [2,3,4],
  #     'n_estimators': [130,150,170],
  #     'warm_start':[True],
  #     'max_features': ['sqrt',20,30,40]}
  # GS=GridSearchCV(RandomForestClassifier(),param_grid=paramgrid,scoring=auc_scorer,verbose=1,n_jobs=7)
  # GS.fit(X_train,y_train)

  # GS.best_params_

  # {'max_depth': 50,
  # 'max_features': 20,
  # 'min_samples_leaf': 1,
  # 'min_samples_split': 2,
  # 'n_estimators': 150,
  # 'warm_start': True}

  # Our best parameters for the random forest gridsearch

  rf=RandomForestClassifier(n_estimators=150,
    max_depth=50, max_features=20, min_samples_leaf=1, 
    min_samples_split=2,warm_start=True)

  rf.fit(X_train, y_train)
  rf.score(X_test,y_test)# 0.9852161785216178

  recall_score(y_test,rf.predict(X_test)) #0.8801169590643275

  y_preds = rf.predict(X_test)

  gb = GradientBoostingClassifier()



  loss='deviance', learning_rate=0.1, n_estimators=100, 
  subsample=1.0, criterion='friedman_mse', min_samples_split=2, 
  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
  min_impurity_decrease=0.0, min_impurity_split=None, init=None,
  random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, 
  warm_start=False, presort='auto', validation_fraction=0.1, 
  n_iter_no_change=None, tol=0.0001)
    
gb_paramgrid = {'n_estimators':[100,150],
'learning_rate': [0.1, 0.05, 0.01],
'min_samples_split':[2,4],
'min_samples_leaf':[1,2],
'max_depth':[20,40],
'max_features': ['sqrt', 20,30]
}

gb_GS=GridSearchCV(GradientBoostingClassifier(),
param_grid=gb_paramgrid,scoring=auc_scorer,verbose=1,n_jobs=7)
gb_GS.fit(X_train,y_train)

# best parameters for gradient boosted model

gb_GS.best_params_


gb_GS.best_estimator_
gbcs=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=20,
                           max_features=20, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=2, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='auto',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
gbcs.fit(X_train, y_train)
gbcs.score(X_test, y_test) #0.9891213389121339
predicted=gbcs.predict(X_test)
{'learning_rate': 0.1,
 'max_depth': 20,
 'max_features': 20,
 'min_samples_leaf': 2,
 'min_samples_split': 2,
 'n_estimators': 100}

pred_probs = gbcs.predict_proba(X_test)[:,1]

threshold = 0.5
predicted = pred_probs >= threshold

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)

print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)

mat = confusion_matrix(y_test, predicted)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')

eclf = VotingClassifier(estimators=[
        ('rf', rf), ('gb', gbcs)],
        voting='soft', weights=[1,2])

eclf.fit(X_train, y_train)

#nlp categories for later

name_cats = ['name', 'description', 'org_desc']

# I = importances(rf, X, y)
# plot_importances(I)


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks.callbacks import EarlyStopping

ES=EarlyStopping(monitor='val_accuracy', patience=12)


model=Sequential()
model.add(Dense(512,input_shape=(45,), activation='sigmoid'))
model.add(Dropout(.2))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dropout(.1))
# model.add(Dense(256, activation='sigmoid'))
# model.add(Dropout(.25))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',
 metrics=['accuracy'])


model.fit(X_res,y_res, epochs=200, batch_size=500,
 verbose=1,callbacks=[ES], validation_data= (X_test, y_test))

predict_proba=model.predict(X_test)
threshold = .5
predicted = predict_proba >= threshold

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)

print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)


# sigmoid all, 512, dropout.25, 64, 256, dropout.25, 64
accuracy: 0.9458856345885635
precision: 0.6557377049180327
recall: 0.9248554913294798

#sigmoid all, 512, drop.2, 64
accuracy: 0.9528591352859135
precision: 0.6997742663656885
recall: 0.8959537572254336

#sigmoid all, 512, 64
accuracy: 0.9523012552301255
precision: 0.7078384798099763
recall: 0.861271676300578


from imblearn.over_sampling import SMOTE 

SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=5,
 m_neighbors='deprecated', out_step='deprecated', kind='deprecated',
  svm_estimator='deprecated', n_jobs=1, ratio=None)
sm=SMOTE()
X_res, y_res = sm.fit_resample(X_train, y_train)

rf.fit(X_res, y_res)
predicted=rf.predict(X_test)

gbcs.fit(X_res,y_res)
predicted=gbcs.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)

print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)


##GRADIENT BOOST + SMOTE
'''
accuracy: 0.9874476987447699
precision: 0.9541284403669725
recall: 0.9122807017543859
'''

