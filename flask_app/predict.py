import pandas as pd
import numpy as np
import datetime
import joblib
# from live import *
import requests

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

def process_data(df):
    cat_features = ['venue_country', 'country', 'payout_type', 'venue_state',
    'email_domain', 'listed', 'currency', 'user_type', 'delivery_method',
    'has_header']

    #These are in Unix(seconds)
    date_features = ['event_created', 'event_end', 'event_published',
    'event_start','user_created',]

    cont_features = ['user_age', 'org_desc_length','body_length',
    'name_length','channels', 'sale_duration',
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
    df['qtotal']=df.ticket_types.apply(lambda x: sum([b['quantity_total'] for b in x]))
    df['avgcost']=df.ticket_types.apply(lambda x: np.mean([b['cost'] for b in x]))
    df['avgcost'] = df['avgcost'].fillna(df['avgcost'].median())

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
    drop_features= ['object_id', 'venue_latitude', 'venue_longitude',
    'description', 'venue_address','name','org_desc', 'ticket_types', 'previous_payouts',
    'org_name','payee_name']

    # Dropping date columns as we already extracted day of week and hour the columns
    for col in date_features:
        drop_features.append(col)

    live_data_drop_cols = ['acct_type',
    'approx_payout_date',
    'gts',
    'num_order',
    'num_payouts',
    'sale_duration2']
    
    for col in live_data_drop_cols:
      drop_features.append(col)

    for col in live_data_drop_cols:
      if col not in df.columns:
          drop_features.remove(col)
    
    try :
      drop_features.append('sequence_number')
    except:
      pass
    df.drop(labels=drop_features,axis=1,inplace=True)

    return df

def predictions(df, estimator, threshold=.5):
    original_df = df.copy()
    if 'sequence_number' not in df.columns:
      df['sequence_number'] = 0
    processed_df = process_data(df)
    preds = estimator.predict_proba(processed_df)[:,1]
    preds = preds >= threshold
    original_df.insert(0, 'predictions', preds)
    # original_df['predictions'] = preds
    return original_df

if __name__ == "__main__":

    client = EventAPIClient()
    raw_data = client.get_data()
    model = joblib.load('gb.pkl')

    raw = pd.DataFrame(raw_data)

    predictions(raw, model)