import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

def preprocess(df):
    print(df.describe())
    print('----------------------------------------------')
    print("Before preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col]==0].shape[0]
        print(col + ": " + str(missing_rows))
    print('----------------------------------------------')

    # Replace 0 values with the mean of the existing values
    df['Age'] = df['age'].replace(0, np.nan)
    df['Gender'] = df['Gender'].replace(0, np.nan)
    df['Smoking'] = df['Smoking'].replace(0, np.nan)
    df['chronic Lung Disease'] = df['chronic Lung Disease'].replace(0, np.nan)
    df['Dry Cough'] = df['Dry Cough'].replace(0, np.nan)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Gender'] = df['Gender'].fillna(df['Gender'].mean())
    df['Smoking'] = df['Smoking'].fillna(df['Smoking'].mean())
    df['chronic Lung Disease'] = df['chronic Lung Disease'].fillna(df['chronic Lung Disease'].mean())
    df['Dry Cough'] = df['Dry Cough'].fillna(df['Dry Cough'].mean())

    print('----------------------------------------------')
    print("After preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col]==0].shape[0]
        print(col + ": " + str(missing_rows))
    print('----------------------------------------------')

    # Standardization
    df_scaled = preprocessing.scale(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    df_scaled['Outcome'] = df['Outcome']
    df = df_scaled
    print(df.describe())

    return df