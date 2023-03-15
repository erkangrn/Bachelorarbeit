import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from keras.losses import CategoricalCrossentropy 
import sys

#df = pd.read_csv('c:/Users/z0044ber/Desktop/Ostfalia/Bachelorarbeit/Bachelorarbeit/Code/Working/SAR_training.csv')
modelStatus = Sequential()
modelStatement = Sequential()

def column_one_hot (dataframe, columns): 
    for column in columns:
        if column in dataframe:
            one_hot = pd.get_dummies(dataframe[column])
            dataframe = dataframe.drop(column,axis = 1)
            dataframe = pd.concat([dataframe, one_hot], axis=1)
    return dataframe

def drop_columns (dataframe, columns):
    for column in columns:
        if column in dataframe.columns:
            dataframe = dataframe.drop(column, axis=1)
    return dataframe

def drop_column (dataframe, column):
    if column in dataframe.columns:
        dataframe = dataframe.drop(column, axis=1)
    return dataframe

def fixMistakes(df):
    maskOpen = ((df['Status'] == 'postponed') | (df['Status'] == 'partly open') |( df['Status'] == 'In creation'))
    maskClosed = (df['Status'] == 'partly closed')
    df.loc[maskOpen, 'Status'] = 'open'
    df.loc[maskClosed, 'Status'] = 'closed'
    df.loc[(df['Status'] == 'non applicable'), 'Status'] = 'not applicable'

    df.loc[df['Version'].str.contains('VICOS_S_D'), 'Product'] = 'VICOS_S_D'
    df.loc[df['Version'].str.contains('VICOS_S_D'), 'Version'] = df['Version'].str[-5:]
    return df

def gatherData(df):
    #global df
    paths = df['Path'].unique()
    accessDB = pd.read_xml("X:/File/DE/bwga024a_IMORA_RM/05_Process_Management/14_Metriken & KPI/KPI-Erhebung/KPI_01-04_General/Data/Input/Input_BWG_Combined_Access.xml")

    for path in paths:
        try:
            if(path == "/ML Realization Projects Algeria"):
                result = accessDB.loc[(accessDB['Type'] == "Real") & (accessDB['Location'] == "BWG") & (accessDB['Offset'] == "/ML Realization Projects Algeria/20006_ML_BM_Boughezoul_MSila")].iloc[0]
            else:
                result = accessDB.loc[(accessDB['Type'] == "Real") & (accessDB['Location'] == "BWG") & ((accessDB['Offset'] == str(path)) | (accessDB['Offset'] == (str(path) + "/")))].iloc[0]           
        except:
            print(str(path) + " has no entry in the AccessDB!")

        mask = df['Path'] == str(path)
        df.loc[mask, 'Project_category'] = result['Project_category']
        df.loc[mask, 'BS'] = result['BS']
        df.loc[mask, 'RU'] = result['RU']
        df.loc[mask, 'ProjectYear'] = result['ProjectYear']
        df.loc[mask, 'section'] = result['section']
        df.loc[mask, 'Project_name'] = result['Project_name']
        df['ProductVersion'] = df["Product"].str.cat(df["Version"], sep = "-")

    df['ProjectYear'] = df['ProjectYear'].astype('int')
    df = df[['Text', 'Product', 'ProductVersion', 'Project_name', 'section', 'Project_category', 'BS', 'RU', 'ProjectYear', 'Status', 'Statement']]
    return df

def oneHot(df):
    products = df['Product'].unique()
    df_product = column_one_hot(df[['Product']], ['Product'])
    projects = df['Project_name'].unique()
    for project in projects:
        for product in products:
            df_product.loc[df['Project_name'] == project, product] = 1 if (df_product.loc[df['Project_name'] == project][product].sum()) >= 1 else 0
    if (not (products[0] in df)):        
        df = df.join(df_product)
    df = drop_column(df, 'Product')
    df = column_one_hot(df, ['ProductVersion', 'Project_name', 'section',
        'Project_category', 'BS', 'RU', 'ProjectYear'])
    return df

def adjustShape(df_training, df_testing):
    length = df_testing.shape[0]
    df_temp = pd.concat([df_training, df_testing], ignore_index=True, sort=False)
    df_temp = df_temp.fillna(0)
    df_testing = df_temp.tail(length)
    return df_testing

def createModel(df_training, text):
    df_training = df_training.loc[df_training['Text'] == str(text)]
    df_training.reset_index(inplace=True, drop=True)
    
    trainX = drop_columns(df_training, ['Status', 'Text', 'Statement'])
    trainYStatus = drop_column(column_one_hot(df_training[['Text', 'Status']], ['Status']), "Text")
    trainYStatement = drop_column(column_one_hot(df_training[['Text', 'Statement']], ['Statement']), "Text")

    modelStatus.add(Input(shape=trainX.shape[1]))
    modelStatus.add(Dense(16, activation='relu'))
    modelStatus.add(Dense(trainYStatus.shape[1], activation='softmax'))
    modelStatus.compile(optimizer='adam',
                loss=CategoricalCrossentropy(),
                metrics=['accuracy'])

    modelStatus.fit(trainX, trainYStatus,
                        batch_size=2,
                        epochs=50,
                        verbose=2,
                        validation_split=0.4)

    modelStatement.add(Input(shape=trainX.shape[1]))
    modelStatement.add(Dense(16, activation='relu'))
    modelStatement.add(Dense(trainYStatement.shape[1], activation='softmax'))
    modelStatement.compile(optimizer='adam',
                loss=CategoricalCrossentropy(),
                metrics=['accuracy'])

    modelStatement.fit(trainX, trainYStatement,
                        batch_size=2,
                        epochs=50,
                        verbose=2,
                        validation_split=0.4)
    
    return modelStatus, modelStatement, trainYStatus, trainYStatement

def predictStatus(modelStatus, row, trainYStatus):
    predictionStatus = modelStatus.predict(row)
    col = 0
    for i in predictionStatus:
        for j in i:
            print (trainYStatus.columns[col] + " " + '{:.1%}'.format(j))
            col += 1

    print ("-----------------------------------------------------")

def predictStatement(modelStatement, row, trainYStatement):
    predictionStatement = modelStatement.predict(row)
    index_max = np.argmax(predictionStatement)
    print (trainYStatement.columns[index_max] + " " + '{:.1%}'.format(predictionStatement[0][index_max]))

def main():
    print("Starting\n")
    df_training = pd.read_csv('c:/Users/z0044ber/Desktop/Ostfalia/Bachelorarbeit/Bachelorarbeit/Code/Working/SAR_training.csv')
    df_result = pd.read_csv('c:/Users/z0044ber/Desktop/Ostfalia/Bachelorarbeit/Bachelorarbeit/Code/Working/SAR_testing.csv')
    df_testing = df_result

    df_training = fixMistakes(df_training)
    df_testing  = fixMistakes(df_testing)

    df_training = gatherData(df_training)
    df_testing  = gatherData(df_testing)

    df_training = oneHot(df_training)
    df_testing  = oneHot(df_testing)

    df_testing = adjustShape(df_training, df_testing)

    print("Training: " + str(df_training.shape))
    print("Testing: " + str(df_testing.shape))

    for i in range(df_testing.shape[0]):
        current = df_testing.iloc[[i]]
        modelStatus, modelStatement, yStatus, yStatement = createModel(df_training, current.iloc[0]['Text'])
        predictStatus(modelStatus, current.drop(columns=['Text', 'Status', 'Statement']), yStatus)
        predictStatement(modelStatement, current.drop(columns=['Text', 'Status', 'Statement']), yStatement)
        sys.exit()


    print("Done\n")

if __name__ == "__main__":
    main()