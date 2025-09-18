import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import errno

def test_dir(value):
    if not os.path.exists(value):
        try:
            os.makedirs(value, 0o755)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


ffile1 = 'configurations/struc_theta_B_010041.txt'
size1 = 51
ffile2 = 'configurations/struc_theta_B_030031.txt'
size2 = 29
ffile3 = 'configurations/struc_theta_B_080116.txt'
size3 = 80
ffile4 = 'configurations/struc_theta_Q_019036.txt'
size4 = 80

test_dir('realizations')

for i in range(1000):
    j = i

    df1=pd.read_csv(ffile1, sep='\s+', header=None)
    df1_totrain = df1.iloc[:, -size1:].to_numpy()

    df2=pd.read_csv(ffile2, sep='\s+', header=None)
    df2_totrain = df2.iloc[:, -size2:].to_numpy()

    df3=pd.read_csv(ffile3, sep='\s+', header=None)
    df3_totrain = df3.iloc[:, -size3:].to_numpy()

    df4=pd.read_csv(ffile4, sep='\s+', header=None)
    df4_totrain = df4.iloc[:, -size4:].to_numpy()


    scaler1 = MinMaxScaler(feature_range=(-1,1))
    scaler2 = MinMaxScaler(feature_range=(-1,1))
    scaler3 = MinMaxScaler(feature_range=(-1,1))
    scaler4 = MinMaxScaler(feature_range=(-1,1))

    norm1 = scaler1.fit_transform(df1_totrain)
    norm2 = scaler2.fit_transform(df2_totrain)
    norm3 = scaler3.fit_transform(df3_totrain)
    norm4 = scaler4.fit_transform(df4_totrain)


    data_ = pd.read_csv('oasis3_healthy_subjects_clean.csv')
    gnn = data_['Gnn']
    X_gender = to_categorical(gnn, num_classes=2)

    print('sex', X_gender)
    Y = data_['AGE']
    mr = data_['MR_ID']



    maes=[]
    # Bin ages for stratification
    n_bins = 10
    age_bins = pd.qcut(Y, q=n_bins, labels=False, duplicates='drop')
    # Preallocate array for predictions
    y_pred = np.empty_like(Y, dtype=float)
    y_true = np.empty_like(Y, dtype=float)


    # Create StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True)


    
    for fold, (train_indices, val_indices) in enumerate(skf.split(Y, age_bins)):    

        # 1. Numerical Feature
        num_input_1 = Input(shape=(size1,), name="B_010041")
        num_dense_1 = Dense(56, activation="relu")(num_input_1)
    

        # 2. Numerical Feature
        num_input_2 = Input(shape=(size2,), name="B_030031")
        num_dense_2 = Dense(16, activation="relu")(num_input_2)
        

        # 3. Numerical Feature
        num_input_3 = Input(shape=(size3,), name="B_080116")
        num_dense_3 = Dense(64, activation="relu")(num_input_3)
        

        # 4. Numerical Feature
        num_input_4 = Input(shape=(size4,), name="Q_019036")
        num_dense_4 = Dense(32, activation="relu")(num_input_4) 
    

        # 5. Categorical Feature
        cat_input = Input(shape=(2,), name="gender")
        cat_dense = Dense(6, activation="relu")(cat_input)



        merged = concatenate([num_dense_1, num_dense_2, num_dense_3, num_dense_4, cat_dense]) #])

        # Fully connected layers for final output
        dense1 = Dense(192, activation="relu", kernel_regularizer=regularizers.l2(0.001))(merged)

        dense1 = BatchNormalization()(dense1)

        dense1 = Dropout(0.3)(dense1)  # More dropout for regularization
        dense2 = Dense(96, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(dense1)
        dense2 = BatchNormalization()(dense2)

        dense2 = Dropout(0.3)(dense2)
        output = Dense(1, activation="relu", name="output")(dense2)


        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9, staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.010111)

        model = Model(inputs=[num_input_1, num_input_2, num_input_3, num_input_4, cat_input], outputs=output)
        model.compile(optimizer=optimizer, loss="huber", metrics=["mae"])
        
        

        TRAIN_INPUT = [norm1[train_indices], norm2[train_indices], norm3[train_indices], 
                       norm4[train_indices], X_gender[train_indices]]

        VALID_INPUT = [norm1[val_indices], norm2[val_indices], norm3[val_indices], 
                       norm4[val_indices], X_gender[val_indices]]

        YTRAIN = Y[train_indices]
        YVALID = Y[val_indices]
        
        history = model.fit(
        TRAIN_INPUT,  # Multi-input data
        YTRAIN,  # Target labels
        epochs=200,  # Number of training epochs
        batch_size=32,  # Batch size
        verbose=0,
        )


        temp = model.predict(VALID_INPUT, verbose=0).ravel()
        y_true[val_indices] = YVALID

        y_pred[val_indices] = temp 


    MAE = mean_absolute_error(y_true,y_pred)
    print(j, MAE)
    

    fff = open('realizations/run%d.txt' % j, 'w')
    for mra,t,p in zip(mr.astype(str), y_true.astype(float), y_pred.astype(float)):
        fff.write('%s %.1f %.1f\n'%(mra,t,p))
    fff.close()


print('it has finished, now do the mean value and get a prediction and error with estimate_metric.py')


