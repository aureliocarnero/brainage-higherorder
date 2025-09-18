import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout,BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers.legacy import SGD  #Legacy version
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping
import errno

def test_dir(value):
    if not os.path.exists(value):
        try:
            os.makedirs(value, 0o755)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise



# Optional: Early stopping
early_stopping = EarlyStopping(
#    monitor='mae',
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)


data_ = pd.read_csv('camcan_healthy_subjects_clean.csv')
gnn = data_['Gnn']
X_gender = gnn 


label_encoder = LabelEncoder()
categorical_encoded = label_encoder.fit_transform(X_gender)
num_categories = len(label_encoder.classes_)
print('num categories',num_categories)


Y = data_['AGE']

mr_id = data_['MR_ID']


ffile1 = 'configurations/struc_theta_B_066107.txt'
size1 = 80
ffile2 = 'configurations/struc_theta_B_030031.txt'
size2 = 29
ffile3 = 'configurations/struc_theta_B_080116.txt'
size3 = 80
ffile4 = 'configurations/fluc_struc_theta_pk_fluc.txt'
size4 = 149

test_dir('realizations')

for kk in range(1000):

    df1=pd.read_csv(ffile1, sep='\s+', header=None)
    df2=pd.read_csv(ffile2, sep='\s+', header=None)
    df3=pd.read_csv(ffile3, sep='\s+', header=None)
    df4=pd.read_csv(ffile4, sep='\s+', header=None)

    array1 = df1.iloc[:, -size1:].to_numpy()
    array2 = df2.iloc[:, -size2:].to_numpy()
    array3 = df3.iloc[:, -size3:].to_numpy()
    array4 = df4.iloc[:, -size4:].to_numpy()



    scalers = {}
    array1_scaled = StandardScaler().fit_transform(array1)
    array2_scaled = StandardScaler().fit_transform(array2)
    array3_scaled = StandardScaler().fit_transform(array3)
    array4_scaled = StandardScaler().fit_transform(array4)



    # 5. Build the neural network architecture
    def create_age_regression_model(array1_shape, array2_shape, array3_shape, array4_shape, num_categories):
        # Input layers for each array
        input1 = Input(shape=(array1_shape,), name='B_066107')
        input2 = Input(shape=(array2_shape,), name='B_030031')
        input3 = Input(shape=(array3_shape,), name='B_080116')
        input4 = Input(shape=(array4_shape,), name='pk_fluc')
        
        # Input layer for categorical data
        categorical_input = Input(shape=(1,), name='categorical_input')


        reg = l1_l2(l1=0.001, l2=0.001)

        # Process each numerical array separately
        # Array 1 branch
        x1 = Dense(128, activation='leaky_relu', kernel_regularizer=reg)(input1)
        x1 = Dropout(0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x1)
        
        # Array 2 branch
        x2 = Dense(128, activation='leaky_relu', kernel_regularizer=reg)(input2)
        x2 = Dropout(0.2)(x2)
        x2 = BatchNormalization()(x2)
        x2 = Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x2)
        
        # Array 3 branch
        x3 = Dense(128, activation='leaky_relu', kernel_regularizer=reg)(input3)
        x3 = Dropout(0.2)(x3)
        x3 = BatchNormalization()(x3)
        x3 = Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x3)
        
        # Array 4 branch
        x4 = Dense(128, activation='leaky_relu', kernel_regularizer=reg)(input4) 
        x4 = BatchNormalization()(x4)
        x4 = Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x4) 

        
        x_cat = Dense(4)(categorical_input)


        merged = Concatenate()([x1, x2, x3, x4, x_cat])
        
        # Additional layers after merging
        x = Dense(128, activation='leaky_relu', kernel_regularizer=reg)(merged)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
        
        # Output layer (single neuron for regression)
        output = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(
            inputs=[input1, input2, input3, input4, categorical_input],
            outputs=output
        )

        initial_lr = 0.005
        decay_steps = 200 * (625 // 32)  
        alpha = 0.0001 / initial_lr  

        lr_schedule = CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=alpha  # controls final LR
        )

        # SGD with momentum + cosine LR schedule
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)


        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean squared error for regression
            metrics=['mae'],  # Mean absolute error
            run_eagerly=False
        )
        model.optimizer.clipnorm = 1.0

        
        return model



    y_pred = np.zeros(len(Y))
    y_true = np.zeros(len(Y))

    mr_true = np.empty_like(mr_id)
    # Bin ages for stratification
    n_bins = 10



    age_bins = pd.qcut(Y, q=n_bins, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=10, shuffle=True)#, random_state=42)
    for fold, (train_indices, val_indices) in enumerate(skf.split(Y, age_bins)):    

        X1_train = array1_scaled[train_indices]
        X1_test = array1_scaled[val_indices]
        X2_train = array2_scaled[train_indices]
        X2_test = array2_scaled[val_indices]
        X3_train = array3_scaled[train_indices]
        X3_test = array3_scaled[val_indices]
        X4_train = array4_scaled[train_indices]
        X4_test = array4_scaled[val_indices]
        cat_train = categorical_encoded[train_indices]
        cat_test = categorical_encoded[val_indices]
        y_train = Y[train_indices]
        y_test = Y[val_indices]

        mr_true[val_indices] = mr_id[val_indices]

        model = create_age_regression_model(
            array1_shape=array1.shape[1],
            array2_shape=array2.shape[1],
            array3_shape=array3.shape[1],
            array4_shape=array4.shape[1],
            num_categories=num_categories
        )

        # 7. Train the model
        history = model.fit(
            x=[X1_train, X2_train, X3_train, X4_train, cat_train],
            y=y_train,
            validation_split=0.1,
            epochs=200,
            batch_size=32,
            verbose=0,
            callbacks=[early_stopping]
        )

        # 8. Evaluate the model
        prediction = model.predict(
            [X1_test, X2_test, X3_test, X4_test, cat_test], 
            verbose=0
        )
        y_pred[val_indices] = prediction.ravel()
        y_true[val_indices] = y_test.ravel()


    fff = open('realizations/run%d.txt' % kk,'w')
    for mr,t,p in zip(mr_true.astype(str), y_true.astype(float), y_pred.astype(float)):
        fff.write('%s %.1f %.1f\n'%(mr,t,p))
    fff.close()


    mae = mean_absolute_error(y_true, y_pred)
    print('MAE KFold run',kk, mae)

print('it has finished, now do the mean value and get a prediction and error with estimate_metric.py')

