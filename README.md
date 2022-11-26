## EX.NO.01
## DATE: 03.09.2022
# <p align="center">Developing a Neural Network Regression Model</p>


## AIM

To develop a neural network regression model for the given dataset.

## THEORY

First we can take the dataset based on one input value and some mathematical calculaus output value.next define the neaural network model in three layers.first layer have four neaurons and second layer have three neaurons,third layer have two neaurons.the neural network model take inuput and produce actual output using regression.

## Neural Network Model

![nn img](https://user-images.githubusercontent.com/75235704/187594061-969542cb-5adc-4eca-bafb-25620b297911.png)




## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

worksheet = gc.open('MyFirstDataset').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])

df.head(n=9)

df.dtypes

df = df.astype({'X':'float'})
df = df.astype({'Y':'float'})

df.dtypes

x=df[['X']].values

x

y=df[['Y']].values

y

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.33,random_state=50)


X_train

#to scale the input from 0 to 1
scaler=MinMaxScaler()

scaler.fit(X_train)

X_train_scaled=scaler.transform(X_train)

X_train_scaled

ai_brain=Sequential([
    Dense(2,activation='relu'),
    Dense(1)
])

ai_brain.compile(optimizer='rmsprop',loss='mse')

ai_brain.fit(x=X_train_scaled,y=Y_train,epochs=20000)

loss_df=pd.DataFrame(ai_brain.history.history)


loss_df.plot()

X_test

X_test_scaled=scaler.transform(X_test)

X_test_scaled

ai_brain.evaluate(X_test_scaled,Y_test)

input=[[10]]

input_scaled=scaler.transform(input)

input_scaled.shape

ai_brain.predict(input_scaled)
```


## Dataset Information
![dl11](https://user-images.githubusercontent.com/75235704/187593563-e631350d-a4ec-45cd-bc61-0190e0824092.PNG)


## OUTPUT

### Training Loss Vs Iteration Plot
![dl2](https://user-images.githubusercontent.com/75235704/187593761-ae76211d-0268-4b51-812c-bb4749378740.PNG)


### Test Data Root Mean Squared Error

![dl 33](https://user-images.githubusercontent.com/75235704/187593808-3752e9de-8f03-46a4-930e-07ef3858f597.PNG)


### New Sample Data Prediction

![dl4](https://user-images.githubusercontent.com/75235704/187593833-bc78b446-8cee-4c88-8c7e-9220ffbeafac.PNG)


## RESULT
Thus a Neural network for Regression model is Implemented.
