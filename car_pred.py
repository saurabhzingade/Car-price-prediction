import pandas as pd

df= pd.read_csv('car.csv')

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['make'] = encoder.fit_transform(df['make'])
df['fueltype'] =encoder.fit_transform(df['fueltype'])
df['aspiration'] =encoder.fit_transform(df['aspiration'])
df['doornumber'] =encoder.fit_transform(df['doornumber'])
df['carbody'] =encoder.fit_transform(df['carbody'])
df['drivewheel'] =encoder.fit_transform(df['drivewheel'])
df['enginelocation'] =encoder.fit_transform(df['enginelocation'])
df['enginetype'] =encoder.fit_transform(df['enginetype'])
df['cylindernumber'] =encoder.fit_transform(df['cylindernumber'])
df['fuelsystem'] =encoder.fit_transform(df['fuelsystem'])

X=df.drop('price',axis=1)
y=df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)

pred=regressor.predict(X_test)
pred_df=pd.DataFrame(pred)

test_df = pd.DataFrame(y_test)
a=[]
for i in range(0,41):
	a.append(i)

test_df=test_df.set_index([a])

main=pd.concat([pred_df,test_df])

















