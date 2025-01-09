import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

df = pd.read_csv('student_data.csv')

X = df['StudyHours'].values.reshape(-1,1)
y = df['ExamScore']
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model = model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
plt.figure(figsize=(10,6))
plt.scatter(X_train,y_train,color='blue', label='Test Data' )
plt.scatter(X_test, y_test,label='Test Data',color='red')
plt.plot(X_train,model.predict(X_train),color='green',label='Regression Line')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.show()
print(mse ,',', r2)