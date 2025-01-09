# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# df = pd.read_csv('Solar panel.csv')
# df = df.iloc[:,[1,2]]
# X = df['Temperature'].values.reshape(-1,1)
# y = df['Efficiency']
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# # trainnig the model
# model = LinearRegression()
# model = model.fit(X_train,y_train)

# # plotting the train data
# plt.scatter(X_train,y_train,color='red')
# plt.plot(X_train, model.predict(X_train),color = 'blue')
# plt.xlabel('Temperature')
# plt.ylabel('Efficiency')
# plt.show()
# # print(df.head())

# # plotting the test data
# plt.scatter(X_test,y_test,color='red')
# plt.plot(X_test,model.predict(X_test),color = 'blue')
# plt.xlabel('temp')
# plt.ylabel( 'efff')
# plt.show()

# # F and T test
# import statsmodels.api as sm
# X = df['Temperature']
# Y = df['Efficiency']

# X = sm.add_constant(X)
# model = sm.OLS(y,X).fit()

# f_stat = model.fvalue
# f_p_value = model.f_pvalue

# t_stat = model.tvalues['Temperature']
# t_p_value = model.pvalues['Temperature']

# print(f'f-statistic : {f_stat:.2f}')
# print(f't-statistic for temoerature : {t_stat:.2f}')

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv('Solar panel.csv')
df = df.iloc[:,[1,2]]

X = df['Temperature'].values.reshape(-1,1)
y = df['Efficiency']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model = LinearRegression()
model = model.fit(X_train,y_train)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,model.predict(X_train))
plt.xlabel('Temperature')
plt.ylabel('Efficiency')
plt.show()


# plotting the tested data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,model.predict(X_test),color='blue')
plt.xlabel('Temperature')
plt.ylabel('Efficiency')
plt.title('Test Data')
plt.show()

#F and T test
X = df['Temperature']
y = df['Efficiency']

X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
 
f_stat = model.fvalue
f_p_value = model.f_pvalue

t_stat = model.tvalues['Temperature']
t_p_value = model.pvalues['Temperature']

print(f'f-statistic: {f_stat}')
print(f't-statistic for temperature: {t_stat}')