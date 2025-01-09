import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

df = pd.read_csv('income_clustering.csv')
plt.scatter(df.Age, df['Income($)'])
plt.xlabel('age')
plt.ylabel('income')
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted

df['cluster'] = y_predicted
df1 = df[df.cluster==0] 
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black',label='cluster2')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],color='purple',marker='*')

plt.legend()
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.show()

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()