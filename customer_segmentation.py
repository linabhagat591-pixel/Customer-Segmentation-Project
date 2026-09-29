import pandas as pd
from sklearn.cluster import KMeans

df=pd.read_csv('customer_data.csv')
model=KMeans(n_clusters=5,n_init=10)
df['Cluster']=model.fit_predict(df[['Income','SpendingScore']])