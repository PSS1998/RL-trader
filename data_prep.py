import pandas as pd 
import numpy as np
  
df = pd.read_csv("data_test.csv")

df = df.drop(['id', 'real_volume'], axis=1)
df['time']= pd.to_datetime(df['time'])
df = df.set_index("time")

#print(df.head())

										 
df = pd.DataFrame(df.resample('5T').agg({'open': 'first', 
                            'high': 'max', 
                            'low': 'min', 
                            'close': 'last',
							'tick_volume': 'sum',
							'spread': 'mean'}))
                            
# df = pd.DataFrame(df.resample('H').agg({'open': 'first', 
                            # 'high': 'max', 
                            # 'low': 'min', 
                            # 'close': 'last',
							# 'tick_volume': 'sum',
							# 'spread': 'mean'}))
							
df.drop(df.head(1).index,inplace=True)
df.drop(df.tail(1).index,inplace=True)
df = df.round({'spread': 0})


df = df.assign(colse_noise=0)
noise = np.random.normal(0,1,len(df))
df['colse_noise'] = df['close'] + noise
# df['colse_noise'] = df['close']

										 
#print(df.head())

df = df.assign(diff_consecutive=df['colse_noise'].shift(-1) - df['colse_noise'])
max_consecutive_diff = df['diff_consecutive'].max()
min_consecutive_diff = df['diff_consecutive'].min()
df = df.drop(['diff_consecutive'], axis=1)

df = df.assign(prediction=0)
df.loc[(df['colse_noise'].shift(-1) - df['colse_noise'])>0, 'prediction'] = 2
df.loc[(df['colse_noise'].shift(-1) - df['colse_noise'])<0, 'prediction'] = 1

df = df.assign(probability=0)
df.loc[df['prediction']==2, 'probability'] = (df['colse_noise'].shift(-1) - df['colse_noise'])/max_consecutive_diff
df.loc[df['prediction']==1, 'probability'] = (df['colse_noise'].shift(-1) - df['colse_noise'])/min_consecutive_diff

#print(df[df['probability'] > 0.05].count())

df.loc[df['probability']<0.02, 'prediction'] = 0
df.loc[df['probability']<0.02, 'probability'] = 0

# print(df.head(30))

df = df.drop(['open', 'high', 'low', 'tick_volume', 'spread'], axis=1)

# print(df.head(30)) 

nan_rows = df[df['close'].isnull()]
df.drop(nan_rows.index,inplace=True)

df.to_csv('out.csv')








