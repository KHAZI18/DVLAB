import matplotlib.pyplot as plt
import pandas as pd

data = {
    'plant_name': ['sunflower','rose','jasmin','fern','cactus', 'Daisy', 'Sunflower', 'Lily', 'Orchid', 'Maple'],
    'sunlight_exposure' : [10,5,12,3,5,11,9,8,4,7],
    'plant_height': [30, 150, 200, 60, 50, 40, 180, 70, 40,20]
}
df = pd.DataFrame(data)
df = df[['sunlight_exposure','plant_height']]
print(df.head())

plt.scatter(df['sunlight_exposure'],df['plant_height'],color='red',marker='*')
plt.title('rel bw sunlight exposure and height')
plt.xlabel('sunlight exp')
plt.ylabel('plant exposure')
plt.show()


#corrilation and coefficient
correlation = df['sunlight_exposure'].corr(df['plant_height'])
print(f'correl bw sunlight exposure and plant height: {correlation}')

threshold = 0.7
if abs(correlation)>=threshold:
    print('there is a significant associate bw sunligtht and plant growth')
else:
    print('there is no significant associate bw sunligtht and plant growth')