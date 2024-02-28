from flask import Flask
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
df=pd.read_csv("daily_weather (1).csv")
print(df.head(5))
data=df.fillna(df.mean())
y=data['high_humidity_3pm'].copy()
x=data[['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am','max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am','rain_duration_9am','relative_humidity_9am',]].copy()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=324)
clf=LinearRegression()
clf.fit(x_train,y_train)


from flask import Flask,render_template,request
app=Flask(__name__)
@app.route('/')
def nvv():
    return render_template('index.html')
@app.route('/prediction',methods=['GET','POST'])
def jyu():
    air_pressure_9am=request.form['air_pressure_9am']
    air_temp_9am=request.form['air_temp_9am']
    avg_wind_direction_9am=request.form['avg_wind_direction_9am']
    avg_wind_speed_9am=request.form['avg_wind_speed_9am']
    max_wind_direction_9am=request.form['max_wind_direction_9am']
    max_wind_speed_9am=request.form['max_wind_speed_9am']
    rain_accumulation_9am=request.form['rain_accumulation_9am']
    rain_duration_9am=request.form['rain_duration_9am']
    relative_humidity_9am=request.form['relative_humidity_9am']
    arr=np.array([air_pressure_9am,air_temp_9am,avg_wind_direction_9am,avg_wind_speed_9am,max_wind_direction_9am,max_wind_speed_9am,rain_accumulation_9am,rain_duration_9am,relative_humidity_9am])
    arr=arr.astype(np.float64)
    pred=clf.predict([arr])
    return render_template('prediction.html',pred=pred)
if __name__=='__main__':
    app.run(debug=True)