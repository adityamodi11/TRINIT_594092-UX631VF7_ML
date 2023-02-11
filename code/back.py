import pandas as pd
from keras.models import load_model
import joblib 
import tensorflow as tf
import numpy as np
import pickle



ann_model = load_model(r"C:\Users\Owner\Tri Nit\model files\ann_model.h5")
filename = r"C:\Users\Owner\Tri Nit\model files\clf_model.sav"
clf_model = pickle.load(open(filename, 'rb'))
scaler = joblib.load(r'C:\Users\Owner\Tri Nit\model files\scaler.save') 
y_train_k = joblib.load(r'C:\Users\Owner\Tri Nit\model files\y_train_k.joblib')
x_train_k = joblib.load(r'C:\Users\Owner\Tri Nit\model files\x_train_k.joblib')



list_of_crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
       'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
       'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

le_name_mapping = {'apple': 0,
 'banana': 1,
 'blackgram': 2,
 'chickpea': 3,
 'coconut': 4,
 'coffee': 5,
 'cotton': 6,
 'grapes': 7,
 'jute': 8,
 'kidneybeans': 9,
 'lentil': 10,
 'maize': 11,
 'mango': 12,
 'mothbeans': 13,
 'mungbean': 14,
 'muskmelon': 15,
 'orange': 16,
 'papaya': 17,
 'pigeonpeas': 18,
 'pomegranate': 19,
 'rice': 20,
 'watermelon': 21}
key_list = list(le_name_mapping.keys())
val_list = list(le_name_mapping.values())


test_data = pd.read_csv(r'C:\Users\Owner\Tri Nit\processed_test_city_weather_soil.csv')
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 100, weights = 'uniform', algorithm = 'auto', metric='minkowski', p =2)
clf.fit(x_train_k, y_train_k)


def crop_map(prediction,lst_crops):
  crop_pred_dic = {}
  for i,j in zip(prediction[0],lst_crops):
    crop_pred_dic[j] = i

  return crop_pred_dic

def pred_recommend(city,month_start,month_end):
    data = test_data.loc[(test_data['city'] == city) & ((test_data['month'] >= month_start)&(test_data['month'] <= month_end))]
    data = data[["N","P","K","tavg","Ph level","rainfall"]]
    data.columns = ['N', 'P', 'K', 'temperature', 'ph', 'rainfall']
    
    data_g = data.groupby('ph')

    # Apply aggregation functions to different columns
    data = data_g.agg({'N': 'mean', 'P': 'mean','K': 'mean','temperature': 'mean', 'rainfall': 'sum'})
    data = data.reset_index()
    test_input_1 = data.loc[0].values.tolist()
    test_input_2 = data.loc[1].values.tolist()
    
    transformed_input_1 = scaler.transform(np.array([test_input_1]))
    transformed_input_2 = scaler.transform(np.array([test_input_2]))
    
    prediction_ann_1 = ann_model.predict(transformed_input_1)
    prediction_ann_2 = ann_model.predict(transformed_input_2)
    

    
    prediction_ann_1 = prediction_ann_1.tolist()
    prediction_ann_2 = prediction_ann_2.tolist()
    
    crop_pred_dic_1 = crop_map(prediction_ann_1,list_of_crops)
    crop_pred_dic_2 = crop_map(prediction_ann_1,list_of_crops)
    
    
    sorted_top_crop_1 = sorted(crop_pred_dic_1.items(), key=lambda x:x[1])
    predicted_crop_1 = sorted_top_crop_1[-1][0]
    
    sorted_top_crop_2 = sorted(crop_pred_dic_2.items(), key=lambda x:x[1])
    predicted_crop_2 = sorted_top_crop_2[-1][0]
    
#     print(prediction_ann_1)
#     print("-----------------------")
#     print(crop_pred_dic_1)
#     print("-----------------------")
#     print(sorted_top_crop_1)

    final_predicted_crops = [predicted_crop_1,predicted_crop_2]
    
    
    
    
    neighbors_clf_1 = clf.kneighbors(transformed_input_1, return_distance=False)
    neighbors_clf_2 = clf.kneighbors(transformed_input_2, return_distance=False)
    
    unique_recommended_values_1 = np.unique(y_train_k[neighbors_clf_1][0])
    unique_recommended_values_2 = np.unique(y_train_k[neighbors_clf_2][0])
    
    

    recommended_crop_lst_1 = []
    recommended_crop_lst_2 = []
 
    for each in unique_recommended_values_1:  
  
      position = val_list.index(each)
      recommended_crop_lst_1.append(key_list[position])
        
    for each in unique_recommended_values_2:  
  
      position = val_list.index(each)
      recommended_crop_lst_2.append(key_list[position])
    
    recommended_crop_lst = recommended_crop_lst_1 + recommended_crop_lst_2
    
    
    
    

    
    return np.unique(final_predicted_crops),np.unique(recommended_crop_lst)



crop_df = pd.read_csv(r"C:\Users\Owner\Tri Nit\crop_recommendation.csv")


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio    


def soil_comp(x):
    reqd_crop_df = crop_df[crop_df.label == x]
    reqd_val_df = reqd_crop_df[['N','P','K','ph']].mean()

    ph = reqd_val_df['ph']

    reqd_val_df = reqd_val_df.iloc[:-1]



    fig = go.Figure(go.Bar(
        y= list(reqd_val_df.index),
        x= list(reqd_val_df.values),
        orientation='h',
        marker=dict(
    color='rgba(164, 163, 204, 0.85)',
    line=dict(
        color='rgba(50, 171, 96, 1.0)',
        width=0.5),
        )))
    fig.update_layout(plot_bgcolor = "white",hovermode="x unified",
    title={
    'text': "Ratio of various elements",
    'y':0.9,
    'x':0.48,
    'xanchor': 'center',
    'yanchor': 'top'},
    xaxis_title="Ratio",
    yaxis_title="Element",
    font=dict(
    family="Courier New, monospace",
    size=20,
    color="white"
      ))

    # fig.show()

    return fig




df_prices = pd.read_excel(r"C:\Users\Owner\Tri Nit\price_data.xlsx")
df_commodity = df_prices.groupby(['commodity'],as_index=False).mean()

def make_price_bar(com):
    price=list(df_commodity[df_commodity['commodity'].isin(com)]['modal_price'])
    # fig=px.bar(x=com,y=price)
    # fig.update_layout(xaxis_title='Crop',yaxis_title='Price',bargap=0.8)
    # fig.show()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x= com,
        y= price,
        opacity = 0.6,
        hovertemplate="%{y}%{_xother}",
        marker_color = 'purple'
    ))

    fig.add_trace(go.Scatter(
        x= com,
        y= price,
        line = dict(shape = 'linear', color = 'rgb(205, 12, 24)', width= 3, dash = 'dot'),
        mode = "lines+markers",
        marker = dict(symbol = "circle", color = 'rgb(205, 12, 24)',size = 6),
        connectgaps = True,
        hovertemplate="%{y}%{_xother}"
    ))

    fig.update_layout(plot_bgcolor = "white",hovermode="x unified",
        title={
        'text': "Price Analysis",
        'y':0.9,
        'x':0.44,
        'xanchor': 'center',
        'yanchor': 'top'},
        xaxis_title="Crop",
        yaxis_title="Price per quintal (in Rs)",
        legend_title="Legend Title",
        font=dict(
        family="Courier New, monospace",
        size=20,
        color="white"
    ))
    return fig





