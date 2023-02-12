# TRINIT_594092-UX631VF7_ML
TRINIT_ML03

## Team Number - 594092-UX631VF7
## Team Members - Akhil Chenna, Aditya Modi, Shubh Sharma

Main Objective - Given the dataset, predict what crop would be best for the farmer to grow based on the GEOLOCATION, SEASON, and PRICE.

Google Drive Link to VIDEO EXPLANATION - https://drive.google.com/file/d/1tHkOQSd3P-TIHfLueVGyJGJZmaMVMk24/view?usp=sharing
Google Drive Link to PRESENTATION - https://docs.google.com/presentation/d/19OneX9NSUs14RbZ8MxDkDoStTZC8UFQd/edit?usp=sharing&ouid=101257776169293568033&rtpof=true&sd=true

Given references and datasets are - 
1) Rainfall in India
2) Historical weather data for Indian cities
3) Crop recommendation dataset
4) All agriculture datasets for India
5) Major Soil type of India Map

Data preprocessing becomes one of the most important steps since there is no proper correlation between the datasets.
Code available in Data_Processing_district_wise_weather_data.ipynb file.

Data preprocessing steps done -
1) Weather Data Indian Cities - Combined datasets of the given Indian cities into a single dataset that includes average value of tavg, tmin, tmax for each month from year 2018 to 2022 (5 years).
2) Transposed Rainfall in India dataset to put months row wise instead of column wise, similar to the previous dataset. Done in order to combimne rainfall parameter of the cities.
3) Created a new dataset on own that includes soil type, Ph level, Nitrogen, Phosphorus, Potassium ratios of soil of each of the cities (referred given map as well). Done in order to correlate with crops given in crop recommendation dataset.
4) Finally, we combined all the datasets into a single one upon which our models were trained. The dataset includes Month, tavg, tmin, tmax, prcp, city, State, Rainfall, Soil Type, Ph level, Nitrogen, Phosphorous, Potassium.

We created two models, one based on Machine Learning - K Nearest Neighbors, and another based on Deep Learning - Artificial Neural Networks.
Code in model_training.ipynb file.

The deep learning model gave us a main crop that should be grown based on user input, and K - NN gave us some other alternative crops that can be grown along with the main crop.

ANN model architecture - 
1) Model made using Tensorflow.
2) First hidden layer neuron units - 32, activation = relu.
3) Second hidden layer neuron units - 128, activation = relu.
4) Output layer neuron units - 22, each corresponding to the 22 crops, activation = softmax.
5) Optimizer = Adam, Loss = Categorical Cross Entropy, Metrics = Accuracy.
6) Batch size = 32, Epochs = 100.

We get an accuracy of around 98 percent during training.

KNN model architecture - 
1) Model made using sklearn.
2) No. of neighbors = 100.
3) Weights = uniform, algorithm = auto, metrics = minkowski.

Output predicted by ANN and KNN is the same i.e. the main crop.

We have created a front end user friendly interface for the user to interact with our models.
We have done it using streamlit framework of Python. A framework used to build and share data apps.
Code in app.py file.

The user inputs Location, Start month, End month (which constitute season together). The model then outputs the main crop, other alternative crops, price of each of the crops, Nitrogen Phosphorous Potassium rations of soil required.

Therefore, this output covers the main objective - predict what crop would be best for the farmer to grow based on the GEOLOCATION, SEASON, and PRICE, and making a user friendly interface.
It also covers brownie points - Suggest a variety of crops that could be cultivated for the geolocations provided
in the dataset and rank them according to the sales price, and improve model by factoring in the various types of soil found in various
parts of India and its general nutritional composition.

File back.py contains all the functions required for the user interface to work such as graphs, crops output, etc.

This concludes our solution to the given problem statement.
