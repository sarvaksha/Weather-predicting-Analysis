import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd  # Import Pandas for reading CSV data

# Load weather data from CSV
weather_data = pd.read_csv('weather_data.csv')

weather_images = {
    'sunny': 'sunny.jpg',
    'rainy': 'rainy.jpg',
    'cloudy': 'cloudy.jpg'
}

X = weather_data[['Temperature', 'Humidity', 'Pressure', 'Wind_Speed']].values.astype(float)
y = weather_data['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
 
predictions = knn.predict(X_test)

accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)

new_data = np.array([[22, 70, 1012, 7]])
predicted_weather = knn.predict(new_data)
print("Predicted weather:")
print(tabulate([np.concatenate((new_data[0], [predicted_weather[0]]))],
               headers=['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Label'], tablefmt='pretty'))

if predicted_weather[0] in weather_images:
    image_path = weather_images[predicted_weather[0]]
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title("Predicted Weather: " + predicted_weather[0])
    plt.axis('off')
    plt.show()
else:
    print("Image not available for the predicted weather condition.")
