from django.shortcuts import render
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import time

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {'accuracy': None, 'plot_data': None})

    k = request.POST.get('k')
    data_points = request.POST.get('data_points', '').strip()

    if not k or not data_points:
        error_message = 'Please provide a valid value for k and data points.'
        return render(request, 'index.html', {'error': error_message})

    try:
        k = int(k)
        if k <= 0:
            error_message = 'Please provide a valid positive value for k.'
            return render(request, 'index.html', {'error': error_message})
    except ValueError:
        error_message = 'Please provide a valid integer value for k.'
        return render(request, 'index.html', {'error': error_message})

    delimiters = [',', ';', '\n', '\r\n']
    for delimiter in delimiters:
        if delimiter in data_points:
            data_points = data_points.replace(delimiter, ',')
    data_points = data_points.split(',')

    if not data_points[0]:
        error_message = 'Please provide valid data points.'
        return render(request, 'index.html', {'error': error_message})

    try:
        data_points = [float(point.strip()) for point in data_points]
        X = np.array(data_points).reshape(-1, 1)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)

        labels = kmeans.predict(X)
        accuracy = accuracy_score(X, labels) if len(X) > 0 else 0.0

        plt.clf()
        plt.scatter(X, np.zeros_like(X), c=labels, cmap='viridis')
        centroids = kmeans.cluster_centers_.flatten()
        plt.scatter(centroids, np.zeros_like(centroids), marker='x', color='red', label='Centroids')
        plt.xlabel('Data Points')
        plt.title('KMeans Clustering')
        plt.legend()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        timestamp = int(time.time())

        return render(request, 'index.html', {'accuracy': accuracy, 'plot_data': plot_data, 'timestamp': timestamp})

    except ValueError:
        error_message = 'Invalid data points format. Please enter numbers separated by commas.'
        return render(request, 'index.html', {'error': error_message})
    
