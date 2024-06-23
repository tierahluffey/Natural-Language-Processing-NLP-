from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# Specifying the file path for the dataset and the output CSV file
input_file_path = 'C:/Users/user/Documents/CODE/final_corrected_homestays.csv'
output_file_path = 'C:/Users/user/Documents/CODE/clustered_homestays.csv'

# Load the dataset
homestay_data = pd.read_csv(input_file_path)

# Extracting the descriptions
descriptions = homestay_data['homestay_description']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

# K-Means Clustering
n_clusters = 4
model = KMeans(n_clusters=n_clusters, random_state=42)
model.fit(tfidf_matrix)

# Assigning the cluster labels to the homestays
cluster_labels = model.labels_
homestay_data['Cluster'] = cluster_labels

# Saving the dataframe with cluster labels to a CSV file
homestay_data.to_csv(output_file_path, index=False)

# Output file path for reference
print(f"Clustered data saved to: {output_file_path}")
