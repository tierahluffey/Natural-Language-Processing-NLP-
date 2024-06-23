
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from flask import request


app = Flask(__name__)

# Load the dataset from the CSV file
data = pd.read_csv("final_corrected_homestays.csv", encoding='iso-8859-1')


# Filter stop words in english vocabulary
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fill any missing descriptions with an empty string
data['homestay_description'] = data['homestay_description'].fillna('')

# Drop rows with missing values in relevant columns
data_clean = data.dropna(subset=['distance_from_KTC', 'type_of_room', 'homestay_price', 'qty_of_bed'])

# Apply TF-IDF vectorization to the 'homestay_description' column for cleaned data
tfidf_matrix = tfidf_vectorizer.fit_transform(data_clean['homestay_description'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#1st CORE - by similar description
def get_recommendations(selected_homestay, cosine_sim=cosine_sim, num_recommendations=40):
    # Check if the selected_homestay exists in the dataset
    if selected_homestay not in data_clean['homestay_name'].values:
        return pd.DataFrame()  # Return an empty DataFrame

    # Get the index of the selected_homestay with the given name
    idx = data_clean[data_clean['homestay_name'] == selected_homestay].index[0]

    # Get the cosine similarity scores of all homestays with the selected_homestay
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Get the indices of the top num_recommendations homestays (excluding the selected_homestay)
    homestay_indices = [i[0] for i in sim_scores if data_clean['homestay_name'].iloc[i[0]] != selected_homestay][:num_recommendations]

    # Return the details (including all columns) of the top num_recommendations recommended homestays
    return data_clean.iloc[homestay_indices]
tfidf_matrix = tfidf_vectorizer.fit_transform(data_clean['homestay_description'])

from flask import request

# 2nd Core - by criteria
@app.route('/recommend_by_words', methods=['POST'])
def recommend_by_words():
    selected_words = request.form.getlist('selected_words')

    if not selected_words:
        return "Please select at least one word criteria."

    # TF-IDF Vector for selected words
    selected_words_tfidf = tfidf_vectorizer.transform(selected_words)

    # Calculate cosine similarity between selected words and homestay descriptions
    cosine_sim_filtered = cosine_similarity(selected_words_tfidf, tfidf_matrix)

    # Sum the cosine similarity scores for each word
    scores = cosine_sim_filtered.sum(axis=0)

    # Convert the scores array to a 1D array for indexing
    scores = scores.ravel()

    # Sort homestays based on the combined score in descending order
    sorted_indices = scores.argsort()[::-1]

    # Get the top recommendations based on the combined score
    num_recommendations = 40
    top_recommendations = data_clean.iloc[sorted_indices[:num_recommendations]]

    return render_template('results_by_review.html', selected_words=selected_words, recommendations=top_recommendations)

# 3rd CORE - by landmark
@app.route('/recommend_by_landmark', methods=['POST'])
def recommend_by_landmark():
    selected_landmark = request.form['selected_landmark']

    # Extract landmarks from the user's input and preprocess
    selected_landmark = selected_landmark.lower()  # Convert to lowercase for case-insensitive matching

    # TF-IDF Vector for selected landmark
    selected_landmark_tfidf = tfidf_vectorizer.transform([selected_landmark])
    
    # Calculate cosine similarity between selected landmark and homestay descriptions
    cosine_sim_landmark = cosine_similarity(selected_landmark_tfidf, tfidf_matrix)

    # Filter homestays that contain the selected landmark in their descriptions
    relevant_homestays = data_clean[data_clean['homestay_description'].str.lower().str.contains(selected_landmark)]

    if relevant_homestays.empty:
        return "No homestays found for the selected landmark."

    # Assign a score based on the presence of the selected landmark
    relevant_homestays['landmark_score'] = relevant_homestays['homestay_description'].str.lower().str.count(selected_landmark)

    # Combine the TF-IDF cosine similarity and keyword matching scores
    relevant_homestays['combined_score'] = (
        cosine_sim_landmark[0][0] + relevant_homestays['landmark_score']
    )

    # Sort homestays by the combined score in descending order
    relevant_homestays = relevant_homestays.sort_values(by=['combined_score'], ascending=False)

    # Get the top recommendations based on the combined score
    num_recommendations = 40
    top_recommendations = relevant_homestays.head(num_recommendations)

    return render_template('result_by_landmark.html', selected_landmark=selected_landmark, recommendations=top_recommendations)

import os

# Read File Path
file_path = os.path.join("USER", "01.txt")
absolute_file_path = os.path.abspath(file_path)

print("Attempting to access file at:", absolute_file_path)
print("Current working directory:", os.getcwd())

if os.path.exists(absolute_file_path):
    with open(absolute_file_path, 'r') as f:
        user_history = f.read().splitlines()
else:
    print("File not found, creating a new file.")
    with open(absolute_file_path, 'w') as f:
        f.write("")  # Create an empty file
    user_history = []


# 4th Core - New function to recommend homestays from user profile@app.route('/recommend_from_history', methods=['GET', 'POST'])
@app.route('/recommend_from_history', methods=['GET', 'POST'])
def recommend_from_history_post():
    # Fetch user_id from form data or URL parameters
    user_id = request.form.get('user_id') or request.args.get('user_id') or '01'

    # Construct the path to the user's history file
    file_path = os.path.join("USER", f"{user_id}.txt")


    # Read the user's history file and extract the highest-rated homestay
    highest_rated_homestay = None
    highest_rating = -1.0

    with open(file_path, 'r') as f:
        for line in f:
            try:
                homestay_name, rating = line.split(',')
                rating = float(rating)
                if rating > highest_rating:
                    highest_rating = rating
                    highest_rated_homestay = homestay_name.strip()
            except ValueError:
                print(f"Error processing line: '{line}'. Line is not correctly formatted.")

    if highest_rated_homestay is None:
        return "No highest-rated homestay found in the user's history."

    # Check if the highest-rated homestay exists in the cleaned dataset
    if highest_rated_homestay not in data_clean['homestay_name'].values:
        return "Highest-rated homestay not found in the current dataset."

    # Find the index of the highest-rated homestay in the cleaned dataset
    idx = data_clean[data_clean['homestay_name'] == highest_rated_homestay].index[0]

    # Calculate cosine similarity between the highest-rated homestay and all homestays in the dataset
    similarity_scores = list(enumerate(cosine_sim[idx]))

    # Sort homestays by similarity score in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top N recommendations (excluding the highest-rated homestay itself)
    num_recommendations = 40
    recommended_indices = [i[0] for i in similarity_scores if i[0] != idx][:num_recommendations]

    # Create a dictionary to store homestay details
    homestay_details = {}
    for idx in recommended_indices:
        homestay_info = data_clean.iloc[idx]
        homestay_details[homestay_info['homestay_name']] = homestay_info.to_dict()

    # Create a list of recommended homestay names
    recommendations = [data_clean.iloc[i]['homestay_name'] for i in recommended_indices]

    return render_template('history.html', homestay_name=highest_rated_homestay, recommendations=recommendations, homestay_details=homestay_details)

# ==================================================

# Directory where user profiles are stored - Create User Profile & Gives Rating
user_profiles_directory = "C:/Users/user/Documents/CODE/USER"

# Load the CSV file and get the homestay names
def get_homestay_names():
    # Assuming your CSV file is in the same directory as your script
    df = pd.read_csv('final_corrected_homestays.csv')
    homestay_names = df['homestay_name'].tolist()
    return homestay_names

# User Ratings List
user_ratings_list = []

# Function to save user ratings to a .txt file
def save_user_ratings(user_id, homestay_name, rating):
    # Define the filename based on the user ID
    filename = os.path.join(user_profiles_directory, f"{user_id}.txt")

    # Format the homestay name and rating
    homestay_rating = f"{homestay_name},{rating}"

    # Append the homestay name and rating to the file
    with open(filename, 'a') as file:
        file.write(f"{homestay_rating}\n")

# User Registration Form - Let users register and create profiles.
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form.get('user_id')

        # Check if the user already exists
        existing_profile = os.path.join(user_profiles_directory, f"{user_id}.txt")
        if os.path.exists(existing_profile):
            return "User already exists!"

        # Create a user profile file
        with open(existing_profile, 'w') as file:
            file.write(f"User ID: {user_id}\n")

        return "User registered successfully!"

    return render_template('register.html')

# User Rating Form - Let registered users insert manual ratings.
@app.route('/rate_homestays', methods=['GET', 'POST'])
def rate_homestays():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        homestay1 = request.form.get('homestay1')
        homestay2 = request.form.get('homestay2')
        homestay3 = request.form.get('homestay3')
        rating1 = int(request.form.get('rating1'))
        rating2 = int(request.form.get('rating2'))
        rating3 = int(request.form.get('rating3'))

        # Save user ratings to the corresponding .txt file
        save_user_ratings(user_id, homestay1, rating1)
        save_user_ratings(user_id, homestay2, rating2)
        save_user_ratings(user_id, homestay3, rating3)

        return "Ratings submitted successfully!"

    # Get the homestay names when it's a GET request
    homestay_names = get_homestay_names()

    return render_template('rate_homestays.html', homestay_names=homestay_names)

#==============================================

@app.route('/', methods=['GET', 'POST'])
def index():
    print(data.columns) 
    homestay_data = data[['homestay_name', 'homestay_description', 'distance_from_KTC', 'type_of_room', 'homestay_price', 'qty_of_bed']]
    homestay_names = homestay_data['homestay_name'].tolist()

    if request.method == 'POST':
        selected_homestay = request.form['homestay_name']
        recommendations = get_recommendations(selected_homestay)
        return render_template('results.html', homestay_name=selected_homestay, recommendations=recommendations)
    
    return render_template('index.html', homestay_data=homestay_data, homestays=homestay_names)

#DEFINE DULU SEMUA NAV LINK DEKAT SINI
@app.route('/original_index')
def original_index():
    return render_template('original_index.html', homestays=data['homestay_name'].tolist())

@app.route('/pop2.html')
def pop2():
    return render_template('pop2.html')

@app.route('/word_input')
def word_input():
    return render_template('word_input.html', common_words=[('clean', 83), ('good facilities', 52), ('number of room', 52), ('price', 45), ('affordable price', 43), ('facilities', 42), ('strategic location', 41), ('house', 40), ('spacious', 31), ('comfortable', 29)])

@app.route('/landmark_input')
def landmark_input():
    return render_template('landmark_input.html', common_words=[
        ('museum terengganu', 116),
        ('chinatown', 55),
        ('batu burok beach', 51),
        ('crystal mosque', 168),
        ('seberang taking', 16),
        ('gong badak', 11),
        ('Drawbridge', 5),
        ('chendering', 5),
        ('pantai marang', 12)
    ])

@app.route('/choose_recommendation')
def choose_recommendation():
    # Add any necessary logic or data retrieval here
    return render_template('choose_recommendation.html')


if __name__ == '__main__':
    app.run(debug=True)
