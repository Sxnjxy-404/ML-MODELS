import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('movie_data.csv')

# Rename columns for consistency
df.rename(columns={
    'WatchFrequency': 'Frequency',
    'GenrePreference': 'Genre'
}, inplace=True)

# Encode Gender and Genre
gender_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

df['Gender'] = gender_encoder.fit_transform(df['Gender'])
df['Genre'] = genre_encoder.fit_transform(df['Genre'])

# Features and Target
X = df[['Age', 'Gender', 'Frequency', 'Genre']]
y = df['Interested']  # 1 = Interested, 0 = Not Interested

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('movie_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as 'movie_model.pkl'")
