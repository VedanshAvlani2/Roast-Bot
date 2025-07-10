import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
import joblib
import os
import datetime

# -------------------------------
# 1. Load & Prepare Dataset
# -------------------------------
df = pd.read_csv("roast_dataset.csv")
df.columns = df.columns.str.strip().str.lower()
roast_styles = df['style'].unique().tolist()

# Train one model per style
style_models = {}
for style in roast_styles:
    sub_df = df[df['style'] == style]
    X = sub_df['input']
    y = sub_df['roast']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', RidgeClassifier())
    ])
    pipeline.fit(X, y)
    style_models[style] = pipeline

# -------------------------------
# 2. Roast Logger Initialization
# -------------------------------
log_file = "roast_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["timestamp", "input", "style", "roast"]).to_csv(log_file, index=False)

# -------------------------------
# 3. Terminal Roast Session
# -------------------------------
print("\n🔥 Welcome to the Savage Roast Generator 🔥")
print("Choose your roast style:")
for idx, style in enumerate(roast_styles, 1):
    print(f"{idx}. {style}")

style_choice = input("\n🎭 Enter style number (e.g., 1): ").strip()
try:
    selected_style = roast_styles[int(style_choice) - 1]
except:
    print("⚠️ Invalid selection. Defaulting to 'savage'")
    selected_style = 'savage'

print("\nLet's begin the roast session 🧠🔥")
print("Type 'exit' at any point to quit.")

while True:
    job = input("\n💼 What is your job? ").strip()
    if job.lower() == 'exit': break

    food = input("🍕 Favorite food? ").strip()
    if food.lower() == 'exit': break

    hobby = input("🎮 Hobby? ").strip()
    if hobby.lower() == 'exit': break

    user_input = f"I am a {job} who loves {food} and enjoys {hobby}."
    prediction = style_models[selected_style].predict([user_input])[0]

    print("\n💥 Roast Result:")
    print(f"🎯 {prediction}")

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input": user_input,
        "style": selected_style,
        "roast": prediction
    }
    pd.DataFrame([log_entry]).to_csv(log_file, mode='a', header=False, index=False)
