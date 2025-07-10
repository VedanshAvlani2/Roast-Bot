# 🔥 Data-Driven Roast Generator

## Overview
This ML-based terminal app delivers savage, sarcastic, or friendly roasts based on a person’s hobbies, job, and favorite food. Built to simulate humor and personalization using text-based features and a trained model.

## Objective
- Take in structured user input (hobby, job, favorite food)
- Predict and generate a roast sentence using trained ML model
- Support different roast "tones" (savage, sarcastic, friendly)

## Dataset
Custom-built dataset with 120 rows of combinations:
- `Hobby`: Reading, Gyming, Gaming, etc.
- `Job`: Engineer, Influencer, etc.
- `Food`: Pizza, Sushi, Kale, etc.
- `Style`: Savage, Sarcastic, Friendly
- `Roast`: Matching roast statements

## Technologies Used
- Python
- pandas, scikit-learn
- TfidfVectorizer, RidgeClassifier
- joblib (for model persistence)

## How to Run
```bash
pip install pandas scikit-learn joblib
python roast_generator.py
```

## Sample Output
```bash
💬 Hobby? gaming
💼 Job? influencer
🍕 Favorite food? sushi

🗯️ Roast (savage):
You call yourself an influencer, but your only follower is the sushi delivery guy.
```

## Features
- Multi-feature ML model: hobby + job + food
- Style-based roast tone selection
- Easy to expand with new roast samples

## Future Enhancements
- Add GPT-based roast mode
- Roast intensity slider (e.g., mild to brutal)
- Voice roast using text-to-speech
