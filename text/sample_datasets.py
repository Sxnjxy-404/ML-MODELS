# Sample Text Classification Datasets Generator
# Run this script to create sample datasets for your text classification app

import pandas as pd
import random

def create_sentiment_dataset():
    """Create a sentiment analysis dataset"""
    positive_texts = [
        "I absolutely love this product! It's amazing!",
        "This is the best purchase I've ever made. Highly recommend!",
        "Excellent quality and fast shipping. Very satisfied!",
        "Outstanding customer service and great value for money.",
        "Perfect! Exactly what I was looking for.",
        "Incredible results! Will definitely buy again.",
        "Five stars! Exceeded my expectations completely.",
        "Fantastic product with amazing features.",
        "Love it! Works perfectly and looks great.",
        "Best quality I've seen in this price range.",
        "Superb craftsmanship and attention to detail.",
        "Highly impressed with the build quality.",
        "Amazing performance and very user-friendly.",
        "Great product, great price, great service!",
        "Wonderful experience from start to finish.",
        "This product is simply outstanding!",
        "Perfect fit and excellent material quality.",
        "Impressed by the fast delivery and packaging.",
        "Great value for money. Very pleased!",
        "Excellent product with top-notch performance."
    ]
    
    negative_texts = [
        "Terrible product. Complete waste of money.",
        "Poor quality and doesn't work as advertised.",
        "Worst purchase ever. Very disappointed.",
        "Cheap materials and broke after one day.",
        "Awful customer service and delayed shipping.",
        "Not worth the price. Very poor quality.",
        "Completely useless. Don't buy this!",
        "Poor design and terrible functionality.",
        "Overpriced for such low quality.",
        "Defective product received. Very frustrated.",
        "Poor packaging and arrived damaged.",
        "Misleading description. Not as promised.",
        "Low quality materials and poor construction.",
        "Terrible experience with this product.",
        "Waste of time and money. Avoid!",
        "Poor performance and unreliable.",
        "Disappointing quality for the price paid.",
        "Not recommended. Poor value for money.",
        "Cheap construction and breaks easily.",
        "Unsatisfied with the product quality."
    ]
    
    neutral_texts = [
        "The product is okay, nothing special.",
        "It's an average product for the price.",
        "Works as expected, nothing more nothing less.",
        "Decent quality but could be better.",
        "It's fine, does what it's supposed to do.",
        "Average performance, meets basic requirements.",
        "Standard quality product, no complaints.",
        "It's okay but there are better alternatives.",
        "Fair quality for a reasonable price.",
        "Acceptable product with standard features.",
        "Basic functionality, nothing extraordinary.",
        "Reasonable quality for the price point.",
        "Standard product, works adequately.",
        "It's alright, meets minimum expectations.",
        "Average build quality and performance.",
        "Decent product but room for improvement.",
        "Satisfactory performance overall.",
        "Standard quality, nothing impressive.",
        "It's fine for basic use.",
        "Adequate product for everyday needs."
    ]
    
    # Create dataset
    texts = positive_texts + negative_texts + neutral_texts
    labels = ['positive'] * len(positive_texts) + ['negative'] * len(negative_texts) + ['neutral'] * len(neutral_texts)
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def create_news_dataset():
    """Create a news category dataset"""
    
    # Sports news
    sports_texts = [
        "The basketball team won their championship game last night with a score of 98-87.",
        "Football season starts next month with high expectations for the home team.",
        "Tennis tournament final will be played this weekend at the stadium.",
        "Olympic swimming records were broken during yesterday's competition.",
        "Baseball playoffs continue with exciting matches scheduled for this week.",
        "Soccer world cup preparations are underway with team selections announced.",
        "Golf championship attracts top players from around the world.",
        "Hockey season ends with spectacular playoff performances.",
        "Marathon runners prepare for the annual city race next month.",
        "Cricket match between rival teams draws massive crowd attendance."
    ]
    
    # Technology news
    technology_texts = [
        "New smartphone released with advanced camera technology and longer battery life.",
        "Artificial intelligence breakthrough announced by leading tech company.",
        "Software update brings new security features to millions of users worldwide.",
        "Electric vehicle sales surge as charging infrastructure expands rapidly.",
        "Cloud computing services experience significant growth in enterprise adoption.",
        "Cybersecurity threats increase as more businesses move operations online.",
        "Quantum computing research achieves major milestone in processing power.",
        "Social media platform introduces new privacy controls for user data.",
        "Robotics technology advances with improved automation capabilities.",
        "Virtual reality applications expand beyond gaming into education and training."
    ]
    
    # Health news
    health_texts = [
        "New medical research shows promising results for cancer treatment.",
        "Health officials recommend increased vaccination rates for flu prevention.",
        "Mental health awareness campaign launches in schools nationwide.",
        "Nutrition study reveals benefits of Mediterranean diet for heart health.",
        "Medical device approval brings new treatment option for patients.",
        "Fitness trends focus on home workouts and outdoor activities.",
        "Healthcare workers receive recognition for pandemic response efforts.",
        "Public health initiative targets childhood obesity prevention programs.",
        "Medical breakthrough offers hope for rare disease patients.",
        "Health insurance changes affect coverage for preventive care services."
    ]
    
    # Politics news
    politics_texts = [
        "Election results show close race between leading candidates.",
        "New legislation proposed to address climate change concerns.",
        "Government announces budget allocation for infrastructure projects.",
        "Political debate highlights differences in healthcare policy approaches.",
        "International relations strengthen through new trade agreement.",
        "Local government implements new policies for urban development.",
        "Voting registration drives encourage citizen participation in democracy.",
        "Political rally draws thousands of supporters to city center.",
        "Congressional hearing addresses national security concerns.",
        "Municipal elections scheduled for next month across the region."
    ]
    
    texts = sports_texts + technology_texts + health_texts + politics_texts
    labels = (['sports'] * len(sports_texts) + 
              ['technology'] * len(technology_texts) + 
              ['health'] * len(health_texts) + 
              ['politics'] * len(politics_texts))
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def create_spam_dataset():
    """Create a spam/ham email dataset"""
    
    spam_texts = [
        "URGENT: Click here to claim your $1000 prize money now!!!",
        "Congratulations! You've won a lottery. Send your bank details immediately.",
        "FREE OFFER: Get rich quick with this amazing opportunity!",
        "CLICK NOW: Limited time offer for miracle weight loss pills.",
        "WINNER: You have been selected for a cash prize. Act fast!",
        "URGENT: Your account will be closed. Verify immediately.",
        "Make $5000 per week working from home. No experience needed!",
        "CONGRATULATIONS! You have won $50000. Claim your prize now!",
        "FREE MONEY: Government grants available. Apply today!",
        "URGENT: Suspicious activity detected. Click to verify account."
    ]
    
    ham_texts = [
        "Hi, are we still meeting for lunch tomorrow at 1 PM?",
        "The quarterly report is ready for your review. Please check email attachment.",
        "Thank you for your order. Your package will arrive in 3-5 business days.",
        "Meeting reminder: Team standup at 9 AM in conference room B.",
        "Your subscription renewal is due next week. Please update payment method.",
        "Happy birthday! Hope you have a wonderful celebration today.",
        "Flight confirmation: Your booking reference is ABC123. Check-in opens 24 hours before.",
        "Doctor appointment reminder: You have an appointment tomorrow at 2 PM.",
        "Weekly newsletter: Here are the top stories from our community.",
        "Password reset request: Click here to reset your password securely."
    ]
    
    # Add more samples to balance the dataset
    spam_texts_extended = spam_texts * 2
    ham_texts_extended = ham_texts * 2
    
    texts = spam_texts_extended + ham_texts_extended
    labels = ['spam'] * len(spam_texts_extended) + ['ham'] * len(ham_texts_extended)
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def create_emotion_dataset():
    """Create an emotion classification dataset"""
    
    joy_texts = [
        "I'm so excited about my vacation next week!",
        "Just got promoted at work! Best day ever!",
        "My baby took their first steps today. So happy!",
        "Finally graduated from university. Feeling amazing!",
        "Won the competition! Can't believe it happened!",
        "Got engaged today! Life is wonderful!",
        "Reunion with old friends was absolutely fantastic!",
        "My book got published! Dreams do come true!",
        "Perfect weather for the beach today. So joyful!",
        "Surprise party was incredible. Feeling loved!"
    ]
    
    sadness_texts = [
        "Lost my beloved pet today. Heartbroken.",
        "Didn't get the job I really wanted. Feeling down.",
        "Moving away from my hometown tomorrow. So sad.",
        "Relationship ended after three years together.",
        "Grandmother passed away yesterday. Miss her already.",
        "Failed my driving test again. Really disappointed.",
        "Best friend is moving to another country.",
        "Project I worked on for months got cancelled.",
        "Feeling lonely during the holiday season.",
        "Missed my flight and ruined vacation plans."
    ]
    
    anger_texts = [
        "Traffic jam made me two hours late for work!",
        "Customer service was absolutely terrible today!",
        "Someone stole my parking spot again. So annoying!",
        "Internet has been down all day. Completely frustrated!",
        "Waited an hour for food that never came. Ridiculous!",
        "Computer crashed and lost all my work. Furious!",
        "Neighbor's loud music kept me awake all night!",
        "Got charged extra fees without any explanation.",
        "Flight delayed for the third time today. Outrageous!",
        "Phone battery died during important call. Irritating!"
    ]
    
    fear_texts = [
        "Worried about the medical test results tomorrow.",
        "Starting new job next week. Feeling anxious.",
        "Dark alley at night makes me nervous.",
        "Scared of flying but have to travel for work.",
        "Worried about financial situation lately.",
        "Afraid of public speaking but have presentation tomorrow.",
        "Concerned about family member's health condition.",
        "Nervous about moving to new city alone.",
        "Worried about job security in current economy.",
        "Scared of heights but need to use elevator."
    ]
    
    texts = joy_texts + sadness_texts + anger_texts + fear_texts
    labels = (['joy'] * len(joy_texts) + 
              ['sadness'] * len(sadness_texts) + 
              ['anger'] * len(anger_texts) + 
              ['fear'] * len(fear_texts))
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return df

# Generate all datasets
if __name__ == "__main__":
    print("Creating sample datasets...")
    
    # Create sentiment dataset
    sentiment_df = create_sentiment_dataset()
    sentiment_df.to_csv('sentiment_dataset.csv', index=False)
    print(f"✅ Sentiment dataset created: {len(sentiment_df)} samples")
    print(f"   Labels: {sentiment_df['label'].value_counts().to_dict()}")
    
    # Create news dataset
    news_df = create_news_dataset()
    news_df.to_csv('news_category_dataset.csv', index=False)
    print(f"✅ News category dataset created: {len(news_df)} samples")
    print(f"   Labels: {news_df['label'].value_counts().to_dict()}")
    
    # Create spam dataset
    spam_df = create_spam_dataset()
    spam_df.to_csv('spam_detection_dataset.csv', index=False)
    print(f"✅ Spam detection dataset created: {len(spam_df)} samples")
    print(f"   Labels: {spam_df['label'].value_counts().to_dict()}")
    
    # Create emotion dataset
    emotion_df = create_emotion_dataset()
    emotion_df.to_csv('emotion_classification_dataset.csv', index=False)
    print(f"✅ Emotion classification dataset created: {len(emotion_df)} samples")
    print(f"   Labels: {emotion_df['label'].value_counts().to_dict()}")
    
    print("\n🎉 All datasets created successfully!")
    print("\nFiles created:")
    print("- sentiment_dataset.csv")
    print("- news_category_dataset.csv") 
    print("- spam_detection_dataset.csv")
    print("- emotion_classification_dataset.csv")
    
    print("\n📊 Dataset Preview:")
    print("\nSentiment Dataset:")
    print(sentiment_df.head(3))
    
    print("\nNews Dataset:")
    print(news_df.head(3))
    
    print("\nSpam Dataset:")
    print(spam_df.head(3))
    
    print("\nEmotion Dataset:")
    print(emotion_df.head(3))