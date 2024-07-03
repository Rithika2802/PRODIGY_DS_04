import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Step 2: Load Data
file_path = r'C:\Users\rithi\OneDrive\Documents\twitter_training.csv' 

# Specify header=None to read the first row as data and not as header names
df = pd.read_csv(file_path, header=None)

# Check the first few rows of your dataframe to understand its structure
print("First few rows of the dataframe:")
print(df.head())

# Assign meaningful column names based on the structure of your data
df.columns = ['ID', 'Author', 'Sentiment', 'Tweet']  # Update with appropriate column names

# Verify the updated column names
print("\nUpdated column names in the dataframe:")
print(df.columns)

# Step 3: Perform Sentiment Analysis
def get_sentiment(text):
    if isinstance(text, str):  # Check if the value is a string
        analysis = TextBlob(text)
        return 'positive' if analysis.sentiment.polarity > 0 else 'negative' if analysis.sentiment.polarity < 0 else 'neutral'
    else:
        return 'unknown'

# Ensure 'Tweet' column exists before applying sentiment analysis
if 'Tweet' in df.columns:
    df['Sentiment'] = df['Tweet'].apply(get_sentiment)
    
    # Step 4: Visualize Sentiment Distribution
    plt.figure(figsize=(8, 6))
    df['Sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    # Step 5: Analyze Sentiment Over Time (if 'Timestamp' column is available)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        plt.figure(figsize=(12, 8))
        df['Sentiment'].resample('D').value_counts().unstack().plot(kind='bar', stacked=True)
        plt.title('Sentiment Over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend(title='Sentiment')
        plt.show()
    else:
        print("Timestamp column not found. Skipping sentiment over time analysis.")
else:
    print("Tweet column not found in the dataframe. Verify column names.")
