#!/usr/bin/env python3
"""
Text Analysis Module for Google Business Reviews Analyzer
Provides additional text analysis capabilities like sentiment analysis and topic extraction.
"""

import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download necessary NLTK resources at module import time
# This ensures all required resources are available
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK resources: {str(e)}")


def preprocess_text(text):
    """Clean and preprocess text for analysis"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove translated by Google sections
    text = re.sub(r'\(Translated by Google\).*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
    
    # Remove (Original) sections
    text = re.sub(r'\(Original\).*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
    
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def analyze_sentiment(df, output_dir):
    """Analyze sentiment of review comments"""
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Filter to reviews with comments
    comments_df = df[df['has_comment']].copy()
    
    if len(comments_df) == 0:
        print("No reviews with comments found for sentiment analysis")
        return
    
    # Apply sentiment analysis
    comments_df['sentiment_scores'] = comments_df['comment'].apply(
        lambda x: sia.polarity_scores(x) if isinstance(x, str) else {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
    )
    
    # Extract sentiment components
    comments_df['sentiment_compound'] = comments_df['sentiment_scores'].apply(lambda x: x['compound'])
    comments_df['sentiment_positive'] = comments_df['sentiment_scores'].apply(lambda x: x['pos'])
    comments_df['sentiment_negative'] = comments_df['sentiment_scores'].apply(lambda x: x['neg'])
    comments_df['sentiment_neutral'] = comments_df['sentiment_scores'].apply(lambda x: x['neu'])
    
    # Categorize sentiment
    comments_df['sentiment_category'] = pd.cut(
        comments_df['sentiment_compound'],
        bins=[-1.1, -0.5, 0.0, 0.5, 1.1],
        labels=['Very Negative', 'Negative', 'Neutral', 'Positive']
    )
    
    # Compare sentiment with star rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='rating', y='sentiment_compound', data=comments_df)
    plt.title('Sentiment Score Distribution by Star Rating')
    plt.xlabel('Star Rating')
    plt.ylabel('Sentiment Compound Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_by_rating.png'), dpi=300)
    plt.close()
    
    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment_category', hue='sentiment_category', data=comments_df, palette='RdYlGn', legend=False)
    plt.title('Distribution of Sentiment Categories')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300)
    plt.close()
    
    # Plot sentiment over time
    monthly_sentiment = comments_df.groupby(pd.Grouper(key='create_time', freq='ME'))['sentiment_compound'].mean()
    
    plt.figure(figsize=(12, 6))
    monthly_sentiment.plot(kind='line', marker='o', color='purple')
    plt.title('Average Sentiment Score by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_over_time.png'), dpi=300)
    plt.close()
    
    # Save sentiment data
    sentiment_summary = {
        "avg_sentiment_score": comments_df['sentiment_compound'].mean(),
        "sentiment_categories": comments_df['sentiment_category'].value_counts().to_dict(),
        "correlation_rating_sentiment": comments_df[['rating', 'sentiment_compound']].corr().iloc[0, 1]
    }
    
    with open(os.path.join(output_dir, 'sentiment_summary.json'), 'w') as f:
        json.dump(sentiment_summary, f, indent=2)
    
    # Add sentiment to the main dataframe
    return comments_df[['name', 'sentiment_compound', 'sentiment_category']]


def extract_common_topics(df, output_dir, n_topics=5):
    """Extract common topics from review comments using LDA"""
    # Filter to reviews with comments
    comments_df = df[df['has_comment']].copy()
    
    if len(comments_df) < 10:  # Need a minimum number for meaningful topic analysis
        print("Not enough reviews with comments for topic modeling")
        return
    
    # Preprocess comments
    comments_df['processed_text'] = comments_df['comment'].apply(preprocess_text)
    
    # Remove empty texts
    comments_df = comments_df[comments_df['processed_text'].str.strip() != '']
    
    if len(comments_df) < 5:
        print("Not enough reviews with valid text after preprocessing")
        return
    
    try:
        # Use CountVectorizer to convert text to document-term matrix
        vectorizer = CountVectorizer(
            max_df=0.95,         # Remove terms that appear in >95% of documents
            min_df=2,            # Remove terms that appear in <2 documents
            stop_words='english',
            max_features=1000    # Limit vocabulary size
        )
        
        # Fit and transform the text data
        dtm = vectorizer.fit_transform(comments_df['processed_text'])
        
        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        
        # Create and fit LDA model
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        
        lda_model.fit(dtm)
        
        # Extract topics
        topics = {}
        for topic_idx, topic in enumerate(lda_model.components_):
            # Get the top words for this topic
            top_words_idx = topic.argsort()[:-11:-1]  # Top 10 words
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f"Topic {topic_idx+1}"] = top_words
        
        # Save topics to file
        with open(os.path.join(output_dir, 'review_topics.json'), 'w') as f:
            json.dump(topics, f, indent=2)
        
        # Create topic visualization
        fig, axes = plt.subplots(n_topics, 1, figsize=(10, n_topics*3), sharex=True)
        
        if n_topics == 1:
            axes = [axes]  # Make it iterable if only one topic
            
        for i, (topic_name, words) in enumerate(topics.items()):
            y_pos = range(len(words))
            topic_probs = lda_model.components_[i][lda_model.components_[i].argsort()[:-11:-1]]
            
            axes[i].barh(y_pos, topic_probs, align='center')
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(words)
            axes[i].invert_yaxis()
            axes[i].set_title(topic_name)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'review_topics.png'), dpi=300)
        plt.close()
        
        print(f"Extracted {n_topics} topics from the reviews")
        
    except Exception as e:
        print(f"Error in topic extraction: {str(e)}")


def analyze_keyword_frequency(df, output_dir):
    """Analyze frequency of keywords in reviews"""
    # Filter to reviews with comments
    comments_df = df[df['has_comment']].copy()
    
    if len(comments_df) == 0:
        print("No reviews with comments found for keyword analysis")
        return
    
    try:
        # Combine all comments
        all_text = ' '.join(comments_df['comment'].fillna(''))
        processed_text = preprocess_text(all_text)
        
        # Tokenize using simple split to avoid punkt_tab dependency
        tokens = processed_text.split()
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_freq = Counter(filtered_tokens)
        most_common = word_freq.most_common(20)
        
        # Create visualization
        words, counts = zip(*most_common)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(words)), counts, align='center')
        plt.yticks(range(len(words)), words)
        plt.gca().invert_yaxis()
        plt.xlabel('Frequency')
        plt.title('Most Common Words in Reviews')
        
        # Add count labels
        for i, (word, count) in enumerate(most_common):
            plt.text(count + 0.5, i, str(count), va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keyword_frequency.png'), dpi=300)
        plt.close()
        
        # Save keyword data
        with open(os.path.join(output_dir, 'keyword_frequency.json'), 'w') as f:
            json.dump(dict(word_freq.most_common(50)), f, indent=2)
            
    except Exception as e:
        print(f"Error in keyword analysis: {str(e)}")


def run_text_analysis(df, output_dir):
    """Run all text analysis functions"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Performing text analysis...")
    
    # Run sentiment analysis
    sentiment_data = analyze_sentiment(df, output_dir)
    
    # Extract topics
    extract_common_topics(df, output_dir)
    
    # Analyze keyword frequency
    analyze_keyword_frequency(df, output_dir)
    
    # Return sentiment data to be added to the main dataframe
    return sentiment_data 