#!/usr/bin/env python3
"""
Google Business Reviews Analyzer
This script processes and visualizes Google Business reviews from JSON files.
"""

import os
import json
import glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import numpy as np

# Import text analysis module
from text_analysis import run_text_analysis

# Import comparative analysis module
try:
    from comparative_analysis import run_comparative_analysis
    COMPARATIVE_AVAILABLE = True
except ImportError:
    COMPARATIVE_AVAILABLE = False
    print("Comparative analysis module not available. Skipping comparative analysis functionality.")

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def load_review_files(input_dir="input"):
    """Load all review JSON files from the input directory"""
    review_files = glob.glob(os.path.join(input_dir, "reviews*.json"))
    
    if not review_files:
        raise ValueError(f"No review files found in {input_dir}")
    
    print(f"Found {len(review_files)} review files")
    return review_files


def parse_reviews(review_files):
    """Parse all reviews from the JSON files into a single DataFrame"""
    all_reviews = []
    
    for file_path in tqdm(review_files, desc="Loading reviews"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'reviews' in data:
                # Add source file to each review for tracking
                for review in data['reviews']:
                    review['source_file'] = os.path.basename(file_path)
                    all_reviews.append(review)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if not all_reviews:
        raise ValueError("No valid reviews found in the provided files")
    
    print(f"Loaded {len(all_reviews)} reviews in total")
    return all_reviews


def create_review_dataframe(reviews):
    """Convert reviews list to a pandas DataFrame with proper data types"""
    df = pd.DataFrame(reviews)
    
    # Extract reviewer name
    df['reviewer_name'] = df['reviewer'].apply(lambda x: x.get('displayName', '') if x else '')
    
    # Convert star ratings to numeric
    rating_map = {
        'ONE': 1,
        'TWO': 2,
        'THREE': 3,
        'FOUR': 4,
        'FIVE': 5
    }
    df['rating'] = df['starRating'].map(rating_map)
    
    # Convert timestamps
    df['create_time'] = pd.to_datetime(df['createTime'])
    df['update_time'] = pd.to_datetime(df['updateTime'])
    
    # Extract date components
    df['create_date'] = df['create_time'].dt.date
    df['create_year'] = df['create_time'].dt.year
    df['create_month'] = df['create_time'].dt.month
    df['create_day'] = df['create_time'].dt.day
    df['create_weekday'] = df['create_time'].dt.day_name()
    
    # Identify if the review was updated
    df['was_updated'] = df['create_time'] != df['update_time']
    
    # Check if review has comment
    df['has_comment'] = df['comment'].notna() & (df['comment'] != '')
    
    return df


def analyze_review_trends(df, output_dir):
    """Analyze review trends over time"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot reviews over time
    reviews_by_month = df.groupby(pd.Grouper(key='create_time', freq='ME')).size()
    
    # Improve month labels
    month_labels = reviews_by_month.index.strftime('%b %Y')
    x_positions = np.arange(len(month_labels))
    
    # Reviews by month chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x_positions, reviews_by_month.values, color='skyblue')
    ax.set_title('Number of Reviews by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Reviews')
    
    # Improve x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(month_labels, rotation=90)
    
    # Show only a subset of ticks if there are too many months
    if len(month_labels) > 12:
        step = max(1, len(month_labels) // 12)
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            if i % step != 0:
                label.set_visible(False)
                
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'reviews_by_month.png'), dpi=300)
    plt.close(fig)
    
    # Plot cumulative reviews over time
    cumulative_reviews = reviews_by_month.cumsum()
    
    # First chart with all data points
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_positions, cumulative_reviews.values, marker='o', color='royalblue')
    
    ax.set_title('Cumulative Number of Reviews Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Number of Reviews')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis with readable month labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(month_labels, rotation=90)
    
    # Show only a subset of ticks if there are too many months
    if len(month_labels) > 12:
        step = max(1, len(month_labels) // 12)
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            if i % step != 0:
                label.set_visible(False)
                
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cumulative_reviews.png'), dpi=300)
    plt.close(fig)
    
    # Create a second version with improved visualization for many data points
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot with fewer markers and a smoother line
    ax.plot(x_positions, cumulative_reviews.values, 
            marker='o', 
            markevery=max(1, len(cumulative_reviews)//15),
            color='royalblue', 
            linewidth=2)
    
    # Add annotations for key milestones
    try:
        milestones = [
            (0, cumulative_reviews.iloc[0], "First Review"),
            (len(cumulative_reviews)//4, cumulative_reviews.iloc[len(cumulative_reviews)//4], 
             f"{int(cumulative_reviews.iloc[len(cumulative_reviews)//4])} Reviews"),
            (len(cumulative_reviews)//2, cumulative_reviews.iloc[len(cumulative_reviews)//2], 
             f"{int(cumulative_reviews.iloc[len(cumulative_reviews)//2])} Reviews"),
            (len(cumulative_reviews)-1, cumulative_reviews.iloc[-1], 
             f"{int(cumulative_reviews.iloc[-1])} Reviews")
        ]
        
        for i, (x_pos, value, label) in enumerate(milestones):
            ax.annotate(label, 
                       xy=(x_pos, value), 
                       xytext=(0, 10 if i % 2 == 0 else -25),
                       textcoords='offset points',
                       ha='center',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    except Exception as e:
        print(f"Warning: Could not add annotations to review growth chart: {str(e)}")
    
    ax.set_title('Growth of Reviews Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Number of Reviews')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis with quarterly labels for better readability
    quarter_indices = []
    quarter_labels = []
    
    for i, date in enumerate(cumulative_reviews.index):
        # Mark approximately quarterly points
        if i == 0 or i == len(cumulative_reviews) - 1 or i % max(1, len(cumulative_reviews) // 8) == 0:
            quarter_indices.append(i)
            quarter_labels.append(date.strftime('%b %Y'))
    
    ax.set_xticks(quarter_indices)
    ax.set_xticklabels(quarter_labels, rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'review_growth.png'), dpi=300)
    plt.close(fig)
    
    # Plot average rating over time
    monthly_avg_rating = df.groupby(pd.Grouper(key='create_time', freq='ME'))['rating'].mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_positions, monthly_avg_rating.values, marker='o', color='green')
    
    ax.set_title('Average Rating by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Rating')
    ax.set_ylim(0, 5.5)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis with readable month labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(month_labels, rotation=90)
    
    # Show only a subset of ticks if there are too many months
    if len(month_labels) > 12:
        step = max(1, len(month_labels) // 12)
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            if i % step != 0:
                label.set_visible(False)
                
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'avg_rating_by_month.png'), dpi=300)
    plt.close(fig)
    
    # Rating distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', hue='rating', data=df, palette='viridis', legend=False)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'), dpi=300)
    plt.close()
    
    # Create cumulative average rating over time chart
    # Sort the dataframe by time
    df_sorted = df.sort_values('create_time')
    
    # Create monthly markers
    monthly_markers = pd.date_range(
        start=df_sorted['create_time'].min(), 
        end=df_sorted['create_time'].max(), 
        freq='ME'
    )
    
    # Calculate cumulative average at each month
    cumulative_ratings = []
    dates = []
    
    for date in monthly_markers:
        # Select all reviews up to this date
        reviews_up_to_date = df_sorted[df_sorted['create_time'] <= date]
        
        if len(reviews_up_to_date) > 0:
            avg_rating = reviews_up_to_date['rating'].mean()
            cumulative_ratings.append(avg_rating)
            dates.append(date)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, cumulative_ratings, marker='o', linestyle='-', color='purple', linewidth=2,
           markersize=4)
    
    # Mark the current overall average
    overall_avg = df['rating'].mean()
    ax.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.7)
    ax.text(dates[0], overall_avg + 0.05, f'Overall Avg: {overall_avg:.2f}', color='red')
    
    # Format x-axis with readable month labels
    month_labels = [d.strftime('%b %Y') for d in dates]
    
    # Show only a subset of labels if there are many months
    if len(month_labels) > 12:
        step = max(1, len(month_labels) // 12)
        visible_dates = dates[::step]
        visible_labels = month_labels[::step]
        ax.set_xticks(visible_dates)
        ax.set_xticklabels(visible_labels, rotation=45)
    else:
        ax.set_xticks(dates)
        ax.set_xticklabels(month_labels, rotation=45)
    
    # Set axis limits
    ax.set_ylim(min(3, min(cumulative_ratings)-0.2), max(5, max(cumulative_ratings)+0.2))
    
    # Add titles and labels
    ax.set_title('Cumulative Average Rating Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Rating (Cumulative)')
    ax.grid(True, alpha=0.3)
    
    # Annotate start and end points
    ax.annotate(f'Start: {cumulative_ratings[0]:.2f}',
               xy=(dates[0], cumulative_ratings[0]),
               xytext=(10, 0),
               textcoords='offset points',
               va='center',
               fontsize=10)
    
    ax.annotate(f'Current: {cumulative_ratings[-1]:.2f}',
               xy=(dates[-1], cumulative_ratings[-1]),
               xytext=(-10, 0),
               textcoords='offset points',
               ha='right',
               va='center',
               fontsize=10)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cumulative_avg_rating.png'), dpi=300)
    plt.close(fig)
    
    # Reviews by day of week
    plt.figure(figsize=(10, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = df['create_weekday'].value_counts().reindex(day_order)
    weekday_counts.plot(kind='bar', color='coral')
    plt.title('Number of Reviews by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reviews_by_weekday.png'), dpi=300)
    plt.close()
    
    # Generate a summary report
    summary = {
        "total_reviews": int(len(df)),
        "avg_rating": float(df['rating'].mean()),
        "rating_counts": {str(k): int(v) for k, v in df['rating'].value_counts().to_dict().items()},
        "percent_with_comments": float((df['has_comment'].sum() / len(df)) * 100),
        "updated_reviews": int(df['was_updated'].sum()),
        "earliest_review": df['create_time'].min().strftime("%Y-%m-%d"),
        "latest_review": df['create_time'].max().strftime("%Y-%m-%d"),
    }
    
    with open(os.path.join(output_dir, 'review_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def analyze_review_content(df, output_dir):
    """Analyze the content of reviews"""
    if not df['has_comment'].any():
        print("No reviews with comments found. Skipping content analysis.")
        return
    
    # Get reviews with comments
    comments_df = df[df['has_comment']].copy()
    
    try:
        # Generate word cloud for positive reviews (4-5 stars)
        from wordcloud import WordCloud
        from nltk.corpus import stopwords
        import nltk
        
        # Download NLTK resources
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        
        # Create word clouds for positive and negative reviews
        positive_comments = ' '.join(comments_df[comments_df['rating'] >= 4]['comment'].fillna(''))
        
        # Configure stop words (common words to exclude)
        stop_words = set(stopwords.words('english'))
        stop_words.update(['translated', 'google', 'original', 'us', 'also', 'really', 'one', 'well'])
        
        # Generate word cloud for positive reviews
        if positive_comments.strip():
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                stopwords=stop_words,
                max_words=100,
                contour_width=3,
                contour_color='steelblue'
            ).generate(positive_comments)
            
            plt.figure(figsize=(10, 7))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Words in Positive Reviews')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'positive_reviews_wordcloud.png'), dpi=300)
            plt.close()
        
        # Generate word cloud for negative reviews (1-3 stars)
        negative_comments = ' '.join(comments_df[comments_df['rating'] <= 3]['comment'].fillna(''))
        
        if negative_comments.strip():
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                stopwords=stop_words,
                max_words=100,
                contour_width=3,
                contour_color='firebrick'
            ).generate(negative_comments)
            
            plt.figure(figsize=(10, 7))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Words in Negative Reviews')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'negative_reviews_wordcloud.png'), dpi=300)
            plt.close()
            
    except Exception as e:
        print(f"Error in content analysis: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Google Business reviews')
    parser.add_argument('--input', '-i', default='input', help='Input directory containing review JSON files')
    parser.add_argument('--output', '-o', default='output', help='Output directory for analysis results')
    parser.add_argument('--skip-text-analysis', action='store_true', help='Skip detailed text analysis')
    
    # Add comparative analysis option
    parser.add_argument('--pivot-date', '-p', help='Date (YYYY-MM-DD) to use as pivot point for comparative analysis')
    
    args = parser.parse_args()
    
    try:
        # Load and process reviews
        review_files = load_review_files(args.input)
        reviews = parse_reviews(review_files)
        df = create_review_dataframe(reviews)
        
        # Perform basic analyses
        summary = analyze_review_trends(df, args.output)
        analyze_review_content(df, args.output)
        
        # Run advanced text analysis if not skipped
        if not args.skip_text_analysis:
            try:
                # Create an advanced analysis subdirectory
                text_output_dir = os.path.join(args.output, 'text_analysis')
                os.makedirs(text_output_dir, exist_ok=True)
                
                # Run text analysis
                sentiment_data = run_text_analysis(df, text_output_dir)
                
                # Merge sentiment data back if available
                if sentiment_data is not None and not sentiment_data.empty:
                    df = df.merge(sentiment_data, on='name', how='left')
            except Exception as e:
                print(f"Warning: Text analysis failed: {str(e)}")
                print("Continuing with basic analysis only...")
        
        # Run comparative analysis if a pivot date is provided
        if args.pivot_date and COMPARATIVE_AVAILABLE:
            try:
                run_comparative_analysis(df, args.pivot_date, args.output)
                print(f"Comparative analysis completed for pivot date: {args.pivot_date}")
            except Exception as e:
                print(f"Warning: Comparative analysis failed: {str(e)}")
                print("Continuing without comparative analysis...")
        
        # Save the processed data
        df.to_csv(os.path.join(args.output, 'processed_reviews.csv'), index=False)
        
        print("\nAnalysis completed successfully!")
        print(f"Total reviews: {summary['total_reviews']}")
        print(f"Average rating: {summary['avg_rating']:.2f}/5.0")
        print(f"Reviews with comments: {summary['percent_with_comments']:.1f}%")
        print(f"Date range: {summary['earliest_review']} to {summary['latest_review']}")
        print(f"\nResults saved to: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 