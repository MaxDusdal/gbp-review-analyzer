#!/usr/bin/env python3
"""
Comparative Analysis Module for Google Business Reviews
Analyze and visualize differences in reviews before and after a specified date.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import json
from matplotlib.ticker import MaxNLocator
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def run_comparative_analysis(df, pivot_date, output_dir):
    """
    Perform comparative analysis of reviews before and after a specific date
    
    Args:
        df: DataFrame containing review data
        pivot_date: Date string in 'YYYY-MM-DD' format
        output_dir: Directory to save analysis results
    """
    # Create output directory if it doesn't exist
    comparative_dir = os.path.join(output_dir, 'comparative_analysis')
    os.makedirs(comparative_dir, exist_ok=True)
    
    # Convert all date columns to naive datetime by removing timezone info
    df_copy = df.copy()
    df_copy['create_time'] = df_copy['create_time'].dt.tz_localize(None)
    df_copy['update_time'] = df_copy['update_time'].dt.tz_localize(None)
    
    # Convert pivot_date to naive datetime
    try:
        pivot_datetime = pd.to_datetime(pivot_date)
    except:
        raise ValueError(f"Invalid date format. Please use YYYY-MM-DD format. Got: {pivot_date}")
    
    # Split data into before and after pivot date
    before_df = df_copy[df_copy['create_time'] < pivot_datetime].copy()
    after_df = df_copy[df_copy['create_time'] >= pivot_datetime].copy()
    
    if len(before_df) == 0 or len(after_df) == 0:
        raise ValueError(f"Not enough data to compare. Before pivot: {len(before_df)} reviews, After pivot: {len(after_df)} reviews")
    
    print(f"\nPerforming comparative analysis around {pivot_date}")
    print(f"Before pivot: {len(before_df)} reviews ({before_df['create_time'].min().strftime('%Y-%m-%d')} to {before_df['create_time'].max().strftime('%Y-%m-%d')})")
    print(f"After pivot: {len(after_df)} reviews ({after_df['create_time'].min().strftime('%Y-%m-%d')} to {after_df['create_time'].max().strftime('%Y-%m-%d')})")
    
    # Run analysis functions
    compare_ratings(before_df, after_df, pivot_date, comparative_dir)
    compare_review_volume(before_df, after_df, pivot_date, comparative_dir)
    compare_sentiment(before_df, after_df, pivot_date, comparative_dir)
    compare_keywords(before_df, after_df, pivot_date, comparative_dir)
    
    # Generate and save summary report
    generate_summary_report(before_df, after_df, pivot_date, comparative_dir)
    
    return comparative_dir


def compare_ratings(before_df, after_df, pivot_date, output_dir):
    """Compare rating distributions before and after pivot date"""
    # Calculate rating statistics
    before_avg = before_df['rating'].mean()
    after_avg = after_df['rating'].mean()
    change_pct = ((after_avg - before_avg) / before_avg) * 100
    
    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare rating data
    rating_labels = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
    before_counts = [len(before_df[before_df['rating'] == i]) for i in range(1, 6)]
    after_counts = [len(after_df[after_df['rating'] == i]) for i in range(1, 6)]
    
    # Convert to percentages
    before_pct = [count / len(before_df) * 100 for count in before_counts]
    after_pct = [count / len(after_df) * 100 for count in after_counts]
    
    x = np.arange(len(rating_labels))
    width = 0.35
    
    # Plot bars
    rects1 = ax.bar(x - width/2, before_pct, width, label=f'Before {pivot_date}', color='skyblue')
    rects2 = ax.bar(x + width/2, after_pct, width, label=f'After {pivot_date}', color='orange')
    
    # Add text
    title = f'Rating Distribution Comparison (Avg: {before_avg:.2f} → {after_avg:.2f}, {change_pct:+.1f}%)'
    ax.set_title(title)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Percentage of Reviews')
    ax.set_xticks(x)
    ax.set_xticklabels(rating_labels)
    ax.legend()
    
    # Add value labels on bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'rating_comparison.png'), dpi=300)
    plt.close(fig)
    
    # Create a second visualization showing before/after average ratings
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(['Before', 'After'], [before_avg, after_avg], color=['skyblue', 'orange'])
    
    # Add annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{height:.2f}', ha='center', va='bottom')
    
    ax.set_ylim(0, 5.5)
    ax.set_title(f'Average Rating Change ({change_pct:+.1f}%)')
    ax.set_ylabel('Average Rating')
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'avg_rating_change.png'), dpi=300)
    plt.close(fig)
    
    # Create cumulative average rating chart comparing before/after
    create_cumulative_rating_comparison(before_df, after_df, pivot_date, output_dir)


def create_cumulative_rating_comparison(before_df, after_df, pivot_date, output_dir):
    """Create a chart showing cumulative average rating over time with a marker at the pivot date"""
    # Combine dataframes and sort by time
    before_df = before_df.copy()
    after_df = after_df.copy()
    
    # Add a marker column to identify before/after periods
    before_df['period'] = 'before'
    after_df['period'] = 'after'
    
    # Combine and sort
    combined_df = pd.concat([before_df, after_df])
    combined_df = combined_df.sort_values('create_time')
    
    # Create monthly markers
    monthly_markers = pd.date_range(
        start=combined_df['create_time'].min(),
        end=combined_df['create_time'].max(),
        freq='ME'
    )
    
    # Calculate cumulative average at each month
    cumulative_ratings = []
    dates = []
    is_after_pivot = []
    
    for date in monthly_markers:
        # Select all reviews up to this date
        reviews_up_to_date = combined_df[combined_df['create_time'] <= date]
        
        if len(reviews_up_to_date) > 0:
            avg_rating = reviews_up_to_date['rating'].mean()
            cumulative_ratings.append(avg_rating)
            dates.append(date)
            is_after_pivot.append(date >= pd.to_datetime(pivot_date))
    
    # Find pivot date index
    pivot_idx = dates.index(min([d for d, after in zip(dates, is_after_pivot) if after], default=dates[-1]))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot with different colors before and after pivot date
    ax.plot(dates[:pivot_idx+1], cumulative_ratings[:pivot_idx+1], 
           marker='o', linestyle='-', color='skyblue', linewidth=2, markersize=4,
           label=f'Before {pivot_date}')
    
    if pivot_idx < len(dates) - 1:  # If there are points after pivot
        ax.plot(dates[pivot_idx:], cumulative_ratings[pivot_idx:], 
               marker='o', linestyle='-', color='orange', linewidth=2, markersize=4,
               label=f'After {pivot_date}')
    
    # Add vertical line at pivot date
    pivot_date_dt = pd.to_datetime(pivot_date)
    if pivot_date_dt >= dates[0] and pivot_date_dt <= dates[-1]:
        ax.axvline(x=pivot_date_dt, color='red', linestyle='--', alpha=0.7)
        ax.text(pivot_date_dt, min(cumulative_ratings) - 0.1, f'Pivot: {pivot_date}', 
               rotation=90, va='bottom', ha='right', color='red')
    
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
    ax.legend()
    
    # Annotate start and current points
    ax.annotate(f'Start: {cumulative_ratings[0]:.2f}',
               xy=(dates[0], cumulative_ratings[0]),
               xytext=(10, 0),
               textcoords='offset points',
               va='center')
    
    ax.annotate(f'Current: {cumulative_ratings[-1]:.2f}',
               xy=(dates[-1], cumulative_ratings[-1]),
               xytext=(-10, 0),
               textcoords='offset points',
               ha='right',
               va='center')
    
    # If there is a visible change at pivot, annotate it
    if pivot_idx > 0 and pivot_idx < len(cumulative_ratings) - 1:
        before_pivot_avg = cumulative_ratings[pivot_idx]
        after_latest_avg = cumulative_ratings[-1]
        change = after_latest_avg - before_pivot_avg
        
        if abs(change) > 0.01:  # Only annotate if there's a noticeable change
            ax.annotate(f'Change: {change:+.2f}',
                       xy=(dates[-1], (before_pivot_avg + after_latest_avg) / 2),
                       xytext=(30, 0),
                       textcoords='offset points',
                       arrowprops=dict(arrowstyle='<->', color='gray'),
                       va='center')
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cumulative_rating_comparison.png'), dpi=300)
    plt.close(fig)


def compare_review_volume(before_df, after_df, pivot_date, output_dir):
    """Compare review volume metrics before and after pivot date"""
    # Calculate time spans
    before_span = (before_df['create_time'].max() - before_df['create_time'].min()).days
    after_span = (after_df['create_time'].max() - after_df['create_time'].min()).days
    
    # Handle edge case of 0 days
    before_span = max(1, before_span)
    after_span = max(1, after_span)
    
    # Calculate average reviews per day
    before_per_day = len(before_df) / before_span
    after_per_day = len(after_df) / after_span
    change_pct = ((after_per_day - before_per_day) / before_per_day) * 100
    
    # Create bar chart comparing review frequency
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(['Before', 'After'], [before_per_day, after_per_day], color=['skyblue', 'orange'])
    
    # Add annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.2f} reviews/day', ha='center', va='bottom')
    
    ax.set_title(f'Review Frequency Change ({change_pct:+.1f}%)')
    ax.set_ylabel('Average Reviews per Day')
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'review_frequency.png'), dpi=300)
    plt.close(fig)
    
    # Create a chart showing review counts over time
    # Group by month and count reviews
    before_monthly = before_df.groupby(pd.Grouper(key='create_time', freq='ME')).size()
    after_monthly = after_df.groupby(pd.Grouper(key='create_time', freq='ME')).size()
    
    # Create combined monthly series with a marker for the pivot date
    all_monthly = pd.concat([before_monthly, after_monthly])
    all_monthly = all_monthly[~all_monthly.index.duplicated()]
    all_monthly = all_monthly.sort_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Convert to numpy arrays for plotting
    dates = all_monthly.index
    counts = all_monthly.values
    x_positions = np.arange(len(dates))
    
    # Find the position that corresponds to the pivot date
    pivot_idx = 0
    for i, date in enumerate(dates):
        if date >= pd.to_datetime(pivot_date):
            pivot_idx = i
            break
    
    # Plot the data with different colors before and after
    ax.bar(x_positions[:pivot_idx], counts[:pivot_idx], color='skyblue', label=f'Before {pivot_date}')
    ax.bar(x_positions[pivot_idx:], counts[pivot_idx:], color='orange', label=f'After {pivot_date}')
    
    # Add vertical line at pivot date
    ax.axvline(x=pivot_idx-0.5, color='red', linestyle='--', alpha=0.7)
    ax.text(pivot_idx, max(counts)*0.9, f'Pivot: {pivot_date}', 
           rotation=90, va='top', ha='right', color='red')
    
    # Format the x-axis with month labels
    month_labels = [d.strftime('%b %Y') for d in dates]
    
    # Show only a subset of labels if there are many
    if len(month_labels) > 12:
        step = max(1, len(month_labels) // 12)
        visible_positions = x_positions[::step]
        visible_labels = [month_labels[i] for i in range(0, len(month_labels), step)]
        ax.set_xticks(visible_positions)
        ax.set_xticklabels(visible_labels, rotation=45)
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(month_labels, rotation=45)
    
    ax.set_title('Review Volume Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Reviews')
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'volume_over_time.png'), dpi=300)
    plt.close(fig)


def compare_sentiment(before_df, after_df, pivot_date, output_dir):
    """Compare sentiment before and after pivot date if sentiment data is available"""
    # Check if sentiment data is available
    if 'sentiment_compound' not in before_df.columns or 'sentiment_compound' not in after_df.columns:
        print("Sentiment data not available for comparative analysis")
        return
    
    # Calculate average sentiment
    before_sentiment = before_df['sentiment_compound'].mean()
    after_sentiment = after_df['sentiment_compound'].mean()
    
    # Handle cases where sentiment is NaN
    before_sentiment = 0 if pd.isna(before_sentiment) else before_sentiment
    after_sentiment = 0 if pd.isna(after_sentiment) else after_sentiment
    
    # Calculate change
    change_abs = after_sentiment - before_sentiment
    change_pct = (change_abs / (abs(before_sentiment) + 1e-10)) * 100  # Avoid division by zero
    
    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(['Before', 'After'], [before_sentiment, after_sentiment], 
                color=['skyblue', 'orange'])
    
    # Add color gradient based on sentiment (-1 to 1 range)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        color = 'green' if height > 0 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2., 
               height + 0.05 if height > 0 else height - 0.1,
               f'{height:.3f}', ha='center', va='bottom', color=color)
    
    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)
    ax.axhline(y=-0.5, color='r', linestyle='--', alpha=0.3)
    
    ax.set_title(f'Sentiment Change ({change_abs:+.3f}, {change_pct:+.1f}%)')
    ax.set_ylabel('Average Sentiment Score (-1 to 1)')
    ax.set_ylim(-1.1, 1.1)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sentiment_change.png'), dpi=300)
    plt.close(fig)


def compare_keywords(before_df, after_df, pivot_date, output_dir):
    """Compare keywords in reviews before and after pivot date"""
    # Check if there are reviews with comments
    if not before_df['has_comment'].any() or not after_df['has_comment'].any():
        print("Not enough reviews with comments for keyword comparison")
        return
    
    # Get reviews with comments
    before_comments = before_df[before_df['has_comment']]['comment'].fillna('')
    after_comments = after_df[after_df['has_comment']]['comment'].fillna('')
    
    # Combine comments
    before_text = ' '.join(before_comments)
    after_text = ' '.join(after_comments)
    
    # Configure stop words (common words to exclude)
    stop_words = set(stopwords.words('english'))
    stop_words.update(['translated', 'google', 'original', 'also', 'really', 'one', 'well'])
    
    try:
        # Generate word clouds
        before_wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            stopwords=stop_words,
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(before_text)
        
        after_wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            stopwords=stop_words,
            max_words=100,
            contour_width=3,
            contour_color='darkorange'
        ).generate(after_text)
        
        # Create side-by-side word clouds
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.imshow(before_wordcloud, interpolation='bilinear')
        ax1.set_title(f'Keywords Before {pivot_date}')
        ax1.axis('off')
        
        ax2.imshow(after_wordcloud, interpolation='bilinear')
        ax2.set_title(f'Keywords After {pivot_date}')
        ax2.axis('off')
        
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'keyword_comparison.png'), dpi=300)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error in keyword comparison: {str(e)}")


def generate_summary_report(before_df, after_df, pivot_date, output_dir):
    """Generate a JSON summary report of the comparative analysis"""
    # Calculate metrics
    before_avg_rating = float(before_df['rating'].mean())
    after_avg_rating = float(after_df['rating'].mean())
    rating_change = after_avg_rating - before_avg_rating
    rating_change_pct = (rating_change / before_avg_rating) * 100
    
    before_span = max(1, (before_df['create_time'].max() - before_df['create_time'].min()).days)
    after_span = max(1, (after_df['create_time'].max() - after_df['create_time'].min()).days)
    
    before_reviews_per_day = len(before_df) / before_span
    after_reviews_per_day = len(after_df) / after_span
    volume_change_pct = ((after_reviews_per_day - before_reviews_per_day) / before_reviews_per_day) * 100
    
    before_with_comments_pct = (before_df['has_comment'].sum() / len(before_df)) * 100
    after_with_comments_pct = (after_df['has_comment'].sum() / len(after_df)) * 100
    
    # Create summary dictionary
    summary = {
        "pivot_date": pivot_date,
        "before_period": {
            "start_date": before_df['create_time'].min().strftime("%Y-%m-%d"),
            "end_date": before_df['create_time'].max().strftime("%Y-%m-%d"),
            "num_reviews": int(len(before_df)),
            "avg_rating": float(before_avg_rating),
            "reviews_per_day": float(before_reviews_per_day),
            "with_comments_pct": float(before_with_comments_pct)
        },
        "after_period": {
            "start_date": after_df['create_time'].min().strftime("%Y-%m-%d"),
            "end_date": after_df['create_time'].max().strftime("%Y-%m-%d"),
            "num_reviews": int(len(after_df)),
            "avg_rating": float(after_avg_rating),
            "reviews_per_day": float(after_reviews_per_day),
            "with_comments_pct": float(after_with_comments_pct)
        },
        "changes": {
            "avg_rating_change": float(rating_change),
            "avg_rating_change_pct": float(rating_change_pct),
            "reviews_per_day_change_pct": float(volume_change_pct)
        }
    }
    
    # Add sentiment metrics if available
    if 'sentiment_compound' in before_df.columns and 'sentiment_compound' in after_df.columns:
        before_sentiment = before_df['sentiment_compound'].mean()
        after_sentiment = after_df['sentiment_compound'].mean()
        
        # Handle NaN values
        before_sentiment = 0 if pd.isna(before_sentiment) else before_sentiment
        after_sentiment = 0 if pd.isna(after_sentiment) else after_sentiment
        
        sentiment_change = after_sentiment - before_sentiment
        sentiment_change_pct = (sentiment_change / (abs(before_sentiment) + 1e-10)) * 100
        
        summary["before_period"]["avg_sentiment"] = float(before_sentiment)
        summary["after_period"]["avg_sentiment"] = float(after_sentiment)
        summary["changes"]["sentiment_change"] = float(sentiment_change)
        summary["changes"]["sentiment_change_pct"] = float(sentiment_change_pct)
    
    # Save summary to file
    with open(os.path.join(output_dir, 'comparative_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    return summary 