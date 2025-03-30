# Google Business Reviews Analyzer

A Python tool for analyzing and visualizing Google Business reviews. This tool can process multiple review files, combine them, analyze trends, and generate visualizations.

## Features

- Load and process review data from Google Business reviews JSON files
- Combine reviews from multiple files into a single dataset
- Analyze review trends over time
- Calculate sentiment scores from review text
- Extract common topics and keywords using NLP techniques
- **Comparative analysis** before and after a specific date (great for measuring impact of changes)
- Generate visualizations including:
  - Rating distribution
  - Reviews by month
  - Average rating over time
  - Cumulative reviews over time
  - Cumulative average rating over time
  - Sentiment analysis
  - Word clouds
  - Keyword frequency

## Installation

1. Clone the repository:
```bash
git clone https://github.com/maxdusdal/gbp-review-analyzer.git
cd gbp-review-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Analysis

To analyze reviews in the `input` directory and save results to the `output` directory:

```bash
python review_analyzer.py
```

With custom input and output directories:

```bash
python review_analyzer.py --input path/to/reviews --output path/to/results
```

Skip advanced text analysis:

```bash
python review_analyzer.py --skip-text-analysis
```

### Comparative Analysis

To analyze how reviews changed before and after a specific date:

```bash
python review_analyzer.py --pivot-date 2022-12-01
```

This creates a `comparative_analysis` folder in your output directory with:
- Rating distribution comparison
- Review frequency comparison
- Review volume over time with a marker for the pivot date
- Cumulative average rating comparison before and after the pivot date
- Sentiment change analysis
- Keyword comparisons with before/after word clouds
- Detailed JSON summary of changes

### Combine Reviews

To combine reviews from multiple files:

```bash
python combine_reviews.py
```

With custom paths:

```bash
python combine_reviews.py --input path/to/reviews --output combined_reviews.json
```

## Input Data

The tool expects Google Business review data in JSON format. Each file should have a structure like:

```json
{
  "reviews": [
    {
      "reviewer": {
        "displayName": "Reviewer Name"
      },
      "starRating": "FIVE",
      "comment": "Review text content",
      "createTime": "2023-02-23T21:20:39.230840Z",
      "updateTime": "2023-02-23T21:20:39.230840Z",
      "name": "accounts/123456789/locations/987654321/reviews/abcdef123"
    },
    ...
  ]
}
```

## Output

The tool generates:

1. CSV file with processed review data
2. JSON summary of review statistics
3. Visualizations (PNG images) including:
   - Review counts by month
   - Average ratings over time
   - Rating distribution
   - Cumulative average rating over time
   - Word clouds for positive and negative reviews
   - Sentiment analysis charts
   - Topic models
   - Comparative analysis charts (when using --pivot-date)

## License

MIT License

## Contributing

Contributions, issues, and feature requests are welcome!

## Acknowledgments

- Uses NLTK for natural language processing
- Uses scikit-learn for topic modeling
- Matplotlib and Seaborn for visualizations 