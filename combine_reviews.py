#!/usr/bin/env python3
"""
Google Business Reviews Combiner
This script combines reviews from multiple JSON files into a single consolidated file.
"""

import os
import json
import glob
import argparse
from tqdm import tqdm


def load_and_combine_reviews(input_dir="input", output_file="reviews-combined.json"):
    """
    Load reviews from all JSON files in the input directory and combine them
    into a single JSON file, removing duplicates.
    """
    # Find all review files
    review_files = glob.glob(os.path.join(input_dir, "reviews*.json"))
    
    if not review_files:
        print(f"No review files found in {input_dir}")
        return False
    
    print(f"Found {len(review_files)} review files")
    
    # Store all reviews in a dictionary with name as key to avoid duplicates
    all_reviews = {}
    
    # Process each file
    for file_path in tqdm(review_files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'reviews' in data:
                # Add each review to the dictionary using its name as key
                for review in data['reviews']:
                    if 'name' in review:
                        review_name = review['name']
                        
                        # If review already exists, keep the one with the latest update time
                        if review_name in all_reviews:
                            existing_time = all_reviews[review_name].get('updateTime', '')
                            new_time = review.get('updateTime', '')
                            
                            if new_time > existing_time:
                                all_reviews[review_name] = review
                        else:
                            all_reviews[review_name] = review
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Convert dictionary values back to a list
    combined_reviews = list(all_reviews.values())
    
    if not combined_reviews:
        print("No valid reviews found in the provided files")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save combined reviews to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"reviews": combined_reviews}, f, indent=2)
    
    print(f"Successfully combined {len(combined_reviews)} unique reviews into {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Combine Google Business reviews from multiple files')
    parser.add_argument('--input', '-i', default='input', help='Input directory containing review JSON files')
    parser.add_argument('--output', '-o', default='output/reviews-combined.json', help='Output file path')
    args = parser.parse_args()
    
    try:
        success = load_and_combine_reviews(args.input, args.output)
        return 0 if success else 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 