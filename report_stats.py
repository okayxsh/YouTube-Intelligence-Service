# /workspace/project/report_stats.py
# Run: python3 report_stats.py  (prints only, no files modified)
# Fill in the [INSERT ...] placeholders in the report with these numbers.

import pandas as pd

CSV_PATH = "youtube_comments_PROCESSED.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    print("=== DATASET OVERVIEW ===")
    print(f"Total comments:              {len(df):,}")
    print(f"Unique videos:               {df['video_id'].nunique()}")
    avg_len = df['clean_text'].astype(str).str.len().mean()
    med_len = df['clean_text'].astype(str).str.len().median()
    print(f"Avg comment length (chars):  {avg_len:.1f}")
    print(f"Median comment length:       {med_len:.0f}")

    print("\n=== COMMENTS PER VIDEO ===")
    print(df['video_id'].value_counts().to_string())

    print("\n=== SENTIMENT DISTRIBUTION ===")
    df['sentiment_label'] = df['sentiment'].apply(
        lambda p: 'positive' if p > 0.1 else ('negative' if p < -0.1 else 'neutral')
    )
    counts = df['sentiment_label'].value_counts()
    pcts = (df['sentiment_label'].value_counts(normalize=True) * 100).round(1)
    summary = pd.DataFrame({'count': counts, 'percent': pcts})
    print(summary.to_string())
    print(f"\nMean polarity: {df['sentiment'].mean():.3f}")
    print(f"Std  polarity: {df['sentiment'].std():.3f}")

    print("\n=== SENTIMENT PER VIDEO (mean polarity) ===")
    per_video = df.groupby('video_id')['sentiment'].agg(
        ['count', 'mean', 'std']
    ).round(3)
    print(per_video.to_string())

    print("\n=== TOP NAMED ENTITIES ===")
    all_entities = (
        df['entities'].dropna().astype(str)
          .str.split(', ').explode()
          .str.replace(r' \((ORG|PERSON|PRODUCT|GPE)\)$', '', regex=True)
    )
    print(f"Total entities extracted:    {len(all_entities):,}")
    print(f"Unique entities:             {all_entities.nunique():,}")
    print("\nTop 15 most-frequent entities:")
    print(all_entities.value_counts().head(15).to_string())

    print("\n=== SAMPLE ROWS PER SENTIMENT CLASS (3 each) ===")
    for label in ['positive', 'neutral', 'negative']:
        sample = df[df['sentiment_label'] == label].head(3)
        print(f"\n--- {label.upper()} ---")
        for _, row in sample.iterrows():
            text = str(row['clean_text'])
            if len(text) > 80:
                text = text[:80] + "..."
            print(f"  [{row['sentiment']:+.2f}] {text}")
            print(f"           entities: {row['entities']}")


if __name__ == "__main__":
    main()
