"""End-to-end verification script on sample reviews dataset."""

import sys
sys.stdout.reconfigure(line_buffering=True)
import onnxruntime

from src.data_loader import load_reviews_from_csv
from src.sentiment import SentimentAnalyzer
from src.embeddings import EmbeddingEngine
from src.clustering import SemanticClusterer
from src.issue_extractor import IssueExtractor
from src.config import SAMPLE_CSV_PATH

def run_verification():
    print("==================================================")
    print(" CUSTOMER REVIEW INTELLIGENCE SYSTEM - END-TO-END ")
    print("==================================================")
    
    print("\n1. Ingesting & Validating Sample Dataset...")
    df, report = load_reviews_from_csv(SAMPLE_CSV_PATH)
    print(f"   Loaded {report.valid_rows} valid reviews from '{SAMPLE_CSV_PATH.name}'.")

    print("\n2. Performing Sentiment Analysis...")
    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_dataframe(df)
    counts = df["sentiment"].value_counts().to_dict()
    print(f"   Sentiment Distribution -> Negative: {counts.get('Negative', 0)}, Neutral: {counts.get('Neutral', 0)}, Positive: {counts.get('Positive', 0)}")

    print("\n3. Generating Semantic Embeddings (FastEmbed BGE-Small ONNX)...")
    engine = EmbeddingEngine()
    embeddings = engine.embed_texts(df["review_text"].tolist())
    print(f"   Generated dense vectors with shape: {embeddings.shape}")

    print("\n4. Grouping Semantically Similar Issues...")
    clusterer = SemanticClusterer(method="agglomerative", distance_threshold=0.45)
    labels = clusterer.fit_predict(embeddings)
    metrics = SemanticClusterer.evaluate_clustering(embeddings, labels)
    print(f"   Clustered into {int(metrics['cluster_count'])} distinct issue topics (Silhouette Score: {metrics['silhouette_score']:.3f})")

    print("\n5. Extracting Issue Groups & Representative Medoid Quotes...")
    extractor = IssueExtractor()
    issue_groups = extractor.extract_issues(df, embeddings, labels)
    enriched_df = extractor.assign_issue_labels_to_dataframe(df, issue_groups, labels)

    print("\n================ TOP IDENTIFIED ISSUES ================")
    for idx, group in enumerate(issue_groups, 1):
        neg = group.sentiment_breakdown.get("Negative", 0)
        neu = group.sentiment_breakdown.get("Neutral", 0)
        pos = group.sentiment_breakdown.get("Positive", 0)
        print(f"\nIssue #{idx}: {group.issue_name}")
        print(f"  • Reviews: {group.review_count} ({group.percentage}%)")
        print(f"  • Sentiment Breakdown: {neg} Negative, {neu} Neutral, {pos} Positive")
        print(f"  • Central Medoid Quote: \"{group.representative_review}\"")
        if group.top_keywords:
            print(f"  • Key Terms: {', '.join(group.top_keywords[:4])}")

    print("\n=======================================================")
    print("   END-TO-END PIPELINE COMPLETED SUCCESSFULLY!         ")
    print("=======================================================")

if __name__ == "__main__":
    run_verification()
