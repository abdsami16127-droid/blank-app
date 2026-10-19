import sys
import traceback

def main():
    try:
        print("1. Importing numpy, pandas, sklearn...")
        import numpy as np
        import pandas as pd
        import sklearn
        print(f"   numpy: {np.__version__}, pandas: {pd.__version__}, sklearn: {sklearn.__version__}")

        print("2. Importing src modules...")
        from src.clustering import SemanticClusterer
        from src.issue_extractor import IssueExtractor
        from src.embeddings import EmbeddingEngine
        print("   Modules imported successfully.")

        print("3. Generating embeddings...")
        engine = EmbeddingEngine()
        texts = [
            "Battery drains in less than 4 hours.",
            "Charge does not last through my commute.",
            "Phone dies quickly after charging.",
            "Package arrived two weeks late.",
            "Shipping tracking was delayed for days.",
        ]
        embs = engine.embed_texts(texts)
        print(f"   Embeddings created. Shape: {embs.shape}")

        print("4. Clustering...")
        clusterer = SemanticClusterer(method="agglomerative", distance_threshold=0.45)
        labels = clusterer.fit_predict(embs)
        print(f"   Cluster labels: {labels}")

        print("5. Extracting issues...")
        df = pd.DataFrame({"review_text": texts, "review_id": range(5)})
        extractor = IssueExtractor()
        groups = extractor.extract_issues(df, embs, labels)
        for g in groups:
            print(f"   * Issue: {g.issue_name} (Count: {g.review_count}) -> Quote: {g.representative_review[:45]}")

        print("\nSUCCESS: All Milestone 3 components verified!")
    except Exception as e:
        print(f"EXCEPTION CAUGHT: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
