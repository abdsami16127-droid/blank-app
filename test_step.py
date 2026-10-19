import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("Starting test...", flush=True)

from src.embeddings import EmbeddingEngine
print("EmbeddingEngine imported", flush=True)

engine = EmbeddingEngine()
print("EmbeddingEngine instantiated", flush=True)

texts = [
    "Battery drains in less than 4 hours.",
    "Charge does not last through my commute.",
    "Phone dies quickly after charging.",
    "Package arrived two weeks late.",
    "Shipping tracking was delayed for days.",
]
embs = engine.embed_texts(texts)
print(f"Embeddings generated: {embs.shape}", flush=True)

from src.clustering import SemanticClusterer
print("SemanticClusterer imported", flush=True)

clusterer = SemanticClusterer(method="agglomerative", distance_threshold=0.45)
labels = clusterer.fit_predict(embs)
print(f"Labels: {labels}", flush=True)

from src.issue_extractor import IssueExtractor
print("IssueExtractor imported", flush=True)

import pandas as pd
df = pd.DataFrame({"review_text": texts, "review_id": range(5)})
extractor = IssueExtractor()
groups = extractor.extract_issues(df, embs, labels)
for g in groups:
    print(f"Issue group: {g.issue_name} ({g.review_count})", flush=True)

print("SUCCESS: ALL STEPS DONE", flush=True)
