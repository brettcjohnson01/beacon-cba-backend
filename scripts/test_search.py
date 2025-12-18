from app.retrieval.search import search

results = search("local hiring requirements", top_k=5)

for r in results:
    print("----")
    print(r["title"])
    print(r["text"][:300])
