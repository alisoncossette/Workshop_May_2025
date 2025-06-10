## Neo4j GraphRAG Retriever Usage Guide

This project demonstrates how to use Neo4j and GraphRAG for advanced retrieval-augmented generation (RAG) with asset manager and cybersecurity risk data.

### Key Scripts & Notebooks

- **PDF Loader Script for Neo4j GraphRAG.py**: Loads and indexes PDF content into Neo4j, including creation of the vector index (`chunkEmbeddings`).
- **Retreivers_notebook.ipynb**: Interactive notebook for running and debugging retrieval queries, with detailed explanations and diagnostics for each step.

### Setup Steps

1. **Neo4j AuraDB Setup**
   - Use the provided connection URI and credentials (see `.env`).
   - Ensure the `chunkEmbeddings` vector index is created. The loader script can do this automatically.

2. **Data Ingestion**
   - Run the loader to ingest PDFs and create `File`, `Chunk`, and related nodes.
   - Upload any additional structured data as needed.

3. **Retrieval & Search**
   - Use the notebook to:
     - Run natural language and Cypher-augmented retrievals.
     - Debug which chunks are being returned by vector search.
     - Explore and validate graph traversals (e.g., chunk-to-company, company-to-risk, manager queries).

### Troubleshooting

- If no results are returned by a query, check:
  - The vector search is returning relevant `Chunk` nodes (see notebook diagnostics).
  - The Cypher pattern matches your schema and data.
  - The vector index is online and populated (`CALL db.indexes()` in Neo4j Browser).
- For Text2CypherRetriever, ensure that generated Cypher does not include markdown code blocks (strip triple backticks if needed).

---

For full explanations of each retrieval pattern and diagnostics, see the notebook.

## Next Steps

- Continue modularizing and testing scripts as you add new retrieval or ingestion patterns.

---

For further troubleshooting or new feature requests, please describe the issue or desired workflow!
