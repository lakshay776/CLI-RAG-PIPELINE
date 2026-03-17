 # CLI RAG Pipeline

This is a simple RAG pipeline that can be used to answer questions from documents.

## Features

- Load documents from a folder
- Convert them into embeddings
- Store them in a vector database
- Retrieve relevant chunks during queries

## Installation

```bash
git clone https://github.com/lakshay776/CLI-RAG-PIPELINE.git
cd CLI-RAG-PIPELINE
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python ingestion.py
python query.py
```
