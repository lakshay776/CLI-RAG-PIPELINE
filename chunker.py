# def chunk_text(text, source=None, chunk_size=500, overlap=100):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append({
#             "text": chunk,
#             "source": source
#         })
#         start += chunk_size - overlap
#     return chunks




from nltk.tokenize import sent_tokenize

def chunk_text(doc, chunk_size=500, overlap=100):
    text = doc["text"]
    source = doc["filename"] # loader.py uses 'filename'

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Check if adding the next sentence exceeds the chunk size
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "source": source
            })
            
            # Start new chunk with overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) > overlap and overlap_sentences:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s)
            
            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += len(sentence)

    # Append the last chunk
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "source": source
        })
        
    return chunks



 




if __name__ == "__main__":
    text="""

    Today I experimented with building a small CLI tool that reads documents
    and answers questions from them. The main idea was simple:

    -   Load files from a folder
    -   Convert them into embeddings
    -   Store them in a vector database
    -   Retrieve relevant chunks during queries

    Even a small prototype revealed how powerful retrieval‑augmented systems
    can be.

    """
    chunks=chunk_text(text)
    for i,chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")