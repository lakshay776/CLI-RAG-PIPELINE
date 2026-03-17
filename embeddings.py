from config import client, embedding_model

def get_embedding(text):
    text = text.strip()
    response = client.models.embed_content(
        model=embedding_model,
        contents=[text]        # wrap in list — bare string causes "empty Part" error
    )
    return response.embeddings[0].values
