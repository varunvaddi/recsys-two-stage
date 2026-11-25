import numpy as np


def embed_texts(texts, encoder, batch_size=128):
    vecs = encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def nearest_items(emb, item_id, k=5):
    sims = emb @ emb[item_id]
    sims[item_id] = -np.inf
    top = np.argsort(-sims)[:k]
    return top, sims[top]