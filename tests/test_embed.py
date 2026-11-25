import numpy as np

from recsys import embed


class FakeEncoder:
    def encode(self, texts, batch_size, show_progress_bar,
               normalize_embeddings, convert_to_numpy):
        vecs = np.array([[len(t), t.count("a"), 1.0] for t in texts])
        if normalize_embeddings:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


def test_embed_texts_shape_dtype_and_unit_norm():
    emb = embed.embed_texts(["alpha", "beta", "gamma game"], FakeEncoder())
    assert emb.shape == (3, 3)
    assert emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_nearest_items_excludes_itself_and_ranks_by_similarity():
    emb = np.array([[1.0, 0.0], [0.9, 0.436], [0.0, 1.0]], dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    top, sims = embed.nearest_items(emb, 0, k=2)
    assert top.tolist() == [1, 2]
    assert sims[0] > sims[1]