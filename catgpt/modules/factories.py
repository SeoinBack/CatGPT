MODEL_TO_EMBEDDING_FN = {
    "albert": "model.albert.embeddings",
    "xlnet": "self.model.transformer.word_embedding",
    'GPT2Model': "self.model.transformer.wte",
}
