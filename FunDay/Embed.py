import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
df=pd.read_csv('intro.csv')
# Example corpus of text data
corpus = df['Self Introduction']
# Preprocess the text data and tokenize into words
processed_corpus = [simple_preprocess(text) for text in corpus]

# Train or load a Word2Vec model
model = gensim.models.Word2Vec(sentences=processed_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Generate sentence embeddings by averaging word embeddings
sentence_embeddings = [np.mean([model.wv[token] for token in sentence], axis=0) for sentence in processed_corpus]

# Convert the sentence embeddings list into a numpy array
sentence_embeddings = np.array(sentence_embeddings)
import umap
# Initialize UMAP model with desired parameters
umap_model = umap.UMAP(n_neighbors=5, n_components=2, metric='cosine')

# Fit the UMAP model to the sentence embeddings
umap_embeddings = umap_model.fit_transform(sentence_embeddings)

# Now, umap_embeddings contains the 2-dimensional representations of the sentence embeddings
print(umap_embeddings)
