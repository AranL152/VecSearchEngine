from sentence_transformers import SentenceTransformer
import json 
import faiss
import numpy as np

#open json data file 
with open("candles.json", "r", encoding="utf-8") as f:
    data = json.load(f)
#import hugging face transformer model, need stronger one 
model = SentenceTransformer("all-MiniLM-L6-v2")


sentences = []

#build sentences for each candle, concatenating name, description, scent notes, and details into one string
for item in data:

    sentence = (
    item["name"] + ". " +
    item["description"] + ". " +
    "Scent notes: " +
    ", ".join(item["scent_notes"]["top"]
              + item["scent_notes"]["heart"]
              + item["scent_notes"]["base"]) + ". " +
    "Free of: " + ", ".join(item["details"]["free_of"]) + ". " +
    "Vessel: " + item["details"]["vessel"] + ". " +
    "Packaging: " + item["details"]["packaging"] + "."
    )

    sentences.append(sentence)

#embed all in one batch, normalized for inner product
embeddings = model.encode(
    sentences, 
    convert_to_numpy=True,
    normalize_embeddings=True
)


embeddings = embeddings.astype(np.float32)

assert embeddings.ndim == 2
assert embeddings.dtype == np.float32  
n, d = embeddings.shape

#build faiss index for cosine similarity search
index = faiss.IndexFlatIP(d)
index.add(embeddings)

#interactive loop to query the model, will print out the name of the candle and the score for each result
while True:

    hardcodedQuery = input("Enter prompt: ")

    queryVector = model.encode(
        [hardcodedQuery],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    assert queryVector.ndim == 2
    assert queryVector.shape[1] == d
    assert queryVector.dtype == np.float32  
    scores, indices = index.search(queryVector, len(data))

    print(f"\nQuery: {hardcodedQuery}\n")

    for score, idx in zip(scores[0], indices[0]):
        candle = data[idx]
        print(f"{candle['name']:10} {score:.3f}")

