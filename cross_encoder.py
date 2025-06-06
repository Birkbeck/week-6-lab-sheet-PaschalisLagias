from sentence_transformers import CrossEncoder

with open('data/sentences.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

model = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall")
scores = model.predict([
    [sentences[0], sentences[0]],
    [sentences[2], sentences[3]],
])

# Convert scores to labels
label_mapping = ["contradiction", "entailment", "neutral"]
labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
print(labels)
