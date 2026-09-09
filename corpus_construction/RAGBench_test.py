import os
import json
import re
import shutil
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

input_file = "ragbench_test.jsonl"
txt_dir = "txt_docs"
os.makedirs(txt_dir, exist_ok=True)
os.environ["LOKY_MAX_CPU_COUNT"] = "8"


with open(input_file, "r", encoding="utf-8") as infile:
    for i, line in enumerate(infile):
        data = json.loads(line.strip())
        documents = data["documents"]
        for j, doc in enumerate(documents):
            doc_title = "_".join(doc.strip().split()[:5])
            doc_title = re.sub(r'[\\/*?:"<>|]', '', doc_title)
            filename = f"doc_{i}_{j}_{doc_title}.txt"
            filepath = os.path.join(txt_dir, filename)
            with open(filepath, "w", encoding="utf-8") as txt_file:
                txt_file.write(doc)


txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
documents = []

for filename in txt_files:
    with open(os.path.join(txt_dir, filename), "r", encoding="utf-8") as file:
        documents.append(file.read())


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)


num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(embeddings)
cluster_labels = clustering_model.labels_

kw_model = KeyBERT()


ori_docs_dir = "Ori-docs"
os.makedirs(ori_docs_dir, exist_ok=True)


for cluster in range(num_clusters):
    cluster_docs = [embeddings[idx] for idx, label in enumerate(cluster_labels) if label == cluster]
    cluster_files = [txt_files[idx] for idx, label in enumerate(cluster_labels) if label == cluster]
    cluster_texts = [documents[idx] for idx, label in enumerate(cluster_labels) if label == cluster]


    sub_num_clusters = min(3, len(cluster_docs))
    sub_clustering_model = KMeans(n_clusters=sub_num_clusters)
    sub_clustering_model.fit(cluster_docs)
    sub_cluster_labels = sub_clustering_model.labels_


    cluster_keywords = []
    for text in cluster_texts:
        keywords = kw_model.extract_keywords(text, top_n=1)
        cluster_keywords.extend([kw[0] for kw in keywords])
    common_keyword = max(set(cluster_keywords), key=cluster_keywords.count)

    cluster_dir = os.path.join(ori_docs_dir, f"cluster_{cluster}_{common_keyword}")
    os.makedirs(cluster_dir, exist_ok=True)


    for sub_cluster in range(sub_num_clusters):
        sub_cluster_docs = [cluster_texts[idx] for idx, sub_label in enumerate(sub_cluster_labels) if
                            sub_label == sub_cluster]
        sub_cluster_files = [cluster_files[idx] for idx, sub_label in enumerate(sub_cluster_labels) if
                             sub_label == sub_cluster]


        sub_cluster_keywords = []
        for text in sub_cluster_docs:
            keywords = kw_model.extract_keywords(text, top_n=1)
            sub_cluster_keywords.extend([kw[0] for kw in keywords])
        sub_common_keyword = max(set(sub_cluster_keywords), key=sub_cluster_keywords.count)

        sub_cluster_dir = os.path.join(cluster_dir, f"subcluster_{sub_cluster}_{sub_common_keyword}")
        os.makedirs(sub_cluster_dir, exist_ok=True)


        for idx, file in enumerate(sub_cluster_files):
            src_file = os.path.join(txt_dir, file)
            shutil.copy(src_file, sub_cluster_dir)

print("Done")
