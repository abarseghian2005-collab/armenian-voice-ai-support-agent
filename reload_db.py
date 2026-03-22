import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

if os.path.exists("./db"):
    shutil.rmtree("./db")
    print("Old DB deleted")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection(name="bank_data")

bank_names = {
    "ameriabank": "Ամերիաբանկ",
    "ardshinbank": "Արդշինբանկ",
    "idbank": "ԱյԴի Բանկ"
}

topic_names = {
    "branches": "մասնաճյուղեր",
    "credits": "վարկեր",
    "deposits": "ավանդներ"
}

data_folder = "./data"
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        name_without_ext = filename.replace(".txt", "")
        bank_label = ""
        topic_label = ""
        for key, val in bank_names.items():
            if name_without_ext.startswith(key):
                bank_label = val
        for key, val in topic_names.items():
            if name_without_ext.endswith(key):
                topic_label = val

        header = f"Բանկ: {bank_label}\nԹեմա: {topic_label}\n\n"

        # Store the ENTIRE file as one chunk + smaller chunks
        # First add the full file as one big chunk for context
        full_chunk = header + text.strip()
        embedding = model.encode(full_chunk[:2000]).tolist()
        collection.add(
            documents=[full_chunk],
            embeddings=[embedding],
            ids=[f"{filename}_full"]
        )

        # Also add smaller chunks for specific questions
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        chunks = []
        current = ""
        for line in lines:
            if len(current) + len(line) < 1500:
                current += line + "\n"
            else:
                if current:
                    chunks.append(header + current.strip())
                current = line + "\n"
        if current:
            chunks.append(header + current.strip())

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk[:2000]).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{filename}_{i}"]
            )
        print(f"Loaded {len(chunks)+1} chunks from {filename}")

print(f"\nTotal chunks: {collection.count()}")