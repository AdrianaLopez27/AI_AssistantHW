import os
import pandas as pd
import openai 
from sklearn.neighbors import NearestNeighbors
import dotenv
from tkinter import *
from tkinter import scrolledtext
import dotenv
import chromadb
from chromadb.config import Settings
  
# Cargar variables de entorno
dotenv.load_dotenv()

# Configuración de OpenAI
#openai.api_type = "azure"
#openai.api_key = os.getenv("AZURE_OPENAI_KEY")
#openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
#openai.api_version = "2023-07-01-preview"
#



client = openai.AzureOpenAI(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_KEY'],
    api_version="2023-10-01-preview",
)

def create_embeddings(text, model=os.environ['AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT']):
    embeddings = client.embeddings.create(input=text, model=model).data[0].embedding
    return embeddings

# Cargar y procesar datos
df = pd.DataFrame(columns=['path', 'text'])
data_paths = ["data/adriana-info.md"]

for path in data_paths:
    with open(path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    df = pd.concat([df, pd.DataFrame([{'path': path, 'text': file_content}])], ignore_index=True)
#

def split_text(text, max_length, min_length):
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) < max_length and len(' '.join(current_chunk)) > min_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

splitted_df = df.copy()
splitted_df['chunks'] = splitted_df['text'].apply(lambda x: split_text(x, 400, 300))
flattened_df = splitted_df.explode('chunks')


chroma_client = chromadb.PersistentClient(path="./DB/")

# Nombre de la colección
collection_name = "embeddings_collection"

# Verificar si la colección existe
collections = chroma_client.list_collections()
collection_names = [col.name for col in collections]

if collection_name not in collection_names:
    # Crear la colección si no existe
    chroma_client.create_collection(name=collection_name)

# Obtener la colección (ya sea recién creada o existente)
collection = chroma_client.get_collection(name=collection_name)
###########################


#embeddings = [create_embeddings(chunk) for chunk in flattened_df['chunks']]
embeddings = []
for chunk in flattened_df['chunks']:
    embeddings.append(create_embeddings(chunk))

flattened_df['embeddings'] = embeddings


# Resetear el índice
flattened_df = flattened_df.reset_index(drop=True)

# Insertar los embeddings en Chroma
for idx, row in flattened_df.iterrows():
    collection.add(
        ids=str(idx),  # Usar el índice como identificador único
        embeddings=row['embeddings'],
        metadatas={"chunk": row['chunks']}
    )

#nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(embeddings)