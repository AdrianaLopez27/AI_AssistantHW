import os
import pandas as pd
import openai
from sklearn.neighbors import NearestNeighbors
from tkinter import *
from tkinter import scrolledtext
import dotenv
import chromadb
from chromadb.config import Settings
####
dotenv.load_dotenv()



client = openai.AzureOpenAI(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_KEY'],
    api_version="2023-10-01-preview",
)


def create_embeddings(text, model=os.environ['AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT']):
    embeddings = client.embeddings.create(input=text, model=model).data[0].embedding
    return embeddings

chroma_client = chromadb.PersistentClient(path="./DB/")

# Nombre de la colección
collection_name = "embeddings_collection"

collection = chroma_client.get_collection(name=collection_name)


def get_response(question):
    #query_vector = create_embeddings(question)
    #distances, indices = nbrs.kneighbors([query_vector])
    #history = [flattened_df['chunks'].iloc[index] for index in indices[0]]
    #history.append(question)
    ####

    # Convert the question to a query vector
    query_vector = create_embeddings(question)

    # Buscar los documentos más similares en Chroma
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=4  # Número de vecinos más cercanos a recuperar
    )
    # Añadir los documentos a la consulta para proporcionar contexto
    history = []
    for metadata in results["metadatas"][0]:
        history.append(metadata["chunk"])

    # Combinar la historia y la entrada del usuario
    history.append(question)



    messages = [
        {"role": "system", "content": f"You are an AI assistant that helps with questions about a person. You will answer the question instead of the person and will deliver the answer as it was a gossip, use this information to answer the question: {history[:-1]}."},
        {"role": "user", "content": history[-1]}
    ]

    response = openai.chat.completions.create(
        model=os.environ['AZURE_OPENAI_COMPLETIONS_DEPLOYMENT'],
        temperature=0.5,
        max_tokens=800,
        messages=messages
    )

    #return response.choices[0].message['content']
    return response.choices[0].message.content




# Crear la interfaz gráfica con Tkinter
def ask_question():
    question = question_entry.get()
    answer = get_response(question)
    response_text.delete('1.0', END)
    response_text.insert(INSERT, answer)

root = Tk()
root.title("AI Assistant")
root.geometry("600x400")

question_label = Label(root, text="Ask a question:")
question_label.pack()



response_text = scrolledtext.ScrolledText(root, width=70, height=20)
response_text.pack()
question_entry = Entry(root, width=150)
question_entry.pack()

ask_button = Button(root, text="Send", command=ask_question)
ask_button.pack()
root.mainloop()