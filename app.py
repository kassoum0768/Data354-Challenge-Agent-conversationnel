import chainlit as cl
from langchain.vectorstores import FAISS
from langchain_cohere import ChatCohere
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv("config.env")

# Initialiser Cohere
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_model = ChatCohere(model="command-r-plus", cohere_api_key=cohere_api_key)

# Classe personnalisée pour les embeddings
class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text):
        return self.model.encode(text, show_progress_bar=False)

# Charger les embeddings et l'index FAISS
embeddings = SentenceTransformersEmbeddings()
vector_store = FAISS.load_local("faiss_index_sentence_transformers", embeddings, allow_dangerous_deserialization=True)

# Fonction pour récupérer les documents pertinents
def get_relevant_documents(query, top_k=3):
    return vector_store.similarity_search(query, k=top_k)

# Fonction pour former le contexte à partir des documents
def form_context_from_documents(query, top_k=1):
    relevant_docs = get_relevant_documents(query, top_k=top_k)
    if not relevant_docs:
        return None, None
    
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    return context, relevant_docs

# Fonction pour générer une réponse
def generate_response(user_question):
    context, relevant_docs = form_context_from_documents(user_question)
    if not context:
        return "Je n'ai pas trouvé suffisamment d'informations pertinentes pour répondre à votre question."

    prompt = f"Contexte:\n{context}\n\nQuestion: {user_question}\n\nRéponse:"
    response = cohere_model.invoke(prompt)
    answer = response.content if response else "Je n'ai pas pu générer de réponse."

    # Ajouter le lien de l'article pertinent
    article_link = relevant_docs[0].metadata.get("url") if relevant_docs else None
    if article_link:
        answer += f"\n\n🔗 [Voir plus]({article_link})"

    return answer

# Message d'accueil affiché dès l'ouverture du chat
WELCOME_MESSAGE = ("Bonjour et bienvenue ! \U0001F44B J'ai été développé par KONE ZANA KASSOUM pour vous fournir des réponses précises sur les articles "
                   "publiés entre le 22 janvier 2025 et le 28 janvier 2025 sur le site de l'Agence Ecofin. "
                   "Posez-moi vos questions sur l’actualité économique, financière et sectorielle. "
                   "Comment puis-je vous aider aujourd’hui ?")

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=WELCOME_MESSAGE).send()

# Gestionnaire d'événements pour Chainlit
@cl.on_message
async def on_message(message: cl.Message):
    user_question = message.content
    response = generate_response(user_question)
    await cl.Message(content=response).send()
