import chromadb
from chromadb.config import Settings
from nltk import collections
from app.models.schemas import Topic
from app.config import settings

# NOT ACID
class chroma_client:

    # Initializes a collection for a user if it doesn't exist, otherwise retrieves it
    def init_collection(self, user_id: str):
        collection = self.client.get_or_create_collection(name=f"user_{user_id}")
        return collection

    # Upserts a topic into the user's collection, creating it if necessary
    def upsert_topic(self, topic_id: str, embedding: list[float], metadata: dict):
        init = self.init_collection(metadata["user_id"])
        init.upsert(ids=[topic_id], embeddings=[embedding], metadatas=[metadata])
        
    def query_similar(self, query_embedding: list[float], n_results: int, user_ids: list[str]):
        # returns the top-n most similar topics across given users
        res = []
        for user_id in user_ids:
            collection = self.init_collection(user_id)
            results = collection.query(query_embeddings=[query_embedding], n_results=n_results, where={"user_id": {"$in": user_ids}})
            res.append(results)
        return res

    def delete_topic(self, topic_id: str):
        # removes a topic from the store
        pass

    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_path)