import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import spacy

class TopicExtractor:

    def setup(self):
        self.kw_model = KeyBERT()
        self.nlp = spacy.load("en_core_web_sm")

    def extract_topics(self, text, top_n=5):
        keywords = self.kw_model.extract_keywords(text, top_n=top_n)
        new_kw = [kw for kw, val in keywords if val > 0.3]

        doc = self.nlp(text)
        entities = {ent.text.lower(): ent.label_ for ent in doc.ents}

        final_mappings = []
        for kw in new_kw:
            entity_type = entities.get(kw.lower(), "other")
            final_mappings.append({
                "content": kw,
                "entity_type": entity_type
            })
        return final_mappings