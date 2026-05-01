from keybert import KeyBERT
import spacy
from app.models.schemas import EntityType, ExtractedTopic

class TopicExtractor:

    def __init__(self):
        self.kw_model = KeyBERT()
        self.nlp = spacy.load("en_core_web_sm")

    def extract_topics(self, text, top_n=5):
        keywords = self.kw_model.extract_keywords(text, top_n=top_n)
        new_kw = [(kw, val) for kw, val in keywords if val > 0.3]

        doc = self.nlp(text)
        entities = {ent.text.lower(): ent.label_ for ent in doc.ents}

        label_map = {
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "FAC": EntityType.LOCATION,
            "PERSON": EntityType.PERSON,
            "EVENT": EntityType.LIFE_EVENT,
            "DATE": EntityType.PLAN,
            "ORG": EntityType.OTHER,
        }

        final_mappings = []
        for kw, score in new_kw:
            spacy_label = entities.get(kw.lower(), None)
            entity_type = label_map.get(spacy_label, EntityType.OTHER)
            final_mappings.append(ExtractedTopic(
                content=kw,
                entity_type=entity_type,
                confidence=score
            ))
        return final_mappings


#if __name__ == "__main__":
#    extractor = TopicExtractor()
#    topics = extractor.extract_topics("I found an apartment in Denver")
#    print(topics)