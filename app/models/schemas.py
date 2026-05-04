
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import time

class EntityType(str, Enum):
    LIFE_EVENT = "life_event"
    LOCATION = "location"
    PERSON = "person"
    PREFERENCE = "preference"
    PLAN = "plan"
    OPINION = "opinion"
    FACT = "fact"
    OTHER = "other" 

class TopicMetadata(BaseModel):
    """Metadata attached to a topic entry."""
    entity_type: EntityType = EntityType.OTHER
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
 
 
class TopicLink(BaseModel):
    """A link between two related topics."""
    topic_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    relationship: Optional[str] = None  # e.g., "related_event", "same_entity"
 
 
class Topic(BaseModel):
    topic_id: str = ""  
    user_id: str
    session_id: str
    content: str  # e.g., "moving to Denver in March"
    embedding: list[float] = Field(default_factory=list)    # vector that goes into ChromaDB
    timestamps: list[float] = Field(default_factory=list)
    access_count: int = 0
    decay_param: float = 0.5
    last_activation: float = 0.0
    created_at: float = Field(default_factory=time.time)
    metadata: TopicMetadata = Field(default_factory=TopicMetadata)
    links: list[TopicLink] = Field(default_factory=list)

# The three components of ACT-R activation
class ActivationBreakdown(BaseModel):
    base_level: float = 0.0       
    spreading: float = 0.0       
    noise: float = 0.0           

# The overall activation result for a topic, including whether it exceeds the retrieval threshold
class ActivationResult(BaseModel):
    topic_id: str
    total_activation: float = 0.0
    breakdown: ActivationBreakdown = Field(default_factory=ActivationBreakdown)
    above_threshold: bool = False

# Context chunks with its embedding and weight
class ContextChunk(BaseModel):
    embedding: list[float]
    weight: float = 1.0
 
# The request model for calculating activation, including all necessary inputs and parameters to do the equation
class ActivationRequest(BaseModel):
    topic_id: str
    timestamps: list[float]
    current_time: float = Field(default_factory=time.time)
    decay: float = 0.5
    query_embedding: list[float]
    topic_embedding: list[float]
    context_chunks: list[ContextChunk] = Field(default_factory=list)
    noise_sigma: float = 0.25

# Retrival Pipeline

# Request for a query from Chroma
class RetrieveRequest(BaseModel):
    session_id: str
    query_embedding: list[float]
    active_users: list[str]
    current_time: float = Field(default_factory=time.time)
    retrieval_threshold: float = -1.0
    max_topics: int = 10
 
# The retrived topic and values for the activation equation
class RetrievedTopic(BaseModel):
    topic_id: str
    user_id: str
    content: str
    activation_breakdown: ActivationBreakdown
    above_threshold: bool
 
# Response from pgvector with numbers and topics
class RetrieveResponse(BaseModel):
    retrieved: list[RetrievedTopic] = Field(default_factory=list)
    pruned_below_threshold: int = 0
    total_candidates_scored: int = 0

# Storage Pipeline
class StoreRequest(BaseModel):
    user_id: str
    session_id: str
    content: str
    embedding: list[float] = Field(default_factory=list)
    entity_type: EntityType = EntityType.OTHER
    confidence: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    links: list[TopicLink] = Field(default_factory=list)

# Response after attempting to store a topic, including whether it was stored and how many links were created
class StoreResponse(BaseModel):
    topic_id: str
    stored: bool = True
    links_created: int = 0
    user_topic_count: int = 0

# Reinforcement Learning Feedback
class ReinforceRequest(BaseModel):
    topic_id: str
    access_timestamp: float = Field(default_factory=time.time)

# Response to recomputed activation
class ReinforceResponse(BaseModel):
    topic_id: str
    access_count: int
    timestamps: list[float]
    new_activation: float

class ExtractRequest(BaseModel):
    user_id: str
    message: str
    session_id: str
    timestamp: float = Field(default_factory=time.time)
 
 
class ExtractedTopic(BaseModel):
    content: str
    embedding: list[float] = Field(default_factory=list)
    entity_type: EntityType
    confidence: float
 
 
class ExtractResponse(BaseModel):
    topics: list[ExtractedTopic] = Field(default_factory=list)
    linked_existing_topics: list[TopicLink] = Field(default_factory=list)

# Vars for ACT-R
class ControlParams(BaseModel):
    decay: float = 0.5
    speech_in_weight: float = 0.6
    retrieval_threshold: float = -1.0
    noise_sigma: float = 0.25
    max_topics_per_user: int = 200
    context_token_budget: int = 4096
 
# User profile that is stored in Mongo
class UserProfile(BaseModel):
    user_id: str
    active_session: Optional[str] = None
    topic_ids: list[str] = Field(default_factory=list)
    retrieval_threshold: float = -1.0
    noise_param: float = 0.25
    speech_in_weight: float = 0.6

class RecentMessage(BaseModel):
    """A recent message from the conversation history."""
    role: str = "user"
    user_id: str
    content: str
    timestamp: float = Field(default_factory=time.time)
 

class ContextBlock(BaseModel):
    role: str = "context"
    content: str
    token_count: int = 0
    sources: list[str] = Field(default_factory=list)  # topic_ids

class ContextWindow(BaseModel):
    system_prompt: str = ""
    context_blocks: list[ContextBlock] = Field(default_factory=list)
    recent_messages: list[RecentMessage] = Field(default_factory=list)
    total_tokens: int = 0
    budget_remaining: int = 0

class ContextBuildRequest(BaseModel):
    session_id: str
    current_message: str
    sender_user_id: str
    active_users: list[str]
    max_tokens: int = 4096