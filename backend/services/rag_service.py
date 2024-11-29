import os
import time
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.utils.config import Config
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.chains import RetrievalQAWithSourcesChain
import faiss
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import logging
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGService:
    """Service for RAG-based retrieval and chat"""
    
    def __init__(self, user_id: str = None, project_id: str = None):
        # Load config
        self.config = Config()
        if not self.config._config:  # If config is empty, load defaults
            self.config.reset()
        
        # Get OpenAI settings with fallbacks
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            openai_api_key = self.config.get('services', 'openai', 'api_key')
            
        openai_api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        if not openai_api_base:
            openai_api_base = self.config.get('services', 'openai', 'api_base', default='https://api.openai.com/v1')
        
        # Initialize embeddings with API key and base URL
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        # Get model settings with fallbacks
        model_name = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        if self.config._config:  # Only try to get from config if it exists
            model_name = self.config.get('services', 'rag', 'model', default='gpt-4-turbo-preview')
            
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base
        )
        self.logger = logging.getLogger(__name__)
        self.persist_dir = Path("tmp_content/vector_store")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_and_chunk_results(self, results_path: Path) -> List[Dict[str, Any]]:
        """
        Load analysis results and convert to documents with parsed metadata
        """
        try:
            with open(results_path) as f:
                data = json.load(f)
                
            documents = []
            
            # Parse metadata from context
            context_metadata = {}
            if 'context' in data:
                try:
                    if isinstance(data['context'], str):
                        context_metadata = eval(data['context'])  # Safely parse string dict
                    elif isinstance(data['context'], dict):
                        context_metadata = data['context']
                except:
                    pass

            # Handle frame results
            if 'results' in data:
                for frame in data['results']:
                    # Convert metadata to readable text
                    metadata_text = "\n".join([
                        f"Frame {frame.get('frame_number', 'unknown')}",
                        f"Timestamp: {frame.get('timestamp', 'unknown')}",
                        f"Analysis ID: {frame.get('analysis_id', 'unknown')}",
                        f"Confidence: {frame.get('confidence', 0.0)}",
                        f"Processing Type: {frame.get('processing_type', 'unknown')}"
                    ])
                    
                    # Combine metadata with content
                    frame_text = f"""
                    {metadata_text}
                    
                    Detections: {frame.get('detections', {})}
                    Scene Analysis: {frame.get('scene_analysis', {})}
                    """
                    documents.append({"text": frame_text, "metadata": frame})
                    
            # Handle scene analysis with metadata
            if 'scene_analysis' in data:
                metadata_text = "\n".join([
                    f"Video: {context_metadata.get('video_file', 'unknown')}",
                    f"Total Frames: {context_metadata.get('total_frames', 'unknown')}",
                    f"Duration: {context_metadata.get('duration', 'unknown')} seconds",
                    f"FPS: {context_metadata.get('fps', 'unknown')}",
                    f"Frames Analyzed: {context_metadata.get('frames_analyzed', [])}",
                    f"Analysis ID: {data.get('analysis_id', 'unknown')}",
                    f"Confidence: {data.get('confidence', 0.0)}"
                ])
                
                scene_text = f"""
                {metadata_text}
                
                Scene Analysis:
                {data['scene_analysis'].get('description', '')}
                """
                documents.append({"text": scene_text, "metadata": {
                    **data.get('scene_analysis', {}),
                    **context_metadata
                }})
                
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading results: {str(e)}")
            return []
            
    def create_knowledge_base(self, results_path: Path) -> Optional[FAISS]:
        """
        Create in-memory vector store from analysis results
        
        Args:
            results_path: Path to JSON results file
            
        Returns:
            FAISS vector store instance or None if creation fails
            
        Notes:
            - Creates a new in-memory vector store for each session
            - Handles complex data types by flattening them to strings
            - More efficient for temporary session-based RAG
        """
        try:
            documents = self._load_and_chunk_results(results_path)
            if not documents:
                return None
                
            # Create documents with parsed metadata
            processed_documents = []
            for doc in documents:
                # Parse and structure metadata
                metadata = {}
                for k, v in doc.get("metadata", {}).items():
                    if isinstance(v, (list, dict)):
                        # Convert complex types to formatted strings
                        if k == 'frame_numbers':
                            metadata[k] = [int(x) for x in eval(str(v))] if isinstance(v, str) else v
                        elif k == 'context' and isinstance(v, str):
                            try:
                                metadata[k] = eval(v)
                            except:
                                metadata[k] = v
                        else:
                            metadata[k] = str(v)
                    else:
                        metadata[k] = v

                # Add required metadata fields
                metadata['source'] = f"frame_{metadata.get('frame_number', 'unknown')}"
                metadata['timestamp'] = metadata.get('timestamp', time.time())
                metadata['analysis_id'] = metadata.get('analysis_id', f"analysis_{int(time.time())}")
                metadata['confidence'] = float(metadata.get('confidence', 0.8))
                metadata['processing_type'] = metadata.get('processing_type', 'scene_analysis')
                
                # Create document with parsed content and metadata
                processed_documents.append(
                    Document(
                        page_content=doc["text"].strip(),
                        metadata=metadata
                    )
                )
            
            # Split into chunks if we have documents
            if not processed_documents:
                self.logger.warning("No documents to process")
                return None
                
            chunks = self.text_splitter.split_documents(processed_documents)
            
            # Create FAISS vector store from documents
            vectordb = FAISS.from_documents(
                chunks,
                self.embeddings
            )
            
            # Add documents to store
            vectordb.add_documents(chunks)
            
            return vectordb
            
        except Exception as e:
            self.logger.error(f"Error creating knowledge base: {str(e)}")
            return None
            
    def get_retrieval_chain(self, vectordb: FAISS) -> RetrievalQAWithSourcesChain:
        """Create retrieval chain with custom prompt"""
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.7
            }
        )

        # Custom prompt template
        prompt_template = """Use the following pieces of context from video analysis results to answer the question.
        Each analysis result contains important metadata that MUST be referenced in your response:
        - Frame numbers: metadata['frame_numbers'] shows which frames were analyzed
        - Total frames: metadata['context'] contains total_frames and duration
        - Analysis details: metadata['analysis_id'] and metadata['confidence']
        - Technical info: metadata['context'] has fps and other details
        - Processing info: metadata['frame_pattern'] and metadata['processing_type']

        Context: {summaries}
        
        Question: {question}
        
        Guidelines for your response:
        1. Start with a metadata summary showing:
           - Number of frames analyzed
           - Which frame numbers were processed
           - Total video frames and duration
           - Analysis timestamp and confidence
        2. Then provide the scene description and analysis
        3. Always cite specific metadata values in your response
        4. Express confidence levels based on metadata['confidence']
        5. If any metadata is missing or conflicts, explicitly note it
        6. Never make up information beyond what's in the context and metadata
        """

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["summaries", "question"]
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT,
                "document_variable_name": "summaries",
                "verbose": True
            }
        )
        
        return chain
        
    def query_knowledge_base(self, query: str, chain: RetrievalQAWithSourcesChain) -> Dict:
        """Query the knowledge base with enhanced source tracking"""
        try:
            response = chain({"question": query})
            
            return {
                "answer": response["answer"],
                "sources": response["sources"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, "score", None)
                    }
                    for doc in response["source_documents"]
                ],
                "chat_history": self.memory.chat_memory.messages
            }
        except Exception as e:
            self.logger.error(f"Error querying knowledge base: {str(e)}")
            return {
                "error": str(e),
                "answer": "Failed to process query"
            }
            
    def _hash_results(self, results_path: Path) -> str:
        """Generate hash of results file for vector store identification"""
        with open(results_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash[:10]  # Use first 10 chars for readability
        
    def _is_store_current(self, store_path: Path, results_hash: str) -> bool:
        """Check if vector store exists and matches results file"""
        metadata_path = store_path / 'metadata.json'
        if not (store_path.exists() and metadata_path.exists()):
            return False
            
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata.get('results_hash') == results_hash
        except Exception as e:
            self.logger.warning(f"Failed to check store currency: {e}")
            return False
            
    def _save_store_metadata(self, store_path: Path, results_hash: str):
        """Save metadata about the vector store"""
        metadata = {
            'results_hash': results_hash,
            'created_at': datetime.now().isoformat(),
            'embeddings_model': 'text-embedding-ada-002'  # Default OpenAI embeddings model
        }
        try:
            with open(store_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving vector store metadata: {str(e)}")
            raise
