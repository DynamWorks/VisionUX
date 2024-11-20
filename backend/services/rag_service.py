import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.utils.config import Config
from langchain_community.vectorstores import Chroma
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
        
        # Initialize embeddings with API key from env or config
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY') or self.config.get('services', {}).get('openai', {}).get('api_key'),
            openai_api_base=os.getenv('OPENAI_API_BASE') or self.config.get('services', {}).get('openai', {}).get('api_base')
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
        self.llm = ChatOpenAI(
            model_name=self.config.get('services', {}).get('rag', {}).get('model', 'gpt-4-turbo-preview'),
            openai_api_key=os.getenv('OPENAI_API_KEY') or self.config.get('services', {}).get('openai', {}).get('api_key'),
            openai_api_base=os.getenv('OPENAI_API_BASE') or self.config.get('services', {}).get('openai', {}).get('api_base')
        )
        self.logger = logging.getLogger(__name__)
        self.persist_dir = Path("tmp_content/vector_store")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_and_chunk_results(self, results_path: Path) -> List[Dict[str, Any]]:
        """
        Load analysis results and split into chunks
        
        Args:
            results_path: Path to JSON results file
            
        Returns:
            List of document dictionaries with text and metadata
            
        Raises:
            ValueError: If results file is invalid or empty
        """
        try:
            with open(results_path) as f:
                data = json.load(f)
                
            # Convert analysis results to documents
            documents = []
            
            # Handle frame results
            if 'results' in data:
                for frame in data['results']:
                    frame_text = f"""
                    Frame {frame.get('frame_number', 0)} at {frame.get('timestamp', 0)}s:
                    Detections: {frame.get('detections', {})}
                    Scene Analysis: {frame.get('scene_analysis', {})}
                    """
                    documents.append({"text": frame_text, "metadata": frame})
                    
            # Handle scene analysis
            if 'scene_analysis' in data:
                scene_text = f"""
                Scene Analysis:
                Description: {data['scene_analysis'].get('description', '')}
                Suggested Pipeline: {data['scene_analysis'].get('suggested_pipeline', [])}
                """
                documents.append({"text": scene_text, "metadata": data['scene_analysis']})
                
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading results: {str(e)}")
            return []
            
    def create_knowledge_base(self, results_path: Path) -> Optional[Chroma]:
        """
        Create or load vector store from analysis results
        
        Args:
            results_path: Path to JSON results file
            
        Returns:
            Chroma vector store instance or None if creation fails
            
        Notes:
            - Creates a new vector store if one doesn't exist for the results
            - Loads existing vector store if available and up to date
            - Updates vector store if results file has changed
        """
        try:
            documents = self._load_and_chunk_results(results_path)
            if not documents:
                return None
                
            # Split documents into chunks
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            chunks = self.text_splitter.create_documents(texts, metadatas=metadatas)
            
            # Create vector store
            # Generate unique ID for these results
            results_hash = self._hash_results(results_path)
            store_path = self.persist_dir / results_hash
            
            # Check if vector store exists and is current
            if self._is_store_current(store_path, results_hash):
                self.logger.info(f"Loading existing vector store from {store_path}")
                vectordb = Chroma(
                    persist_directory=str(store_path),
                    embedding_function=self.embeddings
                )
            else:
                self.logger.info("Creating new vector store")
                vectordb = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=str(store_path)
                )
                vectordb.persist()
                # Save metadata about the store
                self._save_store_metadata(store_path, results_hash)
            
            return vectordb
            
        except Exception as e:
            self.logger.error(f"Error creating knowledge base: {str(e)}")
            return None
            
    def get_retrieval_chain(self, vectordb: Chroma) -> RetrievalQAWithSourcesChain:
        """Create retrieval chain with custom prompt"""
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.7
            }
        )

        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question. 
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        
        Context: {summaries}
        
        Question: {question}
        
        Answer with specific references to frame numbers and timestamps when possible.
        Include confidence levels for any claims you make.
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
