import os
import time
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from google.generativeai import GenerativeModel
import google.generativeai as genai
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
        if not self.config._config:
            self.config.reset()
        
        # Initialize OpenAI
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            openai_api_key = self.config.get('services', 'openai', 'api_key')

        openai_api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')        
        if not openai_api_base:                                                            
             openai_api_base = self.config.get('services', 'openai', 'api_base',            
 default='https://api.openai.com/v1')                        
            
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize Gemini
        self.gemini_api_key = self.config.gemini_api_key
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
                self.gemini_model = GenerativeModel(model_name)
                self.gemini_enabled = True
                self.logger.info(f"Gemini API initialized successfully with model: {model_name}")
            except Exception as e:
                self.gemini_enabled = False
                self.gemini_model = None
                self.logger.error(f"Failed to initialize Gemini API: {e}")
        else:
            self.gemini_enabled = False
            self.gemini_model = None
            self.logger.warning("No Gemini API key found - Gemini features will be disabled")
        
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
            model_name = self.config.get('services', 'rag', 'model', default='gpt-4o-mini')
            
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base
        )
        self.logger = logging.getLogger(__name__)
        self.persist_dir = Path("tmp_content/vector_store")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_and_chunk_results(self, results_path: Path) -> List[Dict[str, Any]]:
        """Load analysis results and convert to documents with parsed metadata"""
        try:
            with open(results_path) as f:
                data = json.load(f)
                
            # Use Gemini to get detailed text representation if available
            if self.gemini_enabled and self.gemini_model:
                prompt = """Analyze this JSON data and provide a detailed text representation that:
                1. Describes the content in natural language
                2. Preserves all important information and relationships
                3. Maintains hierarchical structure
                4. Includes technical details and metadata
                5. Uses clear section headings and organization
                
                JSON data:
                {data}
                """
                
                try:
                    response = self.gemini_model.generate_content(prompt.format(data=json.dumps(data, indent=2)))
                    if response and response.text:
                        text_representation = response.text
                    else:
                        text_representation = json.dumps(data, indent=2)
                except Exception as e:
                    self.logger.error(f"Gemini processing failed: {e}")
                    text_representation = json.dumps(data, indent=2)
            else:
                # Fallback to raw JSON if Gemini not available
                text_representation = json.dumps(data, indent=2)
                
            # Save processed text to knowledgebase
            kb_path = Path("tmp_content/knowledgebase")
            kb_path.mkdir(parents=True, exist_ok=True)
            
            kb_file = kb_path / f"analysis_{int(time.time())}.txt"
            with open(kb_file, 'w') as f:
                f.write(response.text)
                
            # Create document with metadata
            metadata = {
                'source': str(results_path),
                'timestamp': time.time(),
                'type': 'analysis',
                'kb_path': str(kb_file)
            }
            
            return [{
                "text": response.text,
                "metadata": metadata
            }]

            def extract_metadata(obj: Any, prefix: str = "") -> Dict:
                """Extract metadata from JSON object with path context"""
                metadata = {}
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_prefix = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, (dict, list)):
                            metadata.update(extract_metadata(value, current_prefix))
                        else:
                            metadata[current_prefix] = value
                            
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        current_prefix = f"{prefix}[{i}]"
                        if isinstance(item, (dict, list)):
                            metadata.update(extract_metadata(item, current_prefix))
                            
                return metadata

            # Process data with Gemini
            prompt = """Analyze this JSON data and provide a detailed text representation that:
            1. Describes the content in natural language
            2. Preserves all important information and relationships
            3. Maintains hierarchical structure
            4. Includes technical details and metadata
            5. Uses clear section headings and organization
            
            JSON data:
            {data}
            """
            
            response = self.gemini_model.generate_content(prompt.format(data=json.dumps(data, indent=2)))
            
            if not response.text:
                raise ValueError("Failed to get text representation from Gemini")
                
            # Split into chunks
            chunks = self.text_splitter.split_text(response.text)
            
            # Create documents list
            processed_documents = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'source': str(results_path),
                    'timestamp': time.time()
                }
                
                # Extract metadata from the data
                chunk_metadata = extract_metadata(data)
                metadata.update(chunk_metadata)
                
                processed_documents.append({
                    "text": chunk,
                    "metadata": metadata
                })
                
            return processed_documents
            
        except Exception as e:
            self.logger.error(f"Error loading results: {str(e)}")
            return []
            
    def create_knowledge_base(self, results_path: Path) -> Optional[FAISS]:
        """Create vector store from analysis results and chat history"""
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
            
            # Get recent chat history
            chat_dir = Path("tmp_content/chat_history")
            if chat_dir.exists():
                chat_files = sorted(chat_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:3]
                
                for chat_file in chat_files:
                    with open(chat_file) as f:
                        chat_data = json.load(f)
                        
                    # Get text representation of chat
                    chat_prompt = """Convert this chat history to a natural text summary that:
                    1. Captures the key points of discussion
                    2. Maintains conversation flow and context
                    3. Highlights important questions and answers
                    
                    Chat history:
                    {chat}
                    """
                    
                    chat_response = self.gemini_model.generate_content(
                        chat_prompt.format(chat=json.dumps(chat_data, indent=2))
                    )
                    
                    if chat_response.text:
                        # Add chat summary as document
                        chunks.append(Document(
                            page_content=chat_response.text,
                            metadata={
                                'source': str(chat_file),
                                'type': 'chat_history',
                                'timestamp': chat_file.stat().st_mtime
                            }
                        ))
            
            # Create vector store from all documents
            vectordb = FAISS.from_documents(chunks, self.embeddings)
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
