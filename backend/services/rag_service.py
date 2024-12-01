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
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key and self.config._config:  # Only check config if it exists
            gemini_api_key = self.config.get('services', 'gemini', 'api_key')
            if gemini_api_key:
                os.environ['GEMINI_API_KEY'] = gemini_api_key  # Set env var for consistency
            
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
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
            self.logger.warning("No Gemini API key found in environment or config - Gemini features will be disabled")
        
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
                
            # Use Gemini to get detailed text representation
            if not self.gemini_enabled or not self.gemini_model:
                raise ValueError("Gemini model not initialized")

            prompt = """Analyze this video analysis JSON data and create a detailed text summary that:

1. Overview:
   - Video name and timestamp
   - Type of analysis performed
   - Number of frames analyzed
   - Video duration and FPS

2. Technical Details:
   - Frame numbers and timestamps analyzed
   - Resolution and video quality
   - Processing parameters used
   - Any technical challenges or limitations

3. Analysis Results:
   - Scene descriptions and observations
   - Objects and activities detected
   - Changes or patterns observed
   - Confidence levels and certainty

4. Context and Relationships:
   - Temporal relationships between frames
   - Spatial relationships between objects
   - Cause and effect observations
   - Environmental and setting details

Format the output with clear sections and bullet points.
Include specific frame numbers, timestamps, and metrics when available.
Focus on factual observations that can be referenced in future queries.

JSON data to analyze:
{data}
"""

            try:
                response = self.gemini_model.generate_content(prompt.format(data=json.dumps(data, indent=2)))
                if response and response.text:
                    # Parse the response text
                    text_representation = response.text.strip()
                    
                    # # Add metadata section if not included
                    # if 'Metadata' not in text_representation:
                    #     metadata_summary = f"""
                        
                    #     Metadata Summary:
                    #     - Analysis timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
                    #     - Frames analyzed: {len(data.get('frame_numbers', []))}
                    #     - Video duration: {data.get('duration', 'Unknown')} seconds
                    #     - Frame rate: {data.get('fps', 'Unknown')} FPS
                    #     """
                    #     text_representation += metadata_summary
                else:
                    # Fallback to basic text representation
                    text_representation = f"""
                    Analysis Results:
                    - Raw data: {json.dumps(data, indent=2)}
                    """
                    # text_representation = f"""
                    # Analysis Results:
                    # - Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
                    # - Number of frames: {len(data.get('frame_numbers', []))}
                    # - Raw data: {json.dumps(data, indent=2)}
                    # """
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
                f.write(text_representation)
                
            # Create document with metadata
            metadata = {
                'source': str(results_path),
                'timestamp': time.time(),
                'type': 'analysis',
                'kb_path': str(kb_file)
            }
            
            return [{
                "text": text_representation,
                "metadata": metadata
            }]

            
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
            
            # Chat history is now handled separately in query_knowledge_base
            
            # Create or update vector store
            kb_path = Path("tmp_content/knowledgebase")
            store_path = kb_path / 'vector_store'
            
            if store_path.exists():
                # Load existing store and add new documents
                vectordb = FAISS.load_local(str(store_path), self.embeddings)
                vectordb.add_documents(chunks)
            else:
                # Create new store
                vectordb = FAISS.from_documents(chunks, self.embeddings)
            
            # Save updated store
            vectordb.save_local(str(store_path))
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
        prompt_template = """You are discussing video analysis results with a researcher or colleague. 
Use the following analysis context to have an informed, natural conversation about the experiments and findings.

Analysis Context:
{summaries}

Question: {question}

Guidelines for your response:
1. Respond naturally like a knowledgeable colleague discussing research
2. Reference specific frames, timestamps and observations from the analysis
3. Be clear and concise while maintaining a conversational tone
4. Express appropriate uncertainty when information is limited
5. Only use information from the provided analysis context
6. Focus on key insights and interesting findings
7. Feel free to suggest follow-up areas to explore

Format your response to:
1. Directly address the question in a conversational way
2. Support your points with specific evidence from the analysis
3. Highlight interesting patterns or unexpected results
4. Note any limitations or areas needing more investigation
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
        
    def query_knowledge_base(self, query: str, chain: RetrievalQAWithSourcesChain, chat_history: Optional[List[Dict]] = None) -> Dict:
        """Query the knowledge base with enhanced source tracking and chat context"""
        try:
            # Format chat history as context if provided
            chat_context = ""
            if chat_history:
                chat_context = "\nRecent Chat Context:\n"
                for msg in chat_history[-5:]:  # Use last 5 messages
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    chat_context += f"{role}: {content}\n"

            # Combine query with chat context
            enhanced_query = f"""
Question: {query}

{chat_context}
"""
            # Query the chain
            response = chain({"question": enhanced_query})
            
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
                ]
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
