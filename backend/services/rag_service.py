import os
import time
from pathlib import Path
import cv2
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from google.generativeai import GenerativeModel
import google.generativeai as genai
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.utils.config import Config
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import logging
import hashlib
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime
import numpy as np
import shutil

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
        # Create required directories
        self.persist_dir = Path("tmp_content/vector_store")
        self.kb_dir = Path("tmp_content/knowledgebase")
        
        for directory in [self.persist_dir, self.kb_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Initialized directory: {directory}")
        
    def _process_analysis_files(self, new_files: List[Path]) -> List[Dict]:
        """Process analysis files and generate text representations"""
        if not new_files:
            return []

        try:
            # Process each file
            processed_data = []
            for file_path in new_files:
                try:
                    with open(file_path) as f:
                        file_data = json.load(f)
                    if isinstance(file_data, dict):
                        processed_data.append({
                            'data': file_data,
                            'metadata': {
                                'file_path': str(file_path),
                                'timestamp': file_path.stat().st_mtime,
                                'type': 'analysis'
                            }
                        })
                    else:
                        self.logger.warning(f"Invalid data format in {file_path}")
                except Exception as e:
                    self.logger.warning(f"Error processing {file_path}: {e}")
                    continue

            if not processed_data:
                self.logger.warning("No valid analysis data found in files")
                return []

            # Generate text representations if Gemini available
            if self.gemini_enabled and self.gemini_model:
                return self._generate_text_representations(processed_data)
            
            self.logger.info("Gemini not available - using raw analysis data")
            return processed_data

        except Exception as e:
            self.logger.error(f"Error processing analysis files: {e}")
            raise

    def _get_analysis_prompt(self) -> str:
        """Get the analysis prompt template"""
        return """Analyze these video analysis JSON files and create an extremely detailed text representation that includes:

1. Scene Analysis:
   - Comprehensive scene descriptions from all timestamps
   - All identified objects, people, and their positions
   - Every detected activity and event
   - Complete environmental details (lighting, setting, background)
   - Temporal changes and scene evolution

2. Technical Details:
   - Full resolution and frame information
   - All timestamps and frame numbers
   - Complete video metadata
   - Processing parameters used
   - Quality metrics and confidence scores

3. Object & Activity Analysis:
   - Every detected object with confidence scores
   - All spatial relationships between objects
   - Complete activity timelines
   - Movement patterns and trajectories
   - Interaction analysis between objects/people

4. Contextual Information:
   - All relevant background information
   - Environmental conditions
   - Camera movement and perspective changes
   - Scene context and setting details
   - Temporal context and sequence information

5. Analysis Results:
   - All detection results with confidence scores
   - Complete list of identified patterns
   - Every relationship and correlation found
   - All technical measurements and metrics
   - Processing statistics and performance data

6. Metadata & Technical Context:
   - File processing history
   - Analysis timestamps
   - All configuration parameters
   - System performance metrics
   - Processing pipeline details

Generate a comprehensive, structured text representation that captures 100% of the information present in the analysis files. Include all numerical values, timestamps, and technical details exactly as they appear in the source data.

Focus on maximum detail and complete accuracy. Do not summarize or omit any information."""

    def _generate_text_representations(self, analysis_files: List[Dict]) -> List[Dict]:
        """Generate text representations using Gemini with enhanced error handling and retries"""
        if not analysis_files:
            return []
            
        if not self.gemini_enabled or not self.gemini_model:
            raise ValueError("Gemini model not initialized")

        kb_path = Path("tmp_content/knowledgebase")
        kb_path.mkdir(parents=True, exist_ok=True)
        
        processed_texts = []
        max_retries = 3
        retry_delay = 1  # seconds
        
        for file_data in analysis_files:
            for attempt in range(max_retries):
                try:
                    # Enhanced prompt with structured sections
                    prompt_sections = [
                        {"text": "Task: Generate a detailed text representation of video analysis data"},
                        {"text": "Required sections:\n1. Scene Description\n2. Object Analysis\n3. Activity Timeline\n4. Technical Details\n5. Metadata Summary"},
                        {"text": self._get_analysis_prompt()},
                        {"text": f"Analysis JSON data:\n{json.dumps(file_data['data'], indent=2)}"}
                    ]
                    
                    response = self.gemini_model.generate_content(prompt_sections)
                    
                    if not response:
                        raise ValueError("Empty response from model")
                    
                    # Extract and validate text
                    text_representation = response.text if hasattr(response, 'text') else str(response)
                    if not text_representation.strip():
                        raise ValueError("Empty text representation")

                    # Generate unique identifier
                    content_hash = hashlib.sha256(
                        (text_representation + str(time.time())).encode()
                    ).hexdigest()[:16]
                    
                    # Enhanced metadata
                    metadata = {
                        'model': 'gemini',
                        'model_version': getattr(self.gemini_model, 'version', 'unknown'),
                        'source_file': file_data.get('metadata', {}).get('file_path', 'unknown'),
                        'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'content_hash': content_hash,
                        'input_data_hash': hashlib.md5(
                            json.dumps(file_data['data']).encode()
                        ).hexdigest(),
                        'processing_stats': {
                            'attempt': attempt + 1,
                            'timestamp': time.time(),
                            'text_length': len(text_representation)
                        }
                    }
                    
                    # Save response with enhanced structure
                    response_file = kb_path / f"analysis_{content_hash}.json"
                    response_data = {
                        'text': text_representation.strip(),
                        'prompt': prompt_sections,
                        'metadata': metadata,
                        'source_data': {
                            'type': file_data.get('metadata', {}).get('type', 'unknown'),
                            'timestamp': file_data.get('metadata', {}).get('timestamp'),
                            'hash': metadata['input_data_hash']
                        }
                    }
                    
                    with open(response_file, 'w') as f:
                        json.dump(response_data, f, indent=2)
                    
                    self.logger.info(f"Successfully generated and saved text representation: {response_file}")
                    processed_texts.append(response_data)
                    break  # Success - exit retry loop
                    
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        self.logger.error(f"Failed to process file after {max_retries} attempts")

        if not processed_texts:
            self.logger.warning("No analysis files were successfully processed")
            return []

        if not self.gemini_enabled or not self.gemini_model:
            raise ValueError("Gemini model not initialized")

        self.logger.info(f"Created {len(processed_texts)} text representations")
        return processed_texts

    def create_knowledge_base(self, results_path: Path) -> Optional[FAISS]:
        """Create or update vector store from analysis results"""
        try:
            analysis_dir = Path("tmp_content/analysis")
            kb_path = Path("tmp_content/knowledgebase")
            store_path = kb_path / 'vector_store'
            metadata_path = kb_path / 'metadata.json'

            # Create required directories
            kb_path.mkdir(parents=True, exist_ok=True)
            store_path.mkdir(parents=True, exist_ok=True)

            if not analysis_dir.exists():
                self.logger.warning("Analysis directory does not exist")
                return None

            # Get new analysis files
            analysis_files = self._get_new_analysis_files(metadata_path)
            
            # If no analysis files exist, use scene analysis tool
            if not analysis_files and not any(analysis_dir.glob("*.json")):
                self.logger.info("No analysis files found - using scene analysis tool")
                try:
                    from backend.core.analysis_tools import SceneAnalysisTool
                    scene_tool = SceneAnalysisTool(None)  # Pass None since we'll set params directly
                    
                    # Trigger analysis
                    result = scene_tool._run(results_path)
                    if result:
                        return self.create_knowledge_base(results_path)
                        
                except Exception as e:
                    self.logger.error(f"Auto scene analysis failed: {e}")
                    return None
                
            # If no new files but store exists, try loading existing store
            if not analysis_files:
                self.logger.info("No new analysis files - checking existing store")
                if store_path.exists():
                    return self._load_existing_store(store_path)
                else:
                    self.logger.warning("No existing store found")
                    return None

            # Process files and generate text representations
            processed_files = self._process_analysis_files(analysis_files)
            if not processed_files:
                self.logger.warning("No files were successfully processed")
                return self._load_existing_store(store_path)

            # # Generate text representations if Gemini is available
            # text_representations = []
            # if self.gemini_enabled and self.gemini_model:
            #     text_representations = self._generate_text_representations(processed_files)
            # else:
            #     # Use raw analysis data if Gemini not available
            #     text_representations = [{
            #         'text': str(file_data.get('data', {})),
            #         'metadata': file_data.get('metadata', {})
            #     } for file_data in processed_files]


            # Load existing knowledge base texts
            kb_texts = self._load_existing_kb_texts(kb_path)

            # Create chunks from text representations
            chunks = self._create_chunks(kb_texts)

            # Create vector store
            vectordb = self._create_vector_store(chunks, store_path)

            # Save metadata
            self._save_kb_metadata(metadata_path, len(chunks), len(analysis_files))

            return vectordb
            
        except Exception as e:
            self.logger.error(f"Error creating knowledge base: {str(e)}")
            return None

    def _get_new_analysis_files(self, metadata_path: Path) -> List[Path]:
        """Get list of new analysis files to process"""
        processed_files = set()
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    processed_files = set(metadata.get('processed_files', []))
            except Exception as e:
                self.logger.warning(f"Error loading metadata: {e}")

        analysis_files = sorted(Path("tmp_content/analysis").glob("*.json"))
        new_files = []
        
        for file_path in analysis_files:
            file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
            if file_hash not in processed_files:
                new_files.append(file_path)
                processed_files.add(file_hash)

        return new_files

    def _create_chunks(self, text_input: Union[str, List[Dict]], separator: str = "\n\n") -> List[Dict]:
        """Create overlapping chunks from text input with metadata
        
        Args:
            text_input: Either a string or list of text representation dicts
            separator: String to split sections (default paragraph break)
            
        Returns:
            List of chunk dicts with text and metadata
        """
        chunks = []
        chunk_size = 500  # Smaller base chunk size
        min_chunk_size = 300  # Smaller minimum
        max_chunk_size = 800  # Smaller maximum
        overlap = 100  # Add overlap between chunks
        
        def calculate_optimal_chunk_size(text: str) -> int:
            """Calculate optimal chunk size based on text characteristics"""
            avg_sentence_length = len(text.split('.'))
            return min(max_chunk_size, max(min_chunk_size, 
                int(avg_sentence_length * 100)))  # ~100 chars per sentence
        
        def add_chunk_with_context(text: str, index: int, total: int, metadata: Dict) -> None:
            """Add chunk with surrounding context and metadata"""
            chunk_text = text.strip()
            if not chunk_text:
                return
                
            # Add chunk with enhanced metadata
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    'chunk_index': index,
                    'total_chunks': total,
                    'timestamp': time.time(),
                    'chunk_stats': {
                        'length': len(chunk_text),
                        'sentences': len(chunk_text.split('.')),
                        'words': len(chunk_text.split()),
                        'relative_position': index / total
                    }
                }
            })
        
        # Handle single text string
        if isinstance(text_input, str):
            sections = text_input.split("=== Section Break ===" if "=== Section Break ===" in text_input else separator)
            chunk_size = calculate_optimal_chunk_size(text_input)
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                    
                # Split long sections into smaller chunks
                section_chunks = [section[j:j+chunk_size] for j in range(0, len(section), chunk_size)]
                
                for k, chunk in enumerate(section_chunks):
                    add_chunk_with_context(
                        chunk,
                        k + 1,
                        len(section_chunks),
                        {
                            'type': 'analysis',
                            'section': f"section_{i+1}",
                            'total_sections': len(sections),
                            'source_type': 'single_text'
                        }
                    )
                    
        # Handle list of text representations
        else:
            for text_rep in text_input:
                if not isinstance(text_rep, dict) or 'text' not in text_rep:
                    continue
                    
                chunk_size = calculate_optimal_chunk_size(text_rep['text'])
                sections = text_rep['text'].split(separator)
                base_metadata = text_rep.get('metadata', {})
                
                for i, section in enumerate(sections):
                    section_chunks = [section[j:j+chunk_size] for j in range(0, len(section), chunk_size)]
                    
                    for k, chunk in enumerate(section_chunks):
                        add_chunk_with_context(
                            chunk,
                            k + 1,
                            len(section_chunks),
                            {
                                **base_metadata,
                                'section': f"section_{i+1}",
                                'total_sections': len(sections),
                                'source_type': 'text_representation'
                            }
                        )
        
        return chunks

    def _load_existing_kb_texts(self, kb_path: Path) -> List[Dict]:
        """Load existing knowledge base texts"""
        kb_texts = []
        for text_file in kb_path.glob("*.json"):
            try:
                with open(text_file) as f:
                    data = json.load(f)
                    kb_texts.append({
                        "text": data['text'],
                        "metadata": {
                            **data['metadata'],
                            'source': str(text_file),
                            'type': 'knowledge_base'
                        }
                    })
            except Exception as e:
                self.logger.error(f"Error reading KB file {text_file}: {e}")
                
        return kb_texts

    def _create_vector_store(self, documents: List[Dict], store_path: Path) -> Optional[FAISS]:
        """Create FAISS vector store from documents with validation and error handling"""
        try:
            if not documents:
                raise ValueError("No documents provided for vector store creation")

            # Extract texts and metadata
            texts = []
            metadatas = []
            for doc in documents:
                if not isinstance(doc, dict) or "text" not in doc:
                    continue
                texts.append(doc["text"].strip())
                metadatas.append(doc.get("metadata", {}))

            if not texts:
                raise ValueError("No valid texts found in documents")

            # Create embeddings with batching
            batch_size = 32
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)

            # Convert to numpy array and normalize
            embeddings_array = np.array(all_embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)

            # Create FAISS index
            dimension = len(embeddings_array[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)

            # Create document store with metadata
            docstore = InMemoryDocstore({})
            index_to_id = {}
            
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                doc_id = f"doc_{i}"
                index_to_id[i] = doc_id
                docstore.add({
                    doc_id: Document(
                        page_content=text,
                        metadata={
                            **metadata,
                            'doc_id': doc_id,
                            'index': i,
                            'embedding_dim': dimension
                        }
                    )
                })

            # Create FAISS vectorstore with error handling
            vectordb = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_id,
                normalize_L2=True
            )

            # Verify the store was created correctly
            if not vectordb.docstore or not vectordb.index:
                raise ValueError("Vector store creation failed - invalid store state")

            # Save vector store with explicit path handling
            store_path.parent.mkdir(parents=True, exist_ok=True)
            if store_path.exists():
                shutil.rmtree(store_path)
            vectordb.save_local(str(store_path))

            # Verify save was successful
            if not store_path.exists():
                raise ValueError("Vector store save failed - store directory not created")

            # Save metadata about the store
            metadata_path = os.path.join(store_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'num_documents': len(texts),
                    'embedding_dim': dimension,
                    'created_at': time.time(),
                    'documents': [{
                        'id': f"doc_{i}",
                        'metadata': m
                    } for i, m in enumerate(metadatas)]
                }, f, indent=2)

            self.logger.info(f"Created vector store with {len(texts)} documents")
            return vectordb

        except Exception as e:
            self.logger.error(f"Error creating vector store: {e}")
            if store_path.exists():
                shutil.rmtree(store_path)
            return None

    def _save_kb_metadata(self, metadata_path: Path, num_chunks: int, num_files: int):
        """Save knowledge base metadata"""
        kb_stats = {
            'total_chunks': num_chunks,
            'total_files': num_files,
            'embedding_model': 'text-embedding-ada-002',
            'vector_dimensions': 1536,
            'similarity_metric': 'cosine'
        }

        with open(metadata_path, 'w') as f:
            json.dump({
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'kb_stats': kb_stats,
                'last_update': time.time(),
                'version': '1.0'
            }, f, indent=2)
            
    def get_retrieval_chain(self, vectordb: FAISS) -> RetrievalQAWithSourcesChain:
        """Create retrieval chain with custom prompt"""
        if not vectordb:
            raise ValueError("Vector store is required to create retrieval chain")

        # Configure retriever with better defaults
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,  # Increased for better context
                "score_threshold": 0.6,  # Slightly lower threshold
                "fetch_k": 20  # Fetch more candidates
            }
        )

        # Create memory for chat history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Enhanced prompt template
        prompt_template = """You are an AI assistant analyzing video content. Use the following input to answer questions about the video.

{input}

Instructions:
1. Answer based on the context above
2. Be specific and reference timestamps/frames when available
3. If information is incomplete, acknowledge uncertainty
4. Suggest relevant tools when appropriate:
   - scene_analysis: Detailed scene analysis
   - object_detection: Identify and locate objects
   - edge_detection: Highlight visual boundaries

Provide your response in natural language, focusing on being informative and helpful."""

        # Create prompt with single input variable
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["input"]
        )

        # Create chain with enhanced configuration
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            memory=memory,
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True
            }
        )

        # Store chain reference
        self._current_chain = chain
        return chain
        
    def query_knowledge_base(self, query: str, chain: Optional[RetrievalQAWithSourcesChain] = None, chat_history: Optional[List[Dict]] = None) -> Dict:
        """Query the knowledge base with enhanced source tracking and chat context"""
        try:
            # Create chain if not provided
            if not chain:
                vectordb = self._load_existing_store(Path("tmp_content/knowledgebase/vector_store"))
                if not vectordb:
                    return {
                        "answer": "No knowledge base found. Please analyze some content first.",
                        "sources": [],
                        "requires_analysis": True
                    }
                chain = self.get_retrieval_chain(vectordb)

            # Format chat history
            formatted_history = []
            if chat_history:
                for msg in chat_history[-5:]:  # Last 5 messages
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if role == 'user':
                        formatted_history.append(HumanMessage(content=content))
                    elif role == 'assistant':
                        formatted_history.append(AIMessage(content=content))

            # Get vectordb from chain or create new chain if needed
            vectordb = None
            if hasattr(chain, 'retriever'):
                vectordb = chain.retriever.vectorstore
            
            if not vectordb:
                # Try to load existing store
                vectordb = self._load_existing_store(Path("tmp_content/knowledgebase/vector_store"))
                if not vectordb:
                    raise ValueError("No vector store available and could not load from disk")
                # Create new chain with loaded store
                chain = self.get_retrieval_chain(vectordb)

            # Get relevant documents with enhanced parameters 
            relevant_docs = vectordb.similarity_search_with_score( 
                query,
                k=8,  # Increased for better coverage
                score_threshold=0.5,  # Slightly lower threshold for more results
                fetch_k=30,  # Fetch more candidates
                filter=None  # No filtering by default
            )

            # Sort by score and take top 5
            relevant_docs = sorted(relevant_docs, key=lambda x: x[1])[:5]

            # Combine context and query into single input
            # Format context and input text
            context_text = "\n\n".join([doc[0].page_content for doc in relevant_docs])
            history_text = formatted_history if formatted_history else "No previous chat history"
            
            input_text = (
                f"Context: {context_text}\n\n"
                f"Question: {query}\n\n"
                f"Chat History: {history_text}"
            )

            # Query the chain with single input key
            chain_response = chain.invoke({
                "input": input_text
            })

            # Process response
            response = {
                "answer": chain_response.get("answer", "").strip(),
                "sources": chain_response.get("source_documents", []),
                "source_documents": [
                    {
                        "content": doc[0].page_content,
                        "metadata": doc[0].metadata,
                        "score": doc[1]  # Include similarity score
                    }
                    for doc in relevant_docs
                ]
            }

            # Add tool suggestions if relevant
            tool_suggestions = self._analyze_for_tools(query, response["answer"])
            if tool_suggestions:
                response['answer'] += f"\n\n{tool_suggestions}"
                response['suggested_tools'] = tool_suggestions
                response['requires_confirmation'] = True
                response['pending_action'] = tool_suggestions.split()[0]

            return response

        except Exception as e:
            self.logger.error(f"Error querying knowledge base: {str(e)}")
            return {
                "error": str(e),
                "answer": "I encountered an error processing your query. Please try again."
            }
            
    def _hash_results(self, results_path: Path) -> str:
        """Generate hash of results file for vector store identification"""
        with open(results_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash[:10]  # Use first 10 chars for readability
        
    def _load_existing_store(self, store_path: Path) -> Optional[FAISS]:
        """Load existing vector store if available"""
        try:
            if store_path.exists():
                return FAISS.load_local(
                    str(store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Only enable for trusted local files
                )
            return None
        except Exception as e:
            self.logger.error(f"Error loading existing store: {e}")
            return None

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
            
    def _analyze_for_tools(self, query: str, current_answer: str) -> Optional[str]:
        """Analyze query and current answer to suggest relevant tools"""
        tool_patterns = {
            'object_detection': ['object', 'detect', 'find', 'identify', 'locate', 'spot'],
            'scene_analysis': ['describe', 'analyze', 'understand', 'explain', 'what is happening'],
            'edge_detection': ['edge', 'boundary', 'outline', 'shape', 'contour']
        }
        
        query_lower = query.lower()
        suggestions = []
        
        for tool, patterns in tool_patterns.items():
            if any(p in query_lower for p in patterns):
                if tool == 'object_detection':
                    suggestions.append("I can run object detection to identify and locate specific objects in the video.")
                elif tool == 'scene_analysis':
                    suggestions.append("I can perform a detailed scene analysis to better understand what's happening.")
                elif tool == 'edge_detection':
                    suggestions.append("I can enable edge detection to highlight object boundaries and shapes.")
                    
        if suggestions:
            return "\n\nWould you like me to " + " Or ".join(suggestions[:-1] + ["?" if len(suggestions) == 1 else " or " + suggestions[-1] + "?"])
        return None

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
    def _init_doc_stats(self) -> Dict:
        """Initialize document statistics tracking"""
        return {
            'total_files': 0,
            'total_size': 0,
            'oldest_file': None,
            'newest_file': None
        }

    def _process_files(self, files: List[Path], doc_stats: Dict) -> Tuple[List[Dict], Dict]:
        """
        Process individual analysis files
        
        Args:
            files: List of files to process
            doc_stats: Statistics dictionary to update
            
        Returns:
            Tuple of (processed data list, file metadata dict)
        """
        all_data = []
        file_metadata = {}
        
        for file_path in files:
            try:
                if not self._is_valid_file(file_path):
                    continue
                
                file_info = self._get_file_info(file_path)
                self._update_doc_stats(doc_stats, file_info)
                
                data = self._load_json_file(file_path)
                if data:
                    # Create a properly structured analysis data entry
                    analysis_entry = {
                        'data': data,  # Original JSON data
                        'metadata': {
                            'file_path': str(file_path),
                            'timestamp': file_info['mtime'],
                            'size': file_info['size'],
                            'type': 'analysis'
                        }
                    }
                    all_data.append(analysis_entry)
                    file_metadata[str(file_path)] = analysis_entry['metadata']
                    
            except Exception as e:
                self.logger.warning(f"Error processing {file_path}: {e}")
                continue
                
        return all_data, file_metadata
        
    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if file is valid for processing"""
        if not file_path.is_file():
            return False
            
        if file_path.stat().st_size == 0:
            self.logger.warning(f"Skipping empty file: {file_path}")
            return False
            
        return True
        
    def _get_file_info(self, file_path: Path) -> Dict:
        """Get file information"""
        stat = file_path.stat()
        return {
            'size': stat.st_size,
            'mtime': stat.st_mtime
        }
        
    def _update_doc_stats(self, stats: Dict, file_info: Dict) -> None:
        """Update document statistics with file info"""
        stats['total_files'] += 1
        stats['total_size'] += file_info['size']
        
        if not stats['oldest_file'] or file_info['mtime'] < stats['oldest_file']:
            stats['oldest_file'] = file_info['mtime']
            
        if not stats['newest_file'] or file_info['mtime'] > stats['newest_file']:
            stats['newest_file'] = file_info['mtime']
            
    def _load_json_file(self, file_path: Path) -> Optional[Dict]:
        """Load and validate JSON file"""
        try:
            with open(file_path) as f:
                data = json.load(f)
                
            if not isinstance(data, dict):
                raise ValueError(f"Invalid JSON structure in {file_path}")
                
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            self.logger.warning(f"Error loading {file_path}: {e}")
            
        return None
