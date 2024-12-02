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
from typing import List, Dict, Optional, Any, Tuple
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
        # Create required directories
        self.persist_dir = Path("tmp_content/vector_store")
        self.kb_dir = Path("tmp_content/knowledgebase")
        
        for directory in [self.persist_dir, self.kb_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Initialized directory: {directory}")
        
    def _process_analysis_files(self, new_files: List[Path]) -> List[Dict]:
        """
        Process analysis files and generate text representations
        
        Args:
            new_files: List of Path objects pointing to new analysis files
            
        Returns:
            List of processed analysis data dictionaries
            
        Raises:
            ValueError: If no valid data found
        """
        if not new_files:
            return []

        all_data = []
        file_metadata = {}
        doc_stats = self._init_doc_stats()

        try:
            all_data, file_metadata = self._process_files(new_files, doc_stats)
            
            if not all_data:
                self.logger.warning("No valid analysis data found in files")
                return []

            # Don't require Gemini for initial processing
            if not self.gemini_enabled or not self.gemini_model:
                self.logger.warning("Gemini not available - skipping text generation")
                return all_data
                
            return all_data

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
        """Generate text representations using Gemini"""
        if not analysis_files:
            return []
            
        if not self.gemini_enabled or not self.gemini_model:
            raise ValueError("Gemini model not initialized")

        kb_path = Path("tmp_content/knowledgebase")
        kb_path.mkdir(parents=True, exist_ok=True)
        
        processed_texts = []
        
        for file_data in analysis_files:
            try:
                response = self.gemini_model.generate_content([
                    {"text": self._get_analysis_prompt()},
                    {"text": f"Analysis JSON data:\n{json.dumps(file_data['data'], indent=2)}"}
                ])
                
                if not response:
                    self.logger.warning(f"No response received for analysis file")
                    continue
                    
                # Extract text from Gemini response
                text_representation = response.text if hasattr(response, 'text') else str(response)
                if not text_representation.strip():
                    self.logger.warning(f"Empty response for analysis file")
                    continue

                # Save individual response to knowledgebase
                kb_path = Path("tmp_content/knowledgebase")
                kb_path.mkdir(parents=True, exist_ok=True)
                
                timestamp = int(time.time())
                file_hash = hashlib.md5(json.dumps(file_data['data']).encode()).hexdigest()[:10]
                response_file = kb_path / f"analysis_{file_hash}.json"
                
                # Create response data with metadata
                response_data = {
                    'text': text_representation.strip(),
                    'prompt': self._get_analysis_prompt(),
                    'timestamp': timestamp,
                    'metadata': {
                        'model': 'gemini',
                        'source_file': file_data.get('metadata', {}).get('file_path', 'unknown'),
                        'input_data': file_data['data']
                    }
                }
                
                # Save response file
                with open(response_file, 'w') as f:
                    json.dump(response_data, f, indent=2)
                
                self.logger.info(f"Saved Gemini response to {response_file}")
                processed_texts.append(text_representation)

            except Exception as e:
                self.logger.error(f"Failed to process analysis file: {e}")
                continue

        if not processed_texts:
            self.logger.warning("No analysis files were successfully processed")
            return []
                
        # Combine all processed texts with clear section markers
        text_representation = "\n\n=== Section Break ===\n\n".join(processed_texts)

        # Create chunks with metadata
        chunks = []
        sections = text_representation.split("=== Section Break ===")
            
        # Calculate optimal chunk size and overlap
        avg_section_length = sum(len(s.strip()) for s in sections) / len(sections)
        chunk_overlap = min(300, int(avg_section_length * 0.3))  # 30% overlap up to 300 chars
        
        for i, section in enumerate(sections):
                # Skip empty sections
                if not section.strip():
                    continue
                
                chunk_text = section.strip()
            
                # Add context from previous/next sections for overlap
                if i > 0:
                    prev_section = sections[i-1].strip()
                    chunk_text = prev_section[-chunk_overlap:] + "\n\n" + chunk_text
                if i < len(sections) - 1:
                    next_section = sections[i+1].strip()
                    chunk_text = chunk_text + "\n\n" + next_section[:chunk_overlap]
            
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        'source': f"analysis_{i}",  # Use sequential identifier
                        'section': f"section_{i+1}",
                        'timestamp': time.time(),
                        'type': 'analysis',
                        'total_sections': len(sections),
                        'chunk_stats': {
                            'length': len(chunk_text),
                            'overlap': chunk_overlap,
                            'position': i + 1
                        },
                        'doc_stats': {}  # Initialize empty doc stats
                    }
                })

        # Validate results
        if not chunks:
            raise ValueError("No valid chunks created")

        if not self.gemini_enabled or not self.gemini_model:
            raise ValueError("Gemini model not initialized")

        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks

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
            if not analysis_files:
                self.logger.warning("No new analysis files found")
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

            # Generate text representations if Gemini is available
            text_representations = []
            if self.gemini_enabled and self.gemini_model:
                text_representations = self._generate_text_representations(processed_files)
            else:
                # Use raw analysis data if Gemini not available
                text_representations = [{
                    'text': str(file_data.get('data', {})),
                    'metadata': file_data.get('metadata', {})
                } for file_data in processed_files]

            # Create chunks from text representations
            chunks = self._create_chunks(text_representations)

            # Load existing knowledge base texts
            kb_texts = self._load_existing_kb_texts(kb_path)

            # Create vector store
            vectordb = self._create_vector_store(kb_texts + chunks, store_path)

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

    def _create_chunks(self, text_representations: List[Dict]) -> List[Dict]:
        """Create chunks from text representations"""
        chunks = []
        for text_rep in text_representations:
            sections = text_rep['text'].split('\n\n')
            
            # Calculate chunk parameters
            avg_length = sum(len(s) for s in sections) / len(sections)
            chunk_overlap = min(200, int(avg_length * 0.2))
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                    
                chunk_text = section.strip()
                
                # Add overlap context
                if i > 0:
                    chunk_text = sections[i-1][-chunk_overlap:] + "\n\n" + chunk_text
                if i < len(sections) - 1:
                    chunk_text = chunk_text + "\n\n" + sections[i+1][:chunk_overlap]
                    
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **text_rep['metadata'],
                        'section': f"section_{i+1}",
                        'total_sections': len(sections),
                        'chunk_stats': {
                            'length': len(chunk_text),
                            'overlap': chunk_overlap,
                            'position': i + 1
                        }
                    }
                })
                
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

    def _create_vector_store(self, documents: List[Dict], store_path: Path) -> FAISS:
        """Create FAISS vector store from documents"""
        vectordb = FAISS.from_texts(
            texts=[doc["text"].strip() for doc in documents],
            embedding=self.embeddings,
            metadatas=[doc["metadata"] for doc in documents]
        )
        
        store_path.parent.mkdir(parents=True, exist_ok=True)
        vectordb.save_local(str(store_path))
        return vectordb

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
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.7
            }
        )

        # Custom prompt template
        prompt_template = """You are discussing video analysis results with a researcher or colleague.
Use the following analysis context to answer questions about the video.

Analysis Context:
{summaries}

Question: {question}

Guidelines:
1. Keep responses between 30-50 words
2. Be clear and concise
3. Reference specific frames/timestamps
4. Only use information from the context
5. Express uncertainty when needed
6. go above 50 words while still focusing on consise responses only when asked to expand or elaborate.
7. If not enough information is available but is a well known and still context relevant topic, answer but only with facts.
8. If the question is not related to the analysis context, politely decline to answer.

Respond naturally but briefly."""

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
        
    def query_knowledge_base(self, query: str, chain: Optional[RetrievalQAWithSourcesChain] = None, chat_history: Optional[List[Dict]] = None) -> Dict:
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
            # Ensure chain is initialized
            if not chain:
                raise ValueError("RAG chain not initialized. Please create knowledge base first.")
                
            # Query the chain
            try:
                response = chain({"question": enhanced_query})
                
                # Analyze query for tool suggestions
                tool_suggestions = self._analyze_for_tools(enhanced_query, response['answer'])
                if tool_suggestions:
                    response['answer'] += "\n\n" + tool_suggestions
                    response['suggested_tools'] = tool_suggestions
                    
            except Exception as e:
                self.logger.error(f"Chain query error: {e}")
                return {
                    "error": str(e),
                    "answer": "I encountered an error processing your query. Please try again."
                }
            
            # Validate response length
            answer = response["answer"].strip()
            word_count = len(answer.split())
            
            # if word_count < 30:
            #     answer += " " + self.llm.predict("Please expand this response to at least 30 words while maintaining the same meaning: " + answer)
            # elif word_count > 50:
            #     answer = self.llm.predict("Summarize this in 50 words or less while keeping key information: " + answer)
            
            return {
                "answer": answer,
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
