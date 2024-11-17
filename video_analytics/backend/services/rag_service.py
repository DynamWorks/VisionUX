from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalQA
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional

class RAGService:
    """Service for RAG-based retrieval and chat"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.llm = ChatOpenAI(model_name=model_name)
        self.logger = logging.getLogger(__name__)
        
    def _load_and_chunk_results(self, results_path: Path) -> List[Dict]:
        """Load analysis results and split into chunks"""
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
        """Create vector store from analysis results"""
        try:
            documents = self._load_and_chunk_results(results_path)
            if not documents:
                return None
                
            # Split documents into chunks
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            chunks = self.text_splitter.create_documents(texts, metadatas=metadatas)
            
            # Create vector store
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            return vectordb
            
        except Exception as e:
            self.logger.error(f"Error creating knowledge base: {str(e)}")
            return None
            
    def get_retrieval_chain(self, vectordb: Chroma) -> ConversationalRetrievalQA:
        """Create retrieval chain with conversation memory"""
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        chain = ConversationalRetrievalQA.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
        
        return chain
        
    def query_knowledge_base(self, query: str, chain: ConversationalRetrievalQA) -> Dict:
        """Query the knowledge base with conversation history"""
        try:
            response = chain({"question": query})
            
            return {
                "answer": response["answer"],
                "sources": [doc.metadata for doc in response["source_documents"]],
                "chat_history": self.memory.chat_memory.messages
            }
            
        except Exception as e:
            self.logger.error(f"Error querying knowledge base: {str(e)}")
            return {
                "error": str(e),
                "answer": "Failed to process query"
            }
