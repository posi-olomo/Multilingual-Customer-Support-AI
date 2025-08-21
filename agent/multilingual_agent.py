from langchain.prompts import PromptTemplate 
from langdetect import detect, DetectorFactory
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langchain.agents import Tool 
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import csv
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
import getpass
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

class TranslationService:
    """Handles multilingual translation using M2M100 model"""
    
    def __init__(self):
        self.model_name = "facebook/m2m100_418M"
        try:
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
            self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
            logger.info("Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise
    
    def detect_language_name(self, text: str) -> str:
        """Detect language and return standardized name"""
        try:
            # Set seed for consistent results
            DetectorFactory.seed = 0
            code = detect(text)
            
            language_mapping = {
                "en": "english",
                "yo": "yoruba", 
                "ig": "igbo",
                "ha": "hausa",
            }
            
            detected_lang = language_mapping.get(code, "english")
            logger.info(f"Detected language: {detected_lang} (code: {code})")
            return detected_lang
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}. Defaulting to English.")
            return "english"
    
    def translate(self, text: str, src_lang: str, trg_lang: str) -> str:
        """Translate text between languages using M2M100"""
        try:
            # Map language names to M2M100 codes
            lang_codes = {
                "english": "en",
                "yoruba": "yo", 
                "igbo": "ig",
                "hausa": "ha"
            }
            
            src_code = lang_codes.get(src_lang, "en")
            trg_code = lang_codes.get(trg_lang, "en")
            
            # If same language, return original text
            if src_code == trg_code:
                return text
            
            self.tokenizer.src_lang = src_code
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            generated_tokens = self.model.generate(
                **encoded, 
                forced_bos_token_id=self.tokenizer.get_lang_id(trg_code),
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            logger.info(f"Translated from {src_lang} to {trg_lang}")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # Return original text if translation fails
    
    def to_english(self, text: str, source_lang: str) -> str:
        """Translate from any supported language to English"""
        if source_lang == "english":
            return text
        return self.translate(text, source_lang, "english")
    
    def from_english(self, text: str, target_lang: str) -> str:
        """Translate from English to any supported language"""
        if target_lang == "english":
            return text
        return self.translate(text, "english", target_lang)

class CSVStorage:
    """Handles CSV storage for complaints and requests"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.complaints_file = self.base_dir / "complaints.csv"
        self.requests_file = self.base_dir / "requests.csv"
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        complaints_headers = ["id", "timestamp", "query", "language", "status", "source", "priority"]
        requests_headers = ["id", "timestamp", "query", "language", "status", "source", "request_type"]
        
        if not self.complaints_file.exists():
            with open(self.complaints_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(complaints_headers)
                
        if not self.requests_file.exists():
            with open(self.requests_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(requests_headers)
    
    def log_complaint(self, query: str, language: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log a customer complaint to CSV"""
        try:
            complaint_id = f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
            timestamp = datetime.now().isoformat()
            
            meta = metadata or {}
            source = meta.get("source", "chat_agent")
            priority = meta.get("priority", "medium")
            
            with open(self.complaints_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([complaint_id, timestamp, query, language, 'open', source, priority])
            
            logger.info(f"Logged complaint: {complaint_id}")
            return complaint_id
            
        except Exception as e:
            logger.error(f"Failed to log complaint: {e}")
            raise
    
    def log_request(self, query: str, language: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log a customer request to CSV"""
        try:
            request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
            timestamp = datetime.now().isoformat()
            
            meta = metadata or {}
            source = meta.get('source', 'chat_agent')
            request_type = meta.get('request_type', 'general')
            
            with open(self.requests_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([request_id, timestamp, query, language, 'pending', source, request_type])
            
            logger.info(f"Logged request: {request_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
            raise

class KnowledgeBase:
    """Handles vector database operations for Q&A"""
    
    def __init__(self, vector_store_path: str = "data/vectorstore"):
        try:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.db = FAISS.load_local(
                vector_store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("Knowledge base loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            self.db = None
    
    def search(self, query: str, k: int = 3) -> str:
        """Search knowledge base and return context"""
        if not self.db:
            return ""
        
        try:
            docs = self.db.similarity_search(query, k=k)
            if not docs:
                return ""
            
            context = ' '.join([doc.page_content for doc in docs])
            logger.info(f"Found {len(docs)} relevant documents")
            return context
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return ""

class CustomerSupportAgent:
    """Main customer support agent class"""
    
    def __init__(self):
        self.translator = TranslationService()
        self.storage = CSVStorage()
        self.knowledge_base = KnowledgeBase()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        
        # Define priority and category keywords
        self.high_priority_keywords = {
            'urgent', 'emergency', 'terrible', 'awful', 'damaged', 'broken', 
            'worst', 'horrible', 'furious', 'angry', 'refund', 'cancel'
        }
        
        self.booking_keywords = {
            'booking', 'reserve', 'appointment', 'schedule', 'book', 
            'reservation', 'table', 'room', 'meeting'
        }
        
        # Classification prompt
        self.tag_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Classify the following customer message into exactly one category:
            
            Categories:
            - Complaint: Customer is expressing dissatisfaction, reporting problems, or requesting refunds
            - Request: Customer is asking for a service, booking, or action to be taken
            - Question: Customer is seeking information or asking for help understanding something
            
            Message: {text}
            
            Respond with only one word: Complaint, Request, or Question"""
        )
        
        self.qa_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are a helpful customer support assistant. Answer the customer's question clearly and professionally using the provided context.

            Customer Question: {query}
            
            Knowledge Base Context: {context}
            
            Instructions:
            - Provide a clear, helpful answer based on the context
            - If the context doesn't contain relevant information, politely say you don't have that specific information
            - Keep your response concise but complete
            - Be friendly and professional
            
            Answer:"""
        )
    
    def _classify_message(self, query: str) -> str:
        """Classify customer message into complaint, request, or question"""
        try:
            chain = self.tag_prompt | self.llm
            result = chain.invoke({"text": query}).content.strip().lower()
            
            if "complaint" in result:
                return "complaint"
            elif "request" in result:
                return "request"
            elif "question" in result:
                return "question"
            else:
                return "question"  # Default to question
                
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "question"  # Default fallback
    
    def _determine_priority(self, query: str) -> str:
        """Determine priority level for complaints"""
        query_lower = query.lower()
        return 'high' if any(keyword in query_lower for keyword in self.high_priority_keywords) else 'medium'
    
    def _determine_request_type(self, query: str) -> str:
        """Determine request type"""
        query_lower = query.lower()
        return 'booking' if any(keyword in query_lower for keyword in self.booking_keywords) else 'general'
    
    def handle_complaint(self, query: str, language: str) -> str:
        """Handle customer complaint"""
        try:
            priority = self._determine_priority(query)
            complaint_id = self.storage.log_complaint(
                query, 
                language, 
                metadata={"source": "chat_agent", "priority": priority}
            )
            
            response = f"Your complaint has been logged with ID {complaint_id}. We'll get back to you shortly."
            return self.translator.from_english(response, language)
            
        except Exception as e:
            logger.error(f"Failed to handle complaint: {e}")
            error_msg = "We're experiencing technical difficulties. Please try again later."
            return self.translator.from_english(error_msg, language)
    
    def handle_request(self, query: str, language: str) -> str:
        """Handle customer request"""
        try:
            request_type = self._determine_request_type(query)
            request_id = self.storage.log_request(
                query, 
                language,
                metadata={"source": "chat_agent", "request_type": request_type}
            )
            
            response = f"I have noted your request with ID {request_id}. Our team will follow up with you soon."
            return self.translator.from_english(response, language)
            
        except Exception as e:
            logger.error(f"Failed to handle request: {e}")
            error_msg = "We're experiencing technical difficulties. Please try again later."
            return self.translator.from_english(error_msg, language)
    
    def handle_question(self, query: str, language: str) -> str:
        """Handle customer question using knowledge base"""
        try:
            # Translate query to English for knowledge base search
            query_en = self.translator.to_english(query, language)
            
            # Search knowledge base
            context = self.knowledge_base.search(query_en)
            
            if not context:
                response = "Sorry, I couldn't find any information related to your question. Please contact our support team for further assistance."
                return self.translator.from_english(response, language)
            
            # Generate answer using LLM
            chain = self.qa_prompt | self.llm
            answer = chain.invoke({"query": query_en, "context": context}).content
            
            # Translate back to original language
            return self.translator.from_english(answer, language)
            
        except Exception as e:
            logger.error(f"Failed to handle question: {e}")
            error_msg = "Sorry, I'm having trouble processing your question right now. Please try again later."
            return self.translator.from_english(error_msg, language)
    
    def process_message(self, query: str) -> str:
        """Main method to process customer messages"""
        if not query.strip():
            return "Please provide a message for me to help you with."
        
        # Detect language
        language = self.translator.detect_language_name(query)
        logger.info(f"Processing message in {language}")
        
        # Classify message
        message_type = self._classify_message(query)
        logger.info(f"Message classified as: {message_type}")
        
        # Route to appropriate handler
        if message_type == "complaint":
            return self.handle_complaint(query, language)
        elif message_type == "request":
            return self.handle_request(query, language)
        else:  # question
            return self.handle_question(query, language)

def main():
    """Main function for testing the system"""
    print("Customer Support Agent - Enhanced Version")
    print("=" * 50)
    
    try:
        agent = CustomerSupportAgent()
        
        # Test cases
        test_queries = [
            "My order arrived late and the package was damaged!",
            "I'd like to book a table for tomorrow evening",
            "What are your opening hours?",
            "This service is terrible! I want a refund urgently!",
            "Can you help me understand your return policy?",
            "I need to schedule a meeting with customer service"
        ]
        
        print("\nTesting the system...")
        print("-" * 30)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            response = agent.process_message(query)
            print(f"Response: {response}")
            
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        print(f"Error: {e}")

# Uncomment the line below to run the main function when this script is executed
"""
if __name__ == "__main__":
    main()
"""
