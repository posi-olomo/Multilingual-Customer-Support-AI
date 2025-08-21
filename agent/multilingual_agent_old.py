from langchain.prompts import PromptTemplate 
from langdetect import detect, DetectorFactory
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import pipeline
# from langchain.chains import LLMChain 
from langchain.agents import initialize_agent, Tool 
# from langchain.llms import OpenAI 
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # Facebook AI Similarity Search
import faiss
# import pickle 
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
import getpass

# Load environment variables from .env files
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


# Load FAISS index 
# with open("data/vectorstore/index.pkl", "rb") as f:
    # db = pickle.load(f)
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
db = FAISS.load_local("data/vectorstore", embeddings, allow_dangerous_deserialization=True)

"""
# Translation pipelines 
# English <-> Yoruba
translate_en_yo = pipeline("translation", model="Helsinki-NLP/opus-mt-en-yo")
translate_yo_en = pipeline("translation", model="Helsinki-NLP/opus-mt-yo-en")

# English <-> Igbo
translate_en_ig = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ig")
translate_ig_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ig-en")

# English <-> Hausa
translate_en_ha = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ha")
translate_ha_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ha-en")
"""

class CSVStorage:
    def __init__(self, base_dir="data"):
        model_name = "facebook/m2m100_418M"
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok= True)
        self.complaints_file = self.base_dir / "complaints.csv"
        self.requests_file = self.base_dir / "requests.csv"

        # Initialize CSV files with headers if they don't exist
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        complaints_headers = ["id", "timestamp", "query", "language", "status", "source", "priority"]
        requests_headers = ["id", "timestamp", "query", "language", "status", "source", "priority"]

        # Initialize csv files if they don't exist
        if not self.complaints_file.exists():
            with open(self.complaints_file, 'w', newline = '', encoding = 'utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(complaints_headers)

        if not self.requests_file.exists():
            with open(self.requests_file, 'w', newline = '', encoding = 'utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(requests_headers)

    def detect_language_name(self, text):
        """
        Detect language to English, Yoruba, Igbo or Huasa. 
        Fallback to English.
        """
        try:
            code = detect(text)
        except Exception:
            return "english"

        mapping = {
            "en": "english",
            "yo": "yoruba",
            "ig": "igbo",
            "ha": "hausa",
        }
        return mapping.get(code, "english")

    def translate(self, text, src_lang, trg_lang):
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors = "pt")
        generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.get_lang_id(trg_lang))
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    """
    def _hf_out(self, x):
        # Each HF translation return [{'translation_text': '...'}]
        return x[0]["translation_text"]
    """
    def to_english(self, text, lang):
        if lang == "yoruba":
            return self.translate(text, lang, "en")
        elif lang == "igbo":
            return self.translate(text, lang, "en")
        elif lang == "hausa":
            return self.translate(text, lang, "en")
        else:
            return text
    
    def from_english(self, text, lang):
        if lang == "yoruba":
            return self.translate(text, "en", lang)
        elif lang == "igbo":
            return self.translate(text, "en", lang)
        elif lang == "hausa":
            return self.translate(text, "en", lang)
        else:
            return text


    def log_complaint(self, query, lang, metadata=None):

        lang = self.detect_language_name(query)

        # Year, month, day _ Hour, month, Second, milliseconds [:-3]
        # This ensures each complaint ID is unique even if multiple tickets are logged in the same second.
        complaint_id = f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        timestamp = datetime.now().isoformat()

        # Extract metadata or use defaults
        meta = metadata or {}
        source = meta.get("source", "chat_agent")
        priority = meta.get("priority", "medium")

        # Append to CSV
        with open(self.complaints_file, 'a', newline = '', encoding = 'utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([complaint_id, timestamp, query, lang, 'open', source, priority])
        if lang != "english":
            return self.from_english("Your complaint has been logged. We'll get back to you shortly", lang)
        else:
            return "Your complaint has been logged. We'll get back to you shortly"

    def add_request(self, query, lang, metadata = None):
        lang = self.detect_language_name(query)

        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        timestamp = datetime.now().isoformat()

        # Extract metadata or use defaults
        meta = metadata or {}
        source = meta.get('source', 'chat_agent')
        request_type = meta.get('request_type', 'general')

        # Append to CSV 
        with open(self.requests_file, 'a', newline = '', encoding = 'utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([request_id, timestamp, query, lang, 'pending', source, request_type])

        if lang != "english":
            return self.from_english("I have noted your request. Our team will follow up with you soon.", lang)
        else:
            return "I have noted your request. Our team will follow up with you soom."


# Initialize storage 
storage = CSVStorage()

# complaints_db = []
# requests_db = []
# llm = OpenAI(temperature=0)# , model = "gpt-4o-mini") 
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Define the tools
def handle_complaint(query):
    # Auto-detect priority based on keywords
    priority = 'high' if any(word in query.lower() for word in ['urgent', 'terrible', 'awful', 'damaged', 'broken', 'worst', 'horrible']) else 'medium'
    # complaints_db.append(query)
    return storage.log_complaint(query, lang, metadata={"source": "chat_agent", "priority": priority})

def handle_request(query):
    # Auto-categorize request type 
    request_type = 'booking' if any(word in query.lower() for word in ['booking', 'reserve', 'appointment', 'schedule']) else 'general'
    return storage.add_request(query, metadata={"source": 'chat_agent', "request_type":request_type})

def handle_question(query):
    # Detect query language
    lang = storage.detect_language_name(query)

    # Translate to English 
    query_en = storage.to_english(query, lang)

    # Perform similarity search in the knowledge base
    # k=2 means we retrieve the top 2 most similar documents
    docs = db.similarity_search(query_en, k=2)

    if not docs:
        if lang != "english":
            # If no documents found, try translating the query to English
            return storage.from_english("Sorry, I couldn't find any information related to your question.", lang)

        else:
            return "Sorry, I couldn't find any information related to your question."

    context = ' '.join([d.page_content for d in docs])

    prompt = PromptTemplate.from_template(
        "You are a helpful support assistant. \n\n"
        "Question: {query} \n\n"
        "Context (from knowledge base): {context} \n\n"
        "Answer the question clearly using the context above."
    )
    final_prompt = prompt.format(query=query, context=context)

    answer = llm.invoke(final_prompt).content
    final_answer = storage.from_english(answer, lang)
    return final_answer
    # " ".join([d.page_content for d in docs])

# Tool registry
tools = [
    Tool(
        name = "ComplaintHandler",
        func = handle_complaint, 
        description = "Handles customer complaints"
    ),
    Tool(
        name = "RequestHandler",
        func = handle_request,
        description = "Handles customer requests like bookings"
    ),
    Tool(
        name = "QuestionHandler",
        func = handle_question,
        description = "Answers customer questions using the knowledge base"
    )
]
    
# Tagging Chain 


tag_prompt = PromptTemplate(
    input_variables = ["text"],
    template = """ Classify the following customer message into one of the following categories:
    1. Complaint
    2. Request
    3. Question
    4. Other

    Message: {text}
    Category:"""
)

#tag_chain = LLMChain(llm=llm, prompt=tag_prompt)
tag_chain = tag_prompt | llm

# Agent Logic
def customer_support_agent(query):
    # tag = tag_chain.run(query).strip().lower()
    tag = tag_chain.invoke(query).content.strip().lower()

    if "complaint" in tag:
        return handle_complaint(query)
    elif "request" in tag:
        return handle_request(query)
    elif "question" in tag:
        return handle_question(query)
    else:
        return "Thank you for reaching out. Could you please clarify your request?"

if __name__ == "__main__":
    print("Customer Support Agent - Local CSV Storage")
    print("=" * 50)
    
    # Test the system
    print("\nTesting the system...")
    
    response1 = customer_support_agent("My order arrived late and the package was damaged!")
    print(f"Response 1: {response1}")
    
    response2 = customer_support_agent("I'd like to book a table for tomorrow evening")
    print(f"Response 2: {response2}")
    
    response3 = customer_support_agent("What are your opening hours?")
    print(f"Response 3: {response3}")
    
    response4 = customer_support_agent("This service is terrible! I want a refund urgently!")
    print(f"Response 4: {response4}")