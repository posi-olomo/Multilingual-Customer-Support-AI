import streamlit as st
import os
import logging
from pathlib import Path
from datetime import datetime

# Import your multilingual agent classes
try:
    from multilingual_agent import CustomerSupportAgent
except ImportError:
    st.error("Please ensure the multilingual_agent.py file is in the same directory as this script.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multilingual Customer Support",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    border-bottom: 2px solid #f0f2f6;
    margin-bottom: 2rem;
}

.stats-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.language-indicator {
    background-color: #e3f2fd;
    padding: 0.5rem;
    border-radius: 0.25rem;
    border-left: 4px solid #2196f3;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_agent():
    """Initialize the customer support agent"""
    try:
        if "support_agent" not in st.session_state:
            with st.spinner("Loading multilingual support agent..."):
                st.session_state.support_agent = CustomerSupportAgent()
                logger.info("Customer support agent initialized successfully")
        return st.session_state.support_agent
    except Exception as e:
        st.error(f"Failed to initialize support agent: {str(e)}")
        logger.error(f"Agent initialization failed: {e}")
        return None

def load_statistics():
    """Load basic statistics from CSV files"""
    try:
        stats = {"complaints": 0, "requests": 0, "total_messages": 0}
        
        # Count complaints
        complaints_file = Path("data/complaints.csv")
        if complaints_file.exists():
            with open(complaints_file, 'r', encoding='utf-8') as f:
                stats["complaints"] = max(0, len(f.readlines()) - 1)  # Subtract header
        
        # Count requests
        requests_file = Path("data/requests.csv")
        if requests_file.exists():
            with open(requests_file, 'r', encoding='utf-8') as f:
                stats["requests"] = max(0, len(f.readlines()) - 1)  # Subtract header
        
        stats["total_messages"] = stats["complaints"] + stats["requests"]
        return stats
    except Exception as e:
        logger.error(f"Failed to load statistics: {e}")
        return {"complaints": 0, "requests": 0, "total_messages": 0}

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    
    # OpenAI API Key input
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        key="chatbot_api_key", 
        type="password",
        help="Enter your OpenAI API key to enable the chatbot"
    )
    
    # Set the API key as environment variable
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Session Statistics")
    stats = load_statistics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Complaints", stats["complaints"])
        st.metric("Requests", stats["requests"])
    with col2:
        st.metric("Total Messages", stats["total_messages"])
        if "messages" in st.session_state:
            st.metric("Chat Messages", len(st.session_state.messages) - 1)  # Exclude welcome message
    
    st.markdown("---")
    
    # Language support info
    st.markdown("### 🌍 Supported Languages")
    st.markdown("""
    - 🇺🇸 English
    - 🇳🇬 Yoruba  
    - 🇳🇬 Igbo
    - 🇳🇬 Hausa
    """)
    
    st.markdown("---")
    
    # Action buttons
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
        st.rerun()
    
    if st.button("📊 Export Data", help="Export complaints and requests to CSV"):
        st.info("Data is automatically saved in the 'data' folder")
    
    st.markdown("---")
    
    # Links and info
    st.markdown("### 📚 Resources")
    st.markdown("[OpenAI API Keys](https://platform.openai.com/account/api-keys)")
    st.markdown("[Streamlit Documentation](https://docs.streamlit.io)")

# Main content area
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🌍 Multilingual Customer Support")
st.caption("🤖 AI-powered support in English, Yoruba, Igbo, and Hausa")
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hello! I'm your multilingual customer support assistant. I can help you with complaints, requests, and questions in English, Yoruba, Igbo, or Hausa. How can I assist you today?"
    }]

if "last_detected_language" not in st.session_state:
    st.session_state.last_detected_language = "english"

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Show additional info for assistant messages
        if msg["role"] == "assistant" and "metadata" in msg:
            metadata = msg["metadata"]
            col1, col2, col3 = st.columns(3)
            with col1:
                if "message_type" in metadata:
                    st.caption(f"Type: {metadata['message_type'].title()}")
            with col2:
                if "language" in metadata:
                    st.caption(f"Language: {metadata['language'].title()}")
            with col3:
                if "timestamp" in metadata:
                    st.caption(f"Time: {metadata['timestamp']}")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Check if API key is provided
    if not openai_api_key:
        st.error("⚠️ Please add your OpenAI API key in the sidebar to continue.")
        st.stop()
    
    # Initialize agent
    agent = initialize_agent()
    if not agent:
        st.error("❌ Failed to initialize the support agent. Please check your configuration.")
        st.stop()
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Process message with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("Processing your message..."):
            try:
                # Detect language first for UI feedback
                detected_language = agent.translator.detect_language_name(prompt)
                st.session_state.last_detected_language = detected_language
                
                # Classify message type for UI feedback
                message_type = agent._classify_message(prompt)
                
                # Show processing info
                info_container = st.empty()
                info_container.info(f"🔍 Detected: {detected_language.title()} | Type: {message_type.title()}")
                
                # Get response from agent
                response = agent.process_message(prompt)
                
                # Clear processing info
                info_container.empty()
                
                # Display response
                st.write(response)
                
                # Add metadata for display
                metadata = {
                    "message_type": message_type,
                    "language": detected_language,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "metadata": metadata
                })
                
                # Show success message based on type
                if message_type == "complaint":
                    st.success("✅ Complaint logged successfully!")
                elif message_type == "request":
                    st.success("✅ Request recorded successfully!")
                else:
                    st.info("ℹ️ Information provided from knowledge base")
                    
            except Exception as e:
                error_msg = f"Sorry, I encountered an error processing your message: {str(e)}"
                st.error(error_msg)
                logger.error(f"Error processing message: {e}")
                
                # Add error response to chat
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "I'm sorry, I'm experiencing technical difficulties. Please try again or contact support if the problem persists."
                })

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p>🌍 Multilingual Customer Support Agent | Powered by OpenAI & Streamlit</p>
    <p>Supports English, Yoruba, Igbo, and Hausa languages</p>
</div>
""", unsafe_allow_html=True)