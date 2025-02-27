import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from typing import List, Tuple
import requests
import os
import tempfile

class InsuranceDocument:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text = self.load_pdf_with_mupdf()
        self.index, self.chunks, self.model = self.create_faiss_index()
        
    def load_pdf_with_mupdf(self) -> str:
        """
        Enhanced PDF loader using PyMuPDF with better handling of tables and structured content
        """
        text = ""
        doc = fitz.open(self.file_path)
        
        for page_num, page in enumerate(doc, 1):
            # Add page metadata for better context
            text += f"\n=== Page {page_num} ===\n"
            
            # For pages with tables (pages 41+), use a different text extraction mode
            if page_num >= 41:
                # Get text in blocks which better preserves table structure
                blocks = page.get_text("blocks")
                table_text = ""
                for block in blocks:
                    block_text = block[4]  # The text content is at index 4
                    table_text += f"{block_text}\n"
                text += f"TABLE CONTENT:\n{table_text}\n"
            else:
                # Standard text extraction for regular pages
                page_text = page.get_text()
                text += f"{page_text}\n"
                
            # Add section markers for important parts
            if "Annexure" in page_text:
                text += "=== ANNEXURE SECTION START ===\n"
            elif "Exclusions" in page_text:
                text += "=== EXCLUSIONS SECTION START ===\n"
        
        doc.close()
        return text
    
    def create_chunks(self, text: str, chunk_size: int = 2000, overlap: int = 400) -> List[str]:
        """
        Enhanced chunking with better preservation of context and table structure
        """
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            # Preserve section markers in chunks
            if line.startswith("==="):
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = len(line)
                continue
                
            # Special handling for table content
            if "TABLE CONTENT:" in line:
                chunk_size = 1000  # Smaller chunks for tables
            else:
                chunk_size = 2000
                
            if current_length + len(line) > chunk_size:
                chunks.append('\n'.join(current_chunk))
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:] + [line]
                current_length = sum(len(l) for l in current_chunk)
            else:
                current_chunk.append(line)
                current_length += len(line)
                
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
        
    def create_faiss_index(self) -> Tuple[faiss.IndexFlatL2, List[str], SentenceTransformer]:
        """
        Create FAISS index with sentence transformers embedding
        """
        # SentenceTransformers doesn't depend on Keras internally
        model = SentenceTransformer('all-MiniLM-L6-v2')
        chunks = self.create_chunks(self.text)
        chunk_embeddings = model.encode(chunks)
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(chunk_embeddings))
        return index, chunks, model

    def query_document(self, query: str, k: int = 5) -> List[str]:
        """
        Enhanced query function with better context retrieval
        """
        query_embedding = self.model.encode([query])
        k_initial = min(k * 2, len(self.chunks))
        distances, indices = self.index.search(np.array(query_embedding), k_initial)
        
        # Prioritize chunks based on relevance and section importance
        relevant_chunks = []
        table_chunks = []
        annexure_chunks = []
        general_chunks = []
        
        for idx in indices[0]:
            chunk = self.chunks[idx]
            if "TABLE CONTENT:" in chunk:
                table_chunks.append(chunk)
            elif "=== ANNEXURE SECTION" in chunk:
                annexure_chunks.append(chunk)
            else:
                general_chunks.append(chunk)
        
        # Combine chunks in priority order
        relevant_chunks = table_chunks + annexure_chunks + general_chunks
        return relevant_chunks[:k]

SYSTEM_PROMPT="""<Inputs>
{$POLICY_DOCUMENT}
{$USER_QUERY}
</Inputs>


<Instructions Structure>
Your answer should only come from Annexure V table page 50-58.
</Instructions Structure>


<Instructions>
You are an expert insurance policy analyst. Your task is to answer queries about insurance coverage by analyzing ONLY the provided policy document. 


<policy_document>
{$POLICY_DOCUMENT}
</policy_document>


Now, I will analyze the policy document to answer this question:
<query>{$USER_QUERY}</query>

## Analysis Protocol
When answering any query, I will:
Immediately go to the Annexure V tables. Provide the numerical information from these pages. 


I will document my search in the <thinking> section to show which specific pages and sections I examined.
Think step by step and show me your thinking.
Now I will analyze the policy document to answer your specific query.
</Instructions>
"""

class InsuranceChatbot:
    def __init__(self, doc_path: str):
        self.document = InsuranceDocument(doc_path)
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def get_response(self, query: str) -> str:
        relevant_chunks = self.document.query_document(query)
        enhanced_query = self._prepare_query(query, relevant_chunks)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "anthropic/claude-3-sonnet",  # OpenRouter's model name for Claude
            "temperature": 0,  
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": enhanced_query
                }
            ]
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _prepare_query(self, query: str, chunks: List[str]) -> str:
        return f"""Please analyze the following information about {query}.
        Pay special attention to any specific amounts, coverage details, and conditions.
        
        Context:
        {' '.join(chunks)}
        
        Question:
        {query}"""

def main():
    st.set_page_config(page_title="Insurance Policy Assistant", layout="wide")
    
    st.title("Insurance Policy Assistant")
    st.markdown("Upload your insurance policy document and ask questions about your coverage.")
    
    # Initialize session states
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Settings")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload Insurance Policy PDF", type="pdf")
        
        if uploaded_file is not None:
            # Save the uploaded file to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            # Initialize chatbot with progress indicator
            with st.spinner("Processing document... This may take a few minutes."):
                try:
                    st.session_state.chatbot = InsuranceChatbot(
                        doc_path=temp_path
                    )
                    st.session_state.pdf_uploaded = True
                    st.success(f"Document '{uploaded_file.name}' processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        
        st.divider()
        st.markdown("### About")
        st.info(
            "This tool uses AI to analyze insurance policy documents. "
            "Upload your policy document and ask questions about coverage, "
            "exclusions, and benefits."
        )

    # Main chat interface
    if st.session_state.pdf_uploaded:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask about your policy coverage (e.g., 'Is ambulance service covered?')"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response from chatbot
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your policy..."):
                    response = st.session_state.chatbot.get_response(prompt)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Display instruction message if no document is uploaded
        st.info("👈 Please upload your insurance policy document in the sidebar to get started.")
        
        # Sample questions to ask
        with st.expander("Sample questions you can ask after uploading a policy document"):
            st.markdown("""
            - Is emergency ambulance service covered?
            - What's the coverage for dental procedures?
            - Are pre-existing conditions covered?
            - What's the deductible for hospitalization?
            - Is maternity care included in my policy?
            - What are the exclusions for outpatient treatments?
            """)

if __name__ == "__main__":
    main()