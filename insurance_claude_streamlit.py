import streamlit as st

st.write("Checking dependencies...")
try:
    import tensorflow as tf
    st.write(f"TensorFlow version: {tf.__version__}")
    
    import keras
    st.write(f"Keras version: {keras.__version__}")
    
    from sentence_transformers import SentenceTransformer
    st.write("SentenceTransformer imported successfully")
    
    # Continue with the rest of your app
except Exception as e:
    st.error(f"Dependency error: {e}")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from typing import List, Tuple, Dict
import requests
import os
import tempfile

class InsuranceDocument:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text = self.load_pdf_with_ocr()
        self.index, self.chunks, self.model = self.create_faiss_index()
        
    def load_pdf_with_ocr(self) -> str:
        """
        Enhanced PDF loader with better handling of tables and structured content
        """
        text = ""
        with open(self.file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages, 1):
                # Add page metadata for better context
                text += f"\n=== Page {page_num} ===\n"
                
                # Special handling for tables (pages 41+)
                if page_num >= 41:
                    images = convert_from_path(self.file_path, first_page=page_num, last_page=page_num)
                    for img in images:
                        ocr_text = pytesseract.image_to_string(img, lang='eng')
                        # Enhanced table structure preservation
                        text += f"TABLE CONTENT:\n{ocr_text}\n"
                else:
                    page_text = page.extract_text()
                    # Preserve section headers
                    text += f"{page_text}\n"
                
                # Add section markers for important parts
                if "Annexure" in page_text:
                    text += "=== ANNEXURE SECTION START ===\n"
                elif "Exclusions" in page_text:
                    text += "=== EXCLUSIONS SECTION START ===\n"
                    
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
        Create FAISS index with enhanced embedding strategy
        """
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
1.⁠ ⁠Begin with an introduction about the AI's role as an insurance policy analyst
2.⁠ ⁠Provide the POLICY_DOCUMENT structure and details
3.⁠ ⁠Explain the search protocol for analyzing the document
4.⁠ ⁠Detail how to handle numerical values and excerpts
5.⁠ ⁠Specify the exact response format with thinking and answer sections
6.⁠ ⁠Provide rules for determining coverage status
7.⁠ ⁠Include example of proper response format
</Instructions Structure>

<Instructions>
You are an expert insurance policy analyst. Your task is to answer queries about insurance coverage by analyzing ONLY the provided policy document. Never use any external knowledge or information not contained in the document.

<policy_document>
{$POLICY_DOCUMENT}
</policy_document>

Now, I will analyze the policy document to answer this question:
<query>{$USER_QUERY}</query>

## Document Structure
The policy document is structured as follows:
•⁠  ⁠Pages 1-40: General policy terms, conditions, and coverage qualifications
•⁠  ⁠Pages 41-45: Benefit comparison tables
•⁠  ⁠Annexure V (Pages 46-50): Policy Benefit Table with columns for "Coverage Type," "Sum Insured (USD)," "Deductible (USD)," and "International/Domestic Scope"

## Analysis Protocol
When answering any query, I will:
1.⁠ ⁠First search Pages 1-40 for general coverage eligibility and conditions
2.⁠ ⁠Then check Annexure V tables for exact numeric values (sum insured and deductibles)
3.⁠ ⁠Verify against General Exclusions sections to confirm there are no disqualifiers
4.⁠ ⁠Make a final coverage determination based on all relevant information

I will document my search in the <thinking> section to show which specific pages and sections I examined.

## Handling Numerical Information
When reporting financial information:
•⁠  ⁠Sum Insured values will be extracted exactly as stated in the "Sum Insured (USD)" column in Annexure V
•⁠  ⁠Deductible amounts will be reported exactly as shown in the "Deductible (USD)" column
•⁠  ⁠I will note any distinctions between International and Domestic coverage
•⁠  ⁠All values will be presented in the exact format shown in the document (e.g., "USD 500-1,000")

## Response Format
I will structure my response in two parts:

<thinking>
In this section, I will document my analysis process:
1.⁠ ⁠Which pages I searched for general eligibility information
2.⁠ ⁠Which tables in Annexure V I checked for numeric values
3.⁠ ⁠Which exclusion sections I verified
4.⁠ ⁠The reasoning for my final coverage determination
</thinking>

<answer>
Is it covered: [Yes/No/Need more information from user] \n
How much is covered: [Exact USD amount/range from document OR "Need more information from user"] \n
Deductible: [Exact USD amount/options from document OR "Need more information from user"] \n
Exact excerpt: "[Direct quote from the document] (Source: [Specific page/section/table])" \n
</answer>

## Coverage Determination Rules
•⁠  ⁠I will only answer "Yes" to coverage if I find explicit confirmation in the document
•⁠  ⁠I will answer "No" if the document explicitly excludes the queried item/service
•⁠  ⁠I will state "Need more information from user" if:
  - The query is too vague to match to specific policy sections
  - The document contains conditional coverage based on factors not provided in the query
  - Multiple possible interpretations exist and clarification is needed

If I cannot find any relevant information about the query in the document, I will report:
<answer>
Is it covered: Need more information from user
How much is covered: No information found in document
Deductible: No information found in document
Exact excerpt: "No specific information about [query topic] was found in the policy document."
</answer>

## Examples of Proper Analysis
Here is how I would analyze a query about emergency dental coverage:

<thinking>
1.⁠ ⁠Searched Pages 25-30 for dental coverage terms and found reference to "emergency dental services" on Page 28
2.⁠ ⁠Located "Emergency Dental" in Annexure V Table 4a
3.⁠ ⁠Checked exclusions in Section 8.3 and found no exclusions for emergency dental care
4.⁠ ⁠Verified sum insured amount of USD 1,500 in Annexure V Table 4a
5.⁠ ⁠Noted different deductibles for Domestic (USD 100) and International (USD 250) treatments
</thinking>

<answer>
Is it covered: Yes
How much is covered: USD 1,500
Deductible: USD 100 (Domestic) / USD 250 (International)
Exact excerpt: "Emergency dental care: $1,500 sum insured with deductible options per treatment basis (Domestic $100, International $250)" (Source: Annexure V Table 4a)
</answer>

Now I will analyze the policy document to answer your specific query.
</Instructions>"""

class InsuranceChatbot:
    def __init__(self, doc_path: str, api_key: str):
        self.document = InsuranceDocument(doc_path)
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def get_response(self, query: str) -> str:
        relevant_chunks = self.document.query_document(query)
        enhanced_query = self._prepare_query(query, relevant_chunks)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            # "HTTP-Referer": "http://localhost:8000",  # Replace with your actual domain
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "anthropic/claude-3-sonnet",  # OpenRouter's model name for Claude
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

    # def start_chat(self):
    #     print("Welcome to the Insurance Assistant! Type 'exit' to quit.")
    #     while True:
    #         query = input("\nYour question: ")
    #         print(f"You: {query}")
    #         if query.lower() == 'exit':
    #             break
    #         response = self.get_response(query)
    #         print(f"\nAssistant: {response}")
    #         print('\n' + '='*50)

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
    
    # Sidebar for API key and PDF upload
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
                        doc_path=temp_path,
                        api_key=os.getenv('OPENROUTER_API_KEY')
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
