import streamlit as st
import anthropic
import base64
import io
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="PDF Table Extractor",
    page_icon="📄",
    layout="wide"
)

# App title and description
st.title("PDF Table Extractor")
st.markdown("Upload a PDF file to extract tables using Claude 3 Sonnet")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# API key input
api_key = os.getenv("ANTHROPIC_API_KEY")

# Function to process PDF
def process_pdf(pdf_file, api_key):
    # Read and encode the PDF
    pdf_bytes = pdf_file.getvalue()
    pdf_data = base64.b64encode(pdf_bytes).decode("utf-8")
    
    prompt = """
    <Instruction> 
    You are an AI assistant tasked with extracting tables from PDF documents. The input is the PDF file attached at the start of this chat. Analyze ALL PAGES of the PDF and extract ALL tables into CSV Files. 
    Process the ENTIRE document from beginning to end, making sure to extract tables from EVERY page. Don't stop after the first page or first few tables.
    Use "|" as the delimiter and not ",". Be smart in analyzing the table column and when filling up the table. Don't take the tables as it is, analyze them a bit more. Think step by step and don't be lazy. The tables may have merged cells (between rows), span multiple pages, and contain NaN values. Your task is to:
    1. Extract the table data accurately.
    2. Handle merged cells by duplicating the value across the merged rows.
    3. Output the table as a CSV file that can be downloaded.
    4. Extract the complete table.
    5. Give several outputs, each containing 1 table of the PDF. Don't combine all tables into 1 CSV file. But, make sure that you combine the same table into 1 file (some tables span over multiple pages, you need to analyze the structure of the tables).
    6. Use "|" as the delimiter and not ",". Make sure that some of the cells contain a comma inside and it's not split into different columns. 
    7. Add the necessary markdown syntax, such as "---|---" to indicate the parts that are considered as the table header. 

    Instruction on how to extract and analyze the table:  
    1. Remove completely duplicated row, meaning those rows which all columns have the same value as previous row. 
    2. If there exist a column with no input at all in a table, please remove that column and see if the next page's table is a continuation of that table. If yes, then combine them together as 1 table. 
    <\Instruction> 

    Example 1: 
    <Input>
    | Name       | Age | City       |  
    |------------|-----|------------|  
    | John Doe   | 25  | New York   |  
    | Jane Smith |     | Los Angeles|  
    <Input>
    <Output in CSV>
    Name,Age,City 
    John Doe,25,New York 
    Jane Smith,25,Los Angeles
    <Output in CSV>

    Example 2: 
    <Input>
    End of page 1: | Product    | Price | Quantity |  
    |------------|-------|----------|  
    | Apple      | 1.2   | 10       |  
    | Banana     |       | 15       |  
    Page 2: 
    | Orange      | 3.5   | 10       |  
    | Melon     |   2    | 20  |  
    <Input>
    <Output in CSV>
    Product,Price,Quantity
    Apple,1.2,10
    Banana,1.2,15
    Orange, 3.5,10
    Melon,2,20
    <Output in CSV>
    """

    # Send to Claude
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4000,  # Increased token limit to handle larger documents
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data
                        },
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
    )
    
    return message.content[0].text

# Function to extract markdown tables from Claude's response
def extract_tables(response):
    tables = []
    lines = response.split('\n')
    
    # For standard markdown tables
    table_content = []
    collecting = False
    
    # Add support for code blocks with CSV
    csv_content = []
    collecting_csv = False
    
    for line in lines:
        # Start collecting when we see a markdown table structure
        if line.strip().startswith('|') and not collecting:
            collecting = True
            table_content = [line]
        # Continue collecting table content
        elif collecting and line.strip().startswith('|'):
            table_content.append(line)
        # End of table detection
        elif collecting and not line.strip().startswith('|') and len(table_content) > 2:
            tables.append('\n'.join(table_content))
            table_content = []
            collecting = False
        
        # Handle CSV code blocks
        elif line.strip() == "```csv" and not collecting_csv:
            collecting_csv = True
            csv_content = []
        elif collecting_csv and line.strip() != "```":
            # Replace CSV commas with pipes if needed
            if "|" in line:
                csv_content.append(line)  # Already has pipes, add as is
            else:
                # This is a comma-separated line, convert to pipe
                csv_content.append(line.replace(",", "|"))
        elif collecting_csv and line.strip() == "```":
            # End of CSV block
            if csv_content:
                # Convert CSV to markdown table format
                formatted_table = []
                for i, csv_line in enumerate(csv_content):
                    formatted_table.append(f"{csv_line}")
                    # Add header separator after first row
                    if i == 0:
                        # Count pipes to determine number of columns
                        col_count = csv_line.count("|") + 1
                        formatted_table.append("|" + "|".join(["---"] * col_count) + "|")
                
                tables.append('\n'.join(formatted_table))
            csv_content = []
            collecting_csv = False
    
    # Add the last table if there is one
    if collecting and len(table_content) > 2:
        tables.append('\n'.join(table_content))
    if collecting_csv and csv_content:
        formatted_table = []
        for i, csv_line in enumerate(csv_content):
            formatted_table.append(f"{csv_line}")
            if i == 0:
                col_count = csv_line.count("|") + 1
                formatted_table.append("|" + "|".join(["---"] * col_count) + "|")
        
        tables.append('\n'.join(formatted_table))
    
    return tables

# Function to create downloadable files
def get_table_download_link(table_content, index):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"table_{index+1}_{timestamp}.md"
    
    # Create a binary stream
    buf = io.BytesIO()
    buf.write(table_content.encode())
    buf.seek(0)
    
    # Create download button
    return st.download_button(
        label=f"Download Table {index+1} as Markdown",
        data=buf,
        file_name=filename,
        mime="text/markdown"
    )
tables=[]
# Main app logic
if uploaded_file is not None and api_key:
    with st.spinner("Processing PDF with Claude. This may take a minute..."):
        try:
            # Process the file
            response = process_pdf(uploaded_file, api_key)
            
            # Extract tables from the response
            tables = extract_tables(response)
            
            # Display results
            if tables:
                st.success(f"Successfully extracted {len(tables)} tables!")
                
                # Display each table with download option
                for i, table in enumerate(tables):
                    with st.expander(f"Table {i+1}", expanded=True):
                        st.markdown(table)
                        get_table_download_link(table, i)
                
                # Show full response in collapsible section
                with st.expander("View Raw Claude Response"):
                    st.text_area("Response", response, height=400)
            else:
                st.warning("No tables were identified in this PDF. Check the raw response for details.")
                st.text_area("Raw Response", response, height=400)
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.error("Make sure your API key is correct and the PDF is valid.")

elif uploaded_file is not None:
    st.warning("Please enter your Anthropic API key to process the file.")
else:
    st.info("Please upload a PDF file to begin.")

# Add some helpful information
st.markdown("---")
st.markdown("""
### About this tool
This application extracts tables from PDF documents using Claude 3 Sonnet. The extracted tables are displayed in markdown format and can be downloaded as markdown files.

### Notes
- Your API key is used only for processing and is not stored.
- Large PDFs may take longer to process.
- Claude works best with clearly structured tables.
""")

if "processed_files" not in st.session_state:
    st.session_state.processed_files = 0
    
if "extracted_tables" not in st.session_state:
    st.session_state.extracted_tables = 0

if len(tables) > 0:
    st.session_state.processed_files += 1
    st.session_state.extracted_tables += len(tables)
    
with st.sidebar:
    st.header("Usage Statistics")
    st.metric("Files Processed", st.session_state.processed_files)
    st.metric("Tables Extracted", st.session_state.extracted_tables)

# import streamlit as st
# import base64
# import io
# import os
# import json
# import requests
# from datetime import datetime

# # Set page configuration
# st.set_page_config(
#     page_title="PDF Table Extractor",
#     page_icon="📄",
#     layout="wide"
# )

# # App title and description
# st.title("PDF Table Extractor")
# st.markdown("Upload a PDF file to extract tables using various AI models via OpenRouter")

# # File uploader
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# # API key input for OpenRouter
# api_key = os.getenv("OPENROUTER_API_KEY")

# selected_model = "anthropic/claude-3-sonnet"

# # Function to process PDF using OpenRouter
# def process_pdf(pdf_file, api_key, model):
#     # Read and encode the PDF
#     pdf_bytes = pdf_file.getvalue()
#     pdf_data = base64.b64encode(pdf_bytes).decode("utf-8")
    
#     prompt = """
#     <Instruction> 
#     You are an AI assistant tasked with extracting tables from PDF documents. The input is the PDF file attached at the start of this chat. Analyze the PDF and extract the tables into a CSV File. Use "|" as the delimiter and not ",".  Be smart in analyzing the table column and when filling up the table. Don't take the tables as it is, analyze them a bit more. Think step by step and don't be lazy. The tables may have merged cells (between rows), span multiple pages, and contain NaN values. Your task is to:
#     1. Extract the table data accurately.
#     2. Handle merged cells by duplicating the value across the merged rows.
#     3. Output the table as a CSV file that can be downloaded.
#     4. Extract the complete table.
#     5. Give several outputs, each containing 1 table of the PDF. Don't combine all tables into 1 CSV file. But, make sure that you combine the same table into 1 file (some tables span over multiple pages, you need to analyze the structure of the tables).
#     6. Use "|" as the delimiter and not ",". Make sure that some of the cells contain a comma inside and it's not split into different columns. 
#     7. Add the necessary markdown syntax, such as "---|---" to indicate the parts that are considered as the table header. 

#     Instruction on how to extract and analyze the table:  
#     1. Remove completely duplicated row, meaning those rows which all columns have the same value as previous row. 
#     2. If there exist a column with no input at all in a table, please remove that column and see if the next page's table is a continuation of that table. If yes, then combine them together as 1 table. 
#     <\Instruction> 

#     Example 1: 
#     <Input>
#     | Name       | Age | City       |  
#     |------------|-----|------------|  
#     | John Doe   | 25  | New York   |  
#     | Jane Smith |     | Los Angeles|  
#     <Input>
#     <Output in CSV>
#     Name,Age,City 
#     John Doe,25,New York 
#     Jane Smith,25,Los Angeles
#     <Output in CSV>

#     Example 2: 
#     <Input>
#     End of page 1: | Product    | Price | Quantity |  
#     |------------|-------|----------|  
#     | Apple      | 1.2   | 10       |  
#     | Banana     |       | 15       |  
#     Page 2: 
#     | Orange      | 3.5   | 10       |  
#     | Melon     |   2    | 20  |  
#     <Input>
#     <Output in CSV>
#     Product,Price,Quantity
#     Apple,1.2,10
#     Banana,1.2,15
#     Orange, 3.5,10
#     Melon,2,20
#     <Output in CSV>
#     """

#     # OpenRouter API endpoint
#     url = "https://openrouter.ai/api/v1/chat/completions"
    
#     # Headers for OpenRouter
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}",
#         "HTTP-Referer": "https://pdf-table-extractor.com",
#         "X-Title": "PDF Table Extractor"
#     }

#     # Payload for OpenRouter
#     payload = {
#         "model": model,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:application/pdf;base64,{pdf_data}"
#                         }
#                     }
#                 ]
#             }
#         ],
#         "max_tokens": 4096
#     }

#     try:
#         # Make the API request
#         response = requests.post(url, headers=headers, json=payload)
        
#         # Check if the response is successful
#         response.raise_for_status()
        
#         # Process the response
#         response_json = response.json()
        
#         # Debug: Log the response structure
#         st.session_state['debug_response'] = response_json
        
#         # Check if the expected keys exist
#         if 'choices' not in response_json:
#             st.error("API response doesn't contain 'choices' key.")
#             st.json(response_json)
#             return f"Error: API response format doesn't match expected structure.\n\nResponse: {json.dumps(response_json, indent=2)}"
        
#         if len(response_json['choices']) == 0:
#             return "Error: API returned empty choices array."
        
#         if 'message' not in response_json['choices'][0]:
#             st.error("API response doesn't contain 'message' key in the first choice.")
#             st.json(response_json)
#             return f"Error: API response format doesn't match expected structure.\n\nResponse: {json.dumps(response_json, indent=2)}"
        
#         if 'content' not in response_json['choices'][0]['message']:
#             st.error("API response doesn't contain 'content' key in the message.")
#             st.json(response_json)
#             return f"Error: API response format doesn't match expected structure.\n\nResponse: {json.dumps(response_json, indent=2)}"
        
#         return response_json['choices'][0]['message']['content']
        
#     except requests.exceptions.RequestException as e:
#         # Handle request-related errors
#         st.error(f"Request error: {str(e)}")
#         return f"Error communicating with OpenRouter API: {str(e)}"
    
#     except KeyError as e:
#         # Handle missing key errors
#         st.error(f"Response parsing error: Missing key {str(e)}")
#         return f"Error parsing API response: Missing key {str(e)}"
    
#     except Exception as e:
#         # Handle any other errors
#         st.error(f"Unexpected error: {str(e)}")
#         return f"Unexpected error: {str(e)}"

# # Function to extract markdown tables from the response
# def extract_tables(response):
#     tables = []
#     lines = response.split('\n')
    
#     table_content = []
#     collecting = False
    
#     for line in lines:
#         # Start collecting when we see a markdown table structure
#         if line.strip().startswith('|') and not collecting:
#             collecting = True
#             table_content = [line]
#         # Continue collecting table content
#         elif collecting and line.strip().startswith('|'):
#             table_content.append(line)
#         # End of table detection
#         elif collecting and not line.strip().startswith('|') and len(table_content) > 2:
#             tables.append('\n'.join(table_content))
#             table_content = []
#             collecting = False
    
#     # Add the last table if there is one
#     if collecting and len(table_content) > 2:
#         tables.append('\n'.join(table_content))
    
#     return tables

# # Function to create downloadable files
# def get_table_download_link(table_content, index):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"table_{index+1}_{timestamp}.md"
    
#     # Create a binary stream
#     buf = io.BytesIO()
#     buf.write(table_content.encode())
#     buf.seek(0)
    
#     # Create download button
#     return st.download_button(
#         label=f"Download Table {index+1} as Markdown",
#         data=buf,
#         file_name=filename,
#         mime="text/markdown"
#     )

# # Initialize session state for debug information
# if 'debug_response' not in st.session_state:
#     st.session_state['debug_response'] = None

# # Main app logic
# if uploaded_file is not None and api_key:
#     tables = []  # Initialize tables to an empty list
#     with st.spinner(f"Processing PDF with {selected_model} via OpenRouter. This may take a minute..."):
#         try:
#             # Process the file with the selected model
#             response = process_pdf(uploaded_file, api_key, selected_model)
            
#             # Check if response contains error message
#             if response.startswith("Error:"):
#                 st.error(response)
#             else:
#                 # Extract tables from the response
#                 tables = extract_tables(response)
                
#                 # Display results
#                 if tables and len(tables) > 0:
#                     st.success(f"Successfully extracted {len(tables)} tables using {selected_model}!")
                    
#                     # Display each table with download option
#                     for i, table in enumerate(tables):
#                         with st.expander(f"Table {i+1}", expanded=True):
#                             st.markdown(table)
#                             get_table_download_link(table, i)
                    
#                     # Show full response in collapsible section
#                     with st.expander("View Raw Model Response"):
#                         st.text_area("Response", response, height=400)
#                 else:
#                     st.warning(f"No tables were identified in this PDF using {selected_model}. Check the raw response for details.")
#                     st.text_area("Raw Response", response, height=400)
        
#         except Exception as e:
#             st.error(f"Error processing PDF: {str(e)}")
#             st.error("Make sure your API key is correct and the PDF is valid.")

# elif uploaded_file is not None:
#     st.warning("Please enter your OpenRouter API key to process the file.")
# else:
#     st.info("Please upload a PDF file to begin.")

# # Add debug section to view the raw API response
# if st.session_state['debug_response'] is not None:
#     with st.expander("API Response Debug (Raw JSON)", expanded=False):
#         st.json(st.session_state['debug_response'])

# # Add some helpful information
# st.markdown("---")
# st.markdown("""
# ### About this tool
# This application extracts tables from PDF documents using various AI models via OpenRouter. The extracted tables are displayed in markdown format and can be downloaded as markdown files.

# ### Notes
# - Your API key is used only for processing and is not stored.
# - Large PDFs may take longer to process.
# - Different models may produce varying results for the same document.
# - You can switch between models to find the one that works best for your specific PDF.
# - OpenRouter API costs will apply based on the model you select.
# """)

# # Usage metrics
# if "processed_files" not in st.session_state:
#     st.session_state.processed_files = 0
    
# if "extracted_tables" not in st.session_state:
#     st.session_state.extracted_tables = 0

# if len(tables) > 0:
#     st.session_state.processed_files += 1
#     st.session_state.extracted_tables += len(tables)
    
# with st.sidebar:
#     st.header("Usage Statistics")
#     st.metric("Files Processed", st.session_state.processed_files)
#     st.metric("Tables Extracted", st.session_state.extracted_tables)




# import streamlit as st
# import requests
# import os

# # OpenRouter API Settings
# OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# API_KEY = os.getenv("OPENROUTER_API_KEY")  # Replace with your OpenRouter API key

# # SYSTEM PROMPT VARIABLE (Modify this)
# SYSTEM_PROMPT = """
# <Instruction> 
# You are an AI assistant tasked with extracting tables from PDF documents. The input is the PDF file attached at the start of this chat. Analyze the PDF and extract the tables into a CSV File. Use "|" as the delimiter and not ",".  Be smart in analyzing the table column and when filling up the table. Don't take the tables as it is, analyze them a bit more. Think step by step and don't be lazy. The tables may have merged cells (between rows), span multiple pages, and contain NaN values. Your task is to:
# 1. Extract the table data accurately.
# 2. Handle merged cells by duplicating the value across the merged rows.
# 3. Output the table as a CSV file that can be downloaded.
# 4. Extract the complete table.
# 5. Give several outputs, each containing 1 table of the PDF. Don't combine all tables into 1 CSV file. But, make sure that you combine the same table into 1 file (some tables span over multiple pages, you need to analyze the structure of the tables).
# 6. Use "|" as the delimiter and not ",". Make sure that some of the cells contain a comma inside and it's not split into different columns. 
# 7. Add the necessary markdown syntax, such as "---|---" to indicate the parts that are considered as the table header. 

# Instruction on how to extract and analyze the table:  
# 1. Remove completely duplicated row, meaning those rows which all columns have the same value as previous row. 
# 2. If there exist a column with no input at all in a table, please remove that column and see if the next page's table is a continuation of that table. If yes, then combine them together as 1 table. 
# <\Instruction> 

# Example 1: 
# <Input>
# | Name       | Age | City       |  
# |------------|-----|------------|  
# | John Doe   | 25  | New York   |  
# | Jane Smith |     | Los Angeles|  
# <Input>
# <Output in CSV>
# Name,Age,City 
# John Doe,25,New York 
# Jane Smith,25,Los Angeles
# <Output in CSV>

# Example 2: 
# <Input>
# End of page 1: | Product    | Price | Quantity |  
# |------------|-------|----------|  
# | Apple      | 1.2   | 10       |  
# | Banana     |       | 15       |  
# Page 2: 
# | Orange      | 3.5   | 10       |  
# | Melon     |   2    | 20  |  
# <Input>
# <Output in CSV>
# Product,Price,Quantity
# Apple,1.2,10
# Banana,1.2,15
# Orange, 3.5,10
# Melon,2,20
# <Output in CSV>
# """

# # Function to send PDF to Claude via OpenRouter
# def query_claude_with_pdf(file_bytes, file_name):
#     headers = {
#         "Authorization": f"Bearer {API_KEY}"
#     }
#     files = {
#         "file": (file_name, file_bytes, "application/pdf")
#     }
#     payload = {
#         "model": "anthropic/claude-3-sonnet",  # Ensure you're using Claude-3 with PDF support
#         "messages": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": "Please extract the table here."}
#         ],
#         "max_tokens": 1024,
#         "temperature": 0.7
#     }
    
#     response = requests.post(OPENROUTER_API_URL, headers=headers, files=files, data={"payload": str(payload)})
    
#     if response.status_code == 200:
#         return response.json().get("content", "Error processing request")
#     else:
#         return f"Error: {response.status_code}, {response.text}"

# # Streamlit UI
# st.title("Claude PDF Processor (via OpenRouter)")
# st.write("Upload a PDF and get insights using Claude's built-in PDF processor.")

# uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# if uploaded_file:
#     st.write("Processing with Claude...")
#     pdf_bytes = uploaded_file.read()
#     response = query_claude_with_pdf(pdf_bytes, uploaded_file.name)
    
#     st.subheader("Claude's Response")
#     st.write(response)
