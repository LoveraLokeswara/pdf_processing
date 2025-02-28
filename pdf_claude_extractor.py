import streamlit as st
import anthropic
import base64
import io
import os
from datetime import datetime
import pandas as pd
import docx2txt
import openpyxl

# Set page configuration
st.set_page_config(
    page_title="Document Table Extractor",
    page_icon="📄",
    layout="wide"
)

# App title and description
st.title("Document Table Extractor")
st.markdown("Upload a document file to extract tables using Claude 3 Sonnet")

# File uploader with expanded file types
uploaded_file = st.file_uploader("Choose a document file", type=["pdf", "docx", "xlsx", "xls"])

# API key input
api_key = os.getenv("ANTHROPIC_API_KEY")

# Function to process documents
def process_document(file, api_key):
    file_extension = file.name.split('.')[-1].lower()
    
    # Prepare file data based on type
    if file_extension == 'pdf':
        file_bytes = file.getvalue()
        file_data = base64.b64encode(file_bytes).decode("utf-8")
        media_type = "application/pdf"
    elif file_extension == 'docx':
        file_bytes = file.getvalue()
        file_data = base64.b64encode(file_bytes).decode("utf-8")
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif file_extension in ['xlsx', 'xls']:
        file_bytes = file.getvalue()
        file_data = base64.b64encode(file_bytes).decode("utf-8")
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        return "Unsupported file type"
    
    prompt = """
    <Instruction> 
    You are an AI assistant tasked with extracting tables from documents. The input is the document file attached at the start of this chat. Analyze ALL PAGES of the document and extract ALL tables into CSV Files. 
    Process the ENTIRE document from beginning to end, making sure to extract tables from EVERY page. Don't stop after the first page or first few tables.
    Use "|" as the delimiter and not ",". Be smart in analyzing the table column and when filling up the table. Don't take the tables as it is, analyze them a bit more. Think step by step and don't be lazy. The tables may have merged cells (between rows), span multiple pages, and contain NaN values. Your task is to:
    1. Extract the table data accurately.
    2. Handle merged cells by duplicating the value across the merged rows.
    3. Output the table as a CSV file that can be downloaded.
    4. Extract the complete table.
    5. Give several outputs, each containing 1 table of the document. Don't combine all tables into 1 CSV file. But, make sure that you combine the same table into 1 file (some tables span over multiple pages, you need to analyze the structure of the tables).
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
                            "media_type": media_type,
                            "data": file_data
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

# Initialize session state to store processed documents and tables
if 'processed_document' not in st.session_state:
    st.session_state.processed_document = None
    
if 'extracted_tables' not in st.session_state:
    st.session_state.extracted_tables = []

if 'raw_response' not in st.session_state:
    st.session_state.raw_response = ""

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = 0
    
if 'extracted_tables_count' not in st.session_state:
    st.session_state.extracted_tables_count = 0

# Process button
if uploaded_file is not None and api_key:
    # Only process if the file has changed or hasn't been processed yet
    if st.session_state.processed_document != uploaded_file.name:
        process_button = st.button("Process Document")
        if process_button:
            with st.spinner(f"Processing {uploaded_file.name} with Claude. This may take a minute..."):
                try:
                    # Process the file
                    response = process_document(uploaded_file, api_key)
                    
                    # Extract tables from the response
                    tables = extract_tables(response)
                    
                    # Store in session state to avoid reprocessing
                    st.session_state.processed_document = uploaded_file.name
                    st.session_state.extracted_tables = tables
                    st.session_state.raw_response = response
                    
                    if tables:
                        st.session_state.processed_files += 1
                        st.session_state.extracted_tables_count += len(tables)
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    st.error("Make sure your API key is correct and the document is valid.")
    else:
        st.success(f"Document '{uploaded_file.name}' already processed. Displaying extracted tables.")
        
    # Display results if we have tables in session state
    if st.session_state.extracted_tables:
        st.success(f"Successfully extracted {len(st.session_state.extracted_tables)} tables!")
        
        # Display each table with download option
        for i, table in enumerate(st.session_state.extracted_tables):
            with st.expander(f"Table {i+1}", expanded=True):
                st.markdown(table)
                get_table_download_link(table, i)
        
        # Show full response in collapsible section
        with st.expander("View Raw Claude Response"):
            st.text_area("Response", st.session_state.raw_response, height=400)
    elif st.session_state.processed_document == uploaded_file.name:
        st.warning("No tables were identified in this document. Check the raw response for details.")
        st.text_area("Raw Response", st.session_state.raw_response, height=400)

elif uploaded_file is not None:
    st.warning("Please enter your Anthropic API key to process the file.")
else:
    st.info("Please upload a document file (PDF, DOCX, or Excel) to begin.")

# Add some helpful information
st.markdown("---")
st.markdown("""
### About this tool
This application extracts tables from document files using Claude 3 Sonnet. The extracted tables are displayed in markdown format and can be downloaded as markdown files.

### Supported File Types
- PDF documents (.pdf)
- Word documents (.docx) - Currently Unavailable
- Excel spreadsheets (.xlsx, .xls) - Currently Unavailable

### Notes
- Your API key is used only for processing and is not stored.
- Large documents may take longer to process.
- Claude works best with clearly structured tables.
""")

# Display usage statistics in sidebar
with st.sidebar:
    st.header("Usage Statistics")
    st.metric("Files Processed", st.session_state.processed_files)
    st.metric("Tables Extracted", st.session_state.extracted_tables_count)