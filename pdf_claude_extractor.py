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
st.markdown("Upload a document file to extract tables using Claude 3.7 Sonnet")

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
You are an AI assistant tasked with extracting tables from PDF documents. The input is the PDF file attached at the start of this chat. Analyze the PDF and extract the tables into a CSV File. Use “|” as the delimiter and not “,”.  Be smart in analyzing the table column and when filling up the table. Don’t take the tables as it is, analyze them a bit more. Think step by step and don’t be lazy. The tables may have merged cells (between rows), span multiple pages, and contain NaN values. Your task is to:
Extract the table data accurately.
Handle merged cells by duplicating the value across the merged rows.
Output the table as a CSV file that can be downloaded.
Extract the complete table.
Give several outputs, each containing 1 table of the PDF. Don’t combine all tables into 1 CSV file. But, make sure that you combine the same table into 1 file (some tables span over multiple pages, you need to analyze the structure of the tables).
Use “|” as the delimiter and not “,”. Make sure that some of the cells contain a comma inside and it’s not split into different columns. 
Add the necessary markdown syntax, such as “---|---” to indicate the parts that are considered as the table header. 

Instruction on how to extract and analyze the table:  
Remove completely duplicated row, meaning those rows which all columns have the same value as previous row. 
If there exist a column with no input at all in a table, please remove that column and see if the next page’s table is a continuation of that table. If yes, then combine them together as 1 table. 

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
End of page 1: 
| Product    | Price | Quantity |  
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
        model="claude-3-7-sonnet-20250219",
        max_tokens=10000,  # Increased token limit to handle larger documents
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
    
    # For code blocks with table content
    code_block_content = []
    collecting_code_block = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for markdown table format
        if line.strip().startswith('|') and not collecting:
            collecting = True
            table_content = [line]
        elif collecting and line.strip().startswith('|'):
            table_content.append(line)
        elif collecting and not line.strip().startswith('|') and len(table_content) > 2:
            tables.append('\n'.join(table_content))
            table_content = []
            collecting = False
            
        # Check for code blocks that might contain tables
        elif line.strip() == "```" and not collecting_code_block:
            collecting_code_block = True
            code_block_content = []
            # Skip the current line and check if next line contains table data or format info
            i += 1
            if i < len(lines):
                # Skip format identifier like "csv" if present
                if not lines[i].strip().startswith('|') and not '|' in lines[i]:
                    i += 1
            continue
        elif collecting_code_block and '|' in line:
            code_block_content.append(line)
        elif collecting_code_block and line.strip() == "```":
            if code_block_content:
                # Convert to markdown table format if needed
                formatted_table = []
                for j, table_line in enumerate(code_block_content):
                    # If it doesn't start with | but contains |, add | at beginning and end
                    if '|' in table_line and not table_line.strip().startswith('|'):
                        table_line = '|' + table_line + '|'
                    formatted_table.append(table_line)
                    
                    # Add header separator after first row if it doesn't exist
                    if j == 0 and (len(code_block_content) == 1 or not code_block_content[1].strip().startswith('|---')):
                        col_count = table_line.count('|') - 1
                        formatted_table.append('|' + '|'.join(['---'] * col_count) + '|')
                
                tables.append('\n'.join(formatted_table))
            code_block_content = []
            collecting_code_block = False
            
        i += 1
    
    # Handle any remaining content
    if collecting and len(table_content) > 2:
        tables.append('\n'.join(table_content))
    if collecting_code_block and code_block_content:
        formatted_table = []
        for j, table_line in enumerate(code_block_content):
            if '|' in table_line and not table_line.strip().startswith('|'):
                table_line = '|' + table_line + '|'
            formatted_table.append(table_line)
            if j == 0 and (len(code_block_content) == 1 or not code_block_content[1].strip().startswith('|---')):
                col_count = table_line.count('|') - 1
                formatted_table.append('|' + '|'.join(['---'] * col_count) + '|')
        
        tables.append('\n'.join(formatted_table))
    
    # Clean up tables to ensure proper markdown format
    cleaned_tables = []
    for table in tables:
        lines = table.split('\n')
        # Ensure each line starts and ends with |
        formatted_lines = []
        for line in lines:
            if '|' in line:
                parts = line.split('|')
                # If first part is empty (line starts with |), keep it as is
                # Otherwise add a | at the beginning
                if not line.strip().startswith('|'):
                    line = '|' + line
                # If last part is empty (line ends with |), keep it as is
                # Otherwise add a | at the end
                if not line.strip().endswith('|'):
                    line = line + '|'
                formatted_lines.append(line)
        
        if formatted_lines:
            cleaned_tables.append('\n'.join(formatted_lines))
    
    return cleaned_tables

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

        ###ADD DISINI
        # Add this after your existing table extraction and display code
        if not st.session_state.extracted_tables and st.session_state.raw_response:
            # Try to extract tables again with a more direct approach
            raw_response = st.session_state.raw_response
            # Look for code blocks with table data
            code_blocks = []
            current_block = []
            in_code_block = False
            
            for line in raw_response.split('\n'):
                if line.strip() == '```':
                    if in_code_block:
                        if current_block:
                            code_blocks.append('\n'.join(current_block))
                            current_block = []
                        in_code_block = False
                    else:
                        in_code_block = True
                elif in_code_block:
                    # Skip the first line if it's just a format indicator (csv, etc.)
                    if current_block or ('|' in line):
                        current_block.append(line)
            
            # Process the extracted code blocks to convert them to markdown tables
            for i, block in enumerate(code_blocks):
                lines = block.split('\n')
                # Skip first line if it's a format indicator and doesn't contain table data
                start_idx = 0
                if lines and not '|' in lines[0]:
                    start_idx = 1
                    
                if len(lines) > start_idx:
                    table_lines = []
                    # Process each line to ensure it's in markdown table format
                    for j, line in enumerate(lines[start_idx:]):
                        if '|' in line:
                            # Ensure the line starts and ends with |
                            if not line.strip().startswith('|'):
                                line = '|' + line
                            if not line.strip().endswith('|'):
                                line = line + '|'
                            table_lines.append(line)
                            
                            # Add header separator after first row if needed
                            if j == 0 and (len(lines) <= start_idx+1 or not '|---' in lines[start_idx+1]):
                                col_count = line.count('|') - 1
                                table_lines.append('|' + '|'.join(['---'] * col_count) + '|')
                    
                    if table_lines:
                        st.session_state.extracted_tables.append('\n'.join(table_lines))
                        
            if st.session_state.extracted_tables:
                st.success(f"Successfully extracted {len(st.session_state.extracted_tables)} tables!")
                
                # Display each table with download option
                for i, table in enumerate(st.session_state.extracted_tables):
                    with st.expander(f"Table {i+1}", expanded=True):
                        st.markdown(table)
                        get_table_download_link(table, i)
        
    ###END OF ADD

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
This application extracts tables from document files using Claude 3.7 Sonnet. The extracted tables are displayed in markdown format and can be downloaded as markdown files.

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