import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import io
import base64
import tempfile
import os
import traceback
import sys
import numpy as np
from collections import defaultdict

# Configure Streamlit page
st.set_page_config(page_title="PDF Table Extractor", layout="wide")

def handle_duplicate_columns(df):
    """Handle duplicate column names in a DataFrame by adding suffixes."""
    if df.columns.duplicated().any():
        # Get list of all columns
        cols = pd.Series(df.columns)
        
        # For duplicated ones, add a suffix _1, _2, etc.
        for dup in cols[cols.duplicated()].unique(): 
            cols[cols.duplicated()] = [f'{dup}_{i}' for i in range(1, sum(cols == dup) + 1)]
            
        # Assign new column names to the DataFrame
        df.columns = cols
    
    return df

def clean_column_name(col_name):
    """Clean column names by removing unwanted characters and whitespace."""
    if isinstance(col_name, str):
        # Replace newlines with spaces
        col_name = col_name.replace('\n', ' ')
        # Strip leading/trailing whitespace
        col_name = col_name.strip()
        # Remove any other problematic characters
        return col_name
    return col_name

def standardize_column_names(df):
    """Standardize column names to lowercase with underscores for special characters."""
    if isinstance(df.columns, pd.Index):
        df.columns = df.columns.str.lower().str.replace(r"[^a-z0-9]", "_", regex=True)
    return df

def extract_header(df):
    """Extract header rows dynamically by looking at the first few rows."""
    for i in range(min(3, len(df))):  # Look at first 3 rows max
        if df.iloc[i].isnull().sum() <= 2:  # Allow minor NaNs
            return df.iloc[i].fillna("").astype(str).str.strip()
    return df.columns  # Default to existing columns if no header found

def merge_multiline_headers(df, num_header_rows=2):
    """Merge multiple header rows into one if multi-line headers are detected."""
    if len(df) < num_header_rows:
        return df
        
    # Check if the first rows might be a multi-line header
    header_candidates = df.iloc[:num_header_rows]
    merged_headers = []
    
    for col_idx in range(len(df.columns)):
        # Extract column values for this position
        values = [str(row[col_idx]) if not pd.isna(row[col_idx]) else "" 
                  for _, row in header_candidates.iterrows()]
        # Join non-empty values with underscore
        merged_header = "_".join([v for v in values if v]).strip("_")
        if not merged_header:  # If still empty, use original column name
            merged_header = f"col_{col_idx}"
        merged_headers.append(merged_header)
    
    return merged_headers

def fill_forward_nan_values(df):
    """Fill NaN values with the value from the previous row in the same column."""
    return df.fillna(method='ffill')

def extract_tables_from_pdf(pdf_file):
    """Extract tables from PDF file using PyMuPDF with enhanced error handling."""
    tables = []
    temp_file_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_file_path = temp_file.name
        
        if debug_mode:
            st.write(f"Debug: PDF saved to temporary file: {temp_file_path}")
        
        # Open the PDF
        doc = fitz.open(temp_file_path)
        if debug_mode:
            st.write(f"Debug: PDF opened successfully. Pages: {len(doc)}")
        
        # Dictionary to track tables that might span multiple pages
        # Key: tuple of column names, Value: list of tables with these columns
        potential_multi_page_tables = defaultdict(list)
        
        # Process each page
        for page_num, page in enumerate(doc):
            if debug_mode:
                st.write(f"Debug: Processing page {page_num+1}/{len(doc)}")
            
            # Extract tables using PyMuPDF's built-in table detection
            try:
                tab = page.find_tables()
                if debug_mode:
                    st.write(f"Debug: Found {len(tab.tables) if tab.tables else 0} tables on page {page_num+1}")
                
                if tab.tables:
                    for table_num, table in enumerate(tab.tables):
                        try:
                            # Extract cells
                            data = []
                            for row_cells in table.extract():
                                data.append([cell.strip() if isinstance(cell, str) else cell for cell in row_cells])
                            
                            # Convert to DataFrame with proper handling of duplicates
                            df = pd.DataFrame(data)
                            
                            # Use dynamic header extraction approach
                            has_header = False
                            if df.shape[0] > 1:
                                # Try to extract header dynamically
                                if use_multiline_headers and df.shape[0] >= 2:
                                    # Handle potential multi-line headers
                                    header_row = merge_multiline_headers(df, num_header_rows=2)
                                    df.columns = header_row
                                    df = df.iloc[2:].reset_index(drop=True)  # Drop both header rows
                                    has_header = True
                                else:
                                    # Use the improved single-line header detection
                                    header_row = extract_header(df)
                                    df.columns = header_row
                                    has_header = True
                                    # Remove the header row after assigning
                                    df = df.iloc[1:].reset_index(drop=True)
                                
                                # Handle duplicate column names
                                df = handle_duplicate_columns(df)
                                
                                # Standardize column names if option is enabled
                                if standardize_columns:
                                    df = standardize_column_names(df)
                            
                            if not has_header:
                                # Just handle duplicates in default column names
                                df = handle_duplicate_columns(df)
                            
                            # Fill NaN values with values from previous rows
                            if enable_forward_fill:
                                df = fill_forward_nan_values(df)
                            
                            # Convert empty strings to NaN for better display and processing
                            df = df.replace('', np.nan)
                            
                            # Store column names as a tuple for checking multi-page tables
                            column_names_tuple = tuple(df.columns)
                            
                            # Store with metadata
                            table_info = {
                                'page': page_num + 1,
                                'table_num': table_num + 1,
                                'data': df,
                                'raw_data': data,
                                'has_header': has_header,
                                'column_count': len(df.columns)
                            }
                            
                            # Add to the multi-page tracking dictionary
                            potential_multi_page_tables[column_names_tuple].append(table_info)
                            
                            # Also add to the regular tables list
                            tables.append(table_info)
                            
                            if debug_mode:
                                st.write(f"Debug: Successfully processed table {table_num+1} on page {page_num+1}")
                        except Exception as e:
                            st.error(f"Error processing table {table_num+1} on page {page_num+1}: {str(e)}")
                            if debug_mode:
                                st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"Error finding tables on page {page_num+1}: {str(e)}")
                if debug_mode:
                    st.code(traceback.format_exc())
        
        # Process potential multi-page tables
        merged_tables = []
        processed_indices = set()
        
        # First pass: Merge tables with exactly the same columns
        for col_tuple, table_list in potential_multi_page_tables.items():
            if len(table_list) > 1:  # Multiple tables with the same columns
                if debug_mode:
                    st.write(f"Debug: Found potential multi-page table with columns: {col_tuple}")
                
                # Sort by page number
                table_list.sort(key=lambda x: (x['page'], x['table_num']))
                
                # Create a new merged DataFrame
                merged_df = pd.concat([t['data'] for t in table_list], ignore_index=True)
                
                # Create merged table info
                merged_table = {
                    'page': f"{table_list[0]['page']}-{table_list[-1]['page']}",
                    'table_num': table_list[0]['table_num'],
                    'data': merged_df,
                    'raw_data': [t['raw_data'] for t in table_list],
                    'has_header': table_list[0]['has_header'],
                    'column_count': len(merged_df.columns),
                    'merged': True,
                    'merged_from': [tables.index(t) for t in table_list]
                }
                
                merged_tables.append(merged_table)
                
                # Mark these tables as processed
                for t in table_list:
                    processed_indices.add(tables.index(t))
        
        # Second pass: Check for tables without headers that might be continuations
        if detect_continuations:
            for i, table in enumerate(tables):
                if i in processed_indices:
                    continue
                    
                if not table['has_header'] and i > 0:
                    prev_table = tables[i-1]
                    
                    # Check if column counts are similar (within 1)
                    if abs(table['column_count'] - prev_table['column_count']) <= 1:
                        if debug_mode:
                            st.write(f"Debug: Table {i+1} appears to be a continuation of table {i}")
                        
                        # Try to align columns if counts don't match
                        if table['column_count'] != prev_table['column_count']:
                            # Use the table with more columns as reference
                            if prev_table['column_count'] > table['column_count']:
                                reference_cols = prev_table['data'].columns
                                table['data'] = table['data'].reindex(columns=reference_cols)
                            else:
                                reference_cols = table['data'].columns
                                prev_table['data'] = prev_table['data'].reindex(columns=reference_cols)
                        
                        # Merge the tables
                        merged_df = pd.concat([prev_table['data'], table['data']], ignore_index=True)
                        
                        # Create merged table info
                        merged_table = {
                            'page': f"{prev_table['page']}-{table['page']}",
                            'table_num': prev_table['table_num'],
                            'data': merged_df,
                            'raw_data': [prev_table['raw_data'], table['raw_data']],
                            'has_header': prev_table['has_header'],
                            'column_count': len(merged_df.columns),
                            'merged': True,
                            'merged_from': [i-1, i]
                        }
                        
                        merged_tables.append(merged_table)
                        processed_indices.add(i)
                        processed_indices.add(i-1)
        
        # Final tables list: include unprocessed original tables and merged tables
        final_tables = [t for i, t in enumerate(tables) if i not in processed_indices] + merged_tables
        
        # Sort by page number
        final_tables.sort(key=lambda x: x['page'] if isinstance(x['page'], int) else int(str(x['page']).split('-')[0]))
        
        return final_tables
    
    except Exception as e:
        st.error(f"Error extracting tables: {str(e)}")
        if debug_mode:
            st.code(traceback.format_exc())
        return []
    
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                if debug_mode:
                    st.write(f"Debug: Removed temporary file: {temp_file_path}")
            except Exception as e:
                st.warning(f"Could not remove temporary file: {str(e)}")

def extract_tables_from_blocks(pdf_file):
    """Extract tables from PDF using text blocks with improved header detection."""
    tables = []
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        doc = fitz.open(temp_file_path)
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            if debug_mode:
                st.write(f"Debug: Found {len(blocks)} text blocks on page {page_num+1}")
            
            # Simple approach: consider rectangular blocks as potential tables
            if blocks:
                block_data = []
                for block in blocks:
                    text = block[4]
                    lines = text.split('\n')
                    for line in lines:
                        if line.strip():
                            cells = line.split()
                            if len(cells) > 1:  # Consider it a row if it has multiple cells
                                block_data.append(cells)
                
                if block_data:
                    # Create a dataframe from the block data
                    df = pd.DataFrame(block_data)
                    
                    # Apply dynamic header detection
                    has_header = False
                    if df.shape[0] > 1:
                        if use_multiline_headers and df.shape[0] >= 2:
                            # Handle potential multi-line headers
                            header_row = merge_multiline_headers(df, num_header_rows=2)
                            df.columns = header_row
                            df = df.iloc[2:].reset_index(drop=True)
                            has_header = True
                        else:
                            # Use the improved header detection
                            header_row = extract_header(df)
                            df.columns = header_row
                            df = df.iloc[1:].reset_index(drop=True)
                            has_header = True
                        
                        if standardize_columns:
                            df = standardize_column_names(df)
                    
                    # Use numeric column names if configured that way
                    if column_handling == "Use numeric column names":
                        df.columns = [f"Col_{i}" for i in range(len(df.columns))]
                    else:
                        df = handle_duplicate_columns(df)
                    
                    # Fill NaN values if enabled
                    if enable_forward_fill:
                        df = fill_forward_nan_values(df)
                        
                    table_info = {
                        'page': page_num + 1,
                        'table_num': 1,
                        'data': df,
                        'raw_data': block_data,
                        'has_header': has_header,
                        'column_count': len(df.columns)
                    }
                    
                    tables.append(table_info)
        
        return tables
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def get_table_download_link(df, filename="table.csv", label="Download CSV"):
    """Generate a download link for a dataframe."""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ {label}</a>'
        return href
    except Exception as e:
        st.error(f"Error generating CSV download link: {str(e)}")
        return "Download link generation failed"

def get_excel_download_link(tables, filename="tables.xlsx", label="Download Excel"):
    """Generate a download link for multiple tables in Excel format."""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for i, table in enumerate(tables):
                if isinstance(table['page'], int):
                    sheet_name = f"P{table['page']}_T{table['table_num']}"
                else:
                    sheet_name = f"P{table['page']}_T{table['table_num']}"
                # Ensure sheet name is valid for Excel
                sheet_name = sheet_name[:31]  # Excel limit for sheet names
                table['data'].to_excel(writer, sheet_name=sheet_name, index=False)
        
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">⬇️ {label}</a>'
        return href
    except Exception as e:
        st.error(f"Error generating Excel download link: {str(e)}")
        if debug_mode:
            st.code(traceback.format_exc())
        return "Download link generation failed"

# Display version information in sidebar
st.sidebar.title("Settings")
st.sidebar.write(f"PyMuPDF version: {fitz.version}")

# Extraction method option
extraction_method = st.sidebar.radio(
    "Table Extraction Method",
    ["PyMuPDF native (find_tables)", "Simple text blocks analysis"]
)

# Column handling option
column_handling = st.sidebar.radio(
    "Column Name Handling",
    ["Clean and add suffixes to duplicates", "Use numeric column names"]
)

# Header options
st.sidebar.subheader("Header Detection")
use_multiline_headers = st.sidebar.checkbox("Detect multi-line headers", value=True)
standardize_columns = st.sidebar.checkbox("Standardize column names (lowercase + underscores)", value=True)

# Enable forward fill option
enable_forward_fill = st.sidebar.checkbox("Fill NaN values from previous rows", value=True)

# Enable multi-page table detection
detect_continuations = st.sidebar.checkbox("Detect and merge multi-page tables", value=True)

# Debug mode
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# Show environment info if debug mode is on
if debug_mode:
    st.sidebar.subheader("Environment Info")
    st.sidebar.write(f"Python version: {sys.version}")
    st.sidebar.write(f"Pandas version: {pd.__version__}")

# Main UI
st.title("PDF Table Extractor")
st.write("Upload a PDF file to extract tables and download them as CSV or Excel.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}, size: {uploaded_file.size} bytes")
    
    with st.spinner("Extracting tables..."):
        if extraction_method == "PyMuPDF native (find_tables)":
            tables = extract_tables_from_pdf(uploaded_file)
        else:
            tables = extract_tables_from_blocks(uploaded_file)
    
    # Process tables based on column handling option
    if column_handling == "Use numeric column names" and tables:
        for table in tables:
            # Replace all column names with numeric ones
            table['data'].columns = [f"Col_{i}" for i in range(len(table['data'].columns))]
    
    if tables:
        st.success(f"Found {len(tables)} tables/table sections in the PDF!")
        
        # Display merged status information if applicable
        merged_count = sum(1 for t in tables if t.get('merged', False))
        if merged_count > 0:
            st.info(f"{merged_count} multi-page tables were detected and merged.")
        
        # Display all tables with download options
        for i, table_info in enumerate(tables):
            # Show appropriate header based on whether it's a merged table
            if table_info.get('merged', False):
                st.subheader(f"Table {i+1} (Pages {table_info['page']})")
                st.caption("This table was automatically merged from multiple pages")
            else:
                st.subheader(f"Table {i+1} (Page {table_info['page']}, Table #{table_info['table_num']})")
            
            # Display column names for debugging
            if debug_mode:
                st.write("Column names:", list(table_info['data'].columns))
                if table_info.get('merged', False):
                    st.write("Merged from table indices:", table_info['merged_from'])
            
            # Check if dataframe is empty
            if table_info['data'].empty:
                st.warning("This table appears to be empty.")
            else:
                try:
                    # Display the table
                    st.dataframe(table_info['data'])
                except Exception as e:
                    st.error(f"Error displaying table: {str(e)}")
                    if debug_mode:
                        st.code(traceback.format_exc())
                    
                    # Fallback to display as dictionary
                    st.write("Fallback table display:")
                    st.write(table_info['data'].to_dict())
            
            # Display raw extraction result from PyMuPDF
            with st.expander("View raw extraction data"):
                st.write("Raw data extracted by PyMuPDF:")
                st.code(str(table_info['raw_data']))
            
            try:
                # Individual table download
                st.markdown(
                    get_table_download_link(
                        table_info['data'], 
                        filename=f"table_{i+1}_p{table_info['page']}.csv",
                        label=f"Download this table as CSV"
                    ),
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error creating download link: {str(e)}")
            
            st.markdown("---")
        
        # Option to download all tables as Excel
        if len(tables) > 1:
            st.subheader("Download all tables")
            try:
                st.markdown(
                    get_excel_download_link(tables, filename="all_tables.xlsx", label="Download all tables as Excel file"),
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error creating Excel download: {str(e)}")
    else:
        st.warning("No tables were detected in the PDF.")
        
        # Show PDF content as text for debugging
        if debug_mode:
            st.subheader("PDF Text Content (for debugging)")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            try:
                doc = fitz.open(temp_file_path)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    with st.expander(f"Page {page_num+1} Text Content"):
                        st.text(text)
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
            st.info("Try the alternative extraction method in the sidebar if tables aren't being detected properly.")
    
    # Display PDF preview
    with st.expander("Preview PDF"):
        base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

st.markdown("---")
st.info("""
**Feature Overview:**
1. **Dynamic Header Detection**: The app examines the first few rows to identify the best header row instead of assuming the first row is always the header.
2. **Multi-line Header Support**: Can detect and merge multi-line headers into single column names.
3. **Column Name Standardization**: Converts column names to lowercase and replaces special characters with underscores.
4. **NaN Value Filling**: Automatically fills empty cells with values from the previous row in the same column.
5. **Multi-page Table Detection**: Tables spanning multiple pages are automatically detected and merged.
6. **Alternative Extraction Method**: For PDFs where standard table detection fails.

**Troubleshooting Tips:**
1. If headers aren't being correctly identified, try enabling/disabling the multi-line header detection option.
2. For complex PDFs, you might get better results with the debug mode enabled.
3. If the app is having trouble with tables, check the raw PDF content in debug mode.
""")

# Add requirements for easy setup
requirements = """
streamlit==1.32.0
pandas==2.2.0
PyMuPDF==1.23.8
xlsxwriter==3.1.2
numpy==1.26.0
"""

with st.expander("View requirements.txt"):
    st.code(requirements)