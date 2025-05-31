import re
from bs4 import BeautifulSoup

def extract_text_from_html_part(html_part_str):
    """
    Takes an HTML string, removes unwanted tags (like style, script),
    and extracts all text content.
    """
    if not html_part_str or not html_part_str.strip():
        return ""
    
    soup_part = BeautifulSoup(html_part_str, 'html.parser')
    
    # Remove unwanted tags like <style> and <script>
    for unwanted_tag in soup_part(['style', 'script']):
        unwanted_tag.decompose()
        
    # Extract text, joining separated parts with a space, and stripping whitespace
    return soup_part.get_text(separator=' ', strip=True)

def wiki_split_html(html_doc):
    """
    Splits HTML content by <h3> tags and extracts text from each chunk.
    Handles content before the first <h3>, between <h3>s, and after the last <h3>.
    """
    soup = BeautifulSoup(html_doc, 'html.parser')
    
    # Find the main content container
    # Based on your example, it's <div class="mw-parser-output">
    content_container = soup.find('div', class_='mw-parser-output')
    
    if not content_container:
        print("Warning: Main content container '.mw-parser-output' not found.")
        # Fallback: try to process the whole document or a significant part
        # For this example, we'll try to process the body if container not found
        content_container = soup.body if soup.body else soup
        if not content_container:
             print("Error: No processable content found.")
             return []

    # Convert the content container (or fallback) to a string to use regex for splitting
    # We use regex to split by <h3> tags while keeping the <h3> tag as part of the subsequent chunk.
    # The regex (<h3>.*?</h3>) captures the H3 tag and its content.
    # re.split with a capturing group will include the delimiters in the result list.
    html_string_to_split = str(content_container)
    
    # Regex to find <h3> tags. Using re.DOTALL so . matches newlines within tags.
    # Using re.IGNORECASE for robustness, though HTML tags are usually lowercase.
    # The capturing group ( ) ensures that the <h3> tags are kept in the split list.
    parts = re.split(r'(<h3.*?>.*?</h3>)', html_string_to_split, flags=re.DOTALL | re.IGNORECASE)
    
    # Filter out any empty strings that might result from the split
    parts = [p for p in parts if p and p.strip()]

    extracted_chunks = []
    current_h3_title = "Content before first H3" # Default for the very first part
    accumulated_html_for_chunk = ""
    
    # Check if any H3 tag was found to determine how to label the first chunk
    has_h3_tags = any(re.match(r'<h3.*?>', part, flags=re.DOTALL | re.IGNORECASE) for part in parts)

    if not has_h3_tags:
        # If no H3 tags at all, the entire content is one chunk
        text = extract_text_from_html_part(html_string_to_split)
        if text:
            extracted_chunks.append({
                "header_id": "main_content_no_h3",
                "header_text": "Main Content (no H3 found)",
                "extracted_text": text
            })
        return extracted_chunks

    for part_html in parts:
        is_h3_tag_part = re.match(r'<h3.*?>.*?</h3>', part_html, flags=re.DOTALL | re.IGNORECASE)
        
        if is_h3_tag_part:
            # This part is an H3 tag. It signifies the end of the previous accumulated section
            # and the beginning of a new section.
            
            # Process the accumulated HTML for the previous section (if any)
            if accumulated_html_for_chunk.strip():
                text = extract_text_from_html_part(accumulated_html_for_chunk)
                if text:
                    extracted_chunks.append({
                        "header_id": current_h3_title.lower().replace(' ', '_').replace('[edit]', '').replace('(', '').replace(')', ''),
                        "header_text": current_h3_title,
                        "extracted_text": text
                    })
            
            # This H3 part starts a new chunk. Extract its text to use as the new title.
            h3_soup = BeautifulSoup(part_html, 'html.parser')
            h3_tag = h3_soup.find('h3')
            current_h3_title = h3_tag.get_text(strip=True) if h3_tag else "Unnamed H3 Section"
            
            # Start accumulating HTML for the new chunk, beginning with this H3 tag itself.
            accumulated_html_for_chunk = part_html
        else:
            # This part is content that belongs to the current section. Append it.
            accumulated_html_for_chunk += part_html
            
    # After the loop, there might be a final accumulated chunk. Process it.
    if accumulated_html_for_chunk.strip():
        text = extract_text_from_html_part(accumulated_html_for_chunk)
        if text:
            # If the last processed H3 title was "Content before first H3",
            # it means the content didn't actually start with an H3, but we had H3s later.
            # The current_h3_title would be the text of the last H3 found.
            header_for_last_chunk = current_h3_title
            
            extracted_chunks.append({
                "header_id": header_for_last_chunk.lower().replace(' ', '_').replace('[edit]', '').replace('(', '').replace(')', ''),
                "header_text": header_for_last_chunk,
                "extracted_text": text
            })
            
    return extracted_chunks