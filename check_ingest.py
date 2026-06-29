from qdrant_client import QdrantClient

from config import DB_PATH


def verify_qdrant_ingestion():
    db_path = DB_PATH
    collection_name = 'local_docs'
    
    # Initialize the client
    client = QdrantClient(path=db_path)
    
    try:
        # Fetch collection statistics
        info = client.get_collection(collection_name)
        print("--- Qdrant Collection Stats ---")
        print(f'Points stored  : {info.points_count}')
        print(f'Vector size    : {info.config.params.vectors.size}')
        print("-" * 31)

        # Peek at 3 random chunks to sanity-check content
        # scroll() returns a tuple: (list_of_points, next_page_offset)
        results = client.scroll(
            collection_name=collection_name, 
            limit=3, 
            with_payload=True, 
            with_vectors=False
        )

        print("\n--- Content Sanity Check (First 3 Chunks) ---")
        for point in results[0]:
            # Extract metadata and content safely
            metadata = point.payload.get('metadata', {})
            source = metadata.get('source', 'Unknown Source')
            page = metadata.get('page', '?')
            
            # Get text content (adjust 'page_content' if your key is named differently)
            content = point.payload.get('page_content', 'No content found')
            preview = content[:120].replace('\n', ' ')
            
            print(f'[{source} - Pg {page}] {preview}...')
            
    except Exception as e:
        print(f"Error accessing collection: {e}")

if __name__ == "__main__":
    verify_qdrant_ingestion()