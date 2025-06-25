import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os

def prepare_and_load_data_to_chroma(csv_path: str, db_path: str = "./chroma_db"):
    """
    Loads hotel data from a CSV, prepares it, and loads it into a ChromaDB vector store.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # --- Data Cleaning (as done in EDA, simplified for this script) ---
    # Convert date columns (assuming they are still objects if loading fresh)
    for col in ['crawl_date', 'query_time_stamp']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce') # coerce will turn unparseable dates into NaT

    # Fill missing numerical values with median
    for col in ['guest_recommendation', 'site_review_count', 'site_review_rating']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with 'Unknown'
    categorical_fill_cols = [
        'additional_info', 'area', 'hotel_description', 'hotel_facilities',
        'locality', 'point_of_interest', 'qts', 'review_count_by_category',
        'room_area', 'room_facilities', 'similar_hotel', 'hotel_brand'
    ]
    for col in categorical_fill_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')
    # --- End Data Cleaning ---

    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=db_path)
    collection_name = "hotel_recommendations"

    # Ensure the collection is new or empty for a fresh load
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        print(f"Collection {collection_name} does not exist or could not be deleted. Creating new.")

    collection = client.get_or_create_collection(name=collection_name)

    print("Loading SentenceTransformer model...")
    # You might want to use a more powerful model for production, or a cloud-based embedding service
    model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []
    metadatas = []
    ids = []
    embeddings = []

    print("Preparing documents and embeddings...")
    for index, row in df.iterrows():
        # Combine relevant text columns for embedding
        combined_description = f"Hotel Name: {row['property_name']}. " \
                               f"Description: {row['hotel_description']}. " \
                               f"Facilities: {row['hotel_facilities']}. " \
                               f"Room Facilities: {row['room_facilities']}. " \
                               f"Additional Info: {row['additional_info']}. " \
                               f"Points of Interest: {row['point_of_interest']}."

        # Create metadata dictionary
        metadata = {
            "property_id": row['property_id'],
            "property_name": row['property_name'],
            "city": row['city'],
            "state": row['state'],
            "country": row['country'],
            "hotel_star_rating": row['hotel_star_rating'],
            "site_review_rating": row['site_review_rating'],
            "guest_recommendation": row['guest_recommendation'],
            "hotel_category": row['hotel_category'],
            "property_type": row['property_type'],
            "area": row['area'],
            "locality": row['locality'],
            "room_type": row['room_type'],
            "address": row['address'],
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "room_count": row['room_count'],
            "image_count": row['image_count'],
            "site_review_count": row['site_review_count'],
            "pageurl": row['pageurl']
        }

        # Handle potential non-string types for embedding if any slipped through cleaning
        combined_description = str(combined_description)

        documents.append(combined_description)
        metadatas.append(metadata)
        ids.append(str(row['uniq_id'])) # Use uniq_id as a unique identifier

        # Generate embedding for the combined description
        embeddings.append(model.encode(combined_description).tolist())


    # Add documents in batches (ChromaDB can handle large adds, but batching can be safer for very large datasets)
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")


    print(f"Successfully loaded {len(documents)} hotels into ChromaDB collection '{collection_name}'.")
    print(f"Total items in collection: {collection.count()}")

if __name__ == "__main__":
    # Ensure you have the necessary libraries installed:
    # pip install pandas chromadb sentence-transformers

    # Define paths
    csv_file_path = '../data/raw/goibibo_com-travel_sample.csv'
    chroma_db_persist_path = './chroma_db' # This will create a folder in your project root

    # Create the chroma_db directory if it doesn't exist
    if not os.path.exists(chroma_db_persist_path):
        os.makedirs(chroma_db_persist_path)

    prepare_and_load_data_to_chroma(csv_file_path, chroma_db_persist_path)
    print(f"ChromaDB created/updated at: {os.path.abspath(chroma_db_persist_path)}")

    # Example of how to query the database (for testing)
    # client = chromadb.PersistentClient(path=chroma_db_persist_path)
    # collection = client.get_collection(name="hotel_recommendations")
    # results = collection.query(
    #     query_texts=["hotels in Goa with swimming pool"],
    #     n_results=2
    # )
    # print("\nExample Query Results:")
    # print(results) 