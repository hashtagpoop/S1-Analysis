import sec_parser as sp
from sec import download_filing
from latest_ipos import get_latest_ipos
from typing import cast
import os
from llm import LLM
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

MAX_FILINGS_TO_PARSE = 10

# Add argument parser
parser = argparse.ArgumentParser(description='Parse S-1 filings and analyze lockup periods')
parser.add_argument('--analyze', action='store_true', help='Use GPT-4 to analyze lockup texts')
args = parser.parse_args()

# Initialize LLM if analyze flag is set
llm = None
if args.analyze:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    llm = LLM.create(
        provider_type="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

# TODO: need to see if we are missing any lockups or common share offered.
# TODO: need to tune this to make sure we capture all edge cases.
# Method that gets the lockup text along with all the other
# text within the given section so it pulls all the context of lockups.
def extract_lockup_text(
    tree: sp.SemanticTree,
    keywords=("lock-up", "lockup", "lock up")
) -> list[str]:
    # Given that the sec_parser library already gets us the tree structure
    # we can just traverse the tree and collect the text of the nodes that
    # mention the lockup keywords.
    def dfs(node: sp.TreeNode, level=0, collecting=False):
        # Check if the current node's text contains any of the keywords
        node_text = node.semantic_element.text.lower()
        if any(keyword in node_text for keyword in keywords):
            collecting = True

        # If collecting is True, add the node's text to the collected_text list
        if collecting:
            collected_text.append(node.semantic_element.text)

        # Recursively call dfs on each child node
        for child in node.children:
            dfs(child, level + 1, collecting)

    collected_text = []
    # Start the DFS from each top-level section
    top_level_sections = [item for part in tree for item in part.children]
    for section in top_level_sections:
        dfs(section)

    return collected_text

def main():
    could_not_find_filing = []
    latest_ipos = get_latest_ipos()

    for i, row in latest_ipos.iterrows():
        try:
            s1_content = download_filing(ticker=row["ticker"], form="S-1")
            elements: list = sp.Edgar10QParser().parse(s1_content)
            tree: sp.SemanticTree = sp.TreeBuilder().build(elements)
            lockup_texts = extract_lockup_text(tree)
            
            for lockup_text in lockup_texts:
                if args.analyze and llm:
                    print(f"\nAnalyzing lockup texts for {row['ticker']}:")
                    # TODO: the prompt needs a lot more tuning.
                    prompt = f"""
                    Analyze the following lockup text from an S-1 filing and extract (I want this information for each type of share):
                    
                    For example:
                    directors, 100000 shares, 1 year, common, quote here
                    
                    1. The number of shares affected
                    2. The lockup period duration
                    3. The type of shares (common stock, preferred, etc.)
                    4. Who the lockup applies to (directors, employees, etc.)
                    5. Quote where you got it from
                    
                    Format the response as a structured CSV.
                    
                    Text to analyze:
                    {lockup_text}
                    """
                    analysis = llm.complete(prompt)
                    print("\nLLM Analysis:")
                    print(analysis)
                    
                print("-" * 100)
            
        except Exception as e:
            could_not_find_filing.append(row["ticker"])
            print(f"Unable to find S1 filing for {row['ticker']}: {e}")
            
        if i > MAX_FILINGS_TO_PARSE: break
        
    print(f"Could not find filing for the following tickers: {', '.join(could_not_find_filing)}")

"""
render_to_markdown is a function that converts a semantic tree to markdown format, 
with headings representing the hierarchy of the document. It ignores irrelevant elements 
and formats the text appropriately. This will ultimately be used to chunk
in a reasonable hierarchy that feeds into a RAG setup that uses a hybrid semantic
keyword search.
"""
def render_to_markdown(
    tree: sp.SemanticTree,
    ignored_types: tuple = None,
) -> str:
    """
    Convert a semantic tree to markdown format, with headings representing
    the hierarchy of the document.
    """
    ignored_types = ignored_types or (sp.IrrelevantElement,)
    markdown_lines = []

    def _process_node(node: sp.TreeNode, level=0):
        element = node.semantic_element
        if isinstance(element, ignored_types):
            return

        # Determine if this element should be a heading
        is_heading = False
        heading_prefix = ""
        
        # Check if the element is a title element
        if isinstance(element, sp.TitleElement):
            is_heading = True
            # Limit heading levels to a maximum of 6 (markdown standard)
            heading_level = min(level + 1, 6)
            heading_prefix = "#" * heading_level + " "
        
        # Add the node's text to markdown_lines with appropriate formatting
        content = element.text.strip()
        if content:
            if is_heading:
                markdown_lines.append(f"{heading_prefix}{content}")
            else:
                markdown_lines.append(content)
                markdown_lines.append("")  # Add blank line after paragraphs

        # Process all child nodes recursively
        for child in node.children:
            _process_node(child, level + 1)

    # Process each root node in the tree
    for root_node in tree:
        _process_node(root_node)

    return "\n".join(markdown_lines)

def process_and_chunk_document(markdown_content: str, output_file: str = 'output.txt'):
    """
    Process a markdown document and create a parent-child document retrieval system using PostgreSQL.
    This allows for both accurate semantic search (using small chunks) while maintaining
    context (by retrieving larger parent chunks). Both parent and child chunks are stored in PGVector.
    
    Args:
        markdown_content: The markdown content to process
        output_file: The file to write the markdown content to
    """
    from langchain.retrievers import ParentDocumentRetriever
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from langchain_postgres import PGVector
    from langchain.storage import InMemoryStore
    import psycopg
    import os
    
    # PostgreSQL connection settings from environment variables with defaults
    DB_USER = os.getenv('POSTGRES_USER', 'postgres')
    DB_PASS = os.getenv('POSTGRES_PASSWORD', 'postgres')
    DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    DB_PORT = os.getenv('POSTGRES_PORT', '5432')
    DB_NAME = os.getenv('POSTGRES_DB', 'vectordb')
    
    # Create SQLAlchemy connection string
    CONNECTION_STRING = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Create psycopg connection for admin tasks
    admin_conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        dbname=DB_NAME
    )
    
    try:
        # Create vector extension if it doesn't exist
        with admin_conn.cursor() as cursor:
            cursor.execute('CREATE EXTENSION IF NOT EXISTS vector;')
        admin_conn.commit()
    finally:
        admin_conn.close()
    
    # Collection names for parent and child chunks
    PARENT_COLLECTION = "s1_filing_parent_chunks"
    CHILD_COLLECTION = "s1_filing_child_chunks"
    
    # Write the markdown content to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    # Load the markdown file
    loader = TextLoader(output_file)
    documents = loader.load()

    # Create parent and child splitters with markdown-aware separators
    markdown_separators = [
        "\n# ",      # h1
        "\n## ",     # h2
        "\n### ",    # h3
        "\n#### ",   # h4
        "\n##### ",  # h5
        "\n###### ", # h6
        "\n\n",      # paragraphs
        "\n",        # lines
        ". ",        # sentences
        " ",         # words
        ""          # chars
    ]

    # Parent chunks are larger for context
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=markdown_separators,
        length_function=len,
        keep_separator=True
    )
    
    # Child chunks are smaller for better embedding
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=markdown_separators,
        length_function=len,
        keep_separator=True
    )

    try:
        # Initialize embedding function
        embedding_function = OpenAIEmbeddings()
        
        # Debug: Print first document content and metadata
        print("\nFirst document content:", documents[0].page_content[:100])
        print("First document metadata:", documents[0].metadata)
        
        # Debug: Generate and print a sample embedding
        sample_embedding = embedding_function.embed_query(documents[0].page_content[:100])
        print("\nSample embedding dimension:", len(sample_embedding))
        
        # Initialize parent vectorstore with non-empty documents
        parent_vectorstore = PGVector.from_documents(
            documents=documents,  # Use actual documents instead of empty list
            embedding=embedding_function,
            collection_name=PARENT_COLLECTION,
            connection=CONNECTION_STRING,
            pre_delete_collection=True,
            distance_strategy="cosine"
        )
        
        # Initialize child vectorstore with non-empty documents
        child_vectorstore = PGVector.from_documents(
            documents=documents,  # Use actual documents instead of empty list
            embedding=embedding_function,
            collection_name=CHILD_COLLECTION,
            connection=CONNECTION_STRING,
            pre_delete_collection=True,
            distance_strategy="cosine"
        )

        # Initialize the in-memory store for document bytes
        doc_store = InMemoryStore()

        # Create the parent document retriever using both PGVector stores
        retriever = ParentDocumentRetriever(
            vectorstore=child_vectorstore,  # For child chunks
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
            parent_vectorstore=parent_vectorstore,  # For parent chunks
            byte_store=doc_store  # Store for full documents
        )

        # Add documents to the retriever
        retriever.add_documents(documents)
        return retriever, parent_vectorstore, child_vectorstore
        
    except Exception as e:
        print(f"Error setting up document retrieval system: {str(e)}")
        raise

def test():
    try:
        s1_content = download_filing(ticker="TBH", form="S-1")
        elements: list = sp.Edgar10QParser().parse(s1_content)
        tree = sp.TreeBuilder().build(elements)

        # Convert tree to markdown
        markdown_output = render_to_markdown(tree)
        
        # Process and chunk the document
        retriever, parent_vectorstore, child_vectorstore = process_and_chunk_document(markdown_output)
        
        # Example searches to test the retriever
        test_queries = [
            "lockup period",
            "common stock",
            "risk factors",
            "financial statements"
        ]
        
        # Print chunking statistics and sample searches
        print(f"\nChunking Statistics:")
        print(f"Number of parent documents: {len(parent_vectorstore.similarity_search('', k=10000))}")
        
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"Search results for '{query}':")
            results = retriever.invoke(query)
            
            for i, doc in enumerate(results[:2], 1):
                print(f"\nResult {i} (length: {len(doc.page_content)} chars):")
                print(f"Preview: {doc.page_content[:200]}...")
                
    except Exception as e:
        print(f"Error in test function: {str(e)}")
        raise

if __name__ == "__main__":
    test()
