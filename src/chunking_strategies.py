from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker



def chunk_by_semantic(documents: list[Document], embeddings = OpenAIEmbeddings):
    """    
    Splits documents into chunks based on semantic content using embeddings.

    This function takes a list of documents and uses a SemanticChunker, which leverages OpenAIEmbeddings,
    to split each document into semantically coherent chunks. The chunks are determined by the embeddings'
    understanding of the text, aiming to preserve the semantic integrity of the content within each chunk.

    Parameters:
        documents (list[Document]): A list of Document objects to be chunked semantically.
        embeddings (OpenAIEmbeddings, optional): The embeddings model used for semantic chunking. Defaults to OpenAIEmbeddings().

    Returns:
        list[Document]: A list of Document objects, each representing a semantically coherent chunk of the original documents.
    """
    chunks = []
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    for doc in documents:
      docs = text_splitter.create_documents(doc.page_content)
      chunks.extend(docs)
    print(f"--INFO-- Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def chunk_by_recursive_split(documents: list[Document], chunk_size: int = 400):
    """
    Splits documents into chunks of a specified size using a recursive character-based approach.

    This function takes a list of documents and splits each one into smaller chunks based on a specified character count,
    using a RecursiveCharacterTextSplitter. This method does not consider semantic content, and splits are based purely on
    character count.

    Parameters:
        documents (list[Document]): A list of Document objects to be split into smaller chunks.
        chunk_size (int, optional): The desired number of characters in each chunk. Defaults to 400.

    Returns:
        list[Document]: A list of Document objects, each representing a chunk of the original documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks