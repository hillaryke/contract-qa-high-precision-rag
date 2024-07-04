from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker



def chunk_by_semantic(documents: list[Document], embeddings = OpenAIEmbeddings):
    chunks = []
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    for doc in documents:
      docs = text_splitter.create_documents(doc.page_content)
      chunks.extend(docs)
    print(f"--INFO-- Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def chunk_by_recursive_split(documents: list[Document], chunk_size: int = 400):
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