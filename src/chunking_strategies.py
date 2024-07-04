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
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def chunk_by_recursive_split(documents: list[Document], chunk_size: int = 200, chunk_overlap: int = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks
  
def chunk_by_sentence_split(documents: list[Document], chunk_size: int = 200):
    chunks = []
    for doc in documents:
        text = doc.page_content
        sentences = text.split(".")
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "."
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + "."
        if current_chunk:
            chunks.append(current_chunk)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks
  
def chunk_by_paragraph_split(documents: list[Document], chunk_size: int = 200):
    chunks = []
    for doc in documents:
        text = doc.page_content
        paragraphs = text.split("\n")
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < chunk_size:
                current_chunk += paragraph + "\n"
            else:
                chunks.append(current_chunk)
                current_chunk = paragraph + "\n"
        if current_chunk:
            chunks.append(current_chunk)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks
  
def chunk_by_word_split(documents: list[Document], chunk_size: int = 200):
    chunks = []
    for doc in documents:
        text = doc.page_content
        words = text.split(" ")
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) < chunk_size:
                current_chunk += word + " "
            else:
                chunks.append(current_chunk)
                current_chunk = word + " "
        if current_chunk:
            chunks.append(current_chunk)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks
