# Post-processing
def format_docs_to_text(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)