from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall,
    context_relevancy,
    answer_relevancy,
)

from src.utils import extract_qa_pairs_to_df, format_tuple_docs_to_text, format_docs_to_text
# from backend.app.rag.rag_utils import generate_answer
from src.rag_pipeline import generate_answer

def evaluate_metrics(dataset):
  # evaluating dataest on listed metrics
  result = evaluate(
      dataset=dataset,
      metrics=[
          answer_correctness,
          faithfulness,
          context_precision,
          context_recall
      ]
  )


  df_results = result.to_pandas()

  return df_results

def run_evaluation(retriever,
                   file_path="data/evaluation_sets/Robinson_Q&A.docx",
                    llm=None,
                   test_size=None):  # Replace with your actual file path
  df = extract_qa_pairs_to_df(file_path)

  if test_size:
    df = df.head(test_size)
    
  # Change the df columns to list
  questions = df["question"].tolist()
  ground_truths = df["ground_truths"].tolist()


  answers = []
  contexts = []
  # Inference
  for query in questions:
      documents = retriever.get_relevant_documents(query)
      contexts.append(
          [docs.page_content for docs in documents]
      )
      context_text = format_docs_to_text(documents)

      answers.append(generate_answer(query, context_text, llm=llm))
      
  # To dict
  data = {
      "question": questions,
      "answer": answers,
      "contexts": contexts,
      "ground_truth": ground_truths,
  }

  # Convert dict to dataset
  dataset = Dataset.from_dict(data)
  
  results = evaluate_metrics(dataset)
  return results

