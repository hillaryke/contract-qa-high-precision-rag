import docx
import re
import pandas as pd

def format_docs_to_text(docs):
    return "\n\n------------\n\n".join(doc.page_content for doc in docs)

# Extract doc from tuple
def format_tuple_docs_to_text(docs):
    """ Formats a list of (doc, score) tuples into a text string. """
    return "\n\n------------\n\n".join(doc.page_content for doc, _ in docs)  

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def extract_qa_pairs_to_df(file_path):
    """Extracts question-answer pairs from a docx file and saves them to a DataFrame.

    Args:
        file_path (str): The path to the docx file.

    Returns:
        pd.DataFrame: A DataFrame with columns "Question_Number", "Question", and "Answer".
    """

    try:
        doc = docx.Document(file_path)
        questions = []
        answers = []
        question_numbers = []
        current_question = None

        for para in doc.paragraphs:
            # Match questions (e.g., Q1:)
            question_match = re.match(r"(Q\d+[a-z]?):\s*(.*)", para.text)  # Capture question text
            if question_match:
              
                current_question = question_match.group(1)
                
                # Add the question number and text together in one text
                # current_question = question_match.group(1) + ": " + question_match.group(2)  # Combine question number and text

                question_numbers.append(current_question)  # Store question number
                questions.append(question_match.group(2))  # Store question text
                answers.append("")  # Initialize answer
            elif current_question:  # If we're within a question-answer block
                # Remove "A1:" prefix if present
                answer_text = re.sub(r"^A\d+[a-z]?:\s*", "", para.text)  
                answers[-1] += answer_text + "\n"  # Append to the last answer

        # Create DataFrame
        df = pd.DataFrame({"Question": questions, "Answer": answers})
        df.columns = ['question', 'ground_truths']
        
        ### Inlude the question numbers to the dataframe
        # df = pd.DataFrame({"Question_Number": question_numbers, "Question": questions, "Answer": answers})
        return df

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except docx.opc.exceptions.PackageNotFoundError:
        print(f"Error: Invalid docx file at '{file_path}'")
        return None