
from rag import load_db
from langchain.tools import tool  

db = load_db()
@tool
def vector_search(query: str, k=5):
    """
    Search the vector database for documents related to the query.

    This tool performs semantic similarity search using embeddings
    and returns the most relevant documents.

    Args:
        query (str): Natural language query to search for.
        k (int, optional): Number of similar documents to return. Default is 5.

    Returns:
        list: A list of the top-k documents most similar to the query.
    """
    return db.similarity_search(query, k=k)


@tool
def calculate(expr: str):
    """
    Evaluate a mathematical expression.

    Use this tool when a calculation is required such as arithmetic,
    percentages, or numeric expressions.

    Args:
        expr (str): A valid Python mathematical expression
                    (e.g., "25 * 4", "100 / 5", "2 ** 10").

    Returns:
        float | int: The result of the evaluated expression.
    """
    return eval(expr)


@tool
def read_file(path: str):
    """
    Read the contents of a local text file.

    Use this tool when the agent needs to retrieve information
    stored inside a file.

    Args:
        path (str): Path to the file on the local filesystem.

    Returns:
        str: The full text content of the file.
    """
    with open(path, 'r') as f:
        return f.read()


@tool
def run_python(code: str):
    """
    Execute Python code in a restricted environment.

    This tool allows the agent to run Python code for tasks such as
    data processing, calculations, or generating results dynamically.

    The execution environment has no built-in functions available
    for security reasons.

    Args:
        code (str): Python code to execute.

    Returns:
        dict: A dictionary containing variables created during execution.
    """
    allowed_locals = {}

    exec(code, {"__builtins__": {}}, allowed_locals)

    return allowed_locals