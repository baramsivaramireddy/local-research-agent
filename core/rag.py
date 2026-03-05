
from langchain_text_splitters import RecursiveCharacterTextSplitter



# online 
# from langchain_community.embeddings import NomicEmbeddings

# embeddings = NomicEmbeddings(model ="nomic-embed-text")

# offline
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model ="nomic-embed-text")



def chunk_text(text):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 10
    )
    
    return splitter.split_text(text)

from langchain_community.vectorstores import FAISS


def create_db(text):
    
    chunks = chunk_text(text)
    
    return FAISS.from_texts(chunks,embeddings)


def retrieve(db,query):
    return db.similarity_search(query,k=5)

def save_db(db, path="faiss_index"):
    db.save_local(path)
    
def load_db(path="faiss_index"):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
if __name__ == '__main__':
    text = '''
    Use LangChain for:

Real-time data augmentation. Easily connect LLMs to diverse data sources and external/internal systems, drawing from LangChain's vast library of integrations with model providers, tools, vector stores, retrievers, and more.
Model interoperability. Swap models in and out as your engineering team experiments to find the best choice for your application's needs. As the industry frontier evolves, adapt quickly – LangChain's abstractions keep you moving without losing momentum.
Rapid prototyping. Quickly build and iterate on LLM applications with LangChain's modular, component-based architecture. Test different approaches and workflows without rebuilding from scratch, accelerating your development cycle.
Production-ready features. Deploy reliable applications with built-in support for monitoring, evaluation, and debugging through integrations like LangSmith. Scale with confidence using battle-tested patterns and best practices.
Vibrant community and ecosystem. Leverage a rich ecosystem of integrations, templates, and community-contributed components. Benefit from continuous improvements and stay up-to-date with the latest AI developments through an active open-source community.
Flexible abstraction layers. Work at the level of abstraction that suits your needs - from high-level chains for quick starts to low-level components for fine-grained control. LangChain grows with your application's complexity.
LangChain ecosystem
'''

    db = create_db(text)
    save_db(db)
    results = retrieve(db, 'use of langchain')
    
    for r in results:
        print(r.page_content)