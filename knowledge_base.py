from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings





class KnowledgeBase:

    def __init__(self, model_dir, index_path):

        # embedding模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_dir
        )


        # FAISS向量库
        self.db = FAISS.load_local(
            index_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )






    def search(self, query, k=5):

        docs = self.db.similarity_search(
            query,
            k=k
        )

        results=[]

        for doc in docs:
            results.append(
                doc.page_content
            )

        return "\n\n".join(results)