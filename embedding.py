from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    def __init__(self, model_dir):

        self.model = HuggingFaceEmbeddings(
            model_name=model_dir
        )
        
    def embed_text(self,text):

        return self.model.embed_query(
            text
        )

    def embed_documents(self,documents):

        return self.model.embed_documents(
            documents
        )