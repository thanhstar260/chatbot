from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings



# Load model
def load_model(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = "llama",
        max_new_token = 1024,
        temporeature = 0.01
    )
    return llm

# Create prompt
def create_prompt(template):
    # context: tài liệu liên quan trong db
    # question: câu hỏi người dùng
    prompt = PromptTemplate(template=template, input_variables = ["context","question"])
    return prompt

# Create chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain
# Read from db
def read_db(db_path):
    embedding_model = GPT4AllEmbeddings(model_file=r"models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(db_path,embeddings=embedding_model)
    return db


if __name__ == "__main__":
    model_file = r"models/vinallama-7b-chat_q5_0.gguf"
    db_path = r"vectorsores/vec_db"
    db = read_db(db_path=db_path)
    llm = load_model(model_file)

    # create prompt
    template = template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, 
    hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = create_prompt(template)

    llm_chain = create_qa_chain(prompt=prompt,llm=llm,db=db)

    #run
    question = "Tại sao cần chiến lược Học Máy?"

    response = llm_chain.invoke({"query":question})
    print(response)