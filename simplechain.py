from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Cấu hình file

model_file = r".\models\vinallama-7b-chat_q5_0.gguf"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = "llama",
        max_new_token = 1024,
        temporeature = 0.01
    )

    return llm

def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables = ["question"])
    return prompt

# Create simple chain
def create_simple_chain(prompt,llm):
    llm_chain = LLMChain(prompt = prompt, llm = llm)
    return llm_chain

# test run
template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm = load_llm(model_file)
llm_chain = create_simple_chain(prompt,llm)


question = "Hình tam giác có bao nhiêu cạnh"
response = llm_chain.invoke({"question":question})
print(response)
