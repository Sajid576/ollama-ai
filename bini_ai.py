from langchain.llms import Ollama
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class Model():
    LLAMA2="llama2"
    LLAMA2_13B="llama2:13b"
    LLAMA2_70B="llama2:70b"
    MISTRAL="mistral"
    LLAVA="llava"
    VICUNA="vicuna"
    ORCA_MINI="orca-mini"
    CODE_LAMA="codellama"
    STARLING_LM="starling-lm"
    NEURAL_CHAT="neural-chat"
    PHI_2="phi"
    DOLPHIN_PHI="dolphin-phi"


base_url='http://localhost:11434'
knowledge_base_text_file_path="kb.txt"
model_name=Model.LLAMA2_13B
syst_prompt_text_file_path='./sys_prompt.txt'



def load_model(base_url:str,model_name:str):
    ollama = Ollama(base_url=base_url, model=model_name, callback_manager=CallbackManager([StreamingStdOutCallbackHandler]))
    return ollama



def load_knowledge_base(knowledge_base_text_file_path:str):
    print('Loading knowledge base......')
    loader = TextLoader(knowledge_base_text_file_path)
    data = loader.load()
    print('Loading knowledge base Completed.!')
    return data


def load_syst_prompt(path:str):
    print('System prompt loading......')
    file_contents=''
    with open(path, 'r') as file:
        for line in file:
            file_contents+=line
            
    print(file_contents)  
    print('System prompt loading finished......')
    return file_contents


def train_model(data):
    print('Model training started......')
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)


    embeddings = OllamaEmbeddings(base_url=base_url, model=model_name)
    vector_store = Chroma.from_documents(documents=all_splits, embedding=embeddings)

    print('Model training finished.....')

    return vector_store


ollama = load_model(base_url,model_name)

knowledge_base_data = load_knowledge_base(knowledge_base_text_file_path)

system_prompt_data= load_syst_prompt(syst_prompt_text_file_path)

vector_store = train_model(knowledge_base_data)


user_prompt = ""

while user_prompt != "exit":
    user_prompt = input("Ask Bini AI: ")
    docs = vector_store.similarity_search(system_prompt_data+user_prompt, k=1)
    langchain=RetrievalQA.from_chain_type(ollama, retriever=vector_store.as_retriever(search_kwargs={"k": 1}))
    answer = langchain.run(system_prompt_data+user_prompt)
    print(answer)
    print("\n")