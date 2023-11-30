
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from os import environ
from dotenv import load_dotenv
import json
from langchain.vectorstores import Chroma
from glob import glob
import os
from langchain.schema import format_document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain.schema import StrOutputParser
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
import chromadb

load_dotenv()


WX_MODEL = os.environ.get("WX_MODEL")
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }


embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
class vectorDB():
    def __init__(self,collection) -> None:
        self.collection_name = collection
    def vectorstore(self):
        
        vectorstore = Chroma(
                        embedding_function=embeddings,
                        collection_name=self.collection_name,
                        persist_directory=os.environ.get("INDEX_NAME")
                    
            )
        return vectorstore


# qa_chain= RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3})
# )
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [f"Document {idx+1}. \n{format_document(doc, document_prompt)}" for idx, doc in enumerate(docs)]
    return document_separator.join(doc_strings)

def get_collection_list():
    client = chromadb.PersistentClient(path=os.environ.get("INDEX_NAME"))
    return [cols.name for cols in client.list_collections() if cols.name!="GRI"]


def GenerateEsgChain(user_prompt,vector_instance):
    
    params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 30,
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.TEMPERATURE: 0,
    GenParams.STOP_SEQUENCES:["}\n\n","}\n"],
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 1
}

    llm = Model(model_id="meta-llama/llama-2-70b-chat", credentials=creds, params=params, project_id=project_id).to_langchain()

    prompt = PromptTemplate.from_template(
        '''[INST]<<SYS>>
        You are an AI assistant responsible for guiding the review of an ESG (Environmental, Social, Governance) report. Follow the steps below:

        Summarize the content from the provided documents, using the following JSON format:

        {{
        \"Question\": [Specify the question],
        \"Explanation\": [Provide concise, question-specific information from the document, and indicate which document you use to summarize the answer],
        \"Answer\": [Yes, No, or Need further confirmation]
        }}
        AVOID new lines and DO NOT include other information which is not in the context.
        Note: If the document isn't mentioned in Explanation, Answer should be "Need human confirmation".
        By adhering to these rules, you will assist the individuals in charge of reviewing the ESG report in obtaining accurate and valuable information.
        <</SYS>>

        % Documents
        {summarize}
        
        Answer the {question} in JSON: [/INST]'''
    )

    # chain2 = (
    #     {"summarize": qa_chain, "question":itemgetter("question")}
    #     | prompt2
    #     | llm
    #     | StrOutputParser()
    # )
    qa_chain_esg = (
        {
            "summarize": itemgetter("question")|vector_instance.as_retriever(search_kwargs={'k': 3})| _combine_documents,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain_esg.invoke({"question": user_prompt}) 
def TranslateChain(text):
    
    params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.TEMPERATURE: 0,
    GenParams.STOP_SEQUENCES:["。\n\n","\n\n"],
    GenParams.REPETITION_PENALTY: 1
}

    llm = Model(model_id="meta-llama/llama-2-70b-chat", credentials=creds, params=params, project_id=project_id).to_langchain()
    

    prompt =  PromptTemplate.from_template("INST] <<SYS>>\n"\
    "You are a helpful, respectful and honest assistant.\n"\
    "Always answer as helpfully as possible, while being safe.\n"\
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.\n"\
    "Please ensure that your responses are socially unbiased and positive in nature.\n"\
    "\n"\
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.\n"\
    "If you don't know the answer to a question, please don't share false information.\n"\
    "\n"\
    "Translate from English to Chinese as the following example:\n"\
    "English: The Board of Directors is responsible for overseeing the bank's operations, including its financial performance, risk management, and corporate governance practices.\n"\
    "Chinese: 董事會負責監督銀行的運營，包括財務績效、風險管理和公司治理實務。\n\n"\
    "English: The board members' tenure is 3 years, and they can be re-elected for a maximum of 2 consecutive terms.\n"\
    "Chinese: 董事會成員任期3年，最多可連任2屆。\n\n"\
    "English: {original_text}\n"\
    "Chinese: ")
    trans = (
        {"original_text": itemgetter("original_text")}
        | prompt
        | llm
        | StrOutputParser()
    )
    return trans.invoke({"original_text": text}) 
def Generate(prompt, stop_sequences = ["。\n\n","\n\n\n"]):
    params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.TEMPERATURE: 0,
    GenParams.STOP_SEQUENCES:stop_sequences,
    GenParams.REPETITION_PENALTY: 1
}

    llm = Model(model_id="meta-llama/llama-2-70b-chat", credentials=creds, params=params, project_id=project_id).to_langchain()



    return llm(prompt)
def framework():
    items = {
        "環境保護相關議題":[
            "是否響應國內外E&S倡議",
            "企業是否揭露有關溫室氣體排放量資訊，並設立氣候變遷管理目標、定期追蹤達成情形，且評估氣候變遷可能產生的潛在影響(含風險與機會)以及相關因應行動？",
            "企業是否已建立原物料使用管理目標與建立行動方案，並定期追蹤達成情形？",
            "企業是否已設立污染防治行動方案？(空氣、噪音污染管控、水資源管理及廢棄物管理與回收)",
            "企業是否揭露水資源使用量，設立水資源管理目標，並定期追蹤達成情形？",
            "企業是否揭露能源耗用使用情形，制定能源管理目標或提升能源使用效率的行動方案，並定期追蹤達成情形？",
            "企業是否揭露廢棄物料(含有害/非有害)物排放量，制定管理政策或減量目標，並定期追蹤達成情形？",
            "企業是否投資於節能或綠色能源相關環保永續之機器設備，或投資於我國綠能產業(如:再生能源電廠)等，或有發行或投資其資金運用於綠色或社會效益投資計畫並具實質效益之永續發展金融商品，並揭露其投資情形及具體效益？",
            "企業是否揭露如何及從何處取得、消耗和排放水，以及企業透過商業關係與組織活動、產品或服務產生對水相關的衝擊？",
            "企業是否揭露製造主要產品和服務之使用回收再利用的物料？",
            "企業所售出之產品及服務是否符合國內外相關環境保護規範或符合綠色產品設計？"
        ],
        # "裁罰事件":[
        #     "近二年是否發生環保違規情節重大或導致停工/停業者",
        #     "近二年是否發生違反人權(強制勞動、童工問題等)導致停工/停業者",
        #     "近二年是否發生勞工權益糾紛情節重大或導致停工/停業者",
        #     "近二年是否發生公司治理問題情節重大或導致停工/停業者",
        #     "近二年是否發生洗錢或資助資恐活動情節重大或導致停工/停業者",
        #     "近二年是否發生工安事件情節重大或導致停工/停業者",
        #     "企業是否進行色情、野生動物捕殺或棲地破壞、受國際禁限的化學品/藥品/農藥/除草劑或放射性物質之作為"],
        # "公開揭露":[
        #     "是否於公開管道揭露E&S相關資訊(如：企業官網、年報、企業社會責任報告書(備註2)、永續報告書)",
        #     "近一年是否曾獲得外部永續相關獎項",
        #     "是否設有E&S風險專責單位。(如:風險管理委員會、企業社會責任委員會)",
        #     "是否辨識利害關係人關注之E&S議題",
        #     "是否響應國內外E&S倡議(備註3)"
        # ]
        
    }
    return items
# generate_template("如果組織使用與法定名稱不同但眾所皆知的商業名稱時，則宜在其法定名稱外額外報導")