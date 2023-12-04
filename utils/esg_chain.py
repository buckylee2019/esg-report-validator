
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from os import environ
from dotenv import load_dotenv
from genai.model import Credentials
from genai.schemas import GenerateParams
from genai.extensions.langchain.llm import LangChainInterface
import json
from langchain.vectorstores import Milvus
from glob import glob
import os
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from operator import itemgetter
from langchain.schema import StrOutputParser

load_dotenv()


WX_MODEL = os.environ.get("WX_MODEL")
creds = Credentials(os.environ.get("BAM_API_KEY"), "https://bam-api.res.ibm.com/v1")
MILVUS_CONNECTION={"host": os.environ.get("MILVUS_HOST"), "port": os.environ.get("MILVUS_PORT")}


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceHubEmbeddings(
    task="feature-extraction",
    repo_id = repo_id,
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
)
class vectorDB():
    def __init__(self,collection) -> None:
        self.collection_name = collection
    def vectorstore(self):
        
        vectorstore = Milvus(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            connection_args=MILVUS_CONNECTION,
            search_params = {
                "metric_type": "COSINE", 
                "offset": 5, 
                "ignore_growing": False, 
                "params": {"nprobe": 10}
            }
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


def GenerateEsgChain(user_prompt,vector_instance):
    params = GenerateParams(
        decoding_method="greedy",
        max_new_tokens=1024,
        min_new_tokens=1,
        stop_sequences=["}\n\n","}\n"],
        stream=False,
        top_k=50,
        top_p=1,
    )
    llm = LangChainInterface(
                model=WX_MODEL,
                credentials=creds,
                params=params,
            )
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
    params = GenerateParams(
        decoding_method="greedy",
        max_new_tokens=1024,
        min_new_tokens=1,
        stop_sequences=["。\n\n","\n\n"],
        stream=False,
        top_k=50,
        top_p=1,
    )

    llm = LangChainInterface(
                model=WX_MODEL,
                credentials=creds,
                params=params,
            )
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
    params = GenerateParams(
        decoding_method="greedy",
        max_new_tokens=1024,
        min_new_tokens=1,
        stop_sequences=stop_sequences,
        stream=False,
        top_k=50,
        top_p=1,
    )


    WX_MODEL = os.environ.get("WX_MODEL")
    creds = Credentials(os.environ.get("BAM_API_KEY"), "https://bam-api.res.ibm.com/v1")

    llm = LangChainInterface(
                    model=WX_MODEL,
                    credentials=creds,
                    params=params,
                )
    return llm(prompt)
def framework():
    items = {
        "環境保護相關議題":[
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
        "裁罰事件":[
            "近二年是否發生環保違規情節重大或導致停工/停業者",
            "近二年是否發生違反人權(強制勞動、童工問題等)導致停工/停業者",
            "近二年是否發生勞工權益糾紛情節重大或導致停工/停業者",
            "近二年是否發生公司治理問題情節重大或導致停工/停業者",
            "近二年是否發生洗錢或資助資恐活動情節重大或導致停工/停業者",
            "近二年是否發生工安事件情節重大或導致停工/停業者",
            "企業是否進行色情、野生動物捕殺或棲地破壞、受國際禁限的化學品/藥品/農藥/除草劑或放射性物質之作為"],
        "公開揭露":[
            "是否於公開管道揭露E&S相關資訊(如：企業官網、年報、企業社會責任報告書(備註2)、永續報告書)",
            "近一年是否曾獲得外部永續相關獎項",
            "是否設有E&S風險專責單位。(如:風險管理委員會、企業社會責任委員會)",
            "是否辨識利害關係人關注之E&S議題",
            "是否響應國內外E&S倡議(備註3)"   
        ],
        "社會責任議題":[
            "企業是否已建立職業健康安全管理的因應措施與行動方案？（如：員工健康管理活動、安全的工作環境",
            "企業是否已建立勞動權益管理的行動方案？（如：人權政策，申訴、溝通機制及員工之權益）",
            "企業是否已對社區、團體及永續發展議題之關懷？(如：援助弱勢團體、社區義工服務、支持在地發展等）",
            "企業是否有評估對社區之風險或機會並採行相應措施，並將其具體採行措施與實施成效揭露於公司網站、年報或永續報告書？",
            "企業網站、年報或永續報告書是否揭露所制定之供應商管理政策，要求供應商在環保、職業安全衛生或勞動人權等議題遵循相關規範，並說明實施情形？",
            "企業是否揭露員工福利政策？(如：保險、育嬰假、退休制度、員工持股、工作者健康促進、在職訓練…等)",
            "企業是否已制定職場多元化或推動性別平等政策，並揭露其實施情形?",
            "企業是否揭露與營運活動相關人權政策？ (如:禁止歧視行為、反職場霸凌與性騷擾以及自由結社等勞工人權)",
            "企業與其供應商之員工招募是否符合國際或當地勞動相關法規?(無使用童工、強迫或強制勞動事件)",
            "企業提供之產品或服務，是否符合國際或當地社會法令?(包括但不限於顧客健康與安全、產品標示、廣告行銷、客戶隱私保護、資安等)"
            ],
        "公司治理議題":[
            "企業是否設置推動永續發展專(兼)職單位，進行與公司營運相關之環境、社會或公司治理議題之風險評估，訂定相關風險管理政策或策略，且由董事會督導永續發展推動情形，並揭露於公司網站及年報?",
            "企業是否揭露與利害關係人(包含員工、客戶、供應商、社區等)溝通之機制及執行情形，以確認利害關係人對公司之看法及建議，並就所獲資訊訂定策略?",
            "企業是否揭露董監事、經理人以及大股東與公司進行利害關係人交易情形？",
            "企業是否將董事會運作執行情形(如：全體董事之董事會實際出席率等)或相關績效評估結果揭露於公司網站、年報或永續報告書？",
            "企業是否建立獨立董事與內部稽核主管、會計師之單獨溝通機制 (如就公司財務報告及財務業務狀況進行溝通之方式、事項及結果等 )？",
            "企業是否有審計委員會或董事會層級之功能性委員會(如:風險管理委員會)督導風險管理，並訂定經董事會通過之風險管理政策與程序，揭露風險管理組織架構、風險管理程序及其運作情形，且至少一年一次向董事會報告？",
            "企業是否就功能性委員會訂有績效評估辦法，並每年定期進行內部績效評估，並將執行情形及評估結果揭露於公司網站、年報或永續報告書？",
            "企業是否依上市櫃公司重大訊息之查證暨公開處理程序及資訊申報作業辦法等相關規定辦理而未受違約金處分？",
            "企業是否已將「股利政策」、「董監事及經理人績效評估與酬金制度」、「員工權益」，揭露於公司網路、年報或永續報告書？",
            "公司網站、年報或相關公開資訊是否揭露「誠信經營政策、企業社會責任、內部控制制度或相關公司治理守則」，明訂具體作法與防範不誠信行為方案，並設有對公司內、外部人員對於不合法(包括貪汙)與不道德行為的檢舉制度？"
        ]
        
    }
    return items
# generate_template("如果組織使用與法定名稱不同但眾所皆知的商業名稱時，則宜在其法定名稱外額外報導")