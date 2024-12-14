import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# api key 가져오기
load_dotenv(dotenv_path='seok.env')
api_key=os.getenv("GROUP_API_KEY")

llm = ChatOpenAI(model='gpt-4o-mini', api_key=api_key, temperature=0)

dir = "ai_news"

def latest_file(dir, extension=".json"):
    '''가장 최근 생성된 json파일을 불러 온뒤 content의 내용만 추출'''
    files = [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith(extension)]   # json확장자를 가진 파일들.
    
    if not files:   # 폴더에 json없으면 아래 문장 출력.
        raise FileNotFoundError(f"There's no {extension} files.")
    
    latest = max(files, key=os.path.getmtime)   #최근 파일

    # json파일 불러오기
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    news = []
    if isinstance(data, dict):  # JSON이 딕셔너리 형태인 경우
        if "content" in data:
            news.append(data["content"])
    elif isinstance(data, list):  # JSON이 리스트 형태인 경우
        for item in data:
            if isinstance(item, dict) and "content" in item:
                news.append(item["content"])
    else:
        raise ValueError("Unsupported JSON structure.")
    
    return news

def summarize_news(news):
    '''GPT로 news 요약'''
    summaries = []
    for content in news:
        try:
            system_message = SystemMessage(content="You are a summerizing AI. Please answer in Korean. Please reply only based inputs.")
            messages = [system_message]
            messages.append(HumanMessage(content=content))
            response = llm.invoke(messages)
            summaries.append(response.content)
        except Exception as e:
            print(f"오류 발생: {e}")
            summaries.append(None)
    return summaries

news = latest_file(dir) # 원본
llm_news = summarize_news(news) # gpt로 요약한 데이터

from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def get_retriever(data):
    documents = [Document(page_content=text) for text in data]

    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    splits_recur = []
    for doc in documents:
        splits_recur.extend(recursive_text_splitter.split_documents([doc]))
    splits = splits_recur

    # 임베딩 모델 지정
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # 원본 및 요약 데이터 벡터스토어 생성
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    return retriever

retriever = get_retriever(news)
retriever_llm = get_retriever(llm_news)

# 리트리버 참조 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following context."),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])

class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        # print("Debug Output:", output)
        return output

class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가
        # context의 각 문서를 문자열로 결합
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        return {"context": context_text, "question": inputs["question"]}

# RAG체인
rag_chain = {
    "context": retriever,
    "question": DebugPassThrough()
} | DebugPassThrough() | ContextToText() | prompt | llm

rag_chain_llm = {
    "context": retriever_llm,
    "question": DebugPassThrough()
} | DebugPassThrough() | ContextToText() | prompt | llm

'''
# 결과값 도출 테스트

# 결과값 비교
results = {"original": [], "summarized": []}

# 질문
question = "OpenAI에 대한 정보 찾아줘."

# 체인 실행
response = rag_chain.invoke(question)

# 원본 RAG 결과
results["original"].append({"question": question, "answer": response.content})

# 체인 실행
response_llm = rag_chain_llm.invoke(question)

# llm 요약글 RAG 결과
results["summarized"].append({"question": question, "answer": response_llm.content})

print(results)
'''

'''
{'original': [{'question': 'OpenAI에 대한 정보 찾아줘.', 'answer': 
'죄송하지만, 제공된 문서에는 OpenAI에 대한 정보가 포함되어 있지 않 
습니다. 다른 질문이 있으시면 도와드리겠습니다.'}]
'summarized': [{'question': 'OpenAI에 대한 정보 찾아줘.'
'answer': 'OpenAI는 인공 
지능 연구소로, 다양한 AI 모델과 기술을 개발하고 있습니다. 최근에는 
공개적으로 사용 가능한 데이터를 활용하여 AI 모델을 훈련했다고 주장 
했으나, 일부 전문가들은 그 데이터가 실제로는 공개적으로 사용할 수  
있는 콘텐츠가 아니라고 지적하고 있습니다.'}]}
'''