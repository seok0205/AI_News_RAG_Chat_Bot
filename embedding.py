import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# api key 가져오기
load_dotenv(dotenv_path='seok.env')
api_key=os.getenv("GROUP_API_KEY")

llm = ChatOpenAI(model='gpt-4o', api_key=api_key)

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
            response = llm.predict_messages(
                messages=[
                    {"role": "system", "content": "너는 글을 요약하는 것에 뛰어난 능력을 가졌어."},
                    {"role": "user", "content": f"이 글을 간단하게 요약해줘.\n{content}"}
                ],
                max_tokens=200
            )
            summaries.append(response.content)
        except Exception as e:
            print(f"오류 발생: {e}")
            summaries.append(None)
    return summaries

news = latest_file(dir) # 원본
llm_news = summarize_news(news) # gpt로 요약한 데이터

from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# 임베딩 모델 지정
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# 원본 및 요약 데이터 벡터스토어 생성
vectorstore = FAISS.from_texts(texts=news, embedding=embeddings)
vectorstore_llm = FAISS.from_texts(texts=llm_news, embedding=embeddings)

# 원본 및 요약 데이터 리트리버 생성
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
retriever_llm = vectorstore_llm.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# 리트리버 참조 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "제공한 글을 참고하여 질문에 대답하세요. \n{context}")
])

# RAG체인
rag_chain = (
    {"context": retriever} | prompt | llm | StrOutputParser()
)

rag_chain_llm = (
    {"context": retriever_llm} | prompt | llm | StrOutputParser()
)

# 결과값 비교
results = {"original": [], "summarized": []}

# 질문
question = "예상되는 가장 중요한 단어는 무엇입니까?"

# 리트리버에서 질문 중 가장 관련성이 높은 문서를 가져와서 컨텍스트로 활용.
context = retriever.get_relevant_documents(question)
response = llm.predict_messages([
    {"role": "system", "content": f"다음 글을 참고하여 질문에 답하세요.\n{context[0].page_content}"},
    {"role": "user", "content": question}
])

# 원본 RAG 결과
results["original"].append({"question": question, "answer": response.content})

# 리트리버에서 질문 중 가장 관련성이 높은 문서를 가져와서 컨텍스트로 활용.
context_llm = retriever_llm.get_relevant_documents(question)
response_llm = llm.predict_messages([
    {"role": "system", "content": f"다음 글을 참고하여 질문에 답하세요.\n{context_llm[0].page_content}"},
    {"role": "user", "content": question}
])

# llm 요약글 RAG 결과
results["summarized"].append({"question": question, "answer": response_llm.content})

print(results)

'''
결과 ex.
{'original': [{'question': '예상되는 가장 중요한 단어는 무엇입니까?', 'answer': '이 글에서 예상되
는 가장 중요한 단어는 "AI" (인공지능)입니다. 이는 여러 앱들이 AI 기술을 활용하여 사용자에게 새로 
운 경험과 기능을 제공하고 있으며, 이러한 앱들이 구글플레이의 수상작으로 선정된 이유 중 하나이기  
때문입니다. AI는 다양한 분야에서 혁신을 주도하고 있는 핵심 기술입니다.'}], 'summarized': [{'question': '예상되는 가장 중요한 단어는 무엇입니까?', 'answer': '이 문장에서 예상되는 가장 중요한 단어
는 "제미나이 사이드 패널", "한국어", "구글 워크스페이스", "추가 지원", "알파 사용자", "피드백", "최적화", "60일 무료 체험판"입니다. 이 단어들은 구글 클라우드의 새로운 기능 추가와 관련된 주요 정 
보를 제공하고 있습니다.'}]}

ex 2.
{'original': [{'question': '예상되는 가장 중요한 단어는 무엇입니까?', 'answer': '이 글에서 예상되는 가장 
중요한 단어는 "프롬프트"입니다. AI와의 소통에서 프롬프트를 어떻게 작성하고 사용하는지가 중요한 주제로 다뤄
지고 있기 때문입니다.'}], 'summarized': [{'question': '예상되는 가장 중요한 단어는 무엇입니까?', 'answer': '이 글에서 예상되는 가장 중요한 단어는 "AI 신뢰성"입니다. AI의 윤리적 행동 보장과 사회적 영향 관리의 중요
성을 강조하고 있으며, 이는 국가 경쟁력에도 영향을 미치는 핵심 요소로 언급되고 있기 때문입니다.'}]}
'''