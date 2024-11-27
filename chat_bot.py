import os
from dotenv import load_dotenv
import streamlit as st
import embedding as emb

# 환경 변수 로드
load_dotenv(dotenv_path='seok.env')
api_key = os.getenv("API_KEY")

# Streamlit 기본 설정
st.header("compare llm summerized news VS original news")

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if user_input := st.chat_input("질문을 입력하세요! 종료하려면 'exit'을 입력하세요."):
    # 종료 조건 처리
    if user_input.lower() == "exit":
        st.info("채팅을 종료합니다.")
    else:
        # 사용자 메시지를 히스토리에 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # RAG 모델 호출
        with st.chat_message("assistant"):
            try:
                response_llm = emb.rag_chain_llm.invoke(user_input)
                response_llm_content = response_llm.content  # 응답 내용을 가져옵니다.
                response_o = emb.rag_chain.invoke(user_input)
                response_o_content = response_o.content  # 응답 내용을 가져옵니다.
                st.markdown(f"summerized : {response_llm_content}")
                st.markdown(f"\nraw : {response_o_content}")

                # 어시스턴트 메시지를 히스토리에 추가
                st.session_state.messages.append({"role": "assistant", "content": response_llm_content})
                st.session_state.messages.append({"role": "assistant", "content": response_o_content})
            except Exception as e:
                st.error(f"RAG 호출 중 오류가 발생했습니다: {str(e)}")