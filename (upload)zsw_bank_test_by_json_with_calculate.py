import os
import json
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from datetime import datetime


# 환경 변수 설정
OPENAI_API_KEY = "[API KEY 입력]"
CHROMA_PATH = "./chroma_db"
JSON_FILE = "./cleaned"

# .json 추출 및 통합합
def load_document_from_json_from_json_folder(folder_path):
    documents = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                print(f"❌ {file_path} 파일이 존재하지 않습니다")
                continue

            with open(file_path, "r", encoding='utf-8') as f:
                json_data = json.load(f)

            for item in json_data:
                source = item["source"]
                content = item["content"]
                documents.append({"source": source, "content": content})
    if not documents:
        print("❌ 문서가 존재하지 않습니다.")

    return documents
    

def create_chroma_db():
    documents = load_document_from_json_from_json_folder(JSON_FILE)
    if not documents:
        print("❌ 문서가 존재하지 않습니다.")
        return None

    split_docs = [
        Document(page_content=doc["content"], metadata={"source": doc["source"]})
        for doc in documents
    ]

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(split_docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # text-embedding-ada-002
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    print("✅ ChromaDB 저장 완료!")
    return db

def get_qa_chain_with_memory(db, flag=None):
    flag_var = flag

    retriever = db.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.5)

    # 프롬프트 템플릿
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"], 
        template=(
            """
                당신은 은행 대출 전문가 AI입니다.  
                사용자에게 친절하고 상세하게 대출 상품을 안내하며,  
                전세자금대출과 주택담보대출에 관한 정확하고 신뢰성 있는 정보를 제공합니다.  
                다음 '질문'에 대해 제공된 '문서'의 내용을 바탕으로 구체적이고 정확한 답변을 작성해 주세요.  
                상품 추천의 경우, 문서 내용을 중심으로 추천 상품을 안내해 주세요.  
                만약 사용자가 계산을 요청할 경우, 관련 공식을 제공하고, 필요한 값을 수집하여 계산을 진행한 후, 가장 적합한 대출 상품을 추천해 주세요.

                ### 질문:  
                {question}

                ### 문서:  
                {context}

                ### 계산 가능한 항목 ###

                1. 원리금 균등분할 상환  
                - 매월 상환 금액 계산  

                2. 원금균등분할 상환
                - 각 달의 상환 금액 리스트 제공  

                3. 체증식 분할 상환
                - 원리금 균등 상환 금액을 기준으로 점차 증가하는 형태로 계산  
                - 공식:  
                    첫 달의 상환금액을 원리금 균등분할 상환 금액으로 설정하고, 이후 점차적으로 상환 금액을 증가시킴.  
                    - 첫 달: 원리금 균등분할 상환금액  
                    - 둘째 달 이후: 원리금 균등 상환금액 + 증가액  
                    증가액은 상환 개월 수와 대출 잔액을 고려하여 계산됩니다.

                ### 주의 사항 ###  
                - 원금균등분할 상환과 체증식 분할 상환의 경우, 각 개월 수와 상환 금액을 명확히 제공해야 합니다.  
                - 각각의 공식을 이용해 계산하기 위해 필요한 값 중 사용자의 대답에 어떠한 값이 빠져있다면 그 값을 요청해야 합니다. 
                - 필수 값이 누락되었을 경우  
                    만약 제공되지 않은 값이 있을 경우, 계산을 정확히 진행하기 위해 해당 값을 요청합니다: 예시: "상환 기간을 알려주시겠어요?", "대출 원금을 알려주세요.", "연 이자율을 입력해 주세요." 

                ### 답변 ###  
                필수 입력 값 
                사용자가 요청한 계산을 위해서는 몇 가지 값들이 필요합니다. 아래의 정보를 제공해 주시면 계산을 진행할 수 있습니다:

                1. 대출 원금 (P)  
                - 예시: 5천만 원, 1억 원 등

                2. 연 이자율 (r)  
                - 예시: 3%, 5% 등

                3. 상환 기간 (n)  
                - 예시: 5년, 10년 등 (개월 수로 계산됩니다)

                4. 첫 달 상환 금액 (첫 달의 원리금 균등 상환금액이나 체증식 상환 금액의 시작점)**  
                - 예시: 첫 달에 50만 원으로 시작 등

                ### 계산 진행 ###  
                만약 제공되지 않은 값이 있을 경우, 계산을 정확히 진행하기 위해 반드시 해당 값을 요청합니다: 예시: "상환 기간을 알려주시겠어요?", "대출 원금을 알려주세요.", "연 이자율을 입력해 주세요."
                모든 필수 값이 제공되면, 계산을 진행하여 결과를 제공합니다.  
                각 상환 방식에 따라 계산이 이루어지며, 결과는 사용자의 요구에 맞춰 제공합니다.
                계산 과정은 계산 공식만 제공하며, 계산 후 결과 값만 간단하게 제공해주세요.
                사용자 친화적이게 필요하다면 사용자의 질문을 인용해서 답변을 생성해주세요.
            """
        )
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        chain_type="stuff",
        memory=memory,
        chain_type_kwargs={
            "prompt": custom_prompt,
            "verbose": False
        }
    )

    def custom_run(query):
        nonlocal flag_var
        print("플래그 정보:",flag_var)

        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"\n🔎 질문: {query}")
        print(f"📂 검색된 문서 개수: {len(retrieved_docs)}")

        # 은행별로 대출 상품 정리
        bank_products = {}
        sources = set()

        for doc in retrieved_docs:       
            try:
                bank_name = doc.metadata.get("source", "알 수 없음").split(os.sep)[-2]
            except IndexError:
                bank_name = doc.metadata.get("source", "알 수 없음").split("_")[0]
                flag_var = None
            
            sources.add(doc.metadata.get("source", "알 수 없음"))

            if bank_name not in bank_products:
                bank_products[bank_name] = []

            bank_products[bank_name].append(doc.page_content[:700]) # 최대  700자 까지 저장

        bank_info_text = "**은행별 대출 상품 정보:**\n"
        for bank, products in bank_products.items():
            bank_info_text += f"\n🏦 **{bank.upper()}**\n"
            for i, product in enumerate(products, start=1):
                bank_info_text += f"{i}. {product}...\n"

        modified_query = f"{query}\n\n{bank_info_text}"
        print("실행 쿼리: ", modified_query)
        response = qa_chain.run(modified_query)

        response_text = ""
        if flag_var is not None:
            response_text = f"💡 **AI 답변:**\n{response}\n\n📌 **출처:**{', '.join(sources)}"
        else:
            response_text = f"💡 **AI 답변:**\n{response}"

        return response_text

    return custom_run

def generate_feedback(history, user_feedback, qa_chain):
    conversation_log = "\n".join([f"사용자: {q}\nAI: {a}" for q, a in history])
    evaluation_prompt = f"""
        당신은 AI 대화 평가 전문가입니다. 아래 대화를 평가하세요.

        ### 대화 내역 ###
        {conversation_log}

        ### 사용자 피드백 ###
        '{user_feedback}' (👍 만족 / 🤔 보통 / 👎 개선 필요)

        ### 평가 기준 ###
        - 정보의 명확성(1~5)
        - 답변의 명확성(1~5)
        - 사용자 친화성(1~5)
        - 개선이 필요한 부분

        사용자 피드백을 고려하여 AI의 강점과 개선점을 분석하고, 보완할 점을 제안하세요.
    """

    return qa_chain(evaluation_prompt)

def main():
    st.markdown(
        "<h2 style='text-align: center;'>💬 은행 전세자금대출 및 주택담보대출 Q&A 챗봇</h2>",
        unsafe_allow_html=True
    )
    st.write("은행 전세자금대출 및 주택담보대출 관련 질문을 입력하세요!")

    # ChromaDB 초기화
    if not os.path.exists(CHROMA_PATH):
        st.info("문서 DB를 생성 중입니다. 잠시만 기다려 주세요...")
        db = create_chroma_db()
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    qa_chain = get_qa_chain_with_memory(db, "user")

    user_input = st.text_input("질문을 입력하세요:")
    
    if st.button("질문하기") and (user_input not in ["대화 종료"]):
        with st.spinner("답변을 생성 중입니다..."):
            response = qa_chain(user_input)  # 과거 대화 반영됨
            st.success(response)

            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, (user_input, response))

    if user_input in ["대화 종료"]:
        st.success("😊 도움이 되었다니 기쁩니다! 추가 질문이 있으면 언제든 물어보세요.")
        st.write("👉 혹시 이번 답변이 만족스러웠나요? 아래 버튼을 눌러 피드백을 남겨주세요!")

        user_feedback = st.radio("답변이 만족스러웠나요?", ("👍 만족", "🤔 보통", "👎 개선 필요"), index=None)
        if user_feedback:
            if "feedback_processed" not in st.session_state or not st.session_state.feedback_processed:
                with st.spinner("대화 평가 중..."):
                    feedback = generate_feedback(st.session_state.history, user_feedback, qa_chain)
                    st.session_state.feedback_message = feedback

                    st.session_state.feedback_processed = True

                    # if st.button("🔄"): // 준비 안됌
                    #     st.session_state.feedback_processed=False
                    #     st.experimental_rerun()
            if "feedback_message" in st.session_state and st.session_state.feedback_message:
                st.write("🧐 AI 자체 평가 결과:")
                st.write(f"✅ 피드백이 반영되었습니다: {st.session_state.feedback_message}")
                    
        if "history" in st.session_state and st.session_state.history:
            conversation_log = "\n".join([f"사용자: {q}\nAI: {a}" for q, a in st.session_state.history])
                    
            today = datetime.today().strftime('%Y.%m.%d')
            download_filename = f"{today}_conversation_history.txt"
            st.download_button(
                label="대화 내용 다운로드하기",
                data=conversation_log,
                file_name=download_filename,
                mime="text/plain"
            )


    # 이전 대화 기록 표시
    if "history" in st.session_state and st.session_state.history:
        st.write("### 📜 이전 대화 기록")
        for question, answer in st.session_state.history:
            st.write(f"**❓:** {question}")
            st.write(f"**👀:** {answer}")
            st.write("---")

if __name__ == "__main__":
    main()
