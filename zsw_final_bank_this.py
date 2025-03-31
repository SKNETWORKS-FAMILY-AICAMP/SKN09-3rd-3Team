from typing import TypedDict, Annotated, Sequence

import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
import streamlit as st
from datetime import datetime
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API 키
LANGSMITH_PROJECT_NAME = os.getenv("LANGSMITH_PROJECT_NAME")  # Langsmith 프로젝트 이름
CHROMA_PATH = "./chroma_db"
JSON_FILE = "./cleaned"


def load_document_from_json_from_json_folder(folder_path):
    """주어진 폴더에서 모든 JSON 파일을 로드하여 문서 리스트 생성"""
    documents = []

    # 폴더 내 모든 JSON 파일 확인
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                print(f"❌ {file_path} 파일이 존재하지 않습니다")
                continue

            with open(file_path, "r", encoding='utf-8') as f:
                json_data = json.load(f)

            # 각 JSON 파일에서 문서 리스트로 변환
            for item in json_data:
                source = item["source"]
                content = item["content"]
                documents.append({"source": source, "content": content})
    if not documents:
        print("❌ 문서가 존재하지 않습니다.")

    return documents
    

def create_chroma_db():
    """ChromaDB 인덱스를 생성하고 문서를 저장"""
    documents = load_document_from_json_from_json_folder(JSON_FILE)
    if not documents:
        print("❌ 문서가 존재하지 않습니다.")
        return None

    # 문서 내용을 Document 형식으로 변환
    split_docs = [
        Document(page_content=doc["content"], metadata={"source": doc["source"]})
        for doc in documents
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(split_docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # text-embedding-ada-002
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    print("✅ ChromaDB 저장 완료!")
    return db

if not os.path.exists(CHROMA_PATH):
    db = create_chroma_db()
else:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

context_prompt = (
    "대화 기록과 최신 사용자 질문이 주어졌을 때"
    "이 질문이 대화 기록의 맥락을 참조할 수 있으며,"
    "대화 기록 없이 이해할 수 있는 독립적인 질문을 형성하십시오."
    "대화 기록의 맥락 없이 질문에 답하지 마세요,"
    "필요하다면 질문을 다시 표현하고, 그렇지 않으면 그대로 반환하세요."
)

prompt_template = """
    당신은 은행 대출 전문가 AI입니다.  
        사용자에게 친절하고 상세하게 대출 상품을 안내하며,  
        전세자금대출과 주택담보대출에 관한 정확하고 신뢰성 있는 정보를 제공합니다.  
        다음 '질문'에 대해 제공된 '문서'의 내용을 바탕으로 구체적이고 정확한 답변을 작성해 주세요.  
        상품 추천의 경우, 문서 내용을 중심으로 추천 상품을 안내해 주세요.  
        만약 사용자가 계산을 요청할 경우, 관련 공식을 제공하고, 필요한 값을 수집하여 계산을 진행한 후, 가장 적합한 대출 상품을 추천해 주세요.

        ### 질문:  
        {input}

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

retriever = db.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
    
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
    
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"])
        ],
        "context": response["context"],
        "answer": response["answer"]
    }

# 검색된 문서를 출력하는 커스텀 실행 함수
def custom_run(query, retriever, flag=None):
    print("플래그 정보:",flag)
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
            flag = None
            
        sources.add(doc.metadata.get("source", "알 수 없음"))

        if bank_name not in bank_products:
            bank_products[bank_name] = []

        bank_products[bank_name].append(doc.page_content[:700]) # 최대  700자 까지 저장

    # 은행별 대출 상품 정보를 하나의 텍스트로 구성
    bank_info_text = "**은행별 대출 상품 정보:**\n"
    for bank, products in bank_products.items():
        bank_info_text += f"\n🏦 **{bank.upper()}**\n"
        for i, product in enumerate(products, start=1):
            bank_info_text += f"{i}. {product}...\n"

    # QA Chain 실행 (은행별 정보 포함)
    modified_query = f"{query}\n\n{bank_info_text}"

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "zsw123"}}

    response = app.invoke(
        {"input": modified_query},
        config=config
    )

    answer = response["answer"]

    response_text = ""
    if flag is not None:
        response_text = f"💡 **AI 답변:**\n{answer}\n\n📌 **출처:**{', '.join(sources)}"
    else:
        response_text = f"💡 **AI 답변:**\n{answer}"

    return response_text

def main():
    st.markdown(
        "<h2 style='text-align: center;'>💬 은행 전세자금대출 및 주택담보대출 Q&A 챗봇</h2>",
        unsafe_allow_html=True
    )
    st.write("😊 안녕하세요! 은행 대출 관련 질문을 입력하세요!")

    user_input = st.text_input("질문을 입력하세요:")

    if st.button("질문하기") and (user_input not in ["대화 종료", "대화종료"]):
        with st.spinner("답변을 생성 중입니다..."):
            response= custom_run(user_input, retriever, "user")
            st.success(response)

            if "logs" not in st.session_state:
                st.session_state.logs = []
            st.session_state.logs.insert(0, (user_input, response))

    if user_input in ["대화 종료", "대화종료"]:
        st.success("😊 도움이 되었다니 기쁩니다! 추가 질문이 있으면 언제든 물어보세요.")
        if st.button("🔄"):
            st.session_state.feedback_processed=False
            st.rerun()
        
        # 대화 내용 다운로드 기능 추가
        if "logs" in st.session_state and st.session_state.logs:
            # 대화 내역을 텍스트 형식으로 변환
            conversation_log = "\n".join([f"사용자: {q}\nAI: {a}\n\n" for q, a in st.session_state.logs])

            # 다운로드할 텍스트 파일 생성
            today = datetime.today().strftime('%Y%m%d')
            download_filename = f"{today}_conversation_history.txt"
            st.download_button(
                label="대화 내용 다운로드하기",
                data=conversation_log,
                file_name=download_filename,
                mime="text/plain"
            )

    # 이전 대화 기록 표시
    if "logs" in st.session_state and st.session_state.logs:
        st.write("### 📜 이전 대화 기록")
        for question, answer in st.session_state.logs:
            st.write(f"**❓:** {question}")
            st.write(f"**👀:** {answer}")
            st.write("---")

if __name__ == "__main__":
    main()


