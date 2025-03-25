import os
import streamlit as st
from langchain.chains import RetrievalQA
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
OPENAI_API_KEY = "[API Key 입력]"
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"

def load_documents():
    documents = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                print(f"📄 PDF 로드 중: {pdf_path}")

                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                print(f"🔹 {file}에서 {len(docs)}개의 문서 로드 완료")
                documents.extend(docs)
    
    return documents

def create_chroma_db():
    documents = load_documents()
    print(f"📚 총 {len(documents)}개의 문서가 로드되었습니다.")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"🔍 문서 분할 완료: 총 {len(docs)}개의 청크 생성됨")
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # text-embedding-ada-002
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    print("✅ ChromaDB 저장 완료!")
    return db

def get_qa_chain_with_memory(db):
    """과거 대화 내용을 기억하는 QA 체인 생성"""
    retriever = db.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.5)
    
    # 맞춤형 프롬프트 설정
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"], # 입력 변수 정의
        template=(
            """
                당신은 은행 대출 전문가 AI입니다.  
                사용자에게 친절하고 자세하게 대출 상품을 안내하며,  
                전세자금대출과 주택담보대출에 대한 정보를 정확하고 신뢰성 있게 제공합니다. 
                다음 '질문'에 대해 제공된 '문서'의 정보를 기반으로 정확하고 구체적인 답변을 생성해 주세요. 
                문서의 내용을 중심으로 답변해 주세요.

                질문:  
                {question}  

                문서:  
                {context}  

                답변:  
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
        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"\n🔎 질문: {query}")
        print(f"📂 검색된 문서 개수: {len(retrieved_docs)}")

        bank_products = {}
        sources = set()

        for doc in retrieved_docs: 
            bank_name = doc.metadata.get("source", "알 수 없음").split(os.sep)[-2]
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
        print(modified_query)
        response = qa_chain.run(modified_query)

        response_text = f"💡 **AI 답변:**\n{response}\n\n📌 **출처:** {', '.join(sources)}"
        return response_text

    return custom_run

# feedback function
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

    # llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0)
    # return llm.predict(evaluation_prompt)

    return qa_chain(evaluation_prompt) #기존 체인에 학습

def main():
    st.markdown(
        "<h2 style='text-align: center;'>💬 은행 전세자금대출 및 주택담보대출 Q&A 챗봇</h2>",
        unsafe_allow_html=True
    )
    st.write("은행 전세자금대출 및 주택담보대출 관련 질문을 입력하세요!")

    if not os.path.exists(CHROMA_PATH):
        st.info("문서 DB를 생성 중입니다. 잠시만 기다려 주세요...")
        db = create_chroma_db()
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    qa_chain = get_qa_chain_with_memory(db)

    user_input = st.text_input("질문을 입력하세요:")
    
    if st.button("질문하기") and (user_input not in ["대화 종료"]):
        with st.spinner("답변을 생성 중입니다..."):
            response = qa_chain(user_input)  # 과거 대화 반영됨
            st.success("📝 답변:")
            st.write(response)  # PDF 출처 포함

            # 사용자 질문 및 답변 저장
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append((user_input, response))

    if user_input in ["대화 종료"]:
        st.success("😊 도움이 되었다니 기쁩니다! 추가 질문이 있으면 언제든 물어보세요.")
        st.write("👉 혹시 이번 답변이 만족스러웠나요? 아래 버튼을 눌러 피드백을 남겨주세요!")

        # 피드백 버튼 추가
        user_feedback = st.radio("답변이 만족스러웠나요?", ("👍 만족", "🤔 보통", "👎 개선 필요"), index=None)
        if user_feedback:
            if "feedback_processed" not in st.session_state or not st.session_state.feedback_processed:
                with st.spinner("대화 평가 중..."):
                    feedback = generate_feedback(st.session_state.history, user_feedback, qa_chain)
                    st.write("🧐 AI 자체 평가 결과:")
                    st.write(f"✅ 피드백이 반영되었습니다: {feedback}")


                    st.session_state.feedback_processed = True
                    
            if "history" in st.session_state and st.session_state.history:
                conversation_log = "\n".join([f"사용자: {q}\nAI: {a}" for q, a in st.session_state.history])
                
                today = datetime.today().strftime('%Y-%m-%d')
                download_filename = f"{today}_conversation_history.txt"
                st.download_button(
                    label="대화 내용 다운로드하기",
                    data=conversation_log,
                    file_name=download_filename,
                    mime="text/plain"
                )

    if "history" in st.session_state and st.session_state.history:
        st.write("### 📜 이전 대화 기록")
        for question, answer in st.session_state.history:
            st.write(f"**Q:** {question}")
            st.write(f"**A:** {answer}")
            st.write("---")

if __name__ == "__main__":
    main()
