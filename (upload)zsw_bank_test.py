import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


# 환경 변수 설정
OPENAI_API_KEY = '[API KEY 입력]'
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
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # 사용 임베딩 모델: text-embedding-ada-002
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    print("✅ ChromaDB 저장 완료!")
    return db

def get_qa_chain_with_memory(db):
    retriever = db.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.5)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        chain_type="stuff",
        memory=memory
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

def main():
    st.title("💬은행 전세자금대출 Q&A 챗봇")
    st.write("은행 전세자금대출 관련 질문을 입력하세요!")

    # ChromaDB 초기화
    if not os.path.exists(CHROMA_PATH):
        st.info("문서 DB를 생성 중입니다. 잠시만 기다려 주세요...")
        db = create_chroma_db()
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    qa_chain = get_qa_chain_with_memory(db)

    user_input = st.text_input("질문을 입력하세요:")
    
    if st.button("질문하기") and user_input:
        with st.spinner("답변을 생성 중입니다..."):
            response = qa_chain(user_input)  # 과거 대화 반영됨
            st.success("📝 답변:")
            st.write(response)  # PDF 출처 포함

            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append((user_input, response))

    # 이전 대화 기록 표시
    if "history" in st.session_state and st.session_state.history:
        st.write("### 📜 이전 대화 기록")
        for question, answer in st.session_state.history:
            st.write(f"**Q:** {question}")
            st.write(f"**A:** {answer}")
            st.write("---")

if __name__ == "__main__":
    main()
