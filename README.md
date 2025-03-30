# SKN09-3rd-3Team

> SK Networks AI Camp 9기
> 
<br>

# 0. 팀 소개
>
>### 팀명:LLM(Lending Loan Mentor)
>
>
><br>

>### 💲팀원 소개
><table align=center>
>  <tbody>
>    <br>
>      <td align=center><b>서예찬</b></td>
>      <td align=center><b>전성원</b></td>
>     <td align=center><b>조민훈</b></td>
>      <td align=center><b>최재동</b></td>
>    </tr>
>    <tr>
>      
>    
>    </tr>
>    <tr>
>      <td><a href="https://github.com/syc9811"><div align=center>@syc9811</div></a></td>
>      <td><a href="https://github.com/Hack012"><div align=center>@Hack012</div></a></td>
>      <td><a href="https://github.com/alche22"><div align=center>@alche22</div></a></td>
>      <td><a href="https://github.com/Monkakaka"><div align=center>@Monkakaka</div></a></td>
>    </tr>
>  </tbody>
></table>
><br>
>
# 1. 프로젝트 개요

>### 프로젝트 명
>- LLM 기반 대출관련 챗봇 시스템
>
>### 목표
>목표: 사용자에게 대출 상품에 대한 상담, 신청 조건 안내, 상환 계획 계산 등 금융 관련 질문에 실시간으로 답변을 제공하는 것.
>
>주요 역할: 대출 상담, 상품 추천, 사용자 문의 응대.
>
>
>
>### 💲프로젝트 배경
>![Image](https://github.com/user-attachments/assets/38724149-d141-4b31-ac2a-e9316a74091c)
>
>(출처:https://news.nate.com/view/20250323n17586)
>
>최근 강남3구(강남·서초·송파)와 용산구 등의 일부 지역이 토지 거래 허가구역으로 지정되며 KB국민·신한·하나·우리·NH농협 등 주요 은행들은 각기 다른 기준과 절차를 도입하고 있어, 대출 신청 시 필요한 서류와 한도, 금리 등이 은행별로 제각각입니다.예를 들어, 일부 은행은 신규 주택 취득 목적으로만 대출을 허용하는가 하면, 다른 은행은 이미 보유 중인 주택에 대한 전세자금대출을 제한하는 등 세부 정책 차이가 발생하고 있습니다.
>![Image](https://github.com/user-attachments/assets/e05cb257-4bf1-48b2-b828-982d8af33675)
>
>(출처:https://www.yna.co.kr/view/AKR20250322045500002?section=popup)
>
>각 은행의 대출 정책이 자주 변동되고, 지역 및 주택 보유 수에 따른 대출 규제 역시 수시로 바뀌면서 소비자들은 자신에게 맞는 대출 상품을 찾기 어려워졌습니다.게다가 일부 지역의 경우 매입 목적, 전세자금, 기존 주택 보유 여부 등에 따라 대출 가능 여부가 상이하므로, 정확한 정보를 얻지 못하면 잘못된 금융 의사결정을 내릴 위험이 높아집니다.  
>
>
>
>
>
>
>### 💲기대 효과
>빠르게 변동되는 대출 규제 환경에서, LLM 기반 챗봇 시스템은 최신 정보를 지속적으로 반영하고, 사용자별 맞춤형 상담을 제공함으로써 소비자의 혼란을 해소하고 금융기관의 업무 효율을 높이는 핵심 솔루션이 될 것입니다.
>
>
>### 💲요약
>프로젝트 배경: 지역별·주택 보유 수에 따른 대출 조건 차이, 은행별 상이한 대출 정책, 금융 규제 변화로 인해 소비자 혼란이 커지고 있음.
>
>해결 과제: 실시간 업데이트가 가능하고, 복잡한 변수를 종합적으로 고려하여 정확한 정보를 제공할 수 있는 LLM 기반 대출 상담 챗봇의 도입이 요구됨.
>

# 2. 기술 스택

| 분야                   | 기술 및 라이브러리                                                                                                                                                                                                                                       |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 프로그래밍 언어 & 개발환경 | <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white" /> <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=https://gist.githubusercontent.com/yourusername/uniqueid/raw/vscode-logo.svg&logoColor=white" /> <img src="https://img.shields.io/badge/Jupyter-%23FA0F00.svg?style=for-the-badge&logo=Jupyter&logoColor=white" /> |
| 웹 프레임워크            | <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />                                                                                                                                   |
| LLM 체인 및 자연어 처리   |![LangChain](https://img.shields.io/badge/LangChain-005F73?style=for-the-badge&logo=LangChain&logoColor=white)                                                                                                     |
| AI 모델               | <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=OpenAI&logoColor=white" /> <img src="https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=HuggingFace&logoColor=black" />                         |
| 데이터베이스 및 임베딩     | <img src="https://img.shields.io/badge/ChromaDB-FF6F61?style=for-the-badge&logo=https://gist.githubusercontent.com/yourusername/uniqueid/raw/chromadb-logo.svg&logoColor=white" />                                                                                                   |
| 환경변수 관리            | <img src="https://img.shields.io/badge/python_dotenv-000000?style=for-the-badge&logo=Python&logoColor=white" />                                                                                                                                      |
| 문서 로딩               | <img src="https://img.shields.io/badge/PyPDFLoader-4B8BBE?style=for-the-badge&logo=PyPDFLoader&logoColor=white" />                                                                                                                                              |
| 협업 및 형상관리        | <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=Discord&logoColor=white" /> <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white" /> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white" /> |

                                                                                                                                                                                                                            |

<br>

---

# 3. 시스템 아키텍처
><br>
>
>


---
# 4.WBS


---

# 5. 요구사항 명세서
>
>대화형 챗봇 기능
>
    • 사용자의 질문을 분석하여 적절한 대출 상품을 추천해야 한다.

    • 사용자의 요구에 맞춰 상환 금액과 상환 기간을 계산하여 제공해야 한다.

    • RetrievalQA 방식을 활용하여 문서에서 정보를 검색하고 답변을 제공해야 한다.

    • OpenAI API를 활용하여 자연어 기반의 답변을 생성해야 한다.

> 질의응답(QA) 기능
> 
    • 문서에서 대출 상품 관련 정보를 검색하여 사용자 질문에 답변을 제공해야 한다.

    • 질문과 관련된 문서를 우선순위에 따라 검색하고 출처를 명확히 표시해야 한다.

    • 다양한 상환 방식(원리금 균등, 원금 균등, 체증식 분할 등)을 계산하고 그 결과를 제공해야 한다.

    • 대출 상품별 상환 방식을 비교 분석할 수 있어야 한다.
# 6. 데이터 및 전처리 
>
>### 💲데이터 수집 및 선정
>![Image](https://github.com/user-attachments/assets/cd69657a-00c7-4166-8ebf-89aa9deed284)
>![Image](https://github.com/user-attachments/assets/8a489d80-8534-474c-87d7-bcbd6f5a948a)
>![Image](https://github.com/user-attachments/assets/50c27c1e-6da5-4b59-94ec-470b7b3f08b1)
>![Image](https://github.com/user-attachments/assets/bdd81505-689d-4526-aefa-89a15017433e)
>![Image](https://github.com/user-attachments/assets/62b73732-1737-4a25-8b6c-4320742a53bf)
>
>
> #### (국민은행,하나은행,농협은행,우리은행,신한은행 대출상품)
>
>(우리은행 출처:https://spot.wooribank.com/pot/Dream?withyou=POLON0055&cc=c010528:c010531;c012425:c012399&PLM_PDCD=P020000273&PRD_CD=P020000273&HOST_PRD_CD=2031037000079)
>
>(국민은행 출처:https://obank.kbstar.com/quics?page=C103425)
>
>(신한은행 출처:https://bank.shinhan.com/index.jsp#020300000000)
>
>(하나은행 출처:https://www.kebhana.com/cont/mall/mall08/mall0802/mall080204/1462446_115200.jsp?_menuNo=98786)
>
>(농협은행 출처:https://smartmarket.nonghyup.com/servlet/BFLNW0000R.view)

><br>

>### 💲데이터 전처리 


----

# 6. 테스트 계획 및 결과 보고서


---


# 7. 수행결과


---



# 8. 한 줄 회고
>서예찬:
>전성원:
>조민훈:
>최재동:
><br>
