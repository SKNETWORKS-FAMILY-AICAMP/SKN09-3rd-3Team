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
><출처:https://www.joongang.co.kr/article/25322736>
>
>최근 강남3구(강남·서초·송파)와 용산구 등의 일부 지역이 토지 거래 허가구역으로 지정되며 KB국민·신한·하나·우리·NH농협 등 주요 은행들은 각기 다른 기준과 절차를 도입하고 있어, 대출 신청 시 필요한 서류와 한도, 금리 등이 은행별로 제각각입니다.예를 들어, 일부 은행은 신규 주택 취득 목적으로만 대출을 허용하는가 하면, 다른 은행은 이미 보유 중인 주택에 대한 전세자금대출을 제한하는 등 세부 정책 차이가 발생하고 있습니다.
>![Image](https://github.com/user-attachments/assets/e05cb257-4bf1-48b2-b828-982d8af33675)
>
><출처:https://www.yna.co.kr/view/AKR20250322045500002?section=search>
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
| 프로그래밍 언어 & 개발환경          | <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white" />                                                                                                                                     |
| 웹 프레임워크            | <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />                                                                                                                                   |
| LLM 체인 및 자연어 처리   | <img src="https://img.shields.io/badge/LangChain-FF5733?style=for-the-badge&logo=&logoColor=white" />                                                                                                                                |
| AI 모델               |  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=OpenAI&logoColor=white" />                                                                                                                                  |
| 데이터베이스 및 임베딩     |  <img src="https://img.shields.io/badge/Chroma%20DB-2C3E50?style=for-the-badge&logo=&logoColor=white" /><br>                                                                                   |
| 환경변수 관리            | <img src="https://img.shields.io/badge/python_dotenv-000000?style=for-the-badge&logo=Python&logoColor=white" />                                                                                                                   |
| 문서 로딩               |  <img src="https://img.shields.io/badge/PyPDFLoader-4B8BBE?style=for-the-badge&logo=&logoColor=white" />                                                                                                      |
| 협업 및 형상관리        |   <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=Discord&logoColor=white" /> <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white" /> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white" />                                                                                                                                                                                                                                |

<br>

---

# 3. 시스템 아키텍처
><br>
>
>


---
# 4.WBS


---

# 요구사항 명세서

# 6. 데이터 및 전처리 
>
>### 💲데이터 수집 및 선정

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
