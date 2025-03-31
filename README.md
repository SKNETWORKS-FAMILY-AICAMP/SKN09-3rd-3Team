# 0. 팀 소개
  ### 팀명:LLM(Lending Loan Mentor)
<br>

### 💲팀원 소개
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



# 1. 프로젝트 개요

### 프로젝트 명
- LLM 기반 대출관련 챗봇 시스템

### 프로젝트 목표 및 주요 역할
- **목표**: 사용자에게 대출 상품에 대한 상담, 신청 조건 안내, 상환 계획 계산 등 금융 관련 질문에 실시간으로 답변을 제공하는 것.
- **주요 역할**: 대출 상담, 상품 추천, 사용자 문의 응대.


### 💲프로젝트 배경
<div align="center">
  <img src="https://github.com/user-attachments/assets/38724149-d141-4b31-ac2a-e9316a74091c" width="45%" />
  <img src="https://github.com/user-attachments/assets/e05cb257-4bf1-48b2-b828-982d8af33675" width="45%" />
</div>

- 최근 강남3구(강남·서초·송파)와 용산구가 토지거래 허가구역으로 지정되면서, KB국민·신한·하나·우리·NH농협 등 주요 은행들이 각기 다른 대출 기준과 절차를 운영 이에 따라 대출 신청 시 필요한 서류, 한도, 금리 등이 은행별로 상이하며, 일부 은행은 신규 주택 취득 목적 대출만 허용하거나, 보유 주택의 전세자금대출을 제한하는 등 정책 차이가 발생하고 있음.
- 각 은행의 대출 정책이 자주 변동되고, 지역 및 주택 보유 수에 따른 대출 규제가 수시로 변경되면서 소비자들이 자신에게 적합한 대출 상품을 찾기가 점점 어려워지는 상황.
- 특히, 일부 지역에서는 매입 목적, 전세자금 여부, 기존 주택 보유 상태에 따라 대출 가능 여부가 달라지기 때문에, 정확한 정보를 확보하지 못하면 잘못된 금융 의사결정을 내릴 위험이 다분함.

🔗 **출처**
- https://news.nate.com/view/20250323n17586
- https://www.yna.co.kr/view/AKR20250322045500002?section=popup
  

### 💲기대 효과
- 빠르게 변동되는 대출 규제 환경에서, LLM 기반 AI 대출 상담 봇은 실시간 정책 변화를 반영하고 사용자별 맞춤형 상담을 제공하여 최적의 대출 옵션을 안내함.
- 이를 통해 소비자의 혼란을 해소하고, 금융기관의 업무 효율을 높이는 핵심 솔루션이 될 것으로 보임.


### 💲요약
- **프로젝트 배경**: 지역별·주택 보유 수에 따른 대출 조건 차이, 은행별 상이한 대출 정책, 금융 규제 변화로 인해 소비자 혼란이 커지고 있음.

- **해결 과제**: 실시간 업데이트가 가능하고, 복잡한 변수를 종합적으로 고려하여 정확한 정보를 제공할 수 있는 LLM 기반 대출 상담 챗봇의 도입이 요구됨.

<br>


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

<br>


# 3. 시스템 아키텍처

![system_architect](https://github.com/user-attachments/assets/732f96f9-b2f7-400c-8e18-3a809252be30)



---
# 4.WBS
![Image](https://github.com/user-attachments/assets/fd7cc5a1-bb96-4ef5-9809-d22b485a65a0)

<br>


# 5. 요구사항 명세서
>
> **대화형 챗봇 기능**
>
    • 사용자의 질문을 분석하여 적절한 대출 상품을 추천해야 한다.

    • 사용자의 요구에 맞춰 상환 금액과 상환 기간을 계산하여 제공해야 한다.

    • RetrievalQA 방식을 활용하여 문서에서 정보를 검색하고 답변을 제공해야 한다.

    • LLM Model을 활용하여 자연어 기반의 답변을 생성해야 한다.

> **질의응답(QA) 기능**
> 
    • 문서에서 대출 상품 관련 정보를 검색하여 사용자 질문에 답변을 제공해야 한다.

    • 질문과 관련된 문서를 우선순위에 따라 검색하고 출처를 명확히 표시해야 한다.

    • 다양한 상환 방식(원리금 균등, 원금 균등, 체증식 분할 등)을 계산하고 그 결과를 제공해야 한다.

    • 대출 상품별 상환 방식을 비교 분석할 수 있어야 한다.
🔗: [G_대출 추천 AI 시스템 요구사항 명세서.pdf](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN09-3rd-3Team/blob/1ab17cb4d59f86b582932a05cce5634da7a13072/report/G_%EB%8C%80%EC%B6%9C%20%EC%B6%94%EC%B2%9C%20AI%20%EC%8B%9C%EC%8A%A4%ED%85%9C%20%EC%9A%94%EA%B5%AC%EC%82%AC%ED%95%AD%20%EB%AA%85%EC%84%B8%EC%84%9C.pdf)

<br>

    
# 6. 데이터 및 전처리 
>
>### 💲데이터 수집 및 선정
>
><div align="center">
  <img src="https://github.com/user-attachments/assets/cd69657a-00c7-4166-8ebf-89aa9deed284" width="30%" />
  <img src="https://github.com/user-attachments/assets/8a489d80-8534-474c-87d7-bcbd6f5a948a" width="30%" />
  <img src="https://github.com/user-attachments/assets/50c27c1e-6da5-4b59-94ec-470b7b3f08b1" width="30%" />
></div>
><div align="center">
  <img src="https://github.com/user-attachments/assets/bdd81505-689d-4526-aefa-89a15017433e" width="45%" />
  <img src="https://github.com/user-attachments/assets/62b73732-1737-4a25-8b6c-4320742a53bf" width="45%" />
></div>


#### 국민은행,하나은행,농협은행,우리은행,신한은행 대출상품 출처
- 🔗 [하나은행 출처](https://www.kebhana.com/cont/mall/mall08/mall0802/mall080204/1462446_115200.jsp?_menuNo=98786)
- 🔗 [농협은행 출처](https://smartmarket.nonghyup.com/servlet/BFLNW0000R.view)
- 🔗 [국민은행 출처](https://obank.kbstar.com/quics?page=C103425)
- 🔗 [신한은행 출처](https://bank.shinhan.com/index.jsp#020300000000)
- 🔗 [우리은행 출처](https://spot.wooribank.com/pot/Dream?withyou=POLON0055&cc=c010528:c010531;c012425:c012399&PLM_PDCD=P020000273&PRD_CD=P020000273&HOST_PRD_CD=2031037000079)

<br>

>### 💲데이터 전처리 
| 순서 | 내용               | 설명                                                                 |
|------|--------------------|----------------------------------------------------------------------|
| 1    | `\xa0` → 공백 치환 | PDF 추출 시 자주 나타나는 **비표준 공백 문자(줄바꿈 불가 공백)**를 일반 공백으로 치환 |
| 2    | URL 제거           | `http` 또는 `https`로 시작하는 **웹 링크(URL)**를 모두 삭제                           |
| 3    | 공백 정리          | 연속된 **공백, 줄바꿈, 탭** 등을 **하나의 공백**으로 압축하여 깔끔하게 정리               |
| 4    | 특수 문자 제거     | **한글, 영문, 숫자, 일부 문장 부호**(예: . , ! ?)를 제외한 **불필요한 특수 문자 제거**     |
| 5    | 양쪽 공백 제거     | 각 문장의 **양쪽에 붙은 공백**을 제거하여 정돈된 형태로 정리                             |

🔗: [전처리 완료된 db 생성용 데이터](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN09-3rd-3Team/blob/f8348c9b5de3e3ade820db704cad5856266c696f/cleaned/cleaned_data.json)
<br>
<br>

>### 💲학습용 데이터셋 생성
>
  - 위에서 생성한 json파일은 "source"와 "content", 즉 출처와 문서의 내용만을 담고 있음.
  - 파인튜닝용 데이터로는 "instruction","input","output"의 형태를 한 jsonl 파일이 필요
  - 단순 데이터 문서가 아닌 질의 형식의 데이터가 필요했으므로 gpt-4o-mini 모델을 api로 호출해서 분단된 청크 하나당 1~3개의 질의를 생성해 형식에 맞게 출력하게 함.
  - 답변을 jsonl파일에 순차적으로 저장한뒤, `" "`,`\n` 같은 불필요한 문자열 제거

🔗: [파인튜닝용 데이터 셋](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN09-3rd-3Team/blob/f8348c9b5de3e3ade820db704cad5856266c696f/training_data/qa_dataset_cleaned.jsonl)

<br>


# 7. DB 연동  구현 코드

🔗: [zsw_final_bank_this.py](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN09-3rd-3Team/blob/1ab17cb4d59f86b582932a05cce5634da7a13072/zsw_final_bank_this.py)

- 해당 파일의 ```def load_document_from_json_from_json_folder(folder_path):``` 부분과 ```def create_chroma_db():``` 부분 참조
- ```def load_document_from_json_from_json_folder(folder_path):```에서 ```"./cleaned"```폴더의 json파일을 로드해서 리스트 생성
- ```def create_chroma_db():```에서 생성한 리스트를 chroma_db로 변환
- db 연동 확인

![langsmith](https://github.com/user-attachments/assets/e2af3d4b-042f-4234-ad02-0d6b671f9961)



<br>


# 8. 테스트 계획 및 결과 보고서

- **테스트**: 다양한 금융 대출의 질의 테스트
- **목적**
    - 사용자가 다양한 금융 대출에 대한 질문을 입력하면 올바른 답변을 제공하는지 확인한다.
    - 대출 관련 용어 설명, 한도 기준, 신용점수 영향 등을 정확히 안내하는지 검증한다.

- **테스트 시나리오**: 사용자가 대출 관련 다양한 질문을 입력한다.  
  예시)  
    &nbsp;&nbsp;&nbsp; Q1) "DSR(총부채원리금상환비율)이 뭔가요?"  
    &nbsp;&nbsp;&nbsp; Q2) "신용등급이 낮아도 대출 받을 수 있나요?"  
    &nbsp;&nbsp;&nbsp; Q3) "고정금리와 변동금리 중 어떤 걸 선택해야 하나요?"  
    &nbsp;&nbsp;&nbsp; Q4) "마이너스 통장 대출과 일반 신용대출의 차이는?"  
    &nbsp;&nbsp;&nbsp; Q5) "중도 상환 수수료가 적용되는 기준은?"  

- **주의 사항**
    - 시스템이 사용자의 질문에 대해 정확한 정보를 제공하는지 확인한다.
    - 추가적으로 관련 금융 조언이 함께 제공되는지 검토한다.
  
🔗: [기능 테스트 시나리오.pdf 보기](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN09-3rd-3Team/blob/1ab17cb4d59f86b582932a05cce5634da7a13072/report/%EA%B8%B0%EB%8A%A5%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20%EC%8B%9C%EB%82%98%EB%A6%AC%EC%98%A4.pdf)

<br>


# 9. 진행 과정 중 프로그램 개선 노력

**1. 기능 확장 및 학습 데이터 추가**
- PDF 학습 기능 추가: PDF 문서를 분석하고 학습할 수 있도록 기능 확장, 이후 데이터 저장 방식을 .json 포맷으로 변경하여 관리 효율성 향상.

**2. 사용자 경험(UI/UX) 개선**
- 대화 내역 확인 기능 추가: 대화 기록을 하단에서 쉽게 확인할 수 있도록 UI를 조정하여 사용자 편의성 향상
- 대화 내용 다운로드 기능 추가: 과거 대화 내역을 파일로 저장할 수 있도록 하여, 사용자 맞춤형 데이터 활용도를 강화
- 질의 히스토리 정렬 방식 개선: 최신순으로 정렬되도록 변경하여 보다 직관적인 데이터 확인이 가능하게 수정
- 출처 정보 제공 방식 개선: 출처가 없는 데이터의 경우 불필요한 출처 정보를 제공하지 않도록 조정하여, 보다 깔끔한 응답을 제공할 수 있도록 개선

**3. 기술적 개선 및 성능 최적화**
- ChromaDB 연동 및 데이터 추가: 은행별 FAQ 데이터를 ChromaDB에 저장하는 로직을 추가하여, 정보 검색 성능 향상
- 상환 방식별 대출 계산 시스템 추가: 원리금 균등 분할 상환, 원금 균등 분할 상환, 체증식 분할 상환 등 다양한 대출 상환 방식에 맞춰 계산이 가능하도록 기능을 확장
- RAG + Memory 기능 최적화: 기존 함수 로직을 삭제하고, RAG와 Memory Buffer를 활용하여 더욱 정교한 대화 흐름을 유지할 수 있도록 시스템을 개선 노력
- API 보안 강화: 기존의 API Key 하드코딩 방식을 load_dotenv()를 활용한 환경 변수 관리 방식으로 변경하여 보안성 향상
- context 검색어 재생성 기능: "이전 상품 상세설명해줘" 같은 문장을 입력했을때 이전 상품의 명확한 상품명을 context 검색어로 넣기 위해 검색어 생성용 reformer_model을 도입
    - 하나의 답변을 생성하기 위해 llm 모델을 2개 사용하는 것은 **답변생성시간 증가, 비용증가, 구조적 비효율** 등의 문제점이 있다는 지적이 나와 **해당 기능은 폐기** 하고 파인튜닝 진행

**3.1 모델 파인튜닝**
- 좀 더 금융 상담 업무에 특화되고, 일관된 응답을 위해 모델 파인 튜닝 진행
- [KoAlpaca](https://huggingface.co/beomi/KoAlpaca-Polyglot-5.8B) 모델을 사용해서 진행
```LoRA 학습 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```
- 학습 결과
![Image](https://github.com/user-attachments/assets/2bf890de-1c9a-48d8-95b7-b2f524efb6f7)
![Image](https://github.com/user-attachments/assets/34832a5c-380d-4d0d-b7d5-c33b220975c7)
  - 전체적으로는 나쁘지 않은 대답을 보이지만 구체적인 상품의 설명을 제공하지 않음
  - 상세 정보에 대한 대답을 하지 않고 범용적이고 일반화된 응답 위주로 답하는 경향을 보임


<br>


# 10. 수행결과
![결과 화면](https://github.com/user-attachments/assets/5b2bbfc1-83d4-4c9b-afe9-e1871d23c9e7)

🔗: [테스트 결과 보기](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN09-3rd-3Team/tree/main/test_result)

<br>


# 11.문제 해결 및 향후 과제
- **추론이 필요한 질문 처리 개선**: 특정 키워드가 포함된 질문은 정확한 문서를 찾아내는 반면, 맥락을 기반으로 한 추론 질문에서는 정상적인 응답이 어려운 문제가 발견됨. 이를 해결하기 위해 RAG 모델의 검색 및 응답 최적화를 지속적으로 진행예정

- **대화 기억 기능(memory) 추가 개선**: 현재 구현된 Memory 기능이 일부 상황에서 맥락을 정상적으로 유지하지 못하는 문제를 확인. 보다 정교한 Memory 관리 및 저장 방식을 개선할 계획.

- **자가 피드백 기능 재설계**: 기존 코드 수정으로 인해 피드백 기능 로직을 대폭 변경해야 하는 상황이 발생함으로 오작동이 다분함.
  - Feedback 기능 보류: 기존 코드가 대폭 수정됨에 따라 피드백 기능이 보류되었으며, 이후 Prompt 로직을 추가 수정하여 보다 효과적인 방식으로 재도입할 계획

<br>


# 12. 한 줄 회고
> 💰 서예찬:
>
> 💰 전성원: 프로젝트를 할 때 완성도를 높이고 싶어서 계획한 부분이 있는데, 생각보다 예기치 못한 변수들이 발생해서 조금 아쉬운 감이 있습니다. 그래도 협심해서 무사히 프로젝트를 마무리 할 수 있어서 좋았습니다 😊 
>
> 💰 조민훈: 파인튜닝으로 구체적인 응답을 기대했지만 한계가 있었고, 이를 보완하기 위해 RAG와 Agent 기능, 프롬프트 수정을 적용했다. 그러나 여전히 원하는 응답이 어려웠으며, 학습 데이터의 과적합된 응답 패턴이 원인으로 보여 수정이 필요하다.
>
> 💰 최재동:
><br>
