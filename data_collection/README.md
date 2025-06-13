# 데이터 생성, 수집 파이프라인

Simulated student - teacher간 대화를 생성합니다. 

Simulated student는 컴공 전공의 8가지 중 한 가지 페르소나를 가지도록 초기화됩니다. 

Teacher로는 다음 중 하나를 고를 수 있습니다. 
1. Ours
2. Simplified Instruction으로 프롬프팅된 Gemini

# 목적

Gemini vs. Ours를 정량적으로 평가하기 위함입니다. 

## 정량평가 구상안

1. Gemini와 랜덤한 페르소나를 가지는 Student간에 총 8-turn의 대화를 하도록 시켜 데이터셋을 얻습니다. \
여기서 Gemini에 쓰이는 간단한 프롬프트는 아래와 같은 구조를 가집니다. 
```
Simplified Instruction
History
```
Simplified Instruction은 ureka에서 쓰는 instruction과 비슷하지만, 조금 더 간소화된 버전입니다. \
Student의 초기 질문으로는 컴공의 다양한 과목들에서 나올 수 있는 질문들로, 미리 지정되어 있습니다. (`data_collection_student.py` 상단)

이렇게 만든 데이터셋을 다음과 같이 표현하겠습니다.
$$\{(S_1^i, T_1^i, S_2^i, T_2^i, ..., S_7^i, T_7^i)\}$$
여기서 $S_j^i$는 대화 i에서 학생의 j번째 발화를 말합니다. T는 선생님 발화를 말합니다. 

2. 데이터셋에서 랜덤하게 Student 발화까지 포함하게 자른 뒤, Ours, 그리고 Gemini에게 Teacher 응답을 완성해보라고 시킵니다. 예를 들어, 랜덤하게 자른 결과 $(S_1, T_1, S_2, T_2, S_3)$라는 샘플이 나왔다고 합시다. 우리는 이 샘플을 대화 히스토리로 써서 Gemini, ureka에게 응답을 생성하라고 시킵니다. \
Gemini는 이렇게 프롬프팅을 합니다. 
```
Simplified Instruction
History
```
Ureka는 이런 식으로 프롬프팅을 하죠. 
```
Instruction
Examples * N
  History
  Answer
  Reference
History
Answer
Reference
```
Gemini의 응답을 $\hat{T_{G}}$, 우리 응답을 $\hat{T_{U}}$라고 합시다. \
이런 방식으로 $\{(Input^i, \hat{T_{G}}^i, \hat{T_{U}}^i)\}_i$ 형태의 데이터셋을 모읍니다. \
여기서 $Input^i$는 랜덤하게 잘라서 얻은 대화 히스토리를 말합니다. 

3. LLM에게 어떤 쪽이 더 나은 선생님 응답인지 평가하게 합니다. 
평가기준은 1) 의도파악, 2) 설명능력, 3) 질문능력, 4) 읽기 쉬움입니다. 
```
1. Read the following dialogue where a student and teacher engages in a pedagogical conversation. 
2. For the students' last round of response, rate which teacher's instruction is better from four aspects:
• Understanding: Assess whether the teacher correctly understands the student's intention.
• Explanation: Whether the teacher effectively solves the students' problem and provides
appropriate and actionable guidance.
• Language: Evaluate whether the teacher's instruction conforms to the demands of Socratic
teaching, including that it is presented as a question and does not give a direct answer.
• Readability: Assess whether the teacher's instruction is easy to read and not too blunt.

{dialogue history}

[Teacher A]: {Gemini 또는 ours 응답}
[Teacher B]: {나머지 하나의 응답}

Which is better?
(a) Teacher A
(b) Teacher B
(c) Equivalent
```
결과를 정리해 우리 방법의 효과성을 정량적으로 입증합니다. \
적힌 기준들은 충분히 바꿀 수 있습니다. 또한 각각의 기준(의도파악, 설명능력, ...)에 대해 누가 더 잘했느냐를 묻는 방식으로 바꿀 수도 있습니다. 

### 한계
저희가 정말 보여야하는 것은 학생의 입장에서, 문제해결력이 증대되었느냐, 비판적 사고를 더하게 되었느냐입니다. \
하지만 이를 측정하는건 매우 어렵습니다. 그래서 저희는 대신에 누가 더 '좋은 선생님이냐'를 평가합니다. \
혹시 **'학생의 입장에서' 대화를 평가하는 방법에 대해 아이디어가 있으신 분은 제안해주시면 감사하겠습니다.** 

데이터셋을 모을 때 Simple Instruction을 쓰는 이유는 우리 instruction을 쓰면 우리 방법에 치우친 대화가 생성되어 평가가 공정해지지 못할 것이기 때문입니다. \
이 평가 방법은 SocraticLM 논문에서 Overall 점수를 구할 때 사용한 방식과 99% 같습니다. \
다만 논문에서는 3단계를 실제 사람 10명이 수행하지만 저희는 LLM이 하도록 했습니다. 

## 구현사항

선생님과 학생간 대화를 대량으로 수집하는 파이프라인이 구현되어 있습니다. 동작은 간단합니다. 
1. 학생 페르소나를 샘플링합니다. 
2. 새 프로젝트를 만듭니다. 
3. 초기질문 하나를 샘플링합니다. 
4. 초기질문을 선생님에게 던지는 것을 시작으로 선생님, 학생이 대화합니다. 

구현되어 있지 않은 것은 다음과 같습니다. 
1. 프로젝트에 문서 업로드 <- 프로젝트마다 문서를 업로드하는건 비효율적입니다. 대신에 하나의 프로젝트를 만들고, 그 프로젝트에 문서를 대량으로 업로드한 다음, langgraph_service.py의 tool node에서 collection_id를 이 프로젝트 ID로 고정시키는 방식으로 하는게 좋을 것 같습니다. 
2. 구상안의 2단계
3. 구상안의 3단계

특징
- **Diverse Student Personas**: University CS students with varying goal commitment, motivation, and self-efficacy
- **CS-Focused Topics**: Conversations cover algorithms, programming, data structures, AI/ML, and more
- **Concurrent Collection**: Parallel sessions for faster data generation
- **Comprehensive Logging**: JSON logs with conversation history and metadata

## Quick Start

1. **Ureka 서버 실행**

2. **Run data collection:**
   ```bash
   # Simple collection (5 sessions, 3 turns each)
   python data_collection_student.py
   
   # Batch collection with concurrency
   python batch_data_collection.py --sessions 10 --concurrent 3
   
   # Using the runner script
   ./run_data_collection.sh 20 3 5  # 20 sessions, 3 turns, 5 concurrent
   ```

## Usage Options

### Individual Session
```bash
python data_collection_student.py
```

### Batch Collection
```bash
python batch_data_collection.py [OPTIONS]

Options:
  --sessions N      Number of sessions (default: 5)
  --turns N         Student utterances per session (default: 3)
  --concurrent N    Max concurrent sessions (default: 3)
  --delay N         Delay between sessions in seconds (default: 1)
  --output DIR      Output directory (default: data_collection_logs)
  --endpoint END    'simple-chat' for a simple prompted LLM, 'chat' for UREKA
```

### Examples
```bash
# Quick test with 3 sessions
python batch_data_collection.py --sessions 3 --turns 2

# High-throughput collection
python batch_data_collection.py --sessions 50 --concurrent 5 --turns 4
```

## Student Personas

The system generates diverse CS student personas with:

- **Goal Commitment**: Low/High dedication to learning goals
- **Motivation**: Low/High learning motivation  
- **Self-Efficacy**: Low/High confidence in abilities

Topics include: Algorithms, Data Structures, Programming Languages, AI/ML, Software Engineering, Computer Systems, Web Development, and more.

## Output

Data is saved to `data_collection_logs/`:
- `session_*.json`: Individual conversation logs
- `batch_summary_*.json`: Batch statistics and metadata

### Log Structure
```json
{
  "session_id": "session_20231201_143022",
  "project_id": "abc123",
  "student_persona": "motivated CS junior studying algorithms...",
  "student_traits": {
    "goal_commitment": "High",
    "motivation": "Intrinsic", 
    "self_efficacy": "Medium"
  },
  "messages": [...],
  "total_messages": 6,
  "start_time": "2023-12-01T14:30:22",
  "end_time": "2023-12-01T14:31:45"
}
```