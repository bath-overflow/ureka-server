# Dialogue data 생성 ~ Evaluation

## 목적

소크라테스식 질문을 하도록 프롬프팅한 LLM이랑 우리 방법이 다른 것이 무엇이고, 정말 우리 방법이 더 좋은가?를 정량적으로 보이기 위함입니다. 

이상적으로는 각 방법을 이용해 공부했을 때의 학습효과를 측정해야 합니다. 시험을 볼 수도 있고, 배운 내용을 글로 서술해보라고 할 수도 있죠. 하지만 신뢰할만한 방법으로 이를 측정하기 위해서는 여러 사람을 데리고 대규모 연구를 진행해야 합니다. 

이러한 어려움이 있기 때문에 우리는 같은 대화 흐름에서 Ureka, 그리고 Gemini가 선생님 응답을 하라고 한 뒤, 네 가지 측면에서 답변의 질을 측정하고 비교하도록 했습니다.

## 평가 진행 방식

### 개요
1. Simulated student - teacher간 t-turn 대화 데이터를 생성합니다. 
2. N개의 t-turn 대화를 생성한 다음, 

    학생 발화가 마지막이 되도록 랜덤하게 대화를 잘라 M개의 대화 subset을 얻습니다. 
    
3. 각 대화 subset에 대해 Teacher LLM(Gemini)와 Ureka에게 각각 선생님 답을 생성하도록 합니다. 
4. 두 선생님 답을 비교해 누가 더 나은 답을 했는지 결정하게 합니다.

### Student LLM
데이터 생성 과정에서 학생이 “진짜 학생”같이 행동하는 것은 중요합니다. 이를 위해 학생 역할 LLM에게 페르소나를 정의한 다음, 그 페르소나의 학생을 흉내내게 했습니다. 

페르소나를 구성하기 위해 3가지 차원을 정의했습니다. 

1. **Goal Commitment**: Low/High dedication to learning goals
2. **Motivation**: Low/High learning motivation  
3. **Self-Efficacy**: Low/High confidence in abilities

각 차원의 높고 낮음, 그리고 4학년 중 하나의 학년을 결정함에 따라 32종의 다양한 페르소나를 가진 학생을 정의할 수 있습니다. 

학생은 언제나 올바른 답만 하지 않습니다. 오답을 포함해 다양한 종류의 답을 할 수 있도록 답의 종류를 나열해 프롬프트에 포함했습니다. 

사용한 프롬프트는 다음과 같습니다. 

```
You are a {persona}. Based on the previous conversation, generate one of the following:
1. A follow-up question to the teacher
2. A clarification question
3. A new question based on the topic discussed
4. An irrelevant question
5. A correct answer to the teacher's last question
6. A wrong answer to the teacher's last question
Simply generate your response without your selection number.

Previous conversation:
{history_str}
You: 
```

### Teacher LLM
비교군 선생님, 그리고 대화 데이터 생성 과정에서는 ‘간단하게 프롬프팅된’ Gemini를 사용했습니다. 사용한 프롬프트 형식은 다음과 같습니다. 

```
You will be given a dialogue history between a user and a teacher. 
You are the teacher in the dialogue who follows the Socratic style of teaching. 

Dialogue History: 

{dialogue_history}
[Teacher]: 
```

### 평가 방법 상세

1. Gemini(Teacher LLM)과 랜덤한 페르소나를 가지는 Student(Student LLM)간에 총 4-turn의 대화를 하도록 시켜 데이터셋을 얻습니다. (1단계)
    
    대화에서 학생은 첫 질문을 던져야 합니다. 첫 질문으로는 컴퓨터공학과 학생이 4년 과정 동안 배우는 과목에서 나올 수 있는 중간 이상 난이도의, 구체적인 질문을 하도록 했습니다. 이 질문 중 한 가지를 랜덤하게 골라 첫 질문으로 사용합니다. 
    
    사용된 질문은 다음과 같습니다. 
    
    ```
    # Algorithms & Data Structures
    "Why does quicksort have poor worst-case performance, and how do randomized 
    pivots address this issue in practice?",
    "In what scenarios is an AVL tree preferred over a Red-Black tree, and what 
    are the practical trade-offs between the two?",
    "How does the amortized analysis of a dynamic array's append operation 
    justify its average-case O(1) insertion time?",
    "Why can't Dijkstra's algorithm handle negative-weight edges, and how does 
    the Bellman-Ford algorithm overcome this limitation?",
    "When using a hash table, how does the choice between open addressing and 
    separate chaining affect performance and memory usage under high load 
    factors?",
    "Why is the master theorem inapplicable to certain recurrence relations, 
    such as T(n) = 2T(√n) + log n?",
    # Operating Systems
    "How does the copy-on-write mechanism improve the efficiency of fork() in 
    Unix-based systems?",
    "Why are page replacement algorithms like LRU and CLOCK not perfectly 
    accurate in predicting future accesses, and what are their practical 
    limitations?",
    "Explain why spinlocks can degrade system performance under high contention 
    in a multi-core environment.",
    "In what situations is a semaphore preferred over a mutex for inter-process 
    synchronization?",
    'How does the Linux Completely Fair Scheduler approximate "fairness," and 
    what are the trade-offs compared to a strict round-robin scheduler?',
    "Why does increasing the page size in virtual memory lead to higher 
    internal fragmentation, and how does it impact TLB effectiveness?",
    
    # Machine Learning
    "Why does regularization (like L2 or dropout) sometimes fail to prevent 
    overfitting in deep neural networks, despite theoretical guarantees?",
    "How do exploding and vanishing gradients arise in RNNs, and why are gating 
    mechanisms like LSTM and GRU effective?",
    "Why is the area under the ROC curve (AUC) not always a reliable metric for 
    imbalanced datasets?",
    "What are the theoretical and practical reasons behind the failure of 
    k-means to cluster non-convex data?",
    ```
    
    이렇게 만든 데이터셋을 $\{(S_1^i, T_1^i, …, S_t^i, T_t^i)\}_N$로 표현하겠습니다. 
    
2. 데이터셋에서 랜덤하게 Student 발화까지 포함하게 자른 뒤, Ureka, 그리고 Teacher LLM에게 Teacher 응답을 완성해보라고 시킵니다. 예를 들어, 랜덤하게 자른 결과 $\text{Input} = (S_1,T_1,S_2,T_2,S_3)$라는 샘플이 나왔다고 합시다. 우리는 이 샘플을 대화 히스토리로 써서 Teacher LLM, Ureka에게 응답을 생성하라고 시킵니다. (2~3 단계)
    
    Teacher LLM (Gemini) 응답을 $\hat{T_G}$, Ureka 응답을 $\hat{T_U}$이라 합시다. 
    
    이런 방식으로  $\{(\text{Input}_i, \hat{T_G}_i, \hat{T_U}_i)\}_M$ 데이터셋을 모읍니다. 
    
3. Judge LLM에게 어떤 쪽이 더 나은 선생님 응답인지 평가하게 합니다. 평가기준은 1) 의도파악, 2) 설명능력, 3) 질문능력, 4) 읽기 쉬움입니다. (4단계)
    
    ```
    1. Read the following dialogue where a student and teacher engages in a pedagogical conversation. 
    2. For the students' last round of response, rate which teacher's instruction is better from four aspects:
    • Understanding: Assess whether the teacher correctly understands the student's intention.
    • Explanation: Whether the teacher effectively solves the students' problem and provides appropriate and actionable guidance.
    • Language: Evaluate whether the teacher's instruction conforms to the demands of Socratic teaching, including that it is presented as a question and does not give a direct answer.
    • Readability: Assess whether the teacher's instruction is easy to read and not too blunt.
    
    Follow the steps to evaluate. 
    1. Provide a detailed comparison of the two responses for each of the four aspects. 
    2. Provide an overall evaluation considering all four aspects. 
    3. Give your final decision after "###"
    
    Dialogue History:
    {dialogue_history}
    
    Teacher Responses to Evaluate:
    [Teacher A]: {teacher_a_response}
    [Teacher B]: {teacher_b_response}
    
    Which is better?
    (a) Teacher A
    (b) Teacher B
    (c) Equivalent
    
    Your Evaluation: 
    ```
    
    Robust한 평가를 위해 이 단계를 3번 반복해 다수결에 따라 승-패-무승부를 결정합니다.

결과를 정리해 우리 방법의 효과성을 정량적으로 입증합니다. \
적힌 기준들은 충분히 바꿀 수 있습니다. 또한 각각의 기준(의도파악, 설명능력, ...)에 대해 누가 더 잘했느냐를 묻는 방식으로 바꿀 수도 있습니다. 

### 평가 결과

100개의 dialogue data를 모아 200개의 dialogue subset을 구성, 평가를 진행한 결과는 다음과 같습니다. 

|  | 승리 횟수 | 비율 |
| --- | --- | --- |
| Ureka | 123 | 61.5% |
| Gemini | 71 | 35.5% |
| 무승부 | 6 | 3% |

판결 신뢰도
|  | 횟수 | 비율 |
| --- | --- | --- |
| 만장일치 | 96 | 48% |
| 다수일치 (두 번 일치) | 98 | 49% |
| 서로 불일치 | 6 | 3% |

### 한계
저희가 정말 보여야하는 것은 학생의 입장에서, 문제해결력이 증대되었느냐, 비판적 사고를 더하게 되었느냐입니다. \
하지만 이를 측정하는건 매우 어렵습니다. 그래서 저희는 대신에 누가 더 '좋은 선생님이냐'를 평가합니다. \
혹시 **'학생의 입장에서' 대화를 평가하는 방법에 대해 아이디어가 있으신 분은 제안해주시면 감사하겠습니다.** 

데이터셋을 모을 때 Simple Instruction을 쓰는 이유는 우리 instruction을 쓰면 우리 방법에 치우친 대화가 생성되어 평가가 공정해지지 못할 것이기 때문입니다. \
이 평가 방법은 SocraticLM 논문에서 Overall 점수를 구할 때 사용한 방식에서 많은 영감을 얻었습니다. \
다만 논문에서는 3단계를 실제 사람 10명이 수행하지만 저희는 LLM이 하도록 했습니다. 

## 각 스크립트 역할 소개

`student_personas.py`: Student의 다양한 페르소나 정의

`data_collection_student.py`: 임의의 페르소나 가지는 Student LLM 생성, 대화의 초기질문 정의. Data collection 로직

`batch_data_collection.py`:  Student LLM과 Teacher LLM이 대화하도록 하여 dialogue data를 생성 (1단계)

`create_subsets.py`: batch_data_collection으로 얻어진 대화 데이터를 잘라 subset으로 만듦 (2단계)

`collect_endpoint_response.py`: 대화 subset에 대하여 Ureka, Teacher LLM이 각자 응답 생성하도록 하는 파이프라인 (3단계)

`evaluate_teacher_responses.py`: Ureka, Teacher LLM이 각자 생성한 응답을 비교하고 평가하는 파이프라인 (4단계)

`run_robust_evaluation.py`: `evaluate_teacher_response`가 하는 일을 3번 반복하고, 그 결과를 정리

## 사용법 예시

1. **Ureka 서버 실행**

2. **Run data collection:**

```bash
python batch_data_collection.py --turns 4 --output data_collection_logs/dialogues

python create_subsets.py --num_subsets 200 --input_dir data_collection_logs/dialogues --output_dir data_collection_logs/subsets

python collect_endpoint_response.py --input_dir data_collection_logs/subsets --output_dir data_collection_logs/responses

python run_robust_evalaution.py --input_file data_collection_logs/responses/endpoint_responses_20250623_161358.json --output_dir data_collection_logs/evaluations
```