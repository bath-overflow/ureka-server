#!/usr/bin/env python3
"""
Data Collection Script with Student Agent

This script creates a Student agent that communicates with the Teacher agent through the chat API.
It performs the following steps:
1. Creates a new project (which generates a project_id/chat_id)
2. Simulates a student asking questions for 6 total turns (3 student questions + 3 teacher responses)
3. Logs all the chat history

Usage:
    python data_collection_student.py
"""
import asyncio
import json
import os
import random
import uuid
from datetime import datetime
from typing import Dict, List

import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

load_dotenv(override=True)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "None")

llm = init_chat_model(
    model="gemini-2.0-flash", model_provider="google_genai", api_key=GOOGLE_API_KEY
)

# Configuration
SERVER_BASE_URL = "http://localhost:8000"
LOG_DIR = "data_collection_logs"

os.makedirs(LOG_DIR, exist_ok=True)


class ProjectCreate(BaseModel):
    title: str
    description: str


class ChatMessage(BaseModel):
    role: str
    message: str


initial_question_prompt = """
You are a {persona}.
"Generate a question that you might be dealing with while studying in college. Simply return a question.
Some examples of questions you might ask a teacher are:
1. What's the difference between supervised and unsupervised learning?
2. Why is it that hash tables can achieve O(1) average lookup time, but not in the worst case?
3. Why are B+ trees favored over binary search trees for indexing on disk?
"""

student_response_prompt = """
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
You: """

initial_questions = [
    # Algorithms & Data Structures
    "Why does quicksort have poor worst-case performance, and how do randomized pivots address this issue in practice?",
    "In what scenarios is an AVL tree preferred over a Red-Black tree, and what are the practical trade-offs between the two?",
    "How does the amortized analysis of a dynamic array's append operation justify its average-case O(1) insertion time?",
    "Why can't Dijkstra's algorithm handle negative-weight edges, and how does the Bellman-Ford algorithm overcome this limitation?",
    "When using a hash table, how does the choice between open addressing and separate chaining affect performance and memory usage under high load factors?",
    "Why is the master theorem inapplicable to certain recurrence relations, such as T(n) = 2T(√n) + log n?",
    # Operating Systems
    "How does the copy-on-write mechanism improve the efficiency of fork() in Unix-based systems?",
    "Why are page replacement algorithms like LRU and CLOCK not perfectly accurate in predicting future accesses, and what are their practical limitations?",
    "Explain why spinlocks can degrade system performance under high contention in a multi-core environment.",
    "In what situations is a semaphore preferred over a mutex for inter-process synchronization?",
    'How does the Linux Completely Fair Scheduler approximate "fairness," and what are the trade-offs compared to a strict round-robin scheduler?',
    "Why does increasing the page size in virtual memory lead to higher internal fragmentation, and how does it impact TLB effectiveness?",
    # # Computer Networks
    # "Why is TCP slow start necessary, and how does it interact with congestion avoidance and control mechanisms like AIMD?",
    # "Explain the difference between NAT and PAT. How does each affect the ability to host public-facing services?",
    # "In what scenarios would you use a UDP-based protocol instead of TCP, despite the lack of reliability guarantees?",
    # "Why is BGP susceptible to route hijacking, and what mitigations (if any) exist at the protocol level?",
    # "What are the security implications of ARP spoofing, and why is it difficult to prevent on a local network?",
    # # Databases
    # "Why do index-organized tables improve performance for certain types of queries in RDBMSs?",
    # "How does multi-version concurrency control (MVCC) allow for non-blocking reads, and what are the drawbacks in terms of storage and garbage collection?",
    # "In what scenarios does denormalization offer significant performance gains, and what are the maintenance drawbacks?",
    # "How do isolation levels like \"repeatable read\" and \"serializable\" differ in terms of allowed anomalies, and what practical trade-offs are there in OLTP systems?",
    # "What is a phantom read, and how do locking and timestamp-based protocols handle it differently?",
    # Machine Learning
    "Why does regularization (like L2 or dropout) sometimes fail to prevent overfitting in deep neural networks, despite theoretical guarantees?",
    "How do exploding and vanishing gradients arise in RNNs, and why are gating mechanisms like LSTM and GRU effective?",
    "Why is the area under the ROC curve (AUC) not always a reliable metric for imbalanced datasets?",
    "What are the theoretical and practical reasons behind the failure of k-means to cluster non-convex data?",
    # # Programming basics
    # "Why can’t I use == to compare two lists or arrays in C?",
    # "Why does indentation matter in Python, and how do I fix “IndentationError”",
    # "Why does assigning one array to another not copy the values in C?",
    # "Why does assigning one array to another not copy the values in C?",
    # "Why does my array seem to lose its contents after a function call?",
    # "How do I correctly read a line of input from the user in C?",
    # # Python programming
    # "I'm trying to solve an algorithm problem in Python. Given a list of integers, return their sum.\nInput: A list of integers, e.g. [1, 2, 3, 4, 5]\nOutput: The sum, e.g. 15\nMy current code is as follows:\n```python\ndef sum_list(nums):\n    for num in nums:\n        total = 0\n        total += num\n    return total\n\nprint(sum_list([1, 2, 3, 4, 5]))\n```\nMy code only returns 5 instead of 15. I don’t understand why. What am I doing wrong?",
    # "I'm learning C and want to initialize an array with numbers from 1 to 10 and print them.\nInput: None\nOutput: 1 2 3 4 5 6 7 8 9 10 (each number separated by a space)\nMy current code is:\n```c\n#include <stdio.h>\n\nint main() {\n    int arr[10];\n    int i;\n    for(i = 1; i <= 10; i++) {\n        arr[i] = i;\n    }\n    for(i = 1; i <= 10; i++) {\n        printf(\"%d \", arr[i]);\n    }\n    return 0;\n}\n```\nMy code prints weird numbers or sometimes crashes. I don’t understand why it’s not printing 1 to 10.",
    # "I'm trying to write a Python function that returns the largest number in a list.\nInput: [3, 6, 2, 8, 4]\nOutput: 8\nMy code is:\n```python\ndef max_num(nums):\n    max = 0\n    for n in nums:\n        if n > max:\n            max = n\n    return max\n\nprint(max_num([3, 6, 2, 8, 4]))\n```\nMy code works for some cases but fails for lists with all negative numbers, like [-3, -6, -2, -8, -4]—it returns 0. Why?",
    # "In C, I'm trying to write a function that increments an integer variable.\nInput: An integer variable x = 5\nOutput: The value of x should be 6 after calling my function\nMy code is:\n```c\n#include <stdio.h>\n\nvoid increment(int x) {\n    x = x + 1;\n}\n\nint main() {\n    int x = 5;\n    increment(x);\n    printf(\"%d\\n\", x);\n    return 0;\n}\n```\nMy function runs but x is still 5 after calling increment. Shouldn’t it be 6?",
    # "I'm solving a Python problem where I need to return the element at a specific index k from a list.\nInput: [10, 20, 30], k = 3\nOutput: Should be an error, since the largest index is 2\nMy code is:\n```python\ndef get_element(lst, k):\n    return lst[k]\n\nprint(get_element([10, 20, 30], 3))\n```\nI keep getting an IndexError. I thought the third element would be at index 3.",
    # "In C, I need to read a full line of input from the user and print it.\nInput: (User types) hello world\nOutput: The program prints hello world\nMy code is:\n```c\n#include <stdio.h>\n\nint main() {\n    char str[100];\n    scanf(\"%s\", str);\n    printf(\"%s\\n\", str);\n    return 0;\n}\n```\nMy code only prints 'hello' when I input 'hello world'. Why doesn’t it read the whole line?",
    # "I'm trying to write a Python function that doubles every element in a list in-place.\nInput: [1, 2, 3]\nOutput: [2, 4, 6]\nMy code is:\n```python\ndef double(nums):\n    for n in nums:\n        n *= 2\n\nlst = [1, 2, 3]\ndouble(lst)\nprint(lst)\n```\nAfter running my function, lst is still [1, 2, 3]. Why didn’t it change?",
    # "I'm writing a C program to print the string 'Hello', but my program crashes with a segmentation fault.\nInput: None\nOutput: Prints 'Hello'\nMy code is:\n```c\n#include <stdio.h>\n\nint main() {\n    char *str;\n    str[0] = 'H';\n    str[1] = 'e';\n    str[2] = 'l';\n    str[3] = 'l';\n    str[4] = 'o';\n    str[5] = '\\0';\n    printf(\"%s\\n\", str);\n    return 0;\n}\n```\nMy program crashes with a segmentation fault. Why?",
    # "In Python, I want to read two numbers from user input and print their sum.\nInput: 3 5\nOutput: 8\nMy code is:\n```python\na, b = input().split()\nprint(a + b)\n```\nWhen I input 3 5, it prints 35 instead of 8. What did I do wrong?",
    # "I'm trying to sum all numbers from 1 to n using a while loop in C.\nInput: n = 5\nOutput: 15 (1+2+3+4+5)\nMy code is:\n```c\n#include <stdio.h>\n\nint main() {\n    int n = 5, sum = 0;\n    while(n > 0) {\n        sum += n;\n    }\n    printf(\"%d\\n\", sum);\n    return 0;\n}\n```\nMy program never stops running. Why is it stuck in an infinite loop?"
]


class StudentAgent:
    """
    Student agent that simulates a student asking questions to a teacher
    """

    def __init__(self, name: str = "Student", persona: str = None):
        self.name = name
        self.persona = (
            persona
            or "curious undergraduate student learning about AI and machine learning"
        )

    def generate_utterance(
        self, turn_number: int, conversation_history: List[Dict] = None
    ) -> str:
        """
        Generate a question based on the turn number and conversation history
        """
        if turn_number == 1:
            question = random.choice(initial_questions)
            return question
        else:
            if conversation_history:
                # Unpack the conversation history into a string of format "You: <message> Teacher: <message>"
                history_str = "\n".join(
                    (
                        f"You: {msg['message']}"
                        if msg["role"] == "user"
                        else f"Teacher: {msg['message']}"
                    )
                    for msg in conversation_history
                )

                prompt = student_response_prompt.format(
                    persona=self.persona, history_str=history_str
                )

                return llm.invoke(prompt).content


class DataCollectionSession:
    """
    Manages a single data collection session between student and teacher
    """

    def __init__(self, student_agent: StudentAgent):
        self.student_agent = student_agent
        self.project_id = None
        self.chat_history = []
        self.session_log = {
            "session_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')} {uuid.uuid4().hex[:6]}",
            "start_time": datetime.now().isoformat(),
            "student_persona": student_agent.persona,
            "messages": [],
            "metadata": {},
        }

    async def create_project(self) -> str:
        """
        Create a new project and return the project_id
        """
        project_data = ProjectCreate(
            title=f"Data Collection Session - {self.session_log['session_id']}",
            description=f"Automated data collection session with student persona: {self.student_agent.persona}",
        )

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{SERVER_BASE_URL}/projects/",
                    json=project_data.model_dump(),
                    timeout=30.0,
                )
                response.raise_for_status()

                project_response = response.json()
                self.project_id = project_response["id"]

                print(f"✅ Created project: {self.project_id}")
                self.session_log["project_id"] = self.project_id

                return self.project_id

            except httpx.HTTPError as e:
                print(f"❌ Failed to create project: {e}")
                raise

    async def send_message_to_teacher(
        self, message: str, endpoint="simple-chat"
    ) -> str:
        """
        Send a message to the teacher and get the response
        """
        if not self.project_id:
            raise ValueError("Project not created yet")

        message_data = ChatMessage(role="user", message=message)

        async with httpx.AsyncClient() as client:
            try:
                # print(f"📤 Student: {message}")

                response = await client.post(
                    f"{SERVER_BASE_URL}/{endpoint}/{self.project_id}",
                    json=message_data.model_dump(),
                    timeout=60.0,  # Allow longer timeout for AI response
                )
                response.raise_for_status()

                response_data = response.json()
                teacher_response = response_data.get("ai_response", "")

                # print(f"📥 Teacher: {teacher_response[:100]}{'...' if len(teacher_response) > 100 else ''}")

                # Log the exchange
                self.session_log["messages"].extend(
                    [
                        {
                            "role": "user",
                            "message": message,
                            "timestamp": datetime.now().isoformat(),
                        },
                        {
                            "role": "assistant",
                            "message": teacher_response,
                            "timestamp": datetime.now().isoformat(),
                        },
                    ]
                )

                return teacher_response

            except httpx.HTTPError as e:
                print(f"❌ Failed to send message: {e}")
                raise

    async def get_chat_history(self) -> List[Dict]:
        """
        Retrieve the complete chat history
        """
        if not self.project_id:
            return []

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{SERVER_BASE_URL}/chat/{self.project_id}/history", timeout=30.0
                )
                response.raise_for_status()

                history_data = response.json()
                return history_data.get("messages", [])

            except httpx.HTTPError as e:
                print(f"❌ Failed to get chat history: {e}")
                return []

    async def conduct_conversation(
        self, num_turns: int = 3, endpoint="simple-chat"
    ) -> Dict:
        """
        Conduct a conversation between student and teacher for the specified number of turns
        """
        print(f"🎓 Starting conversation with {num_turns} student questions...")

        for turn in range(1, num_turns + 1):
            # print(f"\n--- Turn {turn}/{num_turns} ---")

            # Get conversation history for context
            current_history = await self.get_chat_history()

            # Generate student question
            utterance = self.student_agent.generate_utterance(turn, current_history)

            # Send question and get teacher response
            _ = await self.send_message_to_teacher(utterance, endpoint=endpoint)

            # Add a small delay to be respectful to the server
            await asyncio.sleep(0.5)

        # Get final chat history
        final_history = await self.get_chat_history()
        self.session_log["end_time"] = datetime.now().isoformat()
        self.session_log["total_turns"] = num_turns
        self.session_log["total_messages"] = len(final_history)

        print(f"\n✅ Conversation completed! Total messages: {len(final_history)}")

        return self.session_log

    def save_log(self) -> str:
        """
        Save the session log to a file
        """
        filename = f"{LOG_DIR}/session_{self.session_log['session_id']}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.session_log, f, indent=2, ensure_ascii=False)

        print(f"💾 Session log saved to: {filename}")
        return filename


async def run_data_collection_session(student_persona: str = None, num_turns: int = 3):
    """
    Run a single data collection session
    """
    # Create student agent
    student = StudentAgent(persona=student_persona)

    # Create session
    session = DataCollectionSession(student)

    try:
        # Create project
        await session.create_project()

        # Conduct conversation
        session_log = await session.conduct_conversation(
            num_turns, endpoint="simple-chat"
        )

        # Save results
        log_file = session.save_log()

        print("\n🎉 Data collection session completed successfully!")
        print(f"📋 Project ID: {session.project_id}")
        print(f"📁 Log file: {log_file}")
        print(f"💬 Total messages: {session_log['total_messages']}")

        return session_log

    except Exception as e:
        print(f"❌ Session failed: {e}")
        # Still try to save partial log
        if session.session_log["messages"]:
            session.save_log()
        raise


async def main():
    """
    Main function to run data collection
    """
    print("🚀 Starting AI Teacher-Student Data Collection")
    print("=" * 50)

    # Different student personas for variety
    student_personas = [
        "curious undergraduate student learning about AI and machine learning",
        "graduate student researching deep learning applications",
        "working professional transitioning into data science",
        "high school student interested in computer science and AI",
        "experienced programmer exploring machine learning for the first time",
    ]

    # Run a single session (you can modify this to run multiple sessions)
    persona = random.choice(student_personas)
    print(f"🎭 Student persona: {persona}")

    try:
        session_log = await run_data_collection_session(
            student_persona=persona,
            num_turns=3,  # 3 student questions + 3 teacher responses = 6 total messages
        )

        print("\n📊 Session Summary:")
        print(f"   Start time: {session_log['start_time']}")
        print(f"   End time: {session_log['end_time']}")
        print(f"   Total turns: {session_log['total_turns']}")
        print(f"   Total messages: {session_log['total_messages']}")

    except Exception as e:
        print(f"💥 Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # Run the data collection
    exit_code = asyncio.run(main())
    exit(exit_code)
