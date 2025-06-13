#!/usr/bin/env python3
"""
Batch Data Collection Script

This script runs multiple data collection sessions with different student personas
to generate diverse conversation data between students and the AI teacher.

Usage:
    python batch_data_collection.py --sessions 5 --turns 3
"""

import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from data_collection_student import DataCollectionSession, StudentAgent
from student_personas import StudentPersonaGenerator, StudentTraits


class BatchDataCollector:
    """
    Manages multiple data collection sessions with concurrent execution support
    """

    def __init__(
        self, output_dir: str = "data_collection_logs", max_concurrent: int = 3
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_concurrent = max_concurrent

        # Initialize enhanced persona generator
        self.persona_generator = StudentPersonaGenerator()

        # Semaphore to limit concurrent sessions
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def get_enhanced_persona(self) -> Tuple[StudentTraits, str]:
        """
        Get an enhanced persona with detailed psychological traits
        """
        return self.persona_generator.get_random_persona()

    async def run_batch_collection(
        self,
        num_sessions: int = 5,
        turns_per_session: int = 3,
        delay_between_sessions: int = 1,
        endpoint: str = "simple-chat",
    ) -> List[Dict]:
        """
        Run multiple data collection sessions with optional concurrency
        """
        print("🚀 Starting batch data collection")
        print(f"📊 Sessions: {num_sessions}")
        print(f"🔄 Turns per session: {turns_per_session}")
        print(f"⚡ Concurrent sessions: (max {self.max_concurrent})")
        print(f"⏱️  Delay between sessions: {delay_between_sessions}s")
        print("=" * 60)

        return await self._run_concurrent_sessions(
            num_sessions, turns_per_session, delay_between_sessions, endpoint=endpoint
        )

    async def _run_concurrent_sessions(
        self,
        num_sessions: int,
        turns_per_session: int,
        delay_between_sessions: int,
        endpoint: str = "simple-chat",
    ) -> List[Dict]:
        """
        Run sessions concurrently with rate limiting
        """
        # Create tasks for all sessions
        tasks = []
        for session_num in range(1, num_sessions + 1):
            task = self._run_single_session_with_semaphore(
                session_num,
                turns_per_session,
                delay_between_sessions,
                endpoint=endpoint,
            )
            tasks.append(task)

        # Execute all tasks concurrently
        print(
            f"🔥 Starting {num_sessions} concurrent sessions (max {self.max_concurrent} at once)..."
        )

        # Use asyncio.gather with return_exceptions=True to handle individual failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and separate successful from failed sessions
        batch_results = []
        successful_sessions = 0
        failed_sessions = 0

        for i, result in enumerate(results):
            session_num = i + 1
            if isinstance(result, Exception):
                print(f"❌ Session {session_num} failed: {result}")
                failed_sessions += 1
            else:
                print(f"✅ Session {session_num} completed successfully")
                batch_results.append(result)
                successful_sessions += 1

        return self._finalize_batch_results(
            batch_results,
            successful_sessions,
            failed_sessions,
            num_sessions,
            turns_per_session,
            endpoint,
        )

    async def _run_single_session_with_semaphore(
        self,
        session_num: int,
        turns_per_session: int,
        base_delay: int,
        endpoint: str = "simple-chat",
    ) -> Dict:
        """
        Run a single session with semaphore for rate limiting
        """
        async with self.semaphore:
            # Add some randomization to avoid thundering herd
            delay = base_delay + random.uniform(0, 1)
            await asyncio.sleep(delay)

            print(f"🎯 Starting session {session_num}")

            try:
                session_log = await self._execute_single_session(
                    session_num, turns_per_session, endpoint=endpoint
                )
                print(f"✅ Session {session_num} completed")
                return session_log
            except Exception as e:
                print(f"❌ Session {session_num} failed: {e}")
                raise

    async def _execute_single_session(
        self, session_num: int, turns_per_session: int, endpoint: str = "simple-chat"
    ) -> Dict:
        """
        Execute a single session with the enhanced persona system
        """
        # Get enhanced persona with psychological traits
        traits, persona_description = self.get_enhanced_persona()

        # Create custom student agent
        student = self._create_student_agent(persona_description)
        session = DataCollectionSession(student)

        # Create project
        await session.create_project()

        # Conduct conversation
        session_log = await session.conduct_conversation(
            turns_per_session, endpoint=endpoint
        )

        # Add batch metadata
        session_log["batch_session_number"] = session_num
        session_log["batch_id"] = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_log["student_traits"] = {
            "goal_commitment": traits.goal_commitment,
            "motivation": traits.motivation,
            "self_efficacy": traits.self_efficacy,
        }
        session_log["student_persona"] = persona_description

        # Save individual session log
        session.save_log()

        return session_log

    def _finalize_batch_results(
        self,
        batch_results: List[Dict],
        successful_sessions: int,
        failed_sessions: int,
        total_sessions: int,
        turns_per_session: int,
        endpoint: str = "simple-chat",
    ) -> List[Dict]:
        """
        Finalize batch results and save summary
        """
        # Save batch summary
        batch_summary = self._create_batch_summary(
            batch_results,
            successful_sessions,
            failed_sessions,
            total_sessions,
            turns_per_session,
            endpoint,
        )

        self._save_batch_summary(batch_summary)

        print("\n🎉 Batch collection completed!")
        print(f"✅ Successful sessions: {successful_sessions}")
        print(f"❌ Failed sessions: {failed_sessions}")
        print(f"📊 Success rate: {successful_sessions/total_sessions*100:.1f}%")

        return batch_results

    def _create_student_agent(self, persona: str) -> StudentAgent:
        student = StudentAgent(persona=persona)
        return student

    def _create_batch_summary(
        self,
        batch_results: List[Dict],
        successful: int,
        failed: int,
        total: int,
        turns_per_session: int,
        endpoint: str = "simple-chat",
    ) -> Dict:
        """
        Create a summary of the batch collection
        """
        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "total_sessions": total,
            "successful_sessions": successful,
            "failed_sessions": failed,
            "success_rate": successful / total if total > 0 else 0,
            "turns_per_session": turns_per_session,
            "total_conversations": len(batch_results),
            "total_messages": sum(r.get("total_messages", 0) for r in batch_results),
            "project_ids": [
                r.get("project_id") for r in batch_results if r.get("project_id")
            ],
            "session_files": [f"session_{r['session_id']}.json" for r in batch_results],
            "personas_used": list(r.get("student_persona") for r in batch_results),
            "average_session_duration": self._calculate_average_duration(batch_results),
            "endpoint": endpoint,
        }

    def _calculate_average_duration(self, batch_results: List[Dict]) -> str:
        """
        Calculate average session duration
        """
        durations = []
        for result in batch_results:
            if result.get("start_time") and result.get("end_time"):
                start = datetime.fromisoformat(result["start_time"])
                end = datetime.fromisoformat(result["end_time"])
                duration = (end - start).total_seconds()
                durations.append(duration)

        if durations:
            avg_duration = sum(durations) / len(durations)
            return f"{avg_duration:.1f} seconds"
        return "N/A"

    def _save_batch_summary(self, summary: Dict) -> str:
        """
        Save batch summary to file
        """
        filename = self.output_dir / f"batch_summary_{summary['batch_id']}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📋 Batch summary saved to: {filename}")
        return str(filename)


async def main():
    """
    Main function for batch data collection
    """
    parser = argparse.ArgumentParser(
        description="Run batch AI teacher-student data collection"
    )
    parser.add_argument(
        "--sessions", type=int, default=5, help="Number of sessions to run"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Number of student utterances per session. Total messages will be 2 * turns",
    )
    parser.add_argument(
        "--delay", type=int, default=1, help="Delay between sessions (seconds)"
    )
    parser.add_argument(
        "--output", type=str, default="data_collection_logs", help="Output directory"
    )
    parser.add_argument(
        "--concurrent", type=int, default=3, help="Max concurrent sessions (>=1)"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="simple-chat",
        help="API endpoint for AI teacher. 'simple-chat' for a simple-version prompted LLM, 'chat' for UREKA",
    )

    args = parser.parse_args()

    print("🤖 Batch AI Teacher-Student Data Collection")
    print("=" * 50)
    print("📊 Configuration:")
    print(f"   Sessions: {args.sessions}")
    print(f"   Turns per session: {args.turns}")
    print(f"   Delay between sessions: {args.delay}s")
    print(f"   Output directory: {args.output}")
    print(f"   Max concurrent: {args.concurrent}")
    print(f"   Endpoint: {args.endpoint}")
    print()

    collector = BatchDataCollector(args.output, max_concurrent=args.concurrent)

    try:
        batch_results = await collector.run_batch_collection(
            num_sessions=args.sessions,
            turns_per_session=args.turns,
            delay_between_sessions=args.delay,
            endpoint=args.endpoint,
        )

        print("\n📈 Final Statistics:")
        total_messages = sum(r.get("total_messages", 0) for r in batch_results)
        total_projects = len([r for r in batch_results if r.get("project_id")])

        print(f"   Total projects created: {total_projects}")
        print(f"   Total messages collected: {total_messages}")
        print(
            f"   Average messages per session: {total_messages/len(batch_results):.1f}"
        )

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Batch collection interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Batch collection failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
