#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation Pipeline

This script evaluates teacher responses using an LLM judge to compare
responses from different endpoints based on pedagogical criteria.

Usage:
    python evaluate_teacher_responses.py --input_file endpoint_responses_20250620_182458.json --output_dir evaluation_results
"""

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv(override=True)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "None")

# Initialize LLM for judge
judge_llm = init_chat_model(
    model="gemini-2.0-flash", model_provider="google_genai", api_key=GOOGLE_API_KEY
)


class TeacherResponseEvaluator:
    """
    Evaluates teacher responses using LLM-as-a-judge approach
    """

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load judge prompt template from file
        prompt_file = Path("evaluation_prompt.txt")
        if prompt_file.exists():
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.judge_prompt_template = f.read()
        else:
            raise FileNotFoundError(
                f"Judge prompt template file not found: {prompt_file}"
            )

    def format_dialogue_history(self, messages: List[Dict]) -> str:
        """
        Format the dialogue history for the judge prompt
        """
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("message", "")

            if role == "user":
                formatted_messages.append(f"[Student]: {content}")
            elif role == "assistant":
                formatted_messages.append(f"[Teacher]: {content}")

        return "\n".join(formatted_messages)

    def get_judge_evaluation(
        self, dialogue_history: str, teacher_a: str, teacher_b: str
    ) -> Optional[Dict]:
        """
        Get LLM judge evaluation of two teacher responses
        """
        try:
            # Create judge prompt
            judge_prompt = self.judge_prompt_template.format(
                dialogue_history=dialogue_history,
                teacher_a_response=teacher_a,
                teacher_b_response=teacher_b,
            )

            print("   🤖 Getting judge evaluation...")
            start_time = datetime.now()

            llm_response = judge_llm.invoke(judge_prompt)
            judge_response = (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            )

            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            # Parse the judge response to extract verdict
            verdict = self.parse_judge_verdict(judge_response)

            return {
                "success": True,
                "judge_response": judge_response,
                "verdict": verdict,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"   ❌ Error getting judge evaluation: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def parse_judge_verdict(self, judge_response: str) -> str:
        """
        Parse the final verdict from judge response
        """
        # Look for the final choice pattern
        response_parts = judge_response.rsplit("###", maxsplit=1)
        if len(response_parts) < 2:
            print("   ⚠️  Judge response format unexpected, cannot parse verdict")
            return "unclear"

        verdict_part = response_parts[1]

        if "(a)" in verdict_part or "teacher a" in verdict_part:
            return "teacher_a"
        elif "(b)" in verdict_part or "teacher b" in verdict_part:
            return "teacher_b"
        elif "(c)" in verdict_part or "equivalent" in verdict_part:
            return "equivalent"

        return "unclear"

    def evaluate_single_comparison(self, result: Dict) -> Dict:
        """
        Evaluate a single comparison between two teacher responses
        """
        subset_id = result.get("subset_id")
        print(f"\n📝 Evaluating subset: {subset_id}")

        # Check if we have both responses
        responses = result.get("responses", {})
        chat_response = responses.get("chat")
        simple_chat_response = responses.get("simple-chat")

        if not chat_response or not simple_chat_response:
            print("   ⚠️  Skipping - missing responses")
            return {
                "subset_id": subset_id,
                "success": False,
                "error": "Missing one or both teacher responses",
                "timestamp": datetime.now().isoformat(),
            }

        # Load the original subset to get dialogue history
        try:
            # Find the corresponding subset file
            subset_files = list(
                Path("data_collection_logs/subsets").glob(f"subset_{subset_id}.json")
            )
            if not subset_files:
                raise FileNotFoundError(f"Subset file not found for {subset_id}")

            with open(subset_files[0], "r", encoding="utf-8") as f:
                subset_data = json.load(f)

            messages = subset_data.get("messages", [])
            dialogue_history = self.format_dialogue_history(messages)

        except Exception as e:
            print(f"   ❌ Error loading subset data: {e}")
            return {
                "subset_id": subset_id,
                "success": False,
                "error": f"Failed to load subset data: {e}",
                "timestamp": datetime.now().isoformat(),
            }

        # Randomly assign which response is Teacher A vs Teacher B to avoid bias
        import random

        if random.random() < 0.5:
            teacher_a = chat_response
            teacher_b = simple_chat_response
            teacher_a_type = "chat"
            teacher_b_type = "simple-chat"
        else:
            teacher_a = simple_chat_response
            teacher_b = chat_response
            teacher_a_type = "simple-chat"
            teacher_b_type = "chat"

        # Get judge evaluation
        judge_result = self.get_judge_evaluation(dialogue_history, teacher_a, teacher_b)

        if not judge_result or not judge_result.get("success"):
            error_msg = "Judge evaluation failed"
            if judge_result:
                error_msg = judge_result.get("error", error_msg)
            return {
                "subset_id": subset_id,
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            }

        # Interpret verdict back to original endpoint types
        verdict = judge_result.get("verdict")
        if verdict == "teacher_a":
            winner = teacher_a_type
        elif verdict == "teacher_b":
            winner = teacher_b_type
        elif verdict == "equivalent":
            winner = "equivalent"
        else:
            winner = "unclear"

        evaluation_result = {
            "subset_id": subset_id,
            "success": True,
            "dialogue_history": dialogue_history,
            "teacher_responses": {
                "chat": chat_response,
                "simple-chat": simple_chat_response,
            },
            "judge_assignment": {
                "teacher_a": teacher_a_type,
                "teacher_b": teacher_b_type,
            },
            "judge_evaluation": judge_result,
            "verdict": winner,
            "timestamp": datetime.now().isoformat(),
        }

        print(f"   ✅ Evaluation complete - Winner: {winner}")
        return evaluation_result

    def load_response_data(self, input_file: str) -> List[Dict]:
        """
        Load the response collection data
        """
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            results = data.get("results", [])
            successful_results = [r for r in results if r.get("success", False)]

            print(
                f"📊 Loaded {len(successful_results)} successful response comparisons"
            )
            return successful_results

        except Exception as e:
            print(f"❌ Error loading response data: {e}")
            return []

    def save_evaluation_results(self, evaluations: List[Dict]) -> str:
        """
        Save evaluation results to JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"judge_evaluations_{timestamp}.json"
        output_path = self.output_dir / filename

        # Create summary statistics
        total_evaluations = len(evaluations)
        successful_evaluations = len(
            [e for e in evaluations if e.get("success", False)]
        )

        # Count verdicts
        verdict_counts = {"chat": 0, "simple-chat": 0, "equivalent": 0, "unclear": 0}
        for evaluation in evaluations:
            if evaluation.get("success"):
                verdict = evaluation.get("verdict", "unclear")
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        summary = {
            "evaluation_completed": datetime.now().isoformat(),
            "total_evaluations": total_evaluations,
            "successful_evaluations": successful_evaluations,
            "verdict_distribution": verdict_counts,
            "ours_win_rate": (
                verdict_counts.get("chat", 0) / successful_evaluations
                if successful_evaluations > 0
                else 0
            ),
            "evaluation_success_rate": (
                successful_evaluations / total_evaluations
                if total_evaluations > 0
                else 0
            ),
            "judge_model": "gemini-2.0-flash",
        }

        output_data = {"summary": summary, "evaluations": evaluations}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Evaluation results saved to: {output_path}")
        print("📊 Summary:")
        print(f"   Total evaluations: {total_evaluations}")
        print(f"   Successful evaluations: {successful_evaluations}")
        print("   Verdict distribution:")
        for verdict, count in verdict_counts.items():
            percentage = (
                (count / successful_evaluations * 100)
                if successful_evaluations > 0
                else 0
            )
            print(f"     {verdict}: {count} ({percentage:.1f}%)")

        return str(output_path)

    def run(self, input_file: str, max_evaluations: Optional[int] = None) -> List[Dict]:
        """
        Main execution method
        """
        print("🚀 Starting LLM-as-a-Judge evaluation")
        print("🤖 Judge model: gemini-2.0-flash")
        print(f"📂 Input file: {input_file}")
        print(f"📁 Output directory: {self.output_dir}")
        if max_evaluations:
            print(f"🔢 Max evaluations: {max_evaluations}")
        print("=" * 60)

        # Load response data
        response_results = self.load_response_data(input_file)
        if not response_results:
            print("❌ No response data loaded!")
            return []

        # Limit evaluations if specified
        if max_evaluations and max_evaluations < len(response_results):
            response_results = response_results[:max_evaluations]
            print(f"🔢 Limited to first {max_evaluations} evaluations")

        # Evaluate each comparison
        evaluations = []
        failed_count = 0

        for i, result in enumerate(response_results, 1):
            print(f"\n📊 Progress: {i}/{len(response_results)}")

            try:
                evaluation = self.evaluate_single_comparison(result)
                evaluations.append(evaluation)

                if not evaluation.get("success", False):
                    failed_count += 1

            except KeyboardInterrupt:
                print(f"\n⚠️  Evaluation interrupted by user after {i-1} evaluations")
                break
            except Exception as e:
                print(
                    f"\n💥 Unexpected error evaluating {result.get('subset_id', 'unknown')}: {e}"
                )
                traceback.print_exc()
                failed_count += 1
                continue

        # Save results
        if evaluations:
            self.save_evaluation_results(evaluations)
            print("\n🎉 Evaluation completed!")
            print(f"✅ Processed {len(evaluations)} comparisons")
            if failed_count > 0:
                print(f"⚠️  {failed_count} evaluations had issues")
        else:
            print("❌ No evaluations to save!")

        return evaluations


def main():
    """
    Main function for LLM judge evaluation
    """
    parser = argparse.ArgumentParser(
        description="Evaluate teacher responses using LLM-as-a-judge"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSON file containing response comparisons",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_collection_logs/evaluations",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--max_evaluations",
        type=int,
        help="Maximum number of evaluations to process (for testing)",
    )

    args = parser.parse_args()

    print("⚖️  LLM-as-a-Judge Teacher Response Evaluator")
    print("=" * 50)
    print("📊 Configuration:")
    print(f"   Input file: {args.input_file}")
    print("   Judge model: gemini-2.0-flash")
    print(f"   Output directory: {args.output_dir}")
    if args.max_evaluations:
        print(f"   Max evaluations: {args.max_evaluations}")
    print()

    try:
        evaluator = TeacherResponseEvaluator(output_dir=args.output_dir)
        evaluations = evaluator.run(
            input_file=args.input_file, max_evaluations=args.max_evaluations
        )

        return 0 if evaluations else 1

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Evaluation failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
