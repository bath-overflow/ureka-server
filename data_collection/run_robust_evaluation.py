#!/usr/bin/env python3
"""
Robust LLM-as-a-Judge Evaluation Pipeline

This script runs the teacher response evaluation 3 times and aggregates the results
to provide a more robust evaluation by requiring majority agreement for wins.

Usage:
    python run_robust_evaluation.py --input_file endpoint_responses_20250620_182458.json --output_dir evaluation_results
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class RobustEvaluator:
    """
    Runs the teacher response evaluation multiple times and aggregates results
    """

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_runs = 3  # Fixed to 3 runs
        self.evaluation_files = []

    def extract_output_filename(self, stdout: str) -> Optional[str]:
        """
        Extract the output filename from the stdout of the evaluation script
        """
        # Look for the line that contains "💾 Evaluation results saved to:"
        lines = stdout.split("\n")
        for line in lines:
            if "💾 Evaluation results saved to:" in line:
                # Extract the filename after the colon and space
                parts = line.split("💾 Evaluation results saved to:", 1)
                if len(parts) == 2:
                    filename = parts[1].strip()
                    return filename
        return None

    def run_single_evaluation(
        self, input_file: str, run_number: int, max_evaluations: Optional[int] = None
    ) -> str:
        """
        Run a single evaluation using the existing evaluation script
        """
        print(f"🔄 Starting evaluation run {run_number}/{3}")

        # Construct the command to run the evaluation script
        cmd = [
            sys.executable,
            "evaluate_teacher_responses.py",
            "--input_file",
            input_file,
            "--output_dir",
            str(self.output_dir),
            "--max_workers",
            "4",
        ]

        if max_evaluations:
            cmd.extend(["--max_evaluations", str(max_evaluations)])

        try:
            # Run the evaluation script
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Evaluation run {run_number} completed successfully")

            # Extract the output filename from stdout
            output_file = self.extract_output_filename(result.stdout)
            if output_file and Path(output_file).exists():
                return output_file
            else:
                # Fallback to finding the most recent file if parsing fails
                print(
                    "   ⚠️  Could not parse output filename from stdout, using fallback method"
                )
                eval_files = list(self.output_dir.glob("judge_evaluations_*.json"))
                if eval_files:
                    latest_file = max(eval_files, key=lambda f: f.stat().st_mtime)
                    return str(latest_file)
                else:
                    raise FileNotFoundError("No evaluation output file found")

        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation run {run_number} failed:")
            print(f"   Error: {e}")
            print(f"   Stdout: {e.stdout}")
            print(f"   Stderr: {e.stderr}")
            raise
        except Exception as e:
            print(f"❌ Evaluation run {run_number} failed with unexpected error: {e}")
            raise

    def load_evaluation_results(self, eval_file: str) -> Dict[str, str]:
        """
        Load evaluation results from a JSON file and extract subset_id -> verdict mapping
        """
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            results = {}
            evaluations = data.get("evaluations", [])

            for evaluation in evaluations:
                if evaluation.get("success", False):
                    subset_id = evaluation.get("subset_id")
                    verdict = evaluation.get("verdict")
                    if subset_id and verdict:
                        results[subset_id] = verdict

            print(f"   📊 Loaded {len(results)} successful evaluations")
            return results

        except Exception as e:
            print(f"❌ Error loading evaluation results from {eval_file}: {e}")
            return {}

    def aggregate_results(self, all_results: List[Dict[str, str]]) -> Dict[str, Dict]:
        """
        Aggregate results from multiple evaluation runs
        """
        print(f"🔄 Aggregating results from {len(all_results)} evaluation runs")

        # Collect all subset_ids that appear in any evaluation
        all_subset_ids = set()
        for results in all_results:
            all_subset_ids.update(results.keys())

        aggregated = {}

        for subset_id in all_subset_ids:
            # Collect verdicts for this subset_id from all runs
            verdicts = []
            for results in all_results:
                if subset_id in results:
                    verdicts.append(results[subset_id])

            # Count occurrences of each verdict
            verdict_counts = Counter(verdicts)

            # Determine the final verdict based on majority rule
            final_verdict = self.determine_final_verdict(verdicts, verdict_counts)

            aggregated[subset_id] = {
                "verdicts": verdicts,
                "verdict_counts": dict(verdict_counts),
                "final_verdict": final_verdict,
                "confidence": self.calculate_confidence(verdicts, verdict_counts),
            }

        return aggregated

    def determine_final_verdict(
        self, verdicts: List[str], verdict_counts: Counter
    ) -> str:
        """
        Determine the final verdict based on majority rule for exactly 3 runs
        """
        if len(verdicts) == 0:
            return "no_data"

        if len(verdicts) != 3:
            raise ValueError(f"Expected exactly 3 verdicts, got {len(verdicts)}")

        # For 3 verdicts, check for majority (2 or more)
        most_common = verdict_counts.most_common()
        max_count = most_common[0][1]

        if max_count >= 2:
            return most_common[0][0]
        else:
            # All three verdicts are different - this is a tie
            return "tie"

    def calculate_confidence(self, verdicts: List[str], verdict_counts: Counter) -> str:
        """
        Calculate confidence level based on agreement between runs
        """
        if len(verdicts) == 0:
            return "no_data"

        max_count = verdict_counts.most_common(1)[0][1]
        total_count = len(verdicts)

        if max_count == total_count:
            return "unanimous"
        elif max_count >= 2:
            return "majority"
        else:
            return "no_consensus"

    def save_aggregated_results(
        self, aggregated_results: Dict[str, Dict], evaluation_files: List[str]
    ) -> str:
        """
        Save the aggregated results to a JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robust_evaluation_{timestamp}.json"
        output_path = self.output_dir / filename

        # Create summary statistics
        total_subsets = len(aggregated_results)
        confidence_counts = Counter()
        final_verdict_counts = Counter()

        for subset_data in aggregated_results.values():
            confidence_counts[subset_data["confidence"]] += 1
            final_verdict_counts[subset_data["final_verdict"]] += 1

        # Calculate win rates
        successful_evaluations = sum(
            1
            for data in aggregated_results.values()
            if data["final_verdict"] not in ["no_data", "tie"]
        )

        summary = {
            "evaluation_completed": datetime.now().isoformat(),
            "num_evaluation_runs": 3,
            "evaluation_files": evaluation_files,
            "total_subsets": total_subsets,
            "successful_evaluations": successful_evaluations,
            "final_verdict_distribution": dict(final_verdict_counts),
            "confidence_distribution": dict(confidence_counts),
            "ours_win_rate": (
                final_verdict_counts.get("chat", 0) / successful_evaluations
                if successful_evaluations > 0
                else 0
            ),
            "baseline_win_rate": (
                final_verdict_counts.get("simple-chat", 0) / successful_evaluations
                if successful_evaluations > 0
                else 0
            ),
            "tie_rate": (
                final_verdict_counts.get("tie", 0) / total_subsets
                if total_subsets > 0
                else 0
            ),
            "unanimous_agreement_rate": (
                confidence_counts.get("unanimous", 0) / total_subsets
                if total_subsets > 0
                else 0
            ),
        }

        output_data = {"summary": summary, "aggregated_results": aggregated_results}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Robust evaluation results saved to: {output_path}")
        print("📊 Summary:")
        print(f"   Total subsets: {total_subsets}")
        print(f"   Successful evaluations: {successful_evaluations}")
        print("   Final verdict distribution:")
        for verdict, count in final_verdict_counts.items():
            percentage = (count / total_subsets * 100) if total_subsets > 0 else 0
            print(f"     {verdict}: {count} ({percentage:.1f}%)")
        print("   Confidence distribution:")
        for confidence, count in confidence_counts.items():
            percentage = (count / total_subsets * 100) if total_subsets > 0 else 0
            print(f"     {confidence}: {count} ({percentage:.1f}%)")

        return str(output_path)

    def run(
        self,
        input_file: Optional[str] = None,
        max_evaluations: Optional[int] = None,
        existing_files: Optional[List[str]] = None,
    ) -> str:
        """
        Main execution method - runs multiple evaluations and aggregates results
        Can either run new evaluations or use existing evaluation files
        """
        if existing_files:
            if len(existing_files) != 3:
                raise ValueError(
                    f"Expected exactly 3 existing files, got {len(existing_files)}"
                )

            print("🚀 Starting Robust Evaluation from Existing Files")
            print(f"📁 Output directory: {self.output_dir}")
            print("📊 Using 3 existing evaluation files:")
            for i, file in enumerate(existing_files, 1):
                print(f"   {i}. {file}")
            print("=" * 60)

            evaluation_files = existing_files
            all_results = []

            # Load results from existing files
            for file_path in existing_files:
                try:
                    if not Path(file_path).exists():
                        raise FileNotFoundError(f"File not found: {file_path}")

                    results = self.load_evaluation_results(file_path)
                    if not results:
                        raise ValueError(f"No results loaded from: {file_path}")

                    all_results.append(results)

                except Exception as e:
                    print(f"❌ Failed to load results from {file_path}: {e}")
                    raise

        else:
            print("🚀 Starting Robust LLM-as-a-Judge Evaluation")
            print("🔢 Number of evaluation runs: 3")
            print(f"📂 Input file: {input_file}")
            print(f"📁 Output directory: {self.output_dir}")
            if max_evaluations:
                print(f"🔢 Max evaluations per run: {max_evaluations}")
            print("=" * 60)

            if not input_file:
                raise ValueError("input_file is required when not using existing files")

            evaluation_files = []
            all_results = []

            # Run exactly 3 evaluations
            for run_num in range(1, 4):
                try:
                    eval_file = self.run_single_evaluation(
                        input_file, run_num, max_evaluations
                    )
                    evaluation_files.append(eval_file)

                    # Load and store results
                    results = self.load_evaluation_results(eval_file)
                    all_results.append(results)

                except Exception as e:
                    print(f"❌ Failed to complete evaluation run {run_num}: {e}")
                    raise  # Don't continue if any run fails

        if not all_results:
            print("❌ No successful evaluation runs completed!")
            return ""

        print(f"\n✅ Completed loading {len(all_results)} evaluation runs")

        # Aggregate results
        aggregated_results = self.aggregate_results(all_results)

        # Save aggregated results
        output_file = self.save_aggregated_results(aggregated_results, evaluation_files)

        print("\n🎉 Robust evaluation completed!")
        print(f"📁 Results saved to: {output_file}")

        return output_file


def main():
    """
    Main function for robust evaluation
    """
    parser = argparse.ArgumentParser(
        description="Run robust teacher response evaluation with multiple runs"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input JSON file containing response comparisons (required for new evaluations)",
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
        help="Maximum number of evaluations per run (for testing)",
    )
    parser.add_argument(
        "--existing_files",
        type=str,
        nargs=3,
        help="Exactly 3 existing evaluation JSON files to aggregate (skips running new evaluations)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.existing_files and not args.input_file:
        parser.error("Either --input_file or --existing_files must be provided")

    if args.existing_files and args.input_file:
        print(
            "⚠️  Both --input_file and --existing_files provided. Using existing files and ignoring input_file."
        )

    print("🔁 Robust LLM-as-a-Judge Teacher Response Evaluator")
    print("=" * 55)
    print("📊 Configuration:")
    if args.existing_files:
        print("   Mode: Using existing evaluation files")
        print(f"   Files: {len(args.existing_files)} files")
    else:
        print("   Mode: Running new evaluations")
        print(f"   Input file: {args.input_file}")
        print("   Number of runs: 3")
    print(f"   Output directory: {args.output_dir}")
    if args.max_evaluations:
        print(f"   Max evaluations per run: {args.max_evaluations}")
    print()

    try:
        evaluator = RobustEvaluator(output_dir=args.output_dir)
        output_file = evaluator.run(
            input_file=args.input_file,
            max_evaluations=args.max_evaluations,
            existing_files=args.existing_files,
        )

        return 0 if output_file else 1

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Robust evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
