#!/usr/bin/env python3
"""
Chat Endpoint Response Collector

This script collects responses from both /chat and /simple-chat endpoints
for each chat history subset, allowing comparison between different AI models.

Usage:
    python collect_endpoint_responses.py --base_url http://localhost:8000 --input_dir data_collection_logs/subsets --output_dir response_collection_logs

    # To reuse existing simple-chat responses and only fetch new /chat responses:
    python collect_endpoint_responses.py --base_url http://localhost:8000 --input_dir data_collection_logs/subsets --output_dir response_collection_logs --reuse_simple_chat path/to/existing_responses.json
"""

import argparse
import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import requests


class TeacherResponseCollector:
    """
    Collects responses from different chat endpoints for comparison
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        input_dir: str = "data_collection_logs/subsets",
        output_dir: str = "response_collection_logs",
        reuse_simple_chat_file: Optional[str] = None,
        max_workers: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        # Load existing simple-chat responses if specified
        self.existing_simple_chat_responses = {}
        if reuse_simple_chat_file:
            self.load_existing_simple_chat_responses(reuse_simple_chat_file)

        # API endpoints
        self.set_history_endpoint = f"{self.base_url}/set-chat-history"
        self.chat_endpoint = f"{self.base_url}/chat"
        self.simple_chat_endpoint = f"{self.base_url}/simple-chat"

        # Session for connection reuse - create per thread
        # We'll create sessions in the worker threads to avoid sharing issues

        # Thread-safe printing
        self.print_lock = Lock()

    def thread_safe_print(self, *args, **kwargs):
        """
        Thread-safe print function to avoid overlapping output
        """
        with self.print_lock:
            print(*args, **kwargs)

    def create_session(self) -> requests.Session:
        """
        Create a new session for HTTP requests
        """
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        return session

    def load_existing_simple_chat_responses(self, file_path: str) -> None:
        """
        Load existing simple-chat responses from a previous collection file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract simple-chat responses by subset_id
            if "results" in data:
                for result in data["results"]:
                    subset_id = result.get("subset_id")
                    simple_chat_response = result.get("responses", {}).get(
                        "simple-chat"
                    )

                    if subset_id and simple_chat_response is not None:
                        self.existing_simple_chat_responses[subset_id] = (
                            simple_chat_response
                        )

            print(
                f"📚 Loaded {len(self.existing_simple_chat_responses)} existing simple-chat responses from {file_path}"
            )

        except FileNotFoundError:
            print(f"❌ Error: Reuse file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in reuse file: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading existing responses: {e}")
            raise

    def load_subset_files(self) -> List[Dict]:
        """
        Load all subset JSON files from the input directory
        """
        subset_files = list(self.input_dir.glob("subset_*.json"))
        subsets = []

        print(f"📁 Found {len(subset_files)} subset files")

        for file_path in subset_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    subsets.append(data)
                    print(f"✅ Loaded: {file_path.name}")
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")

        print(f"📊 Successfully loaded {len(subsets)} subsets")
        return subsets

    def create_fresh_chat_id(self, session: requests.Session) -> str:
        """
        Create a new project and return its project ID to use as chat ID
        """
        try:
            project_data = {
                "title": f"Test Project {uuid.uuid4().hex[:8]}",
                "description": "Auto-generated project for endpoint response collection",
            }

            response = session.post(
                f"{self.base_url}/projects/", json=project_data, timeout=30
            )

            if response.status_code == 201:
                project = response.json()
                project_id = project.get("id")
                self.thread_safe_print(f"   ✅ Created project: {project_id}")
                return project_id
            else:
                raise Exception(
                    f"Failed to create project: {response.status_code} - {response.text}"
                )

        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error creating project: {e}")
        except Exception as e:
            if "Failed to create project" in str(e):
                raise  # Re-raise our custom exception
            raise Exception(f"Error creating project: {e}")

    def set_chat_history(
        self, session: requests.Session, chat_id: str, messages: List[Dict]
    ) -> bool:
        """
        Set the chat history for a given chat ID
        """
        try:
            response = session.post(
                f"{self.set_history_endpoint}/{chat_id}", json=messages, timeout=30
            )

            if response.status_code == 200:
                return True
            else:
                self.thread_safe_print(
                    f"❌ Failed to set chat history: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            self.thread_safe_print(f"❌ Error setting chat history: {e}")
            return False

    def get_chat_response(
        self,
        session: requests.Session,
        chat_id: str,
        user_message: str,
        endpoint_type: str = "chat",
    ) -> Optional[str]:
        """
        Get response from either /chat or /simple-chat endpoint
        """
        endpoint = (
            self.chat_endpoint if endpoint_type == "chat" else self.simple_chat_endpoint
        )

        message_data = {"role": "user", "message": user_message}

        try:
            self.thread_safe_print(f"   🤖 Getting {endpoint_type} response...")

            response = session.post(
                f"{endpoint}/{chat_id}", json=message_data, timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("ai_response", "")
            else:
                self.thread_safe_print(
                    f"   ❌ {endpoint_type} request failed: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            self.thread_safe_print(f"   ❌ Error getting {endpoint_type} response: {e}")
            return None

    def process_single_subset(self, subset: Dict) -> Dict:
        """
        Process a single subset and collect responses from both endpoints
        """
        subset_id = subset.get("subset_id")
        self.thread_safe_print(f"\n📝 Processing subset: {subset_id}")

        # Create a session for this thread
        session = self.create_session()

        # Extract the last user message to send to both endpoints
        messages = subset.get("messages", [])
        if not messages or messages[-1].get("role") != "user":
            self.thread_safe_print(
                "   ⚠️  Skipping - subset doesn't end with user message"
            )
            return {
                "subset_id": subset_id,
                "success": False,
                "error": "Subset doesn't end with user message",
                "timestamp": datetime.now().isoformat(),
            }

        last_user_message = messages[-1].get("message", "")
        history_messages = messages[:-1]  # All messages except the last user message

        result = {
            "subset_id": subset_id,
            "timestamp": datetime.now().isoformat(),
            "responses": {},
        }

        # Handle /chat endpoint (always fetch new response)
        self.thread_safe_print("   🔄 Get response from /chat endpoint...")
        chat_id_1 = self.create_fresh_chat_id(session)

        if self.set_chat_history(session, chat_id_1, history_messages):
            chat_response = self.get_chat_response(
                session, chat_id_1, last_user_message, "chat"
            )
            result["responses"]["chat"] = chat_response
        else:
            result["success"] = False
            result["error"] = "Failed to set chat history in chat endpoint"
            return result

        # Handle /simple-chat endpoint (reuse existing or fetch new)
        if subset_id in self.existing_simple_chat_responses:
            self.thread_safe_print("   📚 Reusing existing simple-chat response...")
            result["responses"]["simple-chat"] = self.existing_simple_chat_responses[
                subset_id
            ]
        else:
            self.thread_safe_print("   🔄 Get response from /simple-chat endpoint...")
            chat_id_2 = self.create_fresh_chat_id(session)

            if self.set_chat_history(session, chat_id_2, history_messages):
                simple_chat_response = self.get_chat_response(
                    session, chat_id_2, last_user_message, "simple-chat"
                )
                result["responses"]["simple-chat"] = simple_chat_response
            else:
                result["success"] = False
                result["error"] = "Failed to set chat history in simple-chat endpoint"
                return result

        # Determine overall success
        result["success"] = (
            result["responses"]["chat"] is not None
            and result["responses"]["simple-chat"] is not None
        )

        if result["success"]:
            self.thread_safe_print(
                "   ✅ Successfully collected responses from both endpoints"
            )
        else:
            self.thread_safe_print("   ⚠️  Partial success - some endpoints failed")

        return result

    def save_results(self, results: List[Dict]) -> str:
        """
        Save collection results to JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"endpoint_responses_{timestamp}.json"
        output_path = self.output_dir / filename

        # Create summary statistics
        total_subsets = len(results)
        successful = len([r for r in results if r.get("success", False)])

        summary = {
            "collection_completed": datetime.now().isoformat(),
            "total_subsets_processed": total_subsets,
            "successful": successful,
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
        }

        output_data = {"summary": summary, "results": results}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_path}")
        print("📊 Summary:")
        print(f"   Total subsets: {total_subsets}")
        print(f"   Successful responses: {successful}")

        return str(output_path)

    def run(self, max_subsets: Optional[int] = None) -> List[Dict]:
        """
        Main execution method with parallel processing
        """
        print("🚀 Starting endpoint response collection")
        print(f"🌐 Base URL: {self.base_url}")
        print(f"📂 Input directory: {self.input_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⚡ Max workers: {self.max_workers}")
        if self.existing_simple_chat_responses:
            print(
                f"📚 Reusing {len(self.existing_simple_chat_responses)} existing simple-chat responses"
            )
        if max_subsets:
            print(f"🔢 Max subsets to process: {max_subsets}")
        print("=" * 60)

        # Test server connectivity
        try:
            test_session = self.create_session()
            response = test_session.get(f"{self.base_url}/docs", timeout=5)
            if response.status_code == 200:
                print("✅ Server is reachable")
            else:
                print("⚠️  Server responded but might have issues")
        except Exception as e:
            print(f"❌ Cannot reach server: {e}")
            print("Proceeding anyway...")

        # Load subsets
        subsets = self.load_subset_files()
        if not subsets:
            print("❌ No subsets loaded!")
            return []

        # Limit subsets if specified
        if max_subsets and max_subsets < len(subsets):
            subsets = subsets[:max_subsets]
            print(f"🔢 Limited to first {max_subsets} subsets")

        # Process subsets in parallel
        results = []
        failed_count = 0
        completed_count = 0

        print(
            f"\n🔄 Processing {len(subsets)} subsets with {self.max_workers} workers..."
        )

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_subset = {
                    executor.submit(self.process_single_subset, subset): subset
                    for subset in subsets
                }

                # Process completed tasks as they finish
                for future in as_completed(future_to_subset):
                    subset = future_to_subset[future]
                    subset_id = subset.get("subset_id", "unknown")

                    try:
                        result = future.result()
                        results.append(result)
                        completed_count += 1

                        if result.get("success", False):
                            self.thread_safe_print(
                                f"✅ [{completed_count}/{len(subsets)}] Completed: {subset_id}"
                            )
                        else:
                            self.thread_safe_print(
                                f"⚠️  [{completed_count}/{len(subsets)}] Partial success: {subset_id}"
                            )
                            failed_count += 1

                    except Exception as e:
                        self.thread_safe_print(
                            f"💥 [{completed_count + 1}/{len(subsets)}] Error processing {subset_id}: {e}"
                        )
                        traceback.print_exc()
                        failed_count += 1
                        completed_count += 1

                        # Add a failed result to maintain consistency
                        results.append(
                            {
                                "subset_id": subset_id,
                                "success": False,
                                "error": str(e),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

        except KeyboardInterrupt:
            print(
                f"\n⚠️  Collection interrupted by user after {completed_count} subsets"
            )

        # Save results
        if results:
            self.save_results(results)
            print("\n🎉 Collection completed!")
            print(f"✅ Processed {len(results)} subsets")
            if failed_count > 0:
                print(f"⚠️  {failed_count} subsets had issues")
        else:
            print("❌ No results to save!")

        return results


def main():
    """
    Main function for response collection
    """
    parser = argparse.ArgumentParser(
        description="Collect responses from /chat and /simple-chat endpoints"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data_collection_logs/subsets",
        help="Input directory containing subset JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_collection_logs/responses",
        help="Output directory for response collection results",
    )
    parser.add_argument(
        "--max_subsets",
        type=int,
        help="Maximum number of subsets to process (for testing)",
    )
    parser.add_argument(
        "--reuse_simple_chat",
        type=str,
        help="Path to existing response JSON file to reuse simple-chat responses",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)",
    )

    args = parser.parse_args()

    print("🔀 Chat Endpoint Response Collector")
    print("=" * 40)
    print("📊 Configuration:")
    print(f"   Base URL: {args.base_url}")
    print(f"   Input directory: {args.input_dir}")
    print(f"   Output directory: {args.output_dir}")
    if args.max_subsets:
        print(f"   Max subsets: {args.max_subsets}")
    if args.reuse_simple_chat:
        print(f"   Reuse simple-chat responses from: {args.reuse_simple_chat}")
    print()

    try:
        collector = TeacherResponseCollector(
            base_url=args.base_url,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            reuse_simple_chat_file=args.reuse_simple_chat,
            max_workers=args.max_workers,
        )
        results = collector.run(max_subsets=args.max_subsets)

        return 0 if results else 1

    except KeyboardInterrupt:
        print("\n⚠️  Response collection interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Response collection failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
