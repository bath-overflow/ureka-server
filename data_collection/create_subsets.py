#!/usr/bin/env python3
"""
Conversation Subset Generator

This script reads conversation data from session JSON files and creates random subsets
where each subset ends with a user utterance.

Usage:
    python create_subsets.py --num_subsets 100 --input_dir data_collection_logs --output_dir data_collection_logs/subsets
"""

import argparse
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class ConversationSubsetGenerator:
    """
    Generates random subsets of conversation data ending with user utterances
    """

    def __init__(
        self,
        input_dir: str = "data_collection_logs",
        output_dir: str = "data_collection_logs/subsets",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_session_files(self) -> List[Dict]:
        """
        Load all session JSON files from the input directory
        """
        session_files = list(self.input_dir.glob("session_*.json"))
        conversations = []

        print(f"📁 Found {len(session_files)} session files")

        for file_path in session_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversations.append(data)
                    print(f"✅ Loaded: {file_path.name}")
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")

        print(f"📊 Successfully loaded {len(conversations)} conversations")
        return conversations

    def validate_conversation_format(self, messages: List[Dict]) -> bool:
        """
        Validate that the conversation follows user-assistant alternating pattern
        """
        if len(messages) == 0:
            return False

        # Should start with user and alternate
        for i, message in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if message.get("role") != expected_role:
                return False

        return True

    def get_possible_subsets(self, messages: List[Dict]) -> List[Tuple[int, str]]:
        """
        Get all possible subset endpoints that end with user utterances
        Returns list of (end_index, description) tuples

        Example possible_subsets:
            (1, "u1"), (3, "u1-a1-u2"), (5, "u1-a1-u2-a2-u3")
        """
        if not self.validate_conversation_format(messages):
            return []

        possible_subsets = []

        # Find all user message indices (should be even indices: 0, 2, 4, 6...)
        for i, message in enumerate(messages):
            if message.get("role") == "user":
                turn_number = (i // 2) + 1

                # Generate description dynamically for any conversation length
                if turn_number == 1:
                    description = "u1"
                else:
                    # Build pattern: u1-a1-u2-a2-u3-...-uN
                    parts = []
                    for turn in range(1, turn_number + 1):
                        parts.append(f"u{turn}")
                        if (
                            turn < turn_number
                        ):  # Don't add assistant part after the last user message
                            parts.append(f"a{turn}")
                    description = "-".join(parts)

                possible_subsets.append(
                    (i + 1, description)
                )  # +1 because we want to include this message

        return possible_subsets

    def create_subset(
        self, conversation: Dict, end_index: int, subset_description: str
    ) -> Dict:
        """
        Create a subset of the conversation ending at the specified index
        """
        messages = conversation["messages"][:end_index]

        # Create new subset with original metadata plus subset info
        subset = {
            "subset_id": str(uuid.uuid4())[:8],
            "original_session_id": conversation.get("session_id"),
            "subset_description": subset_description,
            "subset_length": len(messages),
            "created_at": datetime.now().isoformat(),
            "messages": messages,
            # Preserve original metadata
            "original_metadata": {
                "student_persona": conversation.get("student_persona"),
                "project_id": conversation.get("project_id"),
                "student_traits": conversation.get("student_traits"),
                "batch_session_number": conversation.get("batch_session_number"),
                "batch_id": conversation.get("batch_id"),
            },
        }

        return subset

    def generate_random_subsets(
        self, conversations: List[Dict], num_subsets: int
    ) -> List[Dict]:
        """
        Generate N random subsets from the conversations
        """
        print(f"🎲 Generating {num_subsets} random subsets...")

        # Collect all possible subset options
        all_options = []
        for conv in conversations:
            messages = conv.get("messages", [])
            possible_subsets = self.get_possible_subsets(messages)

            for end_index, description in possible_subsets:
                all_options.append((conv, end_index, description))

        if len(all_options) == 0:
            print("❌ No valid conversation subsets found!")
            return []

        print(f"📈 Found {len(all_options)} possible subset options")

        # Randomly sample subsets
        if num_subsets > len(all_options):
            print(
                f"⚠️  Requested {num_subsets} subsets but only {len(all_options)} options available"
            )
            selected_options = all_options
        else:
            selected_options = random.sample(all_options, num_subsets)

        # Generate subsets
        subsets = []
        for conv, end_index, description in selected_options:
            subset = self.create_subset(conv, end_index, description)
            subsets.append(subset)

        return subsets

    def save_subsets(self, subsets: List[Dict]) -> str:
        """
        Save subsets to individual JSON files and create a summary
        """
        print(f"💾 Saving {len(subsets)} subsets...")

        saved_files = []

        # Save individual subset files
        for subset in subsets:
            filename = f"subset_{subset['subset_id']}.json"
            file_path = self.output_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(subset, f, indent=2, ensure_ascii=False)

            saved_files.append(filename)

        # Create summary file
        summary = self.create_summary(subsets, saved_files)
        summary_path = (
            self.output_dir
            / f"subsets_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(subsets)} subset files")
        print(f"📋 Summary saved to: {summary_path}")

        return str(summary_path)

    def create_summary(self, subsets: List[Dict], saved_files: List[str]) -> Dict:
        """
        Create a summary of the generated subsets
        """
        # Count subset types
        subset_counts = {}
        for subset in subsets:
            desc = subset["subset_description"]
            subset_counts[desc] = subset_counts.get(desc, 0) + 1

        # Get original session distribution
        original_sessions = set()
        for subset in subsets:
            original_sessions.add(
                subset["original_metadata"].get("original_session_id")
            )

        return {
            "summary_created": datetime.now().isoformat(),
            "total_subsets": len(subsets),
            "subset_files": saved_files,
            "subset_type_distribution": subset_counts,
            "original_sessions_used": len(original_sessions),
            "average_messages_per_subset": (
                sum(s["subset_length"] for s in subsets) / len(subsets)
                if subsets
                else 0
            ),
            "subset_length_distribution": {
                str(length): len([s for s in subsets if s["subset_length"] == length])
                for length in sorted(set(s["subset_length"] for s in subsets))
            },
        }

    def run(self, num_subsets: int) -> List[Dict]:
        """
        Main execution method
        """
        print("🚀 Starting conversation subset generation")
        print(f"📊 Target subsets: {num_subsets}")
        print(f"📂 Input directory: {self.input_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print("=" * 60)

        # Load conversations
        conversations = self.load_session_files()
        if not conversations:
            print("❌ No conversations loaded!")
            return []

        # Generate subsets
        subsets = self.generate_random_subsets(conversations, num_subsets)
        if not subsets:
            print("❌ No subsets generated!")
            return []

        # Save results
        self.save_subsets(subsets)

        print("\n🎉 Subset generation completed!")
        print(f"✅ Generated {len(subsets)} subsets")
        print(f"📁 Files saved to: {self.output_dir}")

        return subsets


def main():
    """
    Main function for subset generation
    """
    parser = argparse.ArgumentParser(
        description="Generate random subsets of conversation data"
    )
    parser.add_argument(
        "--num_subsets",
        type=int,
        default=100,
        help="Number of random subsets to generate",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data_collection_logs",
        help="Input directory containing session JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_collection_logs/subsets",
        help="Output directory for subset files",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducible results")

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        print(f"🌱 Random seed set to: {args.seed}")

    print("🔀 Conversation Subset Generator")
    print("=" * 40)
    print("📊 Configuration:")
    print(f"   Number of subsets: {args.num_subsets}")
    print(f"   Input directory: {args.input_dir}")
    print(f"   Output directory: {args.output_dir}")
    print()

    try:
        generator = ConversationSubsetGenerator(args.input_dir, args.output_dir)
        subsets = generator.run(args.num_subsets)

        if subsets:
            print("\n📈 Final Statistics:")
            print(f"   Total subsets created: {len(subsets)}")
            total_messages = sum(s["subset_length"] for s in subsets)
            print(f"   Total messages: {total_messages}")
            print(f"   Average messages per subset: {total_messages/len(subsets):.1f}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Subset generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Subset generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
