#!/usr/bin/env python3
"""
Test script to verify the data collection setup is working properly.

This script runs basic connectivity tests and a minimal data collection session.
"""

import asyncio
import sys

import httpx

from data_collection.data_collection_student import DataCollectionSession, StudentAgent


async def test_server_connectivity():
    """Test if the ureka-server is running and accessible"""
    print("🔗 Testing server connectivity...")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/projects/", timeout=10.0)
            response.raise_for_status()
            print("✅ Server is accessible")
            return True
    except httpx.ConnectError:
        print("❌ Cannot connect to server at http://localhost:8000")
        print("   Make sure ureka-server is running:")
        print("   uv run uvicorn server.main:app --host 0.0.0.0 --port 8000")
        return False
    except httpx.HTTPError as e:
        print(f"❌ Server error: {e}")
        return False


async def test_project_creation():
    """Test project creation functionality"""
    print("📋 Testing project creation...")

    try:
        student = StudentAgent("Test Student")
        session = DataCollectionSession(student)

        project_id = await session.create_project()
        print(f"✅ Project created successfully: {project_id}")
        return project_id
    except Exception as e:
        print(f"❌ Project creation failed: {e}")
        return None


async def test_chat_functionality(project_id):
    """Test chat functionality"""
    print("💬 Testing chat functionality...")

    try:
        student = StudentAgent("Test Student")
        session = DataCollectionSession(student)
        session.project_id = project_id

        response = await session.send_message_to_teacher("What is machine learning?")
        print(f"✅ Chat response received: {len(response)} characters")
        return True
    except Exception as e:
        print(f"❌ Chat functionality failed: {e}")
        return False


async def test_chat_history(project_id):
    """Test chat history retrieval"""
    print("📜 Testing chat history retrieval...")

    try:
        student = StudentAgent("Test Student")
        session = DataCollectionSession(student)
        session.project_id = project_id

        history = await session.get_chat_history()
        print(f"✅ Chat history retrieved: {len(history)} messages")
        return True
    except Exception as e:
        print(f"❌ Chat history retrieval failed: {e}")
        return False


async def run_mini_session():
    """Run a minimal data collection session"""
    print("🎓 Running mini data collection session...")

    try:
        student = StudentAgent("Test Student")
        session = DataCollectionSession(student)

        # Create project
        await session.create_project()

        # Send one question
        await session.send_message_to_teacher("What is artificial intelligence?")

        # Get history
        history = await session.get_chat_history()

        print(f"✅ Mini session completed: {len(history)} total messages")
        return True
    except Exception as e:
        print(f"❌ Mini session failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🧪 Data Collection Setup Test")
    print("=" * 40)

    tests_passed = 0
    total_tests = 5

    # Test 1: Server connectivity
    if await test_server_connectivity():
        tests_passed += 1
    else:
        print("\n❌ Server connectivity failed. Please start the ureka-server first.")
        return 1

    print()

    # Test 2: Project creation
    project_id = await test_project_creation()
    if project_id:
        tests_passed += 1
    else:
        print(
            "\n❌ Cannot proceed with remaining tests due to project creation failure."
        )
        return 1

    print()

    # Test 3: Chat functionality
    if await test_chat_functionality(project_id):
        tests_passed += 1

    print()

    # Test 4: Chat history
    if await test_chat_history(project_id):
        tests_passed += 1

    print()

    # Test 5: Mini session
    if await run_mini_session():
        tests_passed += 1

    print()
    print("=" * 40)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Data collection setup is working correctly.")
        print("\nYou can now run:")
        print("  python data_collection_student.py")
        print("  python batch_data_collection.py --sessions 3 --turns 2")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
