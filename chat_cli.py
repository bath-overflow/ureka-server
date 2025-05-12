import asyncio

import websockets
from rich.console import Console
from rich.live import Live
from rich.text import Text

from server.models.chat_model import ChatMessage
from server.services.chat import ChatEvent, ChatInfo

# Create a rich console for nicer output
console = Console()


async def reconnect(chat_id):
    # Attempt to reconnect to the WebSocket server
    uri = "ws://localhost:8000/ws/chat?chat_id=" + chat_id
    console.print("[bold blue]Reconnecting to chat...[/bold blue]")
    try:
        async with websockets.connect(f"{uri}") as websocket:
            console.print("[dim]Reconnected successfully![/dim]")
            return websocket
    except Exception as e:
        console.print(f"[bold red]Reconnect failed: {str(e)}[/bold red]")
        return None


async def chat(websocket: websockets.ClientConnection, chat_id=None):
    # Main interaction loop
    while True:
        # Get user input
        console.print(
            "[bold green]Enter your message (or 'Ctrl+C' to quit):[/bold green]"
        )
        user_input = input("> ")

        # Create and send message
        message = ChatMessage(
            role="user",
            message=user_input,
        )
        await websocket.send(message.model_dump_json())

        # Wait for processing confirmation
        response = await websocket.recv()

        console.print(f"[dim]{response}[/dim]")

        # Collect and display streaming response
        console.print("[bold yellow]AI Response:[/bold yellow]")
        full_response = ""
        response_text = Text()

        with Live(response_text, refresh_per_second=10) as live:
            while True:
                response = await websocket.recv()
                event, content = response.split(": ", 1)

                if event == ChatEvent.SEND_MESSAGE.value:
                    if content == ChatInfo.END_OF_STREAM.value:
                        break
                    full_response += content
                    response_text.append(content)
                    live.update(response_text)

        console.print("\n[dim]--- End of response ---[/dim]\n")


async def chat_in_websocket(chat_id=None):
    # Connect to WebSocket server (adjust URL as needed for your environment)
    uri = (
        "ws://localhost:8000/ws/chat"
        if chat_id is None
        else f"ws://localhost:8000/ws/chat?chat_id={chat_id}"
    )
    console.print("[bold blue]Connecting to chat,,[/bold blue]")

    while True:
        try:
            async with websockets.connect(f"{uri}") as websocket:
                # Handle initial connection message
                response = await websocket.recv()
                console.print(f"[dim]{response}[/dim]")

                # Handle chat_id assignment
                if chat_id is None:
                    response = await websocket.recv()
                    console.print(f"[dim]{response}[/dim]")

                    event, _, id = response.split(": ", 2)
                    if event == ChatEvent.CONNECTED.value:
                        chat_id = id
                    else:
                        console.print(
                            f"[bold red]Something went wrong: {response}[/bold red]"
                        )
                        return

                # Start the chat interaction
                await chat(websocket, chat_id=chat_id)
        except websockets.ConnectionClosedError:
            console.print(
                "[bold red]Connection closed. Attempting to reconnect...[/bold red]"
            )
            websocket = await reconnect(chat_id)
            if websocket:
                await chat(websocket, chat_id=chat_id)


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Chat CLI")
    parser.add_argument(
        "--chat_id",
        type=str,
        help="Chat ID to connect to (optional)",
    )
    args = parser.parse_args()
    chat_id = args.chat_id
    if chat_id:
        console.print(f"[bold blue]Connecting to chat with ID: {chat_id}[/bold blue]")
    else:
        console.print("[bold blue]Connecting to new chat...[/bold blue]")

    try:
        # Run the async function
        asyncio.run(chat_in_websocket(chat_id))
    except KeyboardInterrupt:
        console.print("[bold red]Exiting...[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        import traceback

        traceback.print_exc()
