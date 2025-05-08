import asyncio

import websockets
from rich.console import Console
from rich.live import Live
from rich.text import Text

from server.models.chat import ChatMessage
from server.services.chat import ChatEvent

# Create a rich console for nicer output
console = Console()


async def chat_in_websocket():
    # Connect to WebSocket server (adjust URL as needed for your environment)
    uri = "ws://localhost:8000/ws/chat"
    console.print("[bold blue]Connecting to chat,,[/bold blue]")

    async with websockets.connect(f"{uri}") as websocket:
        # Handle initial connection message
        response = await websocket.recv()
        console.print(f"[dim]{response}[/dim]")

        # Main interaction loop
        while True:
            # Get user input
            console.print(
                "[bold green]Enter your message (or 'exit' to quit):[/bold green]"
            )
            user_input = input("> ")

            if user_input.lower() == "exit":
                break

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
                        if content == "<EOS>":
                            break
                        full_response += content
                        response_text.append(content)
                        live.update(response_text)

            console.print("\n[dim]--- End of response ---[/dim]\n")


if __name__ == "__main__":
    try:
        # Run the async function
        asyncio.run(chat_in_websocket())
    except KeyboardInterrupt:
        console.print("[bold red]Exiting...[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        import traceback

        traceback.print_exc()
