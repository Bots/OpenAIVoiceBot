import os
import json
import base64
import asyncio
import websockets
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket, Request
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Initialize envvironment variables
load_dotenv()

# Initialize colorama
init(autoreset=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", 5050))
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "Always stay positive, keep answers short and consice."
)
VOICE = "alloy"
LOG_EVENT_TYPES = [
    'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
]

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")


def log_conversation(role, message, color=Fore.WHITE):
    """Log conversation in a clean, easy-to-read format."""
    prefix = f"{Fore.CYAN}[{role.upper()}]{Style.RESET_ALL}"
    print(f"{prefix} {color}{message}")


def log_info(message, event_type=None):
    """Log important information."""
    if event_type:
        print(f"{Fore.YELLOW}[INFO - {event_type}]{Style.RESET_ALL} {message}")
    else:
        print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} {message}")


@app.get("/", response_class=HTMLResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response."""
    response = VoiceResponse()
    response.say(
        "Welcome to Bots one's voice bot. Is there something I can help you with?")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    log_info("Client connected")
    await websocket.accept()

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await send_session_update(openai_ws)
        stream_sid = None

        async def receive_from_twilio():
            nonlocal stream_sid
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        log_info(f"Incoming stream started: {
                                 stream_sid}", "Stream")
            except WebSocketDisconnect:
                log_info("Client disconnected", "Connection")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            nonlocal stream_sid
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] == 'response.audio.delta' and response.get('delta'):
                        try:
                            audio_payload = base64.b64encode(
                                base64.b64decode(response['delta'])).decode('utf-8')
                            audio_delta = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": audio_payload
                                }
                            }
                            await websocket.send_json(audio_delta)
                        except Exception as e:
                            log_info(f"Error processing audio data: {
                                     e}", "Error")
                    elif response['type'] == 'conversation.item.input_audio_transcription.completed':
                        log_conversation(
                            "HUMAN", response.get('transcript', ''), Fore.MAGENTA)
                    elif response['type'] == 'response.audio_transcript.done':
                        log_conversation(
                            "AI", response.get('transcript', ''), Fore.MAGENTA)
            except Exception as e:
                log_info(f"Error in send_to_twilio: {e}", "Error")

        await asyncio.gather(receive_from_twilio(), send_to_twilio())


async def send_session_update(openai_ws):
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "input_audio_transcription": {
                "model": "whisper-1",
            },
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    log_info('Sending session update')
    await openai_ws.send(json.dumps(session_update))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
