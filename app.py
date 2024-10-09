import os
from io import BytesIO
import httpx
from openai import AsyncOpenAI
from chainlit.element import ElementBased
import chainlit as cl
from dotenv import load_dotenv
import requests
import furl
import base64

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LITERAL_API_KEY = os.getenv("LITERAL_API_KEY")
SERVER_URL = os.getenv("SERVER_URL")
VOICEFLOW_API_KEY = os.getenv('VOICEFLOW_API_KEY')


print(ELEVENLABS_API_KEY)
print(ELEVENLABS_VOICE_ID)
print(OPENAI_API_KEY)
print(LITERAL_API_KEY)
print(SERVER_URL)
print(VOICEFLOW_API_KEY)

cl.instrument_openai()

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID or not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY, ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set")

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if "voiceflow" in username: 
        return cl.User(
            identifier=username, metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.step(type="tool", name="Speech to Text (OpenAI Whisper)")
async def speech_to_text(audio_file):
    response = await client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


@cl.step(type="tool", name="Conversation Manager (Voiceflow)")
async def generate_text_answer(request):
    user_id = cl.user_session.get("user").identifier.replace("@", "-").replace(".", "-")
    #sessionId = cl.user_session.get("id")
    url = f'https://general-runtime.voiceflow.com/state/user/{user_id}/interact'
    url_encoded = furl.furl(url).url
    print(url_encoded)
    print({ 'request': request })
    response = requests.post(
       url_encoded,
        json={ 'request': request },
        headers={ 
            'Authorization': VOICEFLOW_API_KEY,
            'versionID': 'development'
        },
    )
    response.raise_for_status()
    message = ""
    for trace in response.json():
        print(trace)
        if trace['type'] == 'text':
            message += trace['payload']['message']
    return message

@cl.step(type="tool", name="Text to Speech (Elevelabs)")
async def text_to_speech(text: str, mime_type: str):
    CHUNK_SIZE = 1024

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

    headers = {
    "Accept": mime_type,
    "Content-Type": "application/json",
    "xi-api-key": ELEVENLABS_API_KEY
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()  # Ensure we notice bad responses

        buffer = BytesIO()
        buffer.name = f"output_audio.{mime_type.split('/')[1]}"

        async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
            if chunk:
                buffer.write(chunk)
        
        buffer.seek(0)
        return buffer.name, buffer.read()
        

@cl.on_chat_start
async def start():
    response = await generate_text_answer({ 'type': 'launch' })
    await cl.Message(
        content=response
    ).send()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)
    
    # For now, write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    await cl.Message(
        author="You", 
        type="user_message",
        content=transcription,
        elements=[input_audio_el, *elements]
    ).send()

    text_answer = await generate_text_answer( { 'type': 'text', 'payload': transcription })
    
    output_name, output_audio = await text_to_speech(text_answer, audio_mime_type)
    
    output_audio_el = cl.Audio(
        name=output_name,
        auto_play=True,
        mime=audio_mime_type,
        content=output_audio,
    )
    answer_message = await cl.Message(content=text_answer).send()

    answer_message.elements = [output_audio_el]
    await answer_message.update()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

@cl.step(type="tool", name="Image to Text (OpenAI 4o)")
def image_to_text(image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "You are a food assistant. Please help me identify the food in the image. If there are multiple foods, please list them all. If there are dividers, please list them as well. If it is a plate that has leftovers, please list the leftovers."
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()["choices"][0]["message"]["content"]

@cl.on_message
async def on_message(message: cl.Message):
    has_iamge = False
    message_text = message.content
    if message.elements:
        has_iamge = True
    if has_iamge:
        # Processing images exclusively
        images = [file for file in message.elements if "image" in file.mime]
        print(images)
        # Read the first image
        with open(images[0].path, "r") as f:
            message_text = image_to_text(images[0].path)
            print(message_text)
            pass
    
    text_answer = await generate_text_answer( { 'type': 'text', 'payload': message_text })
    await cl.Message(
        content=text_answer,
    ).send()