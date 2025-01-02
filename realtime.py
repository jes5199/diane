import asyncio
import base64
import sys
import openai
import sounddevice as sd

# We'll use the async client for realtime:
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

# Audio settings
INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
READ_SIZE_SEC = 0.02  # read 20 ms at a time
READ_SIZE_FRAMES = int(INPUT_SAMPLE_RATE * READ_SIZE_SEC)

# New output audio settings
OUTPUT_SAMPLE_RATE = 16000  # TTS audio sample rate
OUTPUT_CHANNELS = 1
OUTPUT_DTYPE = 'int16'

#openai.api_key = "YOUR_OPENAI_API_KEY_HERE"

async def realtime_demo():
    """
    Streams microphone audio to Realtime and plays back TTS audio to your speakers.
    """
    client = AsyncOpenAI()

    # Create a queue for TTS audio playback
    playback_queue = asyncio.Queue()
    
    # Start a background playback task
    playback_task = asyncio.create_task(playback_audio(playback_queue))

    # Initialize transcript accumulator
    acc_text = {}

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview-2024-10-01") as conn:
        print("Connected to Realtime API. Creating session...")

        # Optional: Turn detection approach
        #   "turn_detection": {"type": "server_vad"} (server does voice-activity detection)
        # or you can set it to None to handle “push-to-talk” manually.
        await conn.session.update(session={"turn_detection": {"type": "server_vad"}})
        
        # 2. Fire off a background task to read from your microphone
        #    and append audio to the input_audio_buffer
        mic_task = asyncio.create_task(stream_microphone(conn))

        print("Now listening for server events (transcripts/responses). Press Ctrl+C to stop.\n")

        # 3. Listen for events from the server (transcripts, audio deltas, etc.)
        #    We simply print partial text transcripts. 
        #    If there's TTS audio, you could decode & play it here as well.
        async for event in conn:
            event_type = event.type

            if event_type == "session.created":
                print("Session created:", event.session.id)

            elif event_type == "session.updated":
                # The session can be updated with new settings or state
                pass

            elif event_type == "response.audio_transcript.delta":
                # Accumulate partial text from the transcription
                item_id = event.item_id
                delta_text = event.delta

                if item_id not in acc_text:
                    acc_text[item_id] = delta_text
                else:
                    acc_text[item_id] += delta_text

                # Clear screen line and print partial transcript
                sys.stdout.write(f"\rPartial transcript: {acc_text[item_id]}")
                sys.stdout.flush()

            elif event_type == "response.audio_transcript.done":
                # Get the final accumulated transcript for this item_id
                item_id = event.item_id
                final_text = acc_text.get(item_id, "")
                print(f"\n[Final transcript] {final_text}")
                
                # Clean up the accumulated text for this item
                acc_text.pop(item_id, None)

            elif event_type == "response.audio.delta":
                # Decode TTS audio and queue for playback
                raw_audio = base64.b64decode(event.delta)
                await playback_queue.put(raw_audio)

            # Other events might include "response.text.delta" (if you’re also getting textual output),
            # "response.done", etc.
            # Check the Realtime docs for more event types & usage.

        # Cancel both mic and playback tasks on exit
        mic_task.cancel()
        playback_task.cancel()
        try:
            await mic_task
            await playback_task
        except asyncio.CancelledError:
            pass

async def playback_audio(playback_queue: asyncio.Queue):
    """
    Continuously read raw audio chunks from playback_queue and play them out the speakers.
    """
    print("Setting up audio output...")

    with sd.OutputStream(
        samplerate=OUTPUT_SAMPLE_RATE,
        channels=OUTPUT_CHANNELS,
        dtype=OUTPUT_DTYPE,
        blocksize=1024,
    ) as out_stream:
        try:
            while True:
                # Get the next chunk from the queue
                chunk = await playback_queue.get()
                if chunk is None:
                    continue

                # Write raw bytes to the speaker
                out_stream.write(chunk)
        except asyncio.CancelledError:
            pass

async def stream_microphone(conn: AsyncRealtimeConnection):
    """
    Continuously read from the microphone in small chunks,
    base64-encode, and append to the session's input_audio_buffer.
    """
    print("Opening microphone stream...")

    with sd.InputStream(
        samplerate=INPUT_SAMPLE_RATE, 
        channels=INPUT_CHANNELS, 
        dtype='int16'
    ) as stream:
        try:
            while True:
                # Only attempt to read if we have enough frames
                if stream.read_available < READ_SIZE_FRAMES:
                    await asyncio.sleep(0)
                    continue

                data, _ = stream.read(READ_SIZE_FRAMES)
                b64_chunk = base64.b64encode(data).decode("utf-8")

                # Append audio to the input_audio_buffer
                await conn.input_audio_buffer.append(audio=b64_chunk)

                # Tiny pause to let other coroutines run
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            # The main loop cancels this task on exit
            pass

def main():
    asyncio.run(realtime_demo())

if __name__ == "__main__":
    main()
