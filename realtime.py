import asyncio
import base64
import sys
import openai
import sounddevice as sd

# We'll use the async client for realtime:
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
READ_SIZE_SEC = 0.02  # read 20 ms at a time
READ_SIZE_FRAMES = int(SAMPLE_RATE * READ_SIZE_SEC)

#openai.api_key = "YOUR_OPENAI_API_KEY_HERE"

async def realtime_demo():
    """
    Minimal console-based realtime voice example:
      1) Connect to the Realtime endpoint
      2) Stream microphone audio in
      3) Print partial transcripts / text as they arrive
    """

    # Create an AsyncOpenAI client
    client = AsyncOpenAI()

    # Initialize transcript accumulator at the start
    acc_text = {}

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview-2024-10-01") as conn:
        print("Connected to Realtime API. Creating session...")

        # Optional: Turn detection approach
        #   "turn_detection": {"type": "server_vad"} (server does voice-activity detection)
        # or you can set it to None to handle “push-to-talk” manually.
        await conn.session.update(session={"turn_detection": {"type": "server_vad"}})
        
        # 2. Fire off a background task to read from your microphone
        #    and append audio to the input_audio_buffer
        audio_task = asyncio.create_task(stream_microphone(conn))

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
                # This is TTS audio. You’d decode & queue for playback if desired
                raw_audio = base64.b64decode(event.delta)
                # e.g. pass raw_audio to pyaudio or sounddevice playback
                pass

            # Other events might include "response.text.delta" (if you’re also getting textual output),
            # "response.done", etc.
            # Check the Realtime docs for more event types & usage.

        # If we ever exit the loop, stop reading from the microphone
        audio_task.cancel()
        try:
            await audio_task
        except asyncio.CancelledError:
            pass


async def stream_microphone(conn: AsyncRealtimeConnection):
    """
    Continuously read from the microphone in small chunks,
    base64-encode, and append to the session's input_audio_buffer.
    """
    print("Opening microphone stream...")

    # Start input stream using sounddevice
    # (You might need to check your default device or specify device=... if it fails)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
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
