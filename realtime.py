import asyncio
import base64
import sys
import openai
import sounddevice as sd
import numpy as np

from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

# Audio input settings
INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
READ_SIZE_SEC = 0.02  # read 20 ms at a time
READ_SIZE_FRAMES = int(INPUT_SAMPLE_RATE * READ_SIZE_SEC)

# Audio output settings (for TTS playback)
OUTPUT_SAMPLE_RATE = 16000
OUTPUT_CHANNELS = 1
OUTPUT_DTYPE = "int16"



async def realtime_demo():
    """
    Streams microphone audio to Realtime and plays back TTS audio to your speakers.
    With server-based VAD + create_response=True, you just speak, pause,
    and the server will produce an audio reply.
    """

    client = AsyncOpenAI()

    # Queue for TTS audio chunks
    playback_queue = asyncio.Queue()
    # Start background playback
    playback_task = asyncio.create_task(playback_audio(playback_queue))

    # Accumulate partial transcripts in a dict
    acc_text = {}

    # 1) Connect with parameters that request TTS
    async with client.beta.realtime.connect(
        # CHANGE THIS to a TTS-capable model if needed:
        model="gpt-4o-realtime-preview",
    ) as conn:
        print("Connected to Realtime. Session creation in progress...")
        await conn.session.update(session={"turn_detection": {"type": "server_vad"},
                                            "voice": "alloy"})

        # 2) Start reading from the microphone
        mic_task = asyncio.create_task(stream_microphone(conn))

        print("Now streaming mic input and listening for events. Speak, pause, and wait for TTS.\n")

        async for event in conn:
            event_type = event.type

            # The session is established
            if event_type == "session.created":
                print("Session created:", event.session.id)

            elif event_type == "session.updated":
                # The session can be updated with new state
                pass

            # Real-time partial transcripts
            elif event_type == "response.audio_transcript.delta":
                item_id = event.item_id
                delta_text = event.delta
                acc_text[item_id] = acc_text.get(item_id, "") + delta_text

                # Print partial transcript in place
                sys.stdout.write(f"\rPartial transcript: {acc_text[item_id]}")
                sys.stdout.flush()

            # Final transcript once the server is sure
            elif event_type == "response.audio_transcript.done":
                item_id = event.item_id
                final_text = acc_text.get(item_id, "")
                print(f"\n[Final transcript] {final_text}")
                acc_text.pop(item_id, None)

            # TTS audio
            elif event_type == "response.audio.delta":
                #print(f"Received audio delta: {event.delta}")
                raw_audio = base64.b64decode(event.delta)
                await playback_queue.put(raw_audio)

            elif event_type == "response.done":
                print("[Assistant finished responding. You can talk again or Ctrl+C to quit]\n")

        # If we exit the loop, cancel mic & playback tasks
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
                chunk = await playback_queue.get()
                if chunk is None:
                    continue
                
                # Convert raw bytes to int16 NumPy array
                np_chunk = np.frombuffer(chunk, dtype=np.int16)
                
                # Write the NumPy array to the output stream
                out_stream.write(np_chunk)
        except asyncio.CancelledError:
            pass


async def stream_microphone(conn: AsyncRealtimeConnection):
    """
    Continuously read microphone audio, base64-encode, and append to the session.
    With 'create_response': True in turn_detection, the server will produce a
    response after detecting silence. 
    """
    print("Opening microphone stream...")

    with sd.InputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=INPUT_CHANNELS,
        dtype="int16",
    ) as stream:
        try:
            while True:
                if stream.read_available < READ_SIZE_FRAMES:
                    await asyncio.sleep(0)
                    continue

                data, _ = stream.read(READ_SIZE_FRAMES)
                b64_chunk = base64.b64encode(data).decode("utf-8")

                # Send the chunk to input_audio_buffer
                await conn.input_audio_buffer.append(audio=b64_chunk)

                # Short yield
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass


def main():
    asyncio.run(realtime_demo())


if __name__ == "__main__":
    main()
