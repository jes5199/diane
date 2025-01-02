import asyncio
import base64
import sys
import time
import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI
from hybrid_filter import HybridFilter

# Audio settings
INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
READ_SIZE_SEC = 0.02
READ_SIZE_FRAMES = int(INPUT_SAMPLE_RATE * READ_SIZE_SEC)
OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHANNELS = 1
OUTPUT_DTYPE = "int16"

class AudioProcessor:
    def __init__(self):
        self.filter = HybridFilter(filter_length=96)
        self.output_buffer = np.array([], dtype=np.float32)
        self.output_lock = asyncio.Lock()
        self.samples_processed = 0
        self.last_reset = time.time()
        
    def reset_state(self):
        """Reset the filter state when switching between listening/speaking"""
        self.filter.reset()
        self.output_buffer = np.array([], dtype=np.float32)
        self.samples_processed = 0
        self.last_reset = time.time()
        
    async def process_output(self, raw_audio):
        async with self.output_lock:
            audio_data = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
            self.output_buffer = np.append(self.output_buffer, audio_data)
            if len(self.output_buffer) > 4096:
                self.output_buffer = self.output_buffer[-4096:]
            
    def process_input(self, input_data):
        input_float = input_data.astype(np.float32)
        self.samples_processed += len(input_float)
        
        # Check if we need to reset
        current_time = time.time()
        if (self.samples_processed > INPUT_SAMPLE_RATE * 2 or  # Reset every 2 seconds
            current_time - self.last_reset > 3):               # Or after 3 seconds real time
            self.reset_state()
        
        if len(self.output_buffer) >= len(input_float):
            # Get reference signal
            reference = self.output_buffer[-len(input_float):]
            
            # Process through hybrid filter
            processed_data = self.filter.process(input_float, reference)
            
            return processed_data.astype(np.int16)
        
        return input_data

async def realtime_demo():
    client = AsyncOpenAI()
    playback_queue = asyncio.Queue()
    audio_processor = AudioProcessor()
    
    async def playback_audio():
        with sd.OutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=OUTPUT_CHANNELS,
            dtype=OUTPUT_DTYPE,
            blocksize=1024,
        ) as out_stream:
            while True:
                chunk = await playback_queue.get()
                if chunk is None:
                    continue
                await audio_processor.process_output(chunk)
                np_chunk = np.frombuffer(chunk, dtype=np.int16)
                out_stream.write(np_chunk)

    async def stream_microphone(conn):
        with sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=INPUT_CHANNELS,
            dtype="int16",
        ) as stream:
            while True:
                if stream.read_available < READ_SIZE_FRAMES:
                    await asyncio.sleep(0)
                    continue

                data, _ = stream.read(READ_SIZE_FRAMES)
                processed_data = audio_processor.process_input(data.flatten())
                processed_bytes = processed_data.tobytes()
                b64_chunk = base64.b64encode(processed_bytes).decode("utf-8")
                await conn.input_audio_buffer.append(audio=b64_chunk)
                await asyncio.sleep(0)

    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview",
    ) as conn:
        print("Connected to Realtime. Setting up hybrid echo cancellation...")
        await conn.session.update(session={
            "turn_detection": {"type": "server_vad"},
            "voice": "alloy"
        })

        playback_task = asyncio.create_task(playback_audio())
        mic_task = asyncio.create_task(stream_microphone(conn))

        print("Now streaming with hybrid echo cancellation. Speak, pause, and wait for response.\n")

        acc_text = {}
        try:
            async for event in conn:
                if event.type == "response.audio_transcript.delta":
                    item_id = event.item_id
                    acc_text[item_id] = acc_text.get(item_id, "") + event.delta
                    sys.stdout.write(f"\rPartial transcript: {acc_text[item_id]}")
                    sys.stdout.flush()
                    
                elif event.type == "response.audio_transcript.done":
                    item_id = event.item_id
                    print(f"\n[Final transcript] {acc_text.get(item_id, '')}")
                    acc_text.pop(item_id, None)
                    
                elif event.type == "response.audio.delta":
                    raw_audio = base64.b64decode(event.delta)
                    await playback_queue.put(raw_audio)
                    audio_processor.reset_state()
                    
                elif event.type == "response.done":
                    print("\n[Assistant finished responding]\n")
                    audio_processor.reset_state()
                    
        finally:
            mic_task.cancel()
            playback_task.cancel()
            try:
                await mic_task
                await playback_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    asyncio.run(realtime_demo())