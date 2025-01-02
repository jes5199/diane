import asyncio
import base64
import sys
import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI

# Audio settings
INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
READ_SIZE_SEC = 0.02
READ_SIZE_FRAMES = int(INPUT_SAMPLE_RATE * READ_SIZE_SEC)
OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHANNELS = 1
OUTPUT_DTYPE = "int16"

class NLMSFilter:
    def __init__(self, filter_length=2048, step_size=0.2, epsilon=1e-6):
        self.filter_length = filter_length
        self.step_size = step_size
        self.epsilon = epsilon
        self.weights = np.zeros(filter_length)
        self.reference_buffer = np.zeros(filter_length)
        # Add energy thresholding with lower threshold
        self.energy_threshold = 100  # Reduced threshold
        self.last_error = 0
        
    def update(self, reference_sample, input_sample):
        # Update reference buffer
        self.reference_buffer = np.roll(self.reference_buffer, 1)
        self.reference_buffer[0] = reference_sample
        
        # Calculate filter output
        y = np.dot(self.weights, self.reference_buffer)
        
        # Calculate error
        error = input_sample - y
        
        # Gentler energy-based gating
        signal_energy = np.sum(self.reference_buffer**2)
        if signal_energy < self.energy_threshold:
            error = error * 0.5  # Reduce but don't eliminate
            
        # Light smoothing
        error = 0.9 * error + 0.1 * self.last_error
        self.last_error = error
        
        # Update filter weights
        norm = np.dot(self.reference_buffer, self.reference_buffer) + self.epsilon
        self.weights += self.step_size * error * self.reference_buffer / norm
        
        return error

class AudioProcessor:
    def __init__(self):
        self.nlms_filter = NLMSFilter()
        self.output_buffer = np.array([], dtype=np.int16)
        self.output_lock = asyncio.Lock()
        
    async def process_output(self, raw_audio):
        async with self.output_lock:
            # Convert raw bytes to int16 array
            audio_data = np.frombuffer(raw_audio, dtype=np.int16)
            # Resample if needed (24kHz to 16kHz)
            resampled = audio_data  # Add resampling if needed
            self.output_buffer = np.append(self.output_buffer, resampled)
            
    def process_input(self, input_data):
        # Process each sample through the NLMS filter
        processed = np.zeros_like(input_data)
        for i in range(len(input_data)):
            # Get reference from output buffer if available
            reference = self.output_buffer[0] if len(self.output_buffer) > 0 else 0
            processed[i] = self.nlms_filter.update(reference, input_data[i])
            
            # Remove used reference sample
            if len(self.output_buffer) > 0:
                self.output_buffer = self.output_buffer[1:]
                
        return processed

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
                # Store output audio for echo cancellation
                await audio_processor.process_output(chunk)
                # Play audio
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
                # Apply echo cancellation
                processed_data = audio_processor.process_input(data.flatten())
                # Convert back to bytes and base64 encode
                # Lighter gain control
                processed_data = np.clip(processed_data * 1.2, -32768, 32767)  # Reduced gain
                # Very light noise gate
                noise_gate = np.abs(processed_data) > 50  # Much lower threshold
                processed_data = processed_data * noise_gate
                processed_bytes = processed_data.astype(np.int16).tobytes()
                b64_chunk = base64.b64encode(processed_bytes).decode("utf-8")
                await conn.input_audio_buffer.append(audio=b64_chunk)

    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview",
    ) as conn:
        print("Connected to Realtime. Setting up echo cancellation...")
        await conn.session.update(session={
            "turn_detection": {"type": "server_vad"},
            "voice": "alloy"
        })

        playback_task = asyncio.create_task(playback_audio())
        mic_task = asyncio.create_task(stream_microphone(conn))

        print("Now streaming with echo cancellation. Speak, pause, and wait for response.\n")

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
                    await playback_queue.put(base64.b64decode(event.delta))
                    
                elif event.type == "response.done":
                    print("\n[Assistant finished responding]\n")
                    
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