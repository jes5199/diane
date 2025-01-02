import asyncio
import base64
import sys
import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI
from hybrid_filter import HybridFilter
import json

# Audio settings
INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
READ_SIZE_SEC = 0.02
READ_SIZE_FRAMES = int(INPUT_SAMPLE_RATE * READ_SIZE_SEC)
OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHANNELS = 1
OUTPUT_DTYPE = "int16"

# Add this near the top of the file, after imports
def calculator(operation, a, b):
    """Calculator function that matches the tool definition"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else "Error: Division by zero"

# Add this before the realtime_demo function
FUNCTION_REGISTRY = {
    "calculator": calculator
}

class AudioProcessor:
    def __init__(self):
        self.filter = HybridFilter(filter_length=96)
        self.output_buffer = np.array([], dtype=np.float32)
        self.output_lock = asyncio.Lock()
        self.is_speaking = False  # Track if assistant is speaking
        
    def reset_state(self):
        """Reset the filter state"""
        self.filter.reset()
        self.output_buffer = np.array([], dtype=np.float32)
        
    async def process_output(self, raw_audio):
        async with self.output_lock:
            audio_data = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
            self.output_buffer = np.append(self.output_buffer, audio_data)
            if len(self.output_buffer) > 4096:
                self.output_buffer = self.output_buffer[-4096:]
            
    def process_input(self, input_data):
        #return np.clip(input_data * 1.5, -32768, 32767).astype(np.int16)
        return input_data
        # Just boost volume if assistant isn't speaking
        if not self.is_speaking:
            return np.clip(input_data * 1.5, -32768, 32767).astype(np.int16)
            
        # Apply echo cancellation only when assistant is speaking
        input_float = input_data.astype(np.float32)
        if len(self.output_buffer) >= len(input_float):
            reference = self.output_buffer[-len(input_float):]
            processed_data = self.filter.process(input_float, reference)
            return np.clip(processed_data * 1.5, -32768, 32767).astype(np.int16)
        
        return np.clip(input_data * 1.5, -32768, 32767).astype(np.int16)

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
        model="gpt-4o-mini-realtime-preview-2024-12-17",
    ) as conn:
        print("Connected to Realtime. Setting up conditional echo cancellation...")

        tools = [
            {
                "type": "function",
                "name": "calculator",
                "description": "A calculator that can add, subtract, multiply, and divide",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                },
            }
        ]

        await conn.session.update(session={
            "turn_detection": {"type": "server_vad"},
            "modalities": ["text", "audio"],
            "voice": "alloy",
            "tools": tools,
            "tool_choice": "auto",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
        })

        playback_task = asyncio.create_task(playback_audio())
        mic_task = asyncio.create_task(stream_microphone(conn))

        print("Now streaming. Speak, pause, and wait for response.\n")

        acc_text = {}
        try:
            async for event in conn:
                if event.type == "response.audio_transcript.delta":
                    item_id = event.item_id
                    acc_text[item_id] = acc_text.get(item_id, "") + event.delta
                    sys.stdout.write(f"{event.delta}")
                    sys.stdout.flush()
                    
                elif event.type == "response.audio_transcript.done":
                    item_id = event.item_id
                    print(f"\n[Final transcript] {acc_text.get(item_id, '')}")
                    acc_text.pop(item_id, None)
                    
                elif event.type == "response.audio.delta":
                    # Assistant is starting to speak
                    audio_processor.is_speaking = True
                    raw_audio = base64.b64decode(event.delta)
                    await playback_queue.put(raw_audio)
                    
                elif event.type == "conversation.item.input_audio_transcription.completed":
                    # Print the transcribed user input
                    print(f"\n[User transcript] {event.transcript}")
                    
                elif event.type == "response.content_part.added":
                    pass
                elif event.type == "response.content_part.done":
                    pass
                elif event.type == "response.output_item.done":
                    pass
                elif event.type == "conversation.item.created":
                    pass
                elif event.type == "response.output_item.added":
                    pass
                elif event.type == "response.function_call_arguments.delta":
                    pass
                elif event.type == "response.function_call_arguments.done":
                    pass
                elif event.type == "input_audio_buffer.committed":
                    pass
                elif event.type == "response.created":
                    pass
                elif event.type == "input_audio_buffer.speech_started":
                    print("listening")
                elif event.type == "input_audio_buffer.speech_stopped":
                    print("ok")
                elif event.type == "response.audio.done":
                    pass
                elif event.type == "response.done":
                    # Check if response contains function calls
                    if hasattr(event, 'response') and event.response.output:
                        #print(f"Response: {event.response.output}")
                        for item in event.response.output:
                            if item.type == "function_call":
                                print(f"\n[Function Call] {item.name}")
                                args = json.loads(item.arguments)
                                print(f"Arguments: {json.dumps(args, indent=2)}")
                                
                                # Execute the function if it exists in the registry
                                if item.name in FUNCTION_REGISTRY:
                                    try:
                                        result = FUNCTION_REGISTRY[item.name](**args)
                                        print(f"Result: {result}")
                                        
                                        # Send the result back to the model
                                        await conn.conversation.item.create(item={
                                            "type": "function_call_output",
                                            "call_id": item.call_id,
                                            "output": json.dumps({"result": result})
                                        })

                                        # then send a text message to the model
                                        await conn.response.create(response={
                                            "instructions": "read the result",
                                        })
                                        
                                    except Exception as e:
                                        print(f"Error executing function: {e}")
                                        # Send error back to the model
                                        await conn.conversation.item.create(item={
                                            "type": "function_call_output",
                                            "call_id": item.call_id,
                                            "output": json.dumps({"error": str(e)})
                                        })
                                else:
                                    print(f"Unknown function: {item.name}")
                                    # Send error for unknown function
                                    await conn.conversation.item.create(item={
                                        "type": "function_call_output",
                                        "call_id": item.call_id,
                                        "output": json.dumps({
                                            "error": f"Unknown function: {item.name}"
                                        })
                                    })
                                
                    print("\n[Assistant finished responding]\n")
                    audio_processor.reset_state()
                else:
                    print(f"Unknown event type: {event.type}")
                    print(f"Event: {event}")

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