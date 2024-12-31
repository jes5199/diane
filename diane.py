import rumps
import subprocess
import os
from datetime import datetime
import threading
from openai import OpenAI
from pathlib import Path

class AudioRecorderApp(rumps.App):
    def __init__(self):
        super().__init__("REC")
        self.recording = False
        self.recording_process = None
        self.output_dir = os.path.expanduser("~/Documents/AudioNotes")
        self.obsidian_vault = os.path.expanduser("~/Documents/ObsidianVault")  # Change this to your vault path
        
        # Add menu items
        self.menu = [
            rumps.MenuItem("Toggle Recording", callback=self.toggle_recording, key='d', modifier=3),  # 3 is Cmd+Shift
        ]
        
        # Initialize OpenAI client
        self.client = OpenAI()  # Will use OPENAI_API_KEY environment variable
        
        # Create output directories if they don't exist
        for directory in [self.output_dir, self.obsidian_vault]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def toggle_recording(self, _):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_recording = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
        
        # Start sox recording in a separate process
        self.recording_process = subprocess.Popen([
            "sox", "-d", self.current_recording,
        ])
        
        self.recording = True
        self.title = "[ ● ]"

    def stop_recording(self):
        if self.recording_process:
            self.recording_process.terminate()
            self.recording_process.wait()
            self.recording_process = None
        
        self.recording = False
        self.title = "REC"
        
        # Process the recording in a separate thread to keep UI responsive
        threading.Thread(target=self.process_recording, args=(self.current_recording,)).start()

    def process_recording(self, recording_path):
        try:
            # Transcribe the audio
            with open(recording_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            # Create markdown file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            markdown_content = f"""# Audio Note {timestamp}

{transcription.text}

---
Created: {timestamp}
Source: Audio Recording
"""
            
            # Save markdown file in Obsidian vault
            markdown_filename = f"audio_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            markdown_path = os.path.join(self.obsidian_vault, markdown_filename)
            
            with open(markdown_path, 'w') as f:
                f.write(markdown_content)
            
            # Optional: Delete the audio file to save space
            # os.remove(recording_path)
            
            # Show notification
            rumps.notification(
                title="Transcription Complete",
                subtitle="Audio note has been created",
                message=f"Saved as {markdown_filename}"
            )
            
        except Exception as e:
            rumps.notification(
                title="Error",
                subtitle="Failed to process recording",
                message=str(e))

if __name__ == "__main__":
    app = AudioRecorderApp()
    app.run()