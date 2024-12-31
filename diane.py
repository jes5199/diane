from Foundation import *
from AppKit import *
import subprocess
import os
from datetime import datetime
import threading
from openai import OpenAI
from pathlib import Path

class StatusBarController(NSObject):
    def init(self):
        self = super(StatusBarController, self).init()
        self.recording = False
        self.recording_process = None
        self.output_dir = os.path.expanduser("~/Documents/AudioNotes")
        self.obsidian_vault = os.path.expanduser("~/Documents/projects/")

        # Initialize OpenAI client
        self.client = OpenAI()

        # Create output directories
        for directory in [self.output_dir, self.obsidian_vault]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Create the statusbar item
        self.statusitem = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        
        # Set initial title
        self.statusitem.setTitle_(" ● ")
        
        # Create the menu (empty for now, just handle clicks)
        self.menu = NSMenu.alloc().init()
        
        # Set target for click events
        self.statusitem.setTarget_(self)
        self.statusitem.setAction_('toggleRecording:')
        self.statusitem.setMenu_(self.menu)

        return self

    def toggleRecording_(self, sender):
        if not self.recording:
            self.startRecording()
        else:
            self.stopRecording()

    def startRecording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_recording = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
        
        # Start sox recording
        self.recording_process = subprocess.Popen([
            "sox", "-d", self.current_recording,
        ])
        
        self.recording = True
        self.statusitem.setTitle_("[ ● ]")

    def stopRecording(self):
        if self.recording_process:
            self.recording_process.terminate()
            self.recording_process.wait()
            self.recording_process = None
        
        self.recording = False
        self.statusitem.setTitle_(" ● ")
        
        # Process in background
        threading.Thread(target=self.processRecording, args=(self.current_recording,)).start()

    def processRecording(self, recording_path):
        try:
            # Transcribe
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
            
            markdown_filename = f"audio_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            markdown_path = os.path.join(self.obsidian_vault, markdown_filename)
            
            with open(markdown_path, 'w') as f:
                f.write(markdown_content)
            
            # Show notification
            NSUserNotification = objc.lookUpClass('NSUserNotification')
            NSUserNotificationCenter = objc.lookUpClass('NSUserNotificationCenter')
            
            notification = NSUserNotification.alloc().init()
            notification.setTitle_('Transcription Complete')
            notification.setSubtitle_('Audio note has been created')
            notification.setInformativeText_(f'Saved as {markdown_filename}')
            
            center = NSUserNotificationCenter.defaultUserNotificationCenter()
            center.deliverNotification_(notification)
            
        except Exception as e:
            # Show error notification
            notification = NSUserNotification.alloc().init()
            notification.setTitle_('Error')
            notification.setSubtitle_('Failed to process recording')
            notification.setInformativeText_(str(e))
            
            center = NSUserNotificationCenter.defaultUserNotificationCenter()
            center.deliverNotification_(notification)

def main():
    app = NSApplication.sharedApplication()
    status_bar = StatusBarController.alloc().init()
    app.run()

if __name__ == "__main__":
    main()