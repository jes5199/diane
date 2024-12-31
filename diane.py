from Foundation import *
from AppKit import *
import objc
import subprocess
import os
from datetime import datetime
import threading
from openai import OpenAI
from pathlib import Path

class AppDelegate(NSObject):
    def init(self):
        self = objc.super(AppDelegate, self).init()
        if self is None: return None
        
        self.recording = False
        self.recording_process = None
        self.output_dir = os.path.expanduser("~/Documents/AudioNotes")
        self.obsidian_vault = os.path.expanduser("~/Documents/projects/")
        self._current_recording = None

        # Initialize OpenAI client
        self.client = OpenAI()

        # Create output directories
        for directory in [self.output_dir, self.obsidian_vault]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Create status item
        statusbar = NSStatusBar.systemStatusBar()
        self.statusitem = statusbar.statusItemWithLength_(NSSquareStatusItemLength)
        
        # Create button
        self.statusitem.button().setTitle_(" ● ")
        
        # Set up menu
        self.menu = NSMenu.alloc().init()
        menuitem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit", "terminate:", "")
        self.menu.addItem_(menuitem)
        self.statusitem.setMenu_(self.menu)

        # Set up click handling
        self.statusitem.button().setAction_("handleClick:")
        self.statusitem.button().setTarget_(self)

        return self

    def handleClick_(self, sender):
        if not self.recording:
            self.startRecording()
        else:
            self.stopRecording()

    @objc.python_method
    def startRecording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_recording = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
        
        # Start sox recording
        self.recording_process = subprocess.Popen([
            "sox", "-d", self._current_recording,
        ])
        
        self.recording = True
        self.statusitem.button().setTitle_("[ ● ]")

    @objc.python_method
    def stopRecording(self):
        if self.recording_process:
            self.recording_process.terminate()
            self.recording_process.wait()
            self.recording_process = None
        
        self.recording = False
        self.statusitem.button().setTitle_(" ● ")
        
        recording_path = self._current_recording
        threading.Thread(target=self._process_recording, args=(recording_path,)).start()

    @objc.python_method
    def _process_recording(self, recording_path):
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
            self._show_notification('Transcription Complete', 
                                'Audio note has been created',
                                f'Saved as {markdown_filename}')
            
        except Exception as e:
            self._show_notification('Error',
                                'Failed to process recording',
                                str(e))

    @objc.python_method
    def _show_notification(self, title, subtitle, message):
        notification = NSUserNotification.alloc().init()
        notification.setTitle_(title)
        notification.setSubtitle_(subtitle)
        notification.setInformativeText_(message)
        
        NSUserNotificationCenter.defaultUserNotificationCenter().deliverNotification_(notification)

def main():
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()

if __name__ == "__main__":
    main()