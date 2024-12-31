from Cocoa import (
    NSApplication, NSObject,
    NSStatusBar, NSVariableStatusItemLength,
    NSMenu, NSMenuItem, NSApp,
    NSUserNotification, NSUserNotificationCenter
)
import objc
import subprocess
import os
from datetime import datetime
import threading
from openai import OpenAI

class AppDelegate(NSObject):
    def init(self):
        """Initialize the app delegate, create the status item, set up menu, 
        and prepare for recording and transcription."""
        self = objc.super(AppDelegate, self).init()
        if self is None:
            return None

        # Initialize state variables
        self.recording = False
        self.recording_process = None
        self._current_recording = None
        self.output_dir = os.path.expanduser("~/Documents/AudioNotes")
        self.obsidian_vault = os.path.expanduser("~/Documents/projects/")
        self.client = OpenAI()  # Initialize OpenAI client (adjust as needed)

        # Create necessary directories
        for directory in [self.output_dir, self.obsidian_vault]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Create Status Item
        statusbar = NSStatusBar.systemStatusBar()
        self.statusitem = statusbar.statusItemWithLength_(NSVariableStatusItemLength)

        # Set initial title
        self.statusitem.setTitle_(" ● ")
        self.statusitem.setAction_("statusItemClicked:")
        self.statusitem.setTarget_(self)

        # Create menu
        self.menu = NSMenu.alloc().init()
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit", "terminate:", ""
        )
        self.menu.addItem_(quit_item)

        return self

    def statusItemClicked_(self, sender):
        """Handle status item clicks (left-click toggles recording, right-click opens menu)."""
        event = NSApp.currentEvent()
        if event.type() == 3:  # Right mouse down
            # Show the menu at the status item location
            self.statusitem.popUpStatusItemMenu_(self.menu)
        else:
            # Left click => toggle recording
            if not self.recording:
                self.startRecording()
            else:
                self.stopRecording()

    @objc.python_method
    def startRecording(self):
        """Start audio recording and update status item."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_recording = os.path.join(self.output_dir, f"recording_{timestamp}.wav")

        # Start sox recording
        self.recording_process = subprocess.Popen([
            "sox", "-d", self._current_recording,
        ])

        self.recording = True
        self.statusitem.setTitle_("[ ● ]")  # Show a recording indicator

    @objc.python_method
    def stopRecording(self):
        """Stop the recording process and queue transcription."""
        if self.recording_process:
            self.recording_process.terminate()
            self.recording_process.wait()
            self.recording_process = None

        self.recording = False
        self.statusitem.setTitle_(" ● ")  # Revert the status item title

        # Process the recording in a background thread
        recording_path = self._current_recording
        threading.Thread(target=self._process_recording, args=(recording_path,)).start()

    @objc.python_method
    def _process_recording(self, recording_path):
        """Process the audio file using OpenAI's Whisper model."""
        try:
            with open(recording_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            # Create markdown content
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            markdown_content = f"""# Audio Note {timestamp}
{transcription.text}
---
Created: {timestamp}
Source: Audio Recording
"""

            # Save markdown file
            md_filename = f"audio_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            md_path = os.path.join(self.obsidian_vault, md_filename)
            with open(md_path, 'w') as f:
                f.write(markdown_content)

            # Show success notification
            self._show_notification(
                'Transcription Complete',
                'Audio note has been created',
                f'Saved as {md_filename}'
            )

        except Exception as e:
            # Show error notification
            self._show_notification(
                'Error',
                'Failed to process recording',
                str(e)
            )

    @objc.python_method
    def _show_notification(self, title, subtitle, message):
        """Show a user notification with the given title, subtitle, and message."""
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
