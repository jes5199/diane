from Cocoa import NSStatusBar, NSVariableStatusItemLength, NSApplication, NSObject, NSMenu, NSMenuItem
import objc
import subprocess
import os
from datetime import datetime
import threading
from openai import OpenAI
from pathlib import Path
from PyObjCTools import AppHelper

def ensure_main_thread(func):
    """Decorator to ensure a function is called on the main thread."""
    def wrapper(*args, **kwargs):
        AppHelper.callAfter(func, *args, **kwargs)
    return wrapper

class AppDelegate(NSObject):
    def init(self):
        print("1. Starting init...")
        self = objc.super(AppDelegate, self).init()
        print("2. Super init completed")
        
        if self is None:
            print("ERROR: Super init returned None!")
            return None

        try:
            NSApplication.sharedApplication()
            # Create a basic status item
            print("3. Getting system status bar...")
            statusbar = NSStatusBar.systemStatusBar()
            print("4. System status bar obtained")
            
            print("5. Creating status item...")
            self.statusitem = statusbar.statusItemWithLength_(NSVariableStatusItemLength)
            print("6. Status item created")
            
            print("7. Setting status item title...")
            self.statusitem.setTitle_("●")
            print("8. Status item title set")
            
            # Attach a simple menu
            print("9. Creating menu...")
            self.menu = NSMenu.alloc().init()
            print("10. Menu created")
            
            print("11. Creating quit menu item...")
            quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit", "terminate:", "")
            print("12. Quit menu item created")
            
            print("13. Adding quit item to menu...")
            self.menu.addItem_(quit_item)
            print("14. Quit item added")
            
            print("15. Setting menu to status item...")
            self.statusitem.setMenu_(self.menu)
            print("16. Menu set successfully")

            print("17. Init completed successfully!")
            return self
            
        except Exception as e:
            print(f"ERROR during initialization: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

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
        self.statusitem.setTitle_("[ ● ]")

    @objc.python_method
    def stopRecording(self):
        if self.recording_process:
            self.recording_process.terminate()
            self.recording_process.wait()
            self.recording_process = None
        
        self.recording = False
        self.statusitem.setTitle_(" ● ")
        
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