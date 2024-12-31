from Cocoa import (
    NSApplication, NSObject,
    NSStatusBar, NSVariableStatusItemLength,
    NSMenu, NSMenuItem, NSRect,
    NSView, NSApp
)
import objc
import subprocess
import os
from datetime import datetime
import threading
from openai import OpenAI

# --------------------
# Custom NSView
# --------------------
import sys
from Cocoa import (
    NSColor, NSFont, NSAttributedString,
    NSFontAttributeName, NSForegroundColorAttributeName
)

class StatusView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(StatusView, self).initWithFrame_(frame)
        if self is None:
            return None
        
        self.title = " ● "  # default text
        self.delegate = None  # We'll set this to our AppDelegate
        return self

    def drawRect_(self, rect):
        """Draw the title string in the center of the view."""
        objc.super(StatusView, self).drawRect_(rect)

        attributes = {
            NSFontAttributeName: NSFont.systemFontOfSize_(13),
            NSForegroundColorAttributeName: NSColor.labelColor()
        }
        attr_str = NSAttributedString.alloc().initWithString_attributes_(
            self.title,
            attributes
        )

        str_size = attr_str.size()
        view_width, view_height = self.bounds().size.width, self.bounds().size.height
        x = (view_width - str_size.width) / 2
        y = (view_height - str_size.height) / 2

        attr_str.drawAtPoint_((x, y))

    # Left mouse click => toggle recording
    def mouseDown_(self, event):
        if self.delegate:
            self.delegate.toggleRecording()

    # Right mouse click => show menu
    def rightMouseDown_(self, event):
        if self.delegate and self.delegate.menu:
            self.delegate.statusitem.popUpStatusItemMenu_(self.delegate.menu)

# --------------------
# AppDelegate
# --------------------
class AppDelegate(NSObject):
    def init(self):
        self = objc.super(AppDelegate, self).init()
        if self is None:
            return None

        # State
        self.recording = False
        self.recording_process = None
        self._current_recording = None

        # Output + OpenAI
        self.output_dir = os.path.expanduser("~/Documents/AudioNotes")
        self.obsidian_vault = os.path.expanduser("~/Documents/projects/")
        self.client = OpenAI()

        for d in [self.output_dir, self.obsidian_vault]:
            os.makedirs(d, exist_ok=True)

        # Create the status item
        statusbar = NSStatusBar.systemStatusBar()
        self.statusitem = statusbar.statusItemWithLength_(NSVariableStatusItemLength)

        # Create the custom view
        # Typically ~24–28 wide for the status bar
        frame = ((0, 0), (28, 24))
        self.view = StatusView.alloc().initWithFrame_(frame)
        self.view.delegate = self  # so it can call our toggleRecording, etc.
        self.statusitem.setView_(self.view)

        # Create a simple menu for right-click
        self.menu = NSMenu.alloc().init()
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit", "terminate:", ""
        )
        self.menu.addItem_(quit_item)

        return self

    @objc.python_method
    def toggleRecording(self):
        if not self.recording:
            self.startRecording()
        else:
            self.stopRecording()

    @objc.python_method
    def startRecording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_recording = os.path.join(self.output_dir, f"recording_{timestamp}.wav")

        self.recording_process = subprocess.Popen([
            "sox", "-d", self._current_recording,
        ])
        self.recording = True

        # Update the label to show a recording indicator
        self.view.title = "[ ● ]"
        self.view.setNeedsDisplay_(True)

    @objc.python_method
    def stopRecording(self):
        if self.recording_process:
            self.recording_process.terminate()
            self.recording_process.wait()
            self.recording_process = None

        self.recording = False

        # Revert the label
        self.view.title = " ● "
        self.view.setNeedsDisplay_(True)

        # Process in background
        threading.Thread(target=self._process_recording, args=(self._current_recording,)).start()

    @objc.python_method
    def _process_recording(self, path):
        """Transcribe, then save markdown."""
        try:
            with open(path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            md_fn = f"audio note {now_str}.md"
            md_path = os.path.join(self.obsidian_vault, md_fn)
            with open(md_path, "w") as md:
                md.write(f"{transcription.text}")

            print(f"Saved transcription to {md_fn}")
        except Exception as e:
            print("Error processing:", e)

def main():
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()

if __name__ == "__main__":
    main()
