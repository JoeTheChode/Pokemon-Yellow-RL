"""
Live watcher for Pokemon Yellow RL training.
Opens a window showing the latest saved screenshot every 0.5 seconds.
"""

import time
import tkinter as tk
from tkinter import Label

from PIL import Image, ImageTk, UnidentifiedImageError

from project_paths import SCREENSHOTS_DIR


def watch():
    if not SCREENSHOTS_DIR.exists():
        print(f"No screenshots yet. Waiting for {SCREENSHOTS_DIR}...")
        while not SCREENSHOTS_DIR.exists():
            time.sleep(1)

    root = tk.Tk()
    root.title("Pokemon Yellow RL - Live View")
    label = Label(root)
    label.pack()

    def update():
        files = sorted(SCREENSHOTS_DIR.glob("*.png"))
        if files:
            latest = files[-1]
            try:
                with Image.open(latest) as img:
                    frame = img.resize((480, 432), Image.NEAREST)
            except (FileNotFoundError, PermissionError, UnidentifiedImageError, OSError):
                root.after(500, update)
                return

            photo = ImageTk.PhotoImage(frame)
            label.configure(image=photo)
            label.image = photo
            root.title(f"RL View - {latest.name}")

        root.after(500, update)

    update()
    root.mainloop()


if __name__ == '__main__':
    watch()
