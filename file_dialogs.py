"""Native Windows file/folder picker dialogs (tkinter.filedialog), used by
the start menu's video-selection options instead of typed-in paths.
"""

import os
import tkinter as tk
from tkinter import filedialog

_VIDEO_FILETYPES = (
    ("Видео файлы", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v *.webm"),
    ("Все файлы", "*.*"),
)


def _new_root():
    """A hidden, always-on-top Tk root, so the native dialog reliably pops
    up in front of the console window instead of opening behind it."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def pick_video_file(initialdir=None):
    """Opens the native 'Open File' dialog for a single video.
    Returns the chosen path, or None if the user cancelled."""
    root = _new_root()
    try:
        path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=_VIDEO_FILETYPES,
            initialdir=initialdir or os.getcwd(),
            parent=root,
        )
    finally:
        root.destroy()
    return path or None


def pick_video_files(initialdir=None):
    """Opens the native 'Open Files' dialog with multi-selection enabled.
    Returns a list of chosen paths (empty if the user cancelled)."""
    root = _new_root()
    try:
        paths = filedialog.askopenfilenames(
            title="Выберите видеофайлы",
            filetypes=_VIDEO_FILETYPES,
            initialdir=initialdir or os.getcwd(),
            parent=root,
        )
    finally:
        root.destroy()
    return list(paths)


def pick_folder(initialdir=None):
    """Opens the native 'Select Folder' dialog.
    Returns the chosen folder path, or None if the user cancelled."""
    root = _new_root()
    try:
        path = filedialog.askdirectory(
            title="Выберите папку с видео",
            initialdir=initialdir or os.getcwd(),
            parent=root,
        )
    finally:
        root.destroy()
    return path or None
