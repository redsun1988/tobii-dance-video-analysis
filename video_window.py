import os
import time
import ctypes
from typing import Optional, Tuple

import win32con
import win32gui


def enable_dpi_awareness() -> None:
    """Makes this process DPI-aware so window rects, screen grabs and mouse
    coordinates all agree on physical pixels on HiDPI/multi-monitor setups.

    Without this, Windows silently virtualizes coordinates for non-DPI-aware
    processes, which would throw off the absolute-desktop <-> video-window
    coordinate math the whole gaze mapping depends on.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class WindowNotFoundError(RuntimeError):
    """Raised when the video player window can't be located on screen."""


class VideoPlayerWindow:
    """Launches a video file with the OS-default player and tracks its
    window, so absolute (desktop) gaze coordinates can be converted into the
    player's client-area coordinate space.
    """

    def __init__(self, min_size_px: Tuple[int, int] = (200, 150)):
        self.hwnd: Optional[int] = None
        self._min_w, self._min_h = min_size_px

    @staticmethod
    def _visible_windows() -> set:
        handles = set()

        def _collect(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                handles.add(hwnd)

        win32gui.EnumWindows(_collect, None)
        return handles

    @staticmethod
    def _window_rect_size(hwnd) -> Tuple[int, int]:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        return max(0, right - left), max(0, bottom - top)

    def _is_plausible_player_window(self, hwnd) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return False
        if not win32gui.GetWindowText(hwnd):
            return False
        width, height = self._window_rect_size(hwnd)
        return width >= self._min_w and height >= self._min_h

    def launch_and_locate(self, video_path: str, timeout: float = 15.0, poll_interval: float = 0.3) -> int:
        """Starts the video with the OS-default handler and returns the hwnd
        of the window that most plausibly is the player.

        There is no reliable, player-agnostic way to know which window a
        `start`-ed file will open (name/title depend entirely on whichever
        app is registered for the file type), so this works by diffing the
        set of visible top-level windows before/after launch and preferring
        whichever newly-created window took the foreground (media players
        grab focus on open); if none did, it falls back to the largest
        newly-created window.
        """
        before = self._visible_windows()
        os.startfile(video_path)

        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(poll_interval)
            after = self._visible_windows()
            candidates = [h for h in (after - before) if self._is_plausible_player_window(h)]
            if not candidates:
                continue

            foreground = win32gui.GetForegroundWindow()
            if foreground in candidates:
                chosen = foreground
            else:
                chosen = max(candidates, key=lambda h: self._window_rect_size(h)[0] * self._window_rect_size(h)[1])

            self.hwnd = chosen
            win32gui.ShowWindow(chosen, win32con.SW_MAXIMIZE)
            return chosen

        raise WindowNotFoundError(
            f"Could not locate the video player window within {timeout:.0f}s after launching '{video_path}'."
        )

    def is_alive(self) -> bool:
        return self.hwnd is not None and win32gui.IsWindow(self.hwnd)

    def get_client_bbox_absolute(self) -> Tuple[int, int, int, int]:
        """Returns (left, top, right, bottom) of the window's client area -
        the actual video content, excluding title bar/borders - in absolute
        screen coordinates.
        """
        if not self.is_alive():
            raise WindowNotFoundError("Video player window is no longer available.")
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        left, top = win32gui.ClientToScreen(self.hwnd, (left, top))
        right, bottom = win32gui.ClientToScreen(self.hwnd, (right, bottom))
        return left, top, right, bottom

    def get_title(self) -> str:
        return win32gui.GetWindowText(self.hwnd) if self.hwnd else ""
