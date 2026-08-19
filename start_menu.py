"""Interactive console start menu: collects the viewer's profile, lets the
user pick one video / several videos / a whole folder to run the gaze
analysis on, and exposes a few AppConfig toggles - then drives
PoseGazeApplication for each selected video in turn.

Navigation throughout: arrow keys + Enter to choose, Escape to go back.
Video/folder selection opens native Windows file-picker dialogs (see
file_dialogs.py) rather than asking for a typed path.
"""

import os

import tui
import file_dialogs
from config import AppConfig
from viewer_profile import collect_viewer_profile
from pose_gaze_application import PoseGazeApplication
from eye_gaze_tracker import NoGazeTrackerError

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v', '.webm'}

# (AppConfig attribute, menu label) - toggled on/off from the settings submenu.
SETTINGS_TOGGLES = (
    ("GAZE_SIMULATION_ENABLED", "Симуляция взгляда мышью (без Tobii)"),
    ("VLM_ENABLED", "VLM-анализ 'что привлекло взгляд'"),
    ("IDENTITY_RECONCILE_ENABLED", "Сверка личностей после просмотра"),
    ("DEMOGRAPHICS_ENABLED", "Оценка возраста/пола/телосложения"),
)


class StartMenu:
    def __init__(self):
        self.app_config = AppConfig()
        self.viewer_profile = None

    def run(self):
        while True:
            profile_state = "заполнено" if self.viewer_profile else "не заполнено"
            choice = tui.select_menu(
                "Стартовое меню",
                [
                    (f"Данные о зрителе ({profile_state})", "profile"),
                    ("Выбрать видео и запустить анализ", "video"),
                    ("Настройки", "settings"),
                    ("Выход", "exit"),
                ],
                show_result=False,
            )

            if choice in (tui.CANCELLED, "exit"):
                print("Выход из программы.")
                return
            if choice == "profile":
                self.viewer_profile = collect_viewer_profile(existing=self.viewer_profile)
            elif choice == "video":
                self._video_menu()
            elif choice == "settings":
                self._settings_menu()

    # --- Video selection -----------------------------------------------------

    def _video_menu(self):
        choice = tui.select_menu(
            "Выбор видео",
            [
                ("Один видеофайл", "single"),
                ("Несколько видеофайлов", "multiple"),
                ("Папка с видео (все файлы по очереди)", "folder"),
                ("Назад", "back"),
            ],
            show_result=False,
        )

        if choice in (tui.CANCELLED, "back"):
            return
        if choice == "single":
            self._run_single_file()
        elif choice == "multiple":
            self._run_multiple_files()
        elif choice == "folder":
            self._run_folder()

    def _run_single_file(self):
        path = file_dialogs.pick_video_file()
        if not path:
            print("Выбор файла отменён.")
            return
        self._run_playlist([path])

    def _run_multiple_files(self):
        paths = file_dialogs.pick_video_files()
        if not paths:
            print("Выбор файлов отменён.")
            return
        self._run_playlist(list(paths))

    def _run_folder(self):
        folder = file_dialogs.pick_folder()
        if not folder:
            print("Выбор папки отменён.")
            return
        files = sorted(
            os.path.join(folder, name) for name in os.listdir(folder)
            if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
        )
        if not files:
            print("В указанной папке не найдено видеофайлов.")
            return
        print(f"Найдено видео в папке: {len(files)}")
        self._run_playlist(files)

    def _run_playlist(self, video_paths):
        if not self.viewer_profile:
            print("\nДанные о зрителе ещё не заполнены.")
            answer = tui.confirm("Заполнить их сейчас?", default=True)
            if answer is True:
                self.viewer_profile = collect_viewer_profile()

        for i, video_path in enumerate(video_paths, start=1):
            print(f"\n[{i}/{len(video_paths)}] Видео: {video_path}")
            decision = tui.select_menu(
                f"Готовы начать просмотр '{os.path.basename(video_path)}'?",
                [("Начать", "start"), ("Пропустить это видео", "skip")],
                show_result=False,
            )

            if decision in (tui.CANCELLED,):
                print("Возврат в меню.")
                return
            if decision == "skip":
                print("Видео пропущено.")
                continue

            try:
                app = PoseGazeApplication(viewer_profile=self.viewer_profile)
                app.run(video_path)
            except NoGazeTrackerError as e:
                print(f"Ошибка: {e}")
                print("Возврат в меню.")
                return

        print("\nВсе выбранные видео обработаны.")

    # --- Settings --------------------------------------------------------------

    def _settings_menu(self):
        while True:
            items = [
                (f"[{'Вкл' if getattr(self.app_config, attr) else 'Выкл'}] {label}", attr)
                for attr, label in SETTINGS_TOGGLES
            ]
            items.append(("Назад", "back"))

            choice = tui.select_menu("Настройки", items, show_result=False)
            if choice in (tui.CANCELLED, "back"):
                return

            new_value = not getattr(self.app_config, choice)
            setattr(self.app_config, choice, new_value)
