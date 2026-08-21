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
import webcam
from config import AppConfig
from database import Database
from viewer_profile import collect_viewer_profile, display_profile, PROFILE_FIELDS
from pose_gaze_application import PoseGazeApplication
from video_precomputer import VideoPrecomputer
from eye_gaze_tracker import NoGazeTrackerError

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v', '.webm'}

# (AppConfig attribute, menu label) - toggled on/off from the settings submenu.
SETTINGS_TOGGLES = (
    ("GAZE_SIMULATION_ENABLED", "Симуляция взгляда мышью (без Tobii)"),
    ("VLM_ENABLED", "VLM-анализ 'что привлекло взгляд'"),
    ("IDENTITY_RECONCILE_ENABLED", "Сверка личностей после просмотра"),
    ("DEMOGRAPHICS_ENABLED", "Оценка возраста/пола/телосложения"),
    ("EMOTION_TRACKING_ENABLED", "Трекинг эмоций через веб-камеру"),
    ("FORCE_RECOMPUTE_VIDEO_ANALYSIS", "Игнорировать кеш БД (пересчитывать анализ видео заново)"),
)


class StartMenu:
    def __init__(self):
        self.app_config = AppConfig()
        self.viewer_profile = None
        # Owned for the whole program's lifetime and shared with every
        # PoseGazeApplication run (see _run_playlist) and every DB-backed
        # menu (see _select_existing_user), instead of each opening its own
        # connection and re-running schema setup.
        self.db = Database()

    def run(self):
        try:
            while True:
                profile_state = "заполнено" if self.viewer_profile else "не заполнено"
                choice = tui.select_menu(
                    "Стартовое меню",
                    [
                        (f"Данные о зрителе ({profile_state})", "profile"),
                        ("Выбрать видео и запустить анализ", "video"),
                        ("Предпросчёт кеша видео (высокая точность)", "precompute"),
                        ("Настройки", "settings"),
                        ("Выход", "exit"),
                    ],
                    show_result=False,
                )

                if choice in (tui.CANCELLED, "exit"):
                    print("Выход из программы.")
                    return
                if choice == "profile":
                    self._profile_menu()
                elif choice == "video":
                    self._video_menu()
                elif choice == "precompute":
                    self._precompute_menu()
                elif choice == "settings":
                    self._settings_menu()
        finally:
            self.db.close()

    # --- Viewer profile --------------------------------------------------------

    def _profile_menu(self):
        choice = tui.select_menu(
            "Данные о зрителе",
            [
                ("Выбрать из уже сохранённых в базе", "select_existing"),
                ("Заполнить/изменить вручную", "fill"),
                ("Назад", "back"),
            ],
            show_result=False,
        )

        if choice in (tui.CANCELLED, "back"):
            return
        if choice == "select_existing":
            self._select_existing_user()
        elif choice == "fill":
            self.viewer_profile = collect_viewer_profile(existing=self.viewer_profile)

    def _select_existing_user(self):
        users = self.db.list_users()

        if not users:
            print("\nВ базе данных пока нет сохранённых пользователей.")
            return

        items = [(self._format_user_label(u), u) for u in users]
        items.append(("Назад", None))
        choice = tui.select_menu("Выберите пользователя", items, show_result=False)

        if choice in (tui.CANCELLED, None):
            return

        self.viewer_profile = {field: choice[field] for field in PROFILE_FIELDS}
        print("\nВыбран пользователь:")
        display_profile(self.viewer_profile)

    @staticmethod
    def _format_user_label(user):
        # Includes every profile field (not just a few) plus the creation
        # date, so two stored users differing only in one field - e.g. the
        # same person re-entered with updated dance experience - still show
        # up as distinguishable rows instead of identical-looking entries.
        details = ", ".join(
            str(value) for value in (
                user["gender"], user["age"], user["occupation"], user["dance_experience"]
            ) if value not in (None, "")
        )
        label = f"{user['full_name']} ({details})" if details else user["full_name"]
        created_date = (user["created_at"] or "")[:10]
        return f"{label} - {created_date}" if created_date else label

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
            answer = tui.confirm("Указать их сейчас?", default=True)
            if answer is True:
                self._profile_menu()

        if self.app_config.EMOTION_TRACKING_ENABLED and self.app_config.WEBCAM_INDEX is None:
            print("\nТрекинг эмоций через веб-камеру включён, но камера ещё не выбрана.")
            answer = tui.confirm("Выбрать веб-камеру сейчас?", default=True)
            if answer is True:
                self._select_webcam()

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
                app = PoseGazeApplication(viewer_profile=self.viewer_profile, db=self.db)
                app.run(video_path)
            except NoGazeTrackerError as e:
                print(f"Ошибка: {e}")
                print("Возврат в меню.")
                return

        print("\nВсе выбранные видео обработаны.")

    # --- Cache precompute (no live viewer/player) -------------------------------

    def _precompute_menu(self):
        choice = tui.select_menu(
            "Предпросчёт кеша видео",
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
            path = file_dialogs.pick_video_file()
            if not path:
                print("Выбор файла отменён.")
                return
            self._run_precompute_playlist([path])
        elif choice == "multiple":
            paths = file_dialogs.pick_video_files()
            if not paths:
                print("Выбор файлов отменён.")
                return
            self._run_precompute_playlist(list(paths))
        elif choice == "folder":
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
            self._run_precompute_playlist(files)

    def _run_precompute_playlist(self, video_paths):
        """
        Runs VideoPrecomputer.precompute() over each video in turn - no
        viewer profile, no video player window, always overwrites whatever
        cache already exists for a video (this mode's whole point is a
        deliberate, higher-quality recompute). A fresh VideoPrecomputer (and
        the PoseEstimator/PersonIdentityReconciler/PersonDemographicsEstimator
        it owns) is constructed per video - same reason PoseGazeApplication
        is constructed fresh per video in _run_playlist: their internal
        per-track-id state (and YOLO's persist=True tracker state) must not
        leak from one video into the next.
        """
        print(
            "\nПредпросчёт может занять очень много времени - особенно этап исчерпывающей сверки "
            "личностей (сравнение всех фрагментов ID через локальную VLM, минуты на запрос). "
            "Рекомендуется запускать без присмотра (например, на ночь). Кеш видео будет пересчитан "
            "заново в повышенном качестве, даже если для него уже что-то закешировано ранее."
        )
        if tui.confirm("Начать предпросчёт?", default=True) is not True:
            print("Предпросчёт отменён.")
            return

        for i, video_path in enumerate(video_paths, start=1):
            print(f"\n[{i}/{len(video_paths)}] Видео: {video_path}")
            try:
                precomputer = VideoPrecomputer(db=self.db)
                summary = precomputer.precompute(video_path)
            except KeyboardInterrupt:
                print("\nПредпросчёт прерван пользователем - обработка списка остановлена.")
                return
            except Exception as e:
                print(f"Ошибка при предпросчёте видео '{video_path}': {e}")
                print("Переход к следующему видео.")
                continue

            if summary:
                reconciled = "завершена" if summary['identity_reconciled'] else "не завершена"
                print(f"Готово: {summary['sample_count']} сэмплов позы, сверка личностей {reconciled}, "
                      f"демография определена для {summary['demographics_person_count']} человек(а).")

        print("\nПредпросчёт всех выбранных видео завершён.")

    # --- Settings --------------------------------------------------------------

    def _settings_menu(self):
        while True:
            items = [
                (f"[{'Вкл' if getattr(self.app_config, attr) else 'Выкл'}] {label}", attr)
                for attr, label in SETTINGS_TOGGLES
            ]
            items.append((f"Веб-камера (сейчас: {self._webcam_label()})", "webcam"))
            items.append(("Просмотр камеры и её настройки (баланс белого, яркость...)", "preview_webcam"))
            items.append(("Назад", "back"))

            choice = tui.select_menu("Настройки", items, show_result=False)
            if choice in (tui.CANCELLED, "back"):
                return
            if choice == "webcam":
                self._select_webcam()
                continue
            if choice == "preview_webcam":
                self._preview_webcam()
                continue

            new_value = not getattr(self.app_config, choice)
            setattr(self.app_config, choice, new_value)

    def _webcam_label(self):
        index = self.app_config.WEBCAM_INDEX
        return f"#{index}" if index is not None else "не выбрана"

    def _preview_webcam(self):
        index = self.app_config.WEBCAM_INDEX
        if index is None:
            print("\nСначала нужно выбрать веб-камеру.")
            self._select_webcam()
            index = self.app_config.WEBCAM_INDEX
            if index is None:
                return

        print(f"\nОткрывается окно предпросмотра камеры #{index}.")
        print("В окне предпросмотра: Esc/Q - закрыть, S - открыть настройки камеры "
              "(баланс белого, яркость, экспозиция и т.п. - если камера их поддерживает).")

        if not webcam.preview_camera(index):
            print(f"Не удалось открыть камеру #{index}.")

    def _select_webcam(self):
        print("\nПоиск подключённых веб-камер...")
        cameras = webcam.list_available_cameras()
        if not cameras:
            print("Веб-камеры не найдены.")

        items = [
            (f"Камера #{cam['index']} ({cam['width']}x{cam['height']})", cam['index'])
            for cam in cameras
        ]
        items.append(("Не использовать веб-камеру (отключить трекинг эмоций)", "none"))
        items.append(("Назад", "back"))

        choice = tui.select_menu("Выберите веб-камеру", items, show_result=False)
        if choice in (tui.CANCELLED, "back"):
            return

        self.app_config.WEBCAM_INDEX = None if choice == "none" else choice
        print(f"Веб-камера установлена: {self._webcam_label()}")
