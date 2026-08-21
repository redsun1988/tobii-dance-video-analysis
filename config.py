class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class AppConfig(metaclass=Singleton):
    def __init__(self):
        self.YOLO_MODEL_NAME  = 'yolo26n-pose.pt'
        self.MIN_POSE_CONFIDENCE  = 0.5   # Minimum confidence for a keypoint to be considered valid [1]
        self.DRAW_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to draw a keypoint/connection [1]

        # --- Gaze source --------------------------------------------------------------
        # Имитация взгляда с помощью курсора мыши существует только для проверки работы конвейера
        # без подключенного физического трекера Tobii 4C. По умолчанию она должна быть отключена:

        # в реальной сессии она будет молча выдавать бессмысленную статистику (положение мыши
        # ошибочно принимается за фактический взгляд), вместо того чтобы громко давать сбой. Переключите это значение на
        # True только для целенаправленного запуска теста без оборудования.
        self.GAZE_SIMULATION_ENABLED = False
        # Функция Throttle повторяет попытки, пока недоступен реальный образец взгляда трекера (моргание,
        # кратковременная потеря отслеживания), чтобы длительный сбой оборудования не приводил к бесконечному циклу обработки.
        # полноэкранные снимки без задержки между попытками.
        self.GAZE_UNAVAILABLE_RETRY_DELAY_S = 0.5

        # --- Video window discovery -------------------------------------------------
        # Сколько времени ждать появления окна видеоплеера по умолчанию в ОС после
        # запуска видеофайла, прежде чем сдаться и вернуться к
        # захвату всего экрана.
        self.WINDOW_DETECT_TIMEOUT_S = 15.0
        self.WINDOW_MIN_SIZE_PX = (200, 150)  # Игнорировать маленькие всплывающие окна/уведомления при попытке угадать окно плеера

        # --- Saccade / fixation detection --------------------------------------------
        # Скачок взгляда рассматривается как саккада, если он превышает любой из порогов.
        self.SACCADE_MIN_DISTANCE_PX = 80.0
        self.SACCADE_MIN_VELOCITY_PX_S = 600.0
        # Взгляд должен оставаться на одной и той же новой цели в течение ряда последовательных отсчетов
        # после саккады, прежде чем мы будем считать это подтвержденной фиксацией (фильтрация «дребезга» при неточных наведениях).
        self.FIXATION_CONFIRM_SAMPLES = 2

        # --- Local VLM ("что привлекло взгляд") зона внимания ----------------------
        self.VLM_ENABLED = True
        self.VLM_MODEL_NAME = "muse-glimmer:latest"  # Модель с поддержкой зрения уже скачана локально; после загрузки переключитесь на «llava».
        self.VLM_BASE_URL = "http://192.168.1.12:11434/v1"  # Ollama's OpenAI-compatible endpoint
        self.VLM_API_KEY = "ollama"  # не используется Ollama, но требуется для OpenAI SDK
        self.VLM_PROMPT = (
            "This is a cropped frame from a dance video, taken from the region a viewer "
            "just looked at right after a sudden eye movement. In 1-2 short sentences, "
            "describe what is visible in the crop and what about it (motion, color, body "
            "part, contrast, position) could have attracted the viewer's attention."
        )
        # Эта локальная модель на данном компьютере работает только на центральном процессоре
        # (ускорение силами GPU для Ollama здесь недоступно); обработка одного небольшого
        # фрагмента занимает около 200 секунд, поэтому для тайм-аута необходим значительный запас. 
        # Именно очередь с ограничением размера (см. ниже), а не короткий тайм-аут,
        # обеспечивает отзывчивость цикла захвата данных.
        self.VLM_REQUEST_TIMEOUT_S = 240.0
        self.VLM_QUEUE_MAXSIZE = 2  # очередь, накапливающаяся до того, как новые запросы probe начнут отбрасываться (и регистрироваться как отброшенные)
        self.VLM_COOLDOWN_S = 4.0  # минимальное время перед повторной проверкой той же целевой метки
        self.VLM_JPEG_QUALITY = 85
        self.BACKGROUND_CROP_SIZE_PX = 240  # crop size when gaze lands on no detected person
        self.PERSON_CROP_RATIO = 0.4  # crop size (fraction of person box) when gaze is on a person but no specific part

        # --- Post-hoc person identity reconciliation ----------------------------------
        # При отслеживании поз система может разбивать данные об одном танцоре на несколько идентификаторов (ID),
        # если он оказывается перекрыт другими объектами либо выходит из кадра и возвращается в него. 
        # После завершения воспроизведения система выбирает несколько фрагментов изображения (кропов)
        # для каждого ID и запрашивает у локальной мультимодальной языковой модели (VLM) сравнение
        # кропов одного ID с эталонным кропом другого ID; если большинство результатов указывает
        # на одного и того же человека, эти ID объединяются перед расчетом итоговой статистики. 
        # Сравниваются только те пары ID, для которых была зафиксирована хотя бы одна подтвержденная
        # фиксация взгляда (ID, на который никто не смотрел, не может исказить статистику взгляда),
        # и никогда не сравниваются ID, одновременно присутствовавшие в одном кадре
        # (поскольку они не могут принадлежать одному и тому же человеку).
        self.IDENTITY_RECONCILE_ENABLED = True
        self.IDENTITY_RECONCILE_CROPS_PER_PERSON = 3  # sample crops kept per track ID for cross-ID comparison
        # Доля голосов «за одного и того же человека» (исключительная), необходимая для объединения. 
        # Голоса представляют собой отдельные фрагменты («кропы»), поэтому *фактический*
        # порог согласия определяется с учетом параметра IDENTITY_RECONCILE_CROPS_PER_PERSON
        # и не соответствует в точности указанной доле. Например, при текущем значении
        # в 3 фрагмента доля 0,5 на самом деле требует согласия по 2 из 3 фрагментов
        # (ок. 67%), а не по 50%; при 5 фрагментах потребовалось бы 3 из 5 (60%). 
        # Этот фактический порог меняется при изменении количества фрагментов,
        # даже если само пороговое значение остается неизменным.
        self.IDENTITY_RECONCILE_MAJORITY_THRESHOLD = 0.5
        self.IDENTITY_RECONCILE_PROMPT = (
            "These are two cropped photos from a dance video. The first photo is a "
            "reference image of one tracked person. The second photo may or may not "
            "show the same individual person - judge by clothing, hair, build and "
            "visible skin tone, not by pose or camera angle. Answer with a single "
            "word first, 'yes' or 'no', then a short reason."
        )

        # --- Постфактумная оценка внешнего вида для каждого обнаруженного человека ------------------
        # Один кадр является ненадежным источником для этих оценок (размытие в движении, плохой
        # ракурс, окклюзия), поэтому - как и в случае с согласованием личности выше - это берет
        # несколько фрагментов для каждого ID трека во время воспроизведения видео, а затем после воспроизведения запрашивает
        # у локальной VLM оценку каждого фрагмента независимо, по каждому атрибуту, и проводит
        # голосование большинством голосов по этим оценкам. Оценивается каждый ID трека, который собрал хотя бы
        # один фрагмент, независимо от того, была ли когда-либо зафиксирована фиксация взгляда.
        self.DEMOGRAPHICS_ENABLED = True
        self.DEMOGRAPHICS_CROPS_PER_PERSON = 2  # sampled crops per person for the majority vote
        # Доля ненулевых голосов (строго больше указанного значения), которую должна набрать категория,
        # чтобы быть зафиксированной как значение для данного лица; в противном случае атрибут
        # остается неопределенным (None / «неизвестно») — вместо того чтобы присваивать значение,
        # победившее лишь с минимальным перевесом в ситуации, близкой к равенству голосов.
        self.DEMOGRAPHICS_MAJORITY_THRESHOLD = 0.5
        self.DEMOGRAPHICS_AGE_CATEGORIES = ('child', 'teen', 'young_adult', 'adult', 'senior')
        self.DEMOGRAPHICS_GENDERS = ('male', 'female')
        self.DEMOGRAPHICS_BODY_BUILDS = ('slim', 'athletic', 'average', 'heavy')
        self.DEMOGRAPHICS_CLOTHING_COLORS = (
            'black', 'white', 'gray', 'red', 'orange', 'yellow', 'green', 'blue',
            'purple', 'pink', 'brown', 'multicolor',
        )
        self.DEMOGRAPHICS_PROMPT = (
            "This is a cropped photo of one person from a dance video. Estimate this "
            "person's approximate age category, apparent gender, body build, and the "
            "single most dominant color of their clothing, based on their visible face, "
            "build, hair and clothing. Respond on the first line with exactly four words "
            "separated by commas, in this order: "
            "1) age category from [child, teen, young_adult, adult, senior], "
            "2) gender from [male, female], "
            "3) body build from [slim, athletic, average, heavy], "
            "4) dominant clothing color from [black, white, gray, red, orange, yellow, "
            "green, blue, purple, pink, brown, multicolor]. "
            "For example: 'young_adult, female, athletic, black'. Then give a short "
            "reason on the next line."
        )

        # --- Webcam-based facial emotion tracking --------------------------------------
        # Записывает эмоции тестируемого зрителя во время просмотра через веб-камеру,
        # синхронизированные по времени с данными о взгляде (та же временная ось
        # "секунды с начала сеанса"). В отличие от VLM-модулей выше, здесь используется
        # готовый специализированный классификатор эмоций лица (DeepFace, работает
        # локально, без Ollama) - он на порядки быстрее и надёжнее для этой узкой задачи,
        # чем универсальная VLM, отвечающая словом на промпт.
        self.EMOTION_TRACKING_ENABLED = True
        # Индекс камеры (см. webcam.py / стартовое меню -> Настройки -> Веб-камера).
        # None означает "камера ещё не выбрана" - трекинг эмоций автоматически
        # отключается до тех пор, пока пользователь её не выберет.
        self.WEBCAM_INDEX = None
        # Пауза между запросами к модели распознавания эмоций - анализ одного кадра
        # занимает заметное время на CPU, а лицо не меняет выражение настолько быстро,
        # чтобы требовался покадровый анализ каждого захваченного кадра веб-камеры.
        self.EMOTION_SAMPLE_INTERVAL_S = 0.5
        # Бэкенд детектора лица DeepFace. 'opencv' (каскады Хаара) уже входит в состав
        # пакета и не требует скачивания дополнительных весов при первом запуске -
        # в отличие от 'retinaface'/'mtcnn', которые точнее, но тяжелее и медленнее.
        self.EMOTION_DETECTOR_BACKEND = 'opencv'
        self.EMOTION_LABELS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

        # --- Output -------------------------------------------------------------------
        self.CSV_OUTPUT_DIR = "output"

        # --- Local database (viewer profiles, session results, video analysis cache) --
        self.DB_PATH = "data/app_data.sqlite3"
        # Вычисление данных о позе/траектории движения и демографических характеристиках
        # человека (возраст, пол, телосложение, доминирующий цвет) — ресурсоемкая
        # задача: модель YOLO обрабатывает каждый кадр, а выполнение запроса к VLM
        # занимает минуты. Поэтому после анализа видео его результаты сохраняются
        # в кэше базы данных и используются повторно при любых последующих
        # сеансах работы с тем же видеофайлом, независимо от того, какой именно
        # человек проходит тестирование. Установите значение True, чтобы
        # игнорировать существующий кэш и всегда выполнять вычисления заново
        # в реальном времени (при этом новые результаты впоследствии перезапишут
        # данные в кэше). Это может понадобиться, например, после изменения
        # YOLO_MODEL_NAME или промпта для определения демографических данных/личности,
        # когда кэшированные результаты анализа перестают соответствовать
        # текущим настройкам.
        self.FORCE_RECOMPUTE_VIDEO_ANALYSIS = False

        # --- Video window content-rect correction / precompute -------------------------
        # Большинство видеоплееров (включая штатный проигрыватель Windows) вписывают
        # видео в окно с сохранением пропорций, добавляя чёрные полосы (леттербоксинг/
        # пилларбоксинг), если пропорции окна не совпадают с пропорциями видео. Если
        # нормализовать закешированные координаты по всему захваченному кадру окна
        # (включая эти полосы), они "поплывут" при изменении размера/пропорций окна.
        # При True (по умолчанию) координаты нормализуются относительно реального
        # прямоугольника видео внутри окна (video_geometry.letterboxed_content_rect) —
        # это же делает кеш совместимым между обычным живым просмотром и headless-
        # предпросчётом (video_precomputer.py), который декодирует файл напрямую.
        # Установите False, только если ваш плеер по умолчанию растягивает видео без
        # сохранения пропорций — тогда используется прежнее поведение (координаты
        # относительно всего окна), но такой кеш уже не будет совместим с предпросчётом.
        self.PLAYER_PRESERVES_ASPECT_RATIO = True

        # Частота сэмплирования кадров (кадров в секунду видео-времени) при headless-
        # предпросчёте кеша видео (video_precomputer.py) — выше, чем нерегулярный темп
        # обычного живого захвата экрана, ради более точного трекинга поз.
        self.PRECOMPUTE_FPS = 10.0

        try:
            import CustomTobii4cTracker
            self.TOBII_AVAILABLE = True
        except ImportError:
            self.TOBII_AVAILABLE = False
            if self.GAZE_SIMULATION_ENABLED:
                print("Warning: CustomTobii4cTracker not found. Gaze tracking will be simulated (mouse cursor).")
            else:
                print("Warning: CustomTobii4cTracker not found, and GAZE_SIMULATION_ENABLED is False - "
                      "a real Tobii 4C tracker will be required to start.")

        try:
            import ultralytics  # YOLO-Pose for Human Pose Estimation and Tracking
            self.YOLO_AVAILABLE  = True
        except ImportError:
            print("Error: ultralytics (YOLO) library not found. Pose estimation is unavailable.")
            self.YOLO_AVAILABLE = False

        try:
            import openai  # OpenAI-compatible client used to talk to the local Ollama server
            from openai import OpenAI
            ollama_key = self.VLM_API_KEY
            client_kwargs = {"base_url": self.VLM_BASE_URL, "api_key": ollama_key}
            probe_client = OpenAI(**client_kwargs)
            probe_client.models.list()
            self.OLLAMA_AVAILABLE = True
        except Exception as e:
            print(f"Warning: local Ollama server not reachable at {self.VLM_BASE_URL} ({e}). "
                  "Attention probing on new gaze targets will be disabled.")
            self.OLLAMA_AVAILABLE = False

        try:
            import deepface  # Local facial-emotion classifier for the webcam viewer feed
            self.DEEPFACE_AVAILABLE = True
        except ImportError:
            print("Error: deepface library not found. Webcam emotion tracking is unavailable.")
            self.DEEPFACE_AVAILABLE = False