"""Collects the demographic/self-reported profile of the person whose gaze
is being tracked, so it can be attached to that session's saved statistics
(see GazeAnalyzer.save_to_csv/save_to_excel's `viewer_profile` argument).

Navigation: arrow keys / Enter for the choice fields, Escape at any field
aborts the whole form (falling back to whatever profile was already saved).
"""

import tui

GENDER_CHOICES = ("Мужской", "Женский")
DANCE_EXPERIENCE_CHOICES = (
    "Нет опыта",
    "Начинающий (менее 1 года)",
    "Любитель (более 1 года, для себя)",
    "Опытный (занимается давно и/или выступает)",
    "Профессионал",
)

PROFILE_FIELDS = ("full_name", "gender", "age", "occupation", "dance_experience")

_FIELD_LABELS = {
    "full_name": "ФИО",
    "gender": "Пол",
    "age": "Возраст",
    "occupation": "Род занятий",
    "dance_experience": "Опыт в танцах",
}

_CUSTOM = object()  # sentinel for the "enter it manually" menu item


class _Aborted(Exception):
    """Raised internally when the user presses Escape mid-form."""


def _text(label, default=None, required=False):
    while True:
        value = tui.text_input(label, default=default)
        if value is tui.CANCELLED:
            raise _Aborted()
        if value or not required:
            return value or None
        print("Это поле обязательно для заполнения.")


def _choice(label, choices, default=None):
    items = [(c, c) for c in choices] + [("Другое / указать вручную", _CUSTOM)]
    default_index = choices.index(default) if default in choices else 0
    value = tui.select_menu(label, items, default_index=default_index)
    if value is tui.CANCELLED:
        raise _Aborted()
    if value is _CUSTOM:
        return _text("Введите значение")
    return value


def _age(default=None):
    while True:
        raw = tui.text_input("Возраст", default=str(default) if default is not None else None)
        if raw is tui.CANCELLED:
            raise _Aborted()
        if not raw:
            return default
        if raw.isdigit() and 0 < int(raw) < 130:
            return int(raw)
        print("Введите возраст числом (например, 27).")


def collect_viewer_profile(existing=None):
    """Runs the interactive profile form. `existing` (a profile dict from a
    previous fill) pre-fills each field as its shown default. Returns the
    new profile, or `existing` unchanged if the user aborts with Escape."""
    print("\n--- Данные о зрителе ---")
    existing = existing or {}

    try:
        profile = {
            "full_name": _text("ФИО", default=existing.get("full_name"), required=True),
            "gender": _choice("Пол", GENDER_CHOICES, default=existing.get("gender")),
            "age": _age(default=existing.get("age")),
            "occupation": _text("Род занятий / социальный статус", default=existing.get("occupation")),
            "dance_experience": _choice(
                "Опыт в танцах", DANCE_EXPERIENCE_CHOICES, default=existing.get("dance_experience")
            ),
        }
    except _Aborted:
        print("Заполнение отменено.")
        return existing or None

    print("\nДанные сохранены.")
    display_profile(profile)
    return profile


def display_profile(profile):
    if not profile:
        print("Данные о зрителе ещё не заполнены.")
        return
    for field in PROFILE_FIELDS:
        value = profile.get(field)
        print(f"  {_FIELD_LABELS[field]}: {value if value not in (None, '') else '(не указано)'}")
