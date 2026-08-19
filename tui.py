"""Small arrow-key console UI, built on prompt_toolkit: Up/Down (or j/k) move
the selection, Enter confirms, Escape cancels/goes back. Used everywhere the
program used to ask for a typed number or a typed line.
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

CANCELLED = object()  # returned by select_menu()/text_input() when the user presses Escape

_STYLE = Style.from_dict({
    "title": "bold",
    "selected": "reverse",
    "hint": "fg:#888888 italic",
})

_DEFAULT_HINT = "↑/↓ выбор   Enter подтвердить   Esc назад"


def select_menu(title, items, default_index=0, hint=None, show_result=True):
    """
    items: list of (label, value) pairs.
    Returns the selected value, or CANCELLED if the user pressed Escape/Ctrl-C.
    """
    if not items:
        raise ValueError("select_menu() needs at least one item")

    index = [max(0, min(default_index, len(items) - 1))]
    chosen = {}

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        index[0] = (index[0] - 1) % len(items)
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        index[0] = (index[0] + 1) % len(items)
        event.app.invalidate()

    @kb.add("enter")
    def _confirm(event):
        chosen["value"] = items[index[0]][1]
        chosen["label"] = items[index[0]][0]
        event.app.exit()

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event):
        chosen["value"] = CANCELLED
        event.app.exit()

    def render():
        lines = []
        if title:
            lines.append(("class:title", title + "\n"))
        for i, (label, _value) in enumerate(items):
            if i == index[0]:
                lines.append(("class:selected", f"> {label}\n"))
            else:
                lines.append(("", f"  {label}\n"))
        lines.append(("class:hint", "\n" + (hint or _DEFAULT_HINT)))
        return lines

    control = FormattedTextControl(render, focusable=True)
    window = Window(content=control, always_hide_cursor=True, dont_extend_height=True)
    app = Application(
        layout=Layout(HSplit([window])),
        key_bindings=kb,
        style=_STYLE,
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
    )
    app.run()

    value = chosen.get("value", CANCELLED)
    if show_result and value is not CANCELLED:
        print(f"{title}: {chosen['label']}" if title else chosen["label"])
    return value


def confirm(title, default=True):
    """Yes/No arrow-key prompt. Returns True/False, or CANCELLED on Escape."""
    items = [("Да", True), ("Нет", False)]
    return select_menu(title, items, default_index=0 if default else 1, show_result=False)


def text_input(label, default=None):
    """
    Enter submits the typed text (or the default, if left blank).
    Escape cancels - returns CANCELLED instead of a string.
    """
    kb = KeyBindings()
    cancelled = {"flag": False}

    @kb.add("escape")
    def _cancel(event):
        cancelled["flag"] = True
        event.app.exit(result="")

    message = f"{label} [{default}]: " if default else f"{label}: "
    session = PromptSession(key_bindings=kb)
    try:
        text = session.prompt(message)
    except (EOFError, KeyboardInterrupt):
        return CANCELLED

    if cancelled["flag"]:
        return CANCELLED
    return text.strip() or default
