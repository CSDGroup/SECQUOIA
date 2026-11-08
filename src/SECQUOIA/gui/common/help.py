"""Popup help windows providing a searchable, read only text/Markdown window."""

import textwrap
from html import escape

import markdown as md
import qtawesome as qta
from qtpy import QtCore, QtGui, QtWidgets

from SECQUOIA.config import STYLE, TOOLTIPSTEXT

_MATCH_BG = "#5c4a12"
_MATCH_FG = "#f0f0f0"
_CURRENT_BG = "#ffb02e"
_CURRENT_FG = "#1a1a1a"


_HELP_ICONS = {
    "eraser": ("mdi.eraser", "white"),
    "brush": ("mdi.brush", "white"),
    "lasso": ("mdi.lasso", "white"),
    "pan": ("mdi.cursor-move", "white"),
    "save": ("fa5s.save", "white"),
    "highlighter": ("fa5s.highlighter", "white"),
    "sitemap": ("fa5s.sitemap", "white"),
    "undo": ("fa5s.undo", "white"),
    "metrics": ("fa5s.chart-bar", "white"),
    "arith": ("fa5s.layer-group", "white"),
    "todo": ("fa5s.times-circle", "#e73ce7"),
    "maybe": ("fa5s.question-circle", "#ffbf00"),
    "done": ("fa5s.check-circle", "#1db954"),
}
_ICON_PX = 18


def _visual_html(visual: str | None) -> str:
    """Icon, app style button, or strikethrough sample for the button column."""
    if not visual:
        return ""
    kind, _, value = visual.partition(":")
    if kind == "icon":
        return (
            f'<img src="icon:{value}" width="{_ICON_PX}" height="{_ICON_PX}">'
        )
    if kind == "btn":
        return (
            '<span style="background-color:#3c3c3c; color:#e6e6e6;">'
            f"&nbsp;&nbsp;{escape(value)}&nbsp;&nbsp;</span>"
        )
    return f"<s>{escape(value)}</s>"


def _key_html(key: str) -> str:
    """A key in bold text."""
    return f"<b>{escape(key)}</b>"


class HelpDocs:
    """Centralized help content for the Help popup window."""

    HOTKEY_SECTIONS = [
        (
            "Project Management",
            "",
            [
                (None, ("Ctrl+N",), "Start a new project"),
                (None, ("Ctrl+O",), "Open a previous project"),
                ("icon:save", ("Ctrl+S",), "Save all data"),
            ],
        ),
        (
            "View Menu Shortcuts",
            "",
            [
                (None, ("Ctrl+I",), "Open the Cell Inspector"),
                (None, ("Ctrl+Shift+L",), "Open Lineage Tree parameters"),
                (None, ("Ctrl+Shift+C",), "Open Channels & Masks manager"),
                (None, ("Ctrl+Shift+D",), "Open Plot parameters"),
                (None, ("Ctrl+Shift+=",), "Add new Dynamics plot"),
                (None, ("Ctrl+Shift+-",), "Remove Dynamics plot"),
                (None, ("Ctrl+Shift+T",), "Toggle Show Time Marker"),
                (None, ("Ctrl+Shift+K",), "Toggle Show Tracks"),
                (None, ("Ctrl+Shift+I",), "Toggle Show IDs"),
                (None, ("Ctrl+Shift+B",), "Toggle Tracking bar"),
            ],
        ),
        (
            "Position Navigation",
            "",
            [
                (None, ("Ctrl+Shift+Left",), "Previous position"),
                (None, ("Ctrl+Shift+Right",), "Next position"),
                (None, ("Ctrl+Shift+O",), "Open Select Position window"),
            ],
        ),
        (
            "Tree-IDs & Time Points Navigation",
            "",
            [
                (None, ("Down Arrow",), "Next Tree-ID"),
                (None, ("Up Arrow",), "Previous Tree-ID"),
                (None, ("Right Arrow",), "Next time point"),
                (None, ("Left Arrow",), "Previous time point"),
                (
                    None,
                    ("0",),
                    "Mark current Tree-ID as checked and jump to the next one",
                ),
            ],
        ),
        (
            "Tree-ID List Symbols",
            "",
            [
                ("icon:todo", (), "Not checked yet"),
                ("icon:maybe", (), "Needs further checking"),
                ("icon:done", (), "Checked"),
                ("strike:001", (), "Deactivated"),
            ],
        ),
        (
            "Dynamics Plots",
            "",
            [
                (None, ("Left-click",), "Change current time point"),
                (None, ("Drag rectangle",), "Zoom into region"),
                (
                    None,
                    ("Right-click",),
                    "Context menu (axes, time display, export)",
                ),
                (None, ("Ctrl+Q",), "Reset zoom"),
                (
                    "icon:arith",
                    (),
                    "Mask arithmetic: modify/create masks (dilate, erode, AND, OR, XOR)",
                ),
                (
                    "icon:metrics",
                    (),
                    "Metric calculation: channel/mask arithmetic operations",
                ),
            ],
        ),
        (
            "Lineage Tree Interactions",
            "",
            [
                (
                    None,
                    ("Ctrl+Left-click",),
                    "Select/deselect track or generation",
                ),
                (None, ("Shift+Left-click",), "Move entire tree (pan)"),
                (
                    "icon:highlighter",
                    ("Ctrl+P",),
                    "Activate painting/highlighting mode",
                ),
                (
                    None,
                    ("Ctrl+Left-click",),
                    "In paint mode: paint/unpaint track",
                ),
                (None, ("Ctrl+Shift+P",), "Change highlight colors"),
                ("icon:sitemap", ("Ctrl+A",), "Highlight entire tree"),
                ("icon:undo", ("Ctrl+Shift+A",), "Clear all highlights"),
                (None, ("Drag rectangle",), "Zoom into region"),
                (None, ("Ctrl+Q",), "Reset zoom (also the A button)"),
            ],
        ),
        (
            "Mask Editing (Viewers)",
            "",
            [
                ("icon:eraser", ("1",), "Eraser tool"),
                ("icon:brush", ("2",), "Paint tool"),
                ("icon:lasso", ("3",), "Lasso tool"),
                (None, ("Space",), "Toggle between Eraser & Paint"),
                ("icon:pan", ("6",), "Pan/Zoom tool"),
                (None, ("Alt + +",), "Increase brush size"),
                (None, ("Alt + -",), "Decrease brush size"),
                (
                    None,
                    ("]",),
                    "Grow the selected mask by 1 px (Paint, Erase, Lasso)",
                ),
                (None, ("[",), "Shrink the selected mask by 1 px"),
                (None, ("Ctrl+M",), "Show all masks"),
                (
                    None,
                    ("Ctrl+Shift+M",),
                    "Show only current track's mask (undo Show all masks)",
                ),
                (None, ("M",), "Create a new mask"),
                (None, ("Ctrl+Z",), "Undo"),
                (None, ("Ctrl+Y",), "Redo"),
                (None, ("Right-click",), "Assign mask to current Tree-ID"),
            ],
        ),
        (
            "Cell Fate Labeling",
            "",
            [
                ("btn:Healthy", ("H",), "Mark Healthy"),
                ("btn:Dead", ("D",), "Mark Dead"),
                ("btn:Lost", ("L",), "Mark Lost during tracking"),
                ("btn:Focus", ("F",), "Mark out of focus"),
                ("btn:Signal", ("K",), "Mark low signal"),
            ],
        ),
        (
            "Tracking Toolbar",
            "",
            [
                ("btn:New ID", (), "Create new empty Tree-ID"),
                ("btn:Division", (), "Record cell division"),
                ("btn:Remove Division", (), "Delete division event"),
                ("btn:Split Tree", (), "Split lineage at current time point"),
                ("btn:Fuse Tree", (), "Merge two Tree-IDs"),
                # TODO: enable once tracking undo/redo is tested and added
                # ("btn:Undo", (), "Undo last tracking edit (New ID/Division/Remove Division/Split/Fuse)"),
                # ("btn:Redo", (), "Redo last undone tracking edit"),
            ],
        ),
        (
            "Outlier Navigation",
            "",
            [
                (
                    None,
                    ("Ctrl+Arrow Down",),
                    "Jump to next Tree-ID with outlier or close mask",
                ),
                (
                    None,
                    ("Ctrl+Arrow Up",),
                    "Jump to previous Tree-ID with outlier or close mask",
                ),
                (
                    None,
                    ("Ctrl+Arrow Right",),
                    "Next outlier or close mask in current Tree-ID",
                ),
                (
                    None,
                    ("Ctrl+Arrow Left",),
                    "Previous outlier or close mask in current Tree-ID",
                ),
                (None, ("Ctrl+Shift+H",), "Show current outlier parameters"),
                (None, ("Ctrl+R",), "Reset all outliers and close-mask flags"),
                (
                    None,
                    ("C",),
                    "Mark the flagged outlier or close mask as reviewed and jump to the next one",
                ),
            ],
        ),
        (
            "Analysis and Export Tools",
            "",
            [
                (None, ("Ctrl+H",), "Open Outlier Detection window"),
                (None, ("Ctrl+F",), "Open Cytometric Analysis window"),
                (
                    None,
                    ("Ctrl+E",),
                    "Open Export Single Images or Movies window",
                ),
                (
                    None,
                    ("Right-click",),
                    "On a Dynamics Plot: click export plot",
                ),
                (
                    None,
                    ("Right-click",),
                    "On the Lineage Tree: click export tree",
                ),
            ],
        ),
        (
            "Help",
            "",
            [
                (None, ("F1",), "Open this Hotkeys & Mouse window"),
            ],
        ),
        (
            "Ground Truth Annotation (developer mode)",
            (
                "Only available when started with SECQUOIA_GT_ANNOTATION=1 "
                "(one bar) or =2 (one bar per viewer)."
            ),
            [
                (
                    None,
                    ("X",),
                    "Accept all open masks of the time point as assigned (Missed stays Missed) and jump to the next open time point",
                ),
                (None, ("Enter",), "In Notes: save the note only"),
                (
                    None,
                    ("Right-click", "Ctrl+Z", "Ctrl+Y"),
                    "Paint, erase, right-click, undo or redo to correct the label under the tracking point and record it",
                ),
            ],
        ),
    ]

    WORKFLOWS_MD = (
        "# **Common Workflows**\n\n"
        "## **Start New Analysis**\n"
        "1. **Ctrl+N** → Create project\n"
        "2. Select experiment folder (tTt structure)\n"
        "3. Choose tracking format & segmentation\n"
        "4. Click Load\n"
        "\n"
        "## **Curate Tracking**\n"
        "1. Select Tree-ID from left panel\n"
        "2. Review Dynamics Plots & Lineage Tree\n"
        "3. Edit masks with Paint/Eraser/Lasso (tools 1, 2, 3)\n"
        "4. Correct tracking via right-click or toolbar\n"
        "5. Mark cell fates (Healthy, Dead, Lost, OoF)\n"
        "6. **Ctrl+S** to save\n"
        "\n"
        "## **Outlier Detection**\n"
        "1. Ctrl+H → Open Outlier Detection\n"
        "2. Set threshold rules or sliding window\n"
        "3. Click Apply → Yellow stars mark outliers and Purple stars mark close masks\n"
        "4. Use Ctrl+Arrow keys to navigate\n"
        "5. Press C on a flagged point to mark it reviewed and jump to the next one\n"
        "\n"
        "## **Export Images/Movies**\n"
        "1. Select Tree-ID(s), channels, masks\n"
        "2. Export Data → Image/Movie Exporter\n"
        "3. Configure layout, time window, overlays\n"
        "4. Click Export → Save as PNG/TIF/MP4/GIF\n"
        "\n"
        "## **Cytometric Analysis (without tracking)**\n"
        "1. Ctrl+F → Open Cytometric Analysis\n"
        "2. Select segmentation masks\n"
        "3. Optionally enable BaSiC background correction\n"
        "4. Click Run → CSV files generated per position\n"
    )

    @classmethod
    def hotkeys_html(cls) -> str:
        parts = ["<h2>Hotkeys &amp; Mouse Interactions</h2>"]
        for title, note, rows in cls.HOTKEY_SECTIONS:
            parts.append(f'<h3 style="color:#ffb02e;">{escape(title)}</h3>')
            if note:
                parts.append(f"<p>{escape(note)}</p>")
            parts.append(
                '<table width="100%" cellspacing="0" cellpadding="5" '
                'border="1" style="border-collapse:collapse; '
                'border-color:#444;"><tr style="background-color:#3a3a3a;">'
                '<th width="110" align="left">Button</th>'
                '<th width="190" align="left">Shortcut</th>'
                '<th align="left">Action</th></tr>'
            )
            for i, (visual, keys, action) in enumerate(rows):
                bg = "#2f2f2f" if i % 2 else "#2b2b2b"
                parts.append(
                    f'<tr style="background-color:{bg};">'
                    f'<td align="center">{_visual_html(visual)}</td>'
                    f'<td>{", ".join(_key_html(k) for k in keys)}</td>'
                    f"<td>{escape(action)}</td></tr>"
                )
            parts.append("</table>")
        parts.append(md.markdown(cls.WORKFLOWS_MD))
        return "".join(parts)

    @staticmethod
    def add_icons(doc: QtGui.QTextDocument) -> None:
        """Register the app icons the hotkey tables refer to."""
        for name, (icon, color) in _HELP_ICONS.items():
            pixmap = qta.icon(icon, color=color).pixmap(2 * _ICON_PX)
            doc.addResource(
                QtGui.QTextDocument.ImageResource,
                QtCore.QUrl(f"icon:{name}"),
                pixmap,
            )


class HelpPopup(QtWidgets.QDialog):
    """A non-modal dialog that displays searchable help text, HTML, or Markdown."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        title: str = "Help",
        text: str = "",
    ) -> None:
        """Build the help dialog with a search bar and read-only viewer."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(900, 640)

        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinMaxButtonsHint
            | QtCore.Qt.WindowSystemMenuHint
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Search row
        row = QtWidgets.QHBoxLayout()
        root.addLayout(row)
        row.addWidget(QtWidgets.QLabel("Find:"))

        self.search_edit = QtWidgets.QLineEdit(
            placeholderText=TOOLTIPSTEXT.PLACEHOLDERTEXT
        )
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setToolTip(TOOLTIPSTEXT.EDIT)
        row.addWidget(self.search_edit, 2)

        self.btn_next = QtWidgets.QPushButton("Find Next")
        self.btn_next.setToolTip(TOOLTIPSTEXT.BTN_NEXT)
        self.btn_prev = QtWidgets.QPushButton("Find Prev")
        self.btn_prev.setToolTip(TOOLTIPSTEXT.BTN_PREV)
        row.addWidget(self.btn_next)
        row.addWidget(self.btn_prev)

        # Read only editor
        self.text = QtWidgets.QTextEdit()
        self.text.setAcceptRichText(False)
        self.text.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        help_font = QtGui.QFont(STYLE.FONT_FAMILY)
        help_font.setPixelSize(STYLE.FONT_SIZE_HELP)
        self.text.setFont(help_font)
        self.text.setReadOnly(True)
        root.addWidget(self.text, 10)

        # Status line
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color: #888;")
        root.addWidget(self.status)

        self.search_edit.returnPressed.connect(self.find_next)
        self.btn_next.clicked.connect(self.find_next)
        self.btn_prev.clicked.connect(self.find_prev)
        self.search_edit.installEventFilter(self)  # capture Shift+Enter
        self.text.cursorPositionChanged.connect(self._on_cursor_moved)
        self._match_info = ""
        self._match_range = None
        self._own_move = False

        # shortcuts
        QtWidgets.QShortcut(
            QtGui.QKeySequence.Find, self, activated=self._focus_search
        )  # Ctrl+F
        QtWidgets.QShortcut(
            QtGui.QKeySequence.FindNext, self, activated=self.find_next
        )  # F3
        QtWidgets.QShortcut(
            QtGui.QKeySequence.FindPrevious, self, activated=self.find_prev
        )  # Shift+F3
        QtWidgets.QShortcut(
            QtGui.QKeySequence("Shift+Return"), self, activated=self.find_prev
        )  # Shift+Enter
        QtWidgets.QShortcut(
            QtGui.QKeySequence("Esc"), self, activated=self._clear_search
        )

        if text:
            self.text.setPlainText(text)
        self._update_status()

    def load_text(
        self, text: str, *, rich: bool = False, as_markdown: bool = False
    ) -> None:
        """Load content into the viewer.

        Args:
            text: Content to display.
            rich: If True, render as HTML.
            as_markdown: If True, render as Markdown (takes precedence over ``rich``).
        """
        text = textwrap.dedent(text).strip()
        self.text.setExtraSelections([])
        self._match_info = ""
        self._match_range = None

        if as_markdown:
            if hasattr(self.text, "setMarkdown"):
                self.text.setAcceptRichText(True)
                self.text.setMarkdown(text)
            else:
                self.text.setAcceptRichText(True)
                self.text.setHtml(md.markdown(text))
            self._update_status()
            return

        if rich:
            self.text.setAcceptRichText(True)
            self.text.setHtml(text)
        else:
            self.text.setAcceptRichText(False)
            self.text.setPlainText(text)

        self._update_status()

    # Search flow
    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Intercept Shift+Enter in the search box to trigger a backward search."""
        if (
            obj is self.search_edit
            and isinstance(event, QtGui.QKeyEvent)
            and event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter)
            and (event.modifiers() & QtCore.Qt.ShiftModifier)
        ):
            self.find_prev()
            return True
        return super().eventFilter(obj, event)

    def _focus_search(self) -> None:
        """Focus the search box and select its current text."""
        self.search_edit.setFocus()
        self.search_edit.selectAll()

    def _clear_search(self) -> None:
        """Clear the search box and any active selection in the viewer."""
        self.search_edit.clear()
        cur = self.text.textCursor()
        cur.clearSelection()
        self.text.setTextCursor(cur)
        self.text.setExtraSelections([])
        self._match_info = ""
        self._match_range = None
        self._flash_status("Cleared")

    def _selection(
        self, cursor: QtGui.QTextCursor, *, current: bool
    ) -> QtWidgets.QTextEdit.ExtraSelection:
        """Build one painted range."""
        selection = QtWidgets.QTextEdit.ExtraSelection()
        selection.cursor = cursor
        fmt = QtGui.QTextCharFormat()
        fmt.setBackground(QtGui.QColor(_CURRENT_BG if current else _MATCH_BG))
        fmt.setForeground(QtGui.QColor(_CURRENT_FG if current else _MATCH_FG))
        selection.format = fmt
        return selection

    def _highlight_matches(self, current: QtGui.QTextCursor) -> None:
        """Paint every occurrence of the search term."""
        needle = self.search_edit.text().strip()
        doc = self.text.document()
        start, end = current.selectionStart(), current.selectionEnd()

        selections = []
        index = 0
        cursor = QtGui.QTextCursor(doc)
        while True:
            cursor = doc.find(needle, cursor, self._find_flags())
            if cursor.isNull():
                break
            is_current = (
                cursor.selectionStart() == start
                and cursor.selectionEnd() == end
            )
            if is_current:
                index = len(selections) + 1
            selections.append(self._selection(cursor, current=is_current))

        self.text.setExtraSelections(selections)
        self._match_info = f"match {index} of {len(selections)}"

    def _find_flags(
        self, backwards: bool = False
    ) -> QtGui.QTextDocument.FindFlag:
        """Return case insensitive find flags, optionally searching backward."""
        flags = QtGui.QTextDocument.FindFlag()
        if backwards:
            flags |= QtGui.QTextDocument.FindBackward
        return flags

    def _search(self, backwards: bool = False) -> bool:
        """Search for the current term, wrapping around the document if needed."""
        needle = self.search_edit.text().strip()
        if not needle:
            self._focus_search()
            return False

        doc = self.text.document()
        flags = self._find_flags(backwards)

        if self._match_range is None:
            start = self.text.textCursor().position()
        else:
            start = self._match_range[0 if backwards else 1]

        hit = doc.find(needle, start, flags)
        if hit.isNull():
            wrap = doc.characterCount() - 1 if backwards else 0
            hit = doc.find(needle, wrap, flags)
            if hit.isNull():
                self.text.setExtraSelections([])
                self._match_info = ""
                self._match_range = None
                QtWidgets.QApplication.beep()
                self._flash_status("No matches")
                return False

        self._match_range = (hit.selectionStart(), hit.selectionEnd())
        self._park_caret(hit.selectionStart())
        self._highlight_matches(hit)
        self.text.ensureCursorVisible()
        self._update_status()
        return True

    def find_next(self) -> None:
        """Find the next occurrence of the search term."""
        self._search(False)

    def find_prev(self) -> None:
        """Find the previous occurrence of the search term."""
        self._search(True)

    def _park_caret(self, position: int) -> None:
        """Move the caret without selecting anything."""
        caret = QtGui.QTextCursor(self.text.document())
        caret.setPosition(position)
        self._own_move = True
        try:
            self.text.setTextCursor(caret)
        finally:
            self._own_move = False

    def _on_cursor_moved(self) -> None:
        """Drop the search anchor when the user moves the caret themselves."""
        if not self._own_move:
            self._match_range = None
        self._update_status()

    def _update_status(self) -> None:
        self.status.setText(self._match_info)

    def _flash_status(self, msg: str, ms: int = 900) -> None:
        self.status.setText(msg)
        QtCore.QTimer.singleShot(ms, self._update_status)
