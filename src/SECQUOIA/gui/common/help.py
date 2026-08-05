"""Pop-up help windows providing a searchable, read-only text/Markdown window."""

import textwrap

import markdown as md
from qtpy import QtCore, QtGui, QtWidgets

from SECQUOIA.config import STYLE, TOOLTIPSTEXT

_MATCH_BG = "#5c4a12"
_MATCH_FG = "#f0f0f0"
_CURRENT_BG = "#ffb02e"
_CURRENT_FG = "#1a1a1a"


class HelpDocs:
    """Centralized help text for the Help Popup window."""

    HOTKEYS_MOUSE_MD = (
        "# **Hotkeys & Mouse Interactions**\n\n"
        "## **Project Management**\n"
        "- **Ctrl+N** → Start a new project\n"
        "- **Ctrl+O** → Open a previous project\n"
        "- **Ctrl+S** → Save all data\n"
        "\n"
        "## **Analysis Tools**\n"
        "- **Ctrl+H** → Open Outlier Detection window\n"
        "- **Ctrl+F** → Open Cytometric Analysis window\n"
        "- **Ctrl+E** → Open Export Single Images or Movies window\n"
        "\n"
        "## **Navigation (Tree-IDs & Time Points)**\n"
        "- **Down Arrow** → Next Tree-ID\n"
        "- **Up Arrow** → Previous Tree-ID\n"
        "- **Right Arrow** → Next time point\n"
        "- **Left Arrow** → Previous time point\n"
        "\n"
        "## **Position Navigation**\n"
        "- **Ctrl+Shift+Left** → Previous position\n"
        "- **Ctrl+Shift+Right** → Next position\n"
        "- **Ctrl+Shift+O** → Open Select Position window\n"
        "\n"
        "## **Outlier Navigation**\n"
        "- **Ctrl+Arrow Down** → Jump to next Tree-ID with outlier\n"
        "- **Ctrl+Arrow Up** → Jump to previous Tree-ID with outlier\n"
        "- **Ctrl+Arrow Right** → Next outlier in current Tree-ID\n"
        "- **Ctrl+Arrow Left** → Previous outlier in current Tree-ID\n"
        "- **Ctrl+Shift+H** → Show current outlier parameters\n"
        "- **Ctrl+R** → Reset all outliers\n"
        "\n"
        "## **Cell Fate Labeling**\n"
        "- **Healthy** button or H → Mark Healthy\n"
        "- **Dead** button or D → Mark Dead\n"
        "- **Lost** button or L → Mark Lost during tracking\n"
        "- **Out of Focus** button or F → Mark out of focus\n"
        "- **Signal** button or K → Mark low signal\n"
        "\n"
        "## **Mask Editing (Viewers)**\n"
        "- **1** → Eraser tool\n"
        "- **2** → Paint tool\n"
        "- **Space** → Toggle between Eraser & Paint\n"
        "- **6** → Pan/Zoom tool\n"
        "- **Alt + +** → Increase brush size\n"
        "- **Alt + -** → Decrease brush size\n"
        "- **Ctrl+M** → Show all masks\n"
        "- **Ctrl+Shift+M** → Show only current track's mask (undo Show all masks)\n"
        "- **M** → Create a new mask\n"
        "- **Ctrl+Z** → Undo\n"
        "- **Ctrl+Y** → Redo\n"
        "- **Right-click** → Assign mask to current Tree-ID\n"
        "\n"
        "## **Lineage Tree Interactions**\n"
        "- **Ctrl+Left-click** → Select/deselect track or generation\n"
        "- **Shift+Left-click** → Move entire tree (pan)\n"
        "- **Ctrl+P** → Activate painting/highlighting mode\n"
        "- **Ctrl+Left-click** (in paint mode) → Paint/unpaint track\n"
        "- **Ctrl+Shift+P** → Change highlight colors\n"
        "- **Ctrl+A** → Highlight entire tree\n"
        "- **Ctrl+Shift+A** → Clear all highlights\n"
        "- **Drag rectangle** → Zoom into region\n"
        "- **Ctrl+Q** or **A button** → Reset zoom\n"
        "\n"
        "## **Dynamics Plots**\n"
        "- **Left-click** → Change current time point\n"
        "- **Drag rectangle** → Zoom into region\n"
        "- **Right-click** → Context menu (axes, time display, export)\n"
        "- **Ctrl+Q** → Reset zoom\n"
        "\n"
        "## **Tracking (with Tracking Toolbar)**\n"
        "- **New ID** button → Create new empty Tree-ID\n"
        "- **Division** button → Record cell division\n"
        "- **Remove Division** button → Delete division event\n"
        "- **Split Tree** button → Split lineage at current time point\n"
        "- **Fuse Tree** button → Merge two Tree-IDs\n"
        "- **Ctrl+Right-click** → Add Tree-ID to Fuse dialog\n"
        # TODO: enable once tracking undo/redo is tested and added
        # "- **Undo** button → Undo last tracking edit (New ID/Division/Remove Division/Split/Fuse)\n"
        # "- **Redo** button → Redo last undone tracking edit\n"
        "\n"
        "## **View Menu Shortcuts**\n"
        "- **Ctrl+I** → Open the Cell Inspector\n"
        "- **Ctrl+Shift+L** → Open Lineage Tree parameters\n"
        "- **Ctrl+Shift+C** → Open Channels & Masks manager\n"
        "- **Ctrl+Shift+D** → Open Plot parameters\n"
        "- **Ctrl+Shift+=** → Add new Dynamics plot\n"
        "- **Ctrl+Shift+-** → Remove Dynamics plot\n"
        "- **Ctrl+Shift+T** → Toggle Show Time Marker\n"
        "- **Ctrl+Shift+K** → Toggle Show Tracks\n"
        "- **Ctrl+Shift+I** → Toggle Show IDs\n"
        "- **Ctrl+Shift+B** → Toggle Tracking bar\n"
        "\n"
        "## **Help**\n"
        "- **F1** → Open this Hotkeys & Mouse window\n"
        "\n"
        "## **Export & Mask Operations**\n"
        "- **Export Data → Export Single Images or Movies**\n"
        "- **Right-click on Dynamics Plot** → Click export plot\n"
        "- **Right-click on Lineage Tree** → Click export tree\n"
        "- **Mask arithmetic button** → Modify/create masks (dilate, erode, AND, OR, XOR)\n"
        "- **Metric calculation button** → Channel/mask arithmetic operations\n"
        "\n"
        "## **Tree-ID List Symbols**\n"
        "- **Purple X** → Not checked yet\n"
        "- **Yellow ?** → Needs further checking\n"
        "- **Green ✓** → Checked\n"
        "- **Strikethrough** → Deactivated\n"
        "\n"
        "\n"
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
        "3. Edit masks with Paint/Eraser (tools 1, 2)\n"
        "4. Correct tracking via right-click or toolbar\n"
        "5. Mark cell fates (Healthy, Dead, Lost, OoF)\n"
        "6. **Ctrl+S** to save\n"
        "\n"
        "## **Outlier Detection**\n"
        "1. Ctrl+H → Open Outlier Detection\n"
        "2. Set threshold rules or sliding window\n"
        "3. Click Apply → Yellow stars mark outliers\n"
        "4. Use Ctrl+Arrow keys to navigate\n"
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
        self.text.setFont(QtGui.QFont(STYLE.FONT_FAMILY, STYLE.FONT_SIZE_HELP))
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
        """Update the status line with the current line."""
        cur = self.text.textCursor()
        line = cur.blockNumber() + 1
        col = cur.positionInBlock() + 1
        extra = f"  |  {self._match_info}" if self._match_info else ""
        self.status.setText(f"Ln {line}, Col {col}{extra}")

    def _flash_status(self, msg: str, ms: int = 900) -> None:
        """Show a status message, then restore the position display."""
        self.status.setText(msg)
        QtCore.QTimer.singleShot(ms, self._update_status)
