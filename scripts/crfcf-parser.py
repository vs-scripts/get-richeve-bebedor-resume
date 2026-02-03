"""CRFCF Parser - Parses CRFCF documents into AST per PEG grammar"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import re


class NodeType(Enum):
    """AST node types for CRFCF document elements"""
    DOCUMENT = "document"
    BEGIN_MARKER = "begin_marker"
    END_MARKER = "end_marker"
    DISCLAIMER = "disclaimer"
    MAIN_SECTION = "main_section"
    SUBSECTION = "subsection"
    SPECIFIC_SECTION = "specific_section"
    SECTION_HEADER = "section_header"
    SECTION_BODY = "section_body"
    PARAGRAPH = "paragraph"
    ORDERED_LIST = "ordered_list"
    UNORDERED_LIST = "unordered_list"
    LIST_ITEM = "list_item"
    FOOTER_NOTES = "footer_notes"


@dataclass
class ASTNode:
    """AST node with type, value, children, and metadata"""
    node_type: NodeType
    value: Optional[str] = None
    children: List['ASTNode'] = field(default_factory=list)
    line: Optional[int] = None
    level: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {"type": self.node_type.value}
        if self.value is not None:
            result["value"] = self.value
        if self.children:
            result["children"] = [
                child.to_dict() for child in self.children
            ]
        if self.line is not None:
            result["line"] = self.line
        if self.level is not None:
            result["level"] = self.level
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class CRFCFParser:
    """Parses CRFCF files into AST. Assumes 4-space indents."""

    INDENT_SIZE = 4

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.position: int = 0
        self.current_line: int = 1

    def parse_file(self, filepath: str) -> ASTNode:
        """Parse CRFCF file into AST. Raises FileNotFoundError."""
        # Read file contents
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into lines (preserve empty lines)
        self.lines = content.splitlines()

        # Reset parser state
        self.position = 0
        self.current_line = 1

        begin_marker = self._parse_begin_marker()
        disclaimer = self._parse_disclaimer()
        sections = self._parse_sections()
        footer_notes = self._parse_footer_notes()
        end_marker = self._parse_end_marker()

        children = [begin_marker, disclaimer] + sections
        if footer_notes:
            children.append(footer_notes)
        children.append(end_marker)

        return ASTNode(NodeType.DOCUMENT, children=children)

    def _peek_line(self, offset: int = 0) -> Optional[str]:
        """Return line at position+offset without consuming."""
        idx = self.position + offset
        if idx < len(self.lines):
            return self.lines[idx]
        return None

    def _consume_line(self) -> Optional[str]:
        """Return current line and advance position."""
        if self.position < len(self.lines):
            line = self.lines[self.position]
            self.position += 1
            self.current_line += 1
            return line
        return None

    def _get_indent_level(self, line: str) -> int:
        """Calculate indent level (spaces / 4)."""
        spaces = len(line) - len(line.lstrip(' '))
        return spaces // self.INDENT_SIZE

    def _parse_begin_marker(self) -> ASTNode:
        """Parse BEGIN-CRFCF marker. Raises SyntaxError if invalid."""
        line = self._consume_line()
        marker = "<" + ("-" * 31) + ">"
        tag = "[ BEGIN-CRFCF ]"
        expected = f"|{marker}{tag}{marker}|"

        if line != expected:
            raise SyntaxError(
                f"Invalid begin marker at line {self.current_line}"
            )
        return ASTNode(NodeType.BEGIN_MARKER, line=self.current_line)

    def _parse_end_marker(self) -> ASTNode:
        """Parse ENDED-CRFCF marker. Raises SyntaxError if invalid."""
        line = self._consume_line()
        marker = "<" + ("-" * 31) + ">"
        tag = "[ ENDED-CRFCF ]"
        expected = f"|{marker}{tag}{marker}|"

        if line and line != expected:
            raise SyntaxError(
                f"Invalid end marker at line {self.current_line}"
            )
        return ASTNode(NodeType.END_MARKER, line=self.current_line)

    def _parse_disclaimer(self) -> ASTNode:
        """Parse disclaimer text between begin marker and sections."""
        while self._peek_line() == '':
            self._consume_line()

        lines = []
        start = self.current_line

        while self._peek_line() is not None:
            line = self._peek_line()
            if re.match(r'^\d+\.\s+\w+:', line):
                break
            if line == '' and self._peek_line(1) == '':
                break
            lines.append(self._consume_line())

        content = '\n'.join(lines).strip()
        return ASTNode(
            NodeType.DISCLAIMER,
            value=content,
            line=start
        )

    def _parse_sections(self) -> List[ASTNode]:
        """Parse all sections until end marker."""
        sections = []

        while self._peek_line() is not None:
            line = self._peek_line()
            if line and line.startswith('|<---'):
                break
            if line == '':
                self._consume_line()
                continue

            section = self._parse_section()
            if section:
                sections.append(section)

        return sections

    def _parse_section(self) -> Optional[ASTNode]:
        """Parse section by type: main, subsection, or specific."""
        line = self._peek_line()
        if not line:
            return None

        if re.match(r'^(\d+)\.\s\s(.+):$', line):
            return self._parse_main_section()
        if re.match(r'^(\d+\.\d+)\s(.+):$', line):
            return self._parse_subsection()

        stripped = line.lstrip()
        if stripped.startswith('- ') and stripped[2:].endswith(':'):
            return self._parse_specific_section()

        return None

    def _parse_main_section(self) -> ASTNode:
        """Parse numbered main section (e.g., '1.  Title:')."""
        header_line = self._consume_line()
        start = self.current_line

        match = re.match(r'^(\d+)\.\s\s(.+):$', header_line)
        num = match.group(1)
        title = match.group(2)

        if self._peek_line() == '':
            self._consume_line()

        body = self._parse_section_body(level=1)

        header = ASTNode(
            NodeType.SECTION_HEADER,
            value=f"{num}.  {title}",
            metadata={"number": num, "title": title},
            line=start
        )

        return ASTNode(
            NodeType.MAIN_SECTION,
            children=[header, body],
            line=start
        )

    def _parse_subsection(self) -> ASTNode:
        """Parse numbered subsection (e.g., '1.1 Title:')."""
        header_line = self._consume_line()
        start = self.current_line

        match = re.match(r'^(\d+\.\d+)\s(.+):$', header_line)
        num = match.group(1)
        title = match.group(2)

        if self._peek_line() == '':
            self._consume_line()

        body = self._parse_section_body(level=1)

        header = ASTNode(
            NodeType.SECTION_HEADER,
            value=f"{num} {title}",
            metadata={"number": num, "title": title},
            line=start
        )

        return ASTNode(
            NodeType.SUBSECTION,
            children=[header, body],
            line=start
        )

    def _parse_specific_section(self) -> ASTNode:
        """Parse indented specific section (e.g., '    - Title:')."""
        header_line = self._consume_line()
        start = self.current_line
        indent = self._get_indent_level(header_line)

        title = header_line.strip()[2:-1]

        if self._peek_line() == '':
            self._consume_line()

        body = self._parse_section_body(level=indent + 1)

        header = ASTNode(
            NodeType.SECTION_HEADER,
            value=title,
            level=indent,
            line=start
        )

        return ASTNode(
            NodeType.SPECIFIC_SECTION,
            children=[header, body],
            level=indent,
            line=start
        )

    def _parse_section_body(self, level: int) -> ASTNode:
        """Parse section body: paragraphs, lists. Stops at headers."""
        children = []

        while self._peek_line() is not None:
            line = self._peek_line()

            if re.match(r'^\d+\.\s\s\w+:', line):
                break
            if re.match(r'^\d+\.\d+\s\w+:', line):
                break
            if line and line.startswith('|<---'):
                break
            if line == '':
                self._consume_line()
                continue

            if line.lstrip().startswith('- '):
                node = self._parse_unordered_list()
                if node:
                    children.append(node)
                continue

            if re.match(r'^\s*[A-Za-z0-9]+\.\s+', line):
                node = self._parse_ordered_list()
                if node:
                    children.append(node)
                continue

            para = self._parse_paragraph()
            if para:
                children.append(para)

        return ASTNode(NodeType.SECTION_BODY, children=children)

    def _parse_paragraph(self) -> Optional[ASTNode]:
        """Parse paragraph: consecutive non-blank, non-list lines."""
        lines = []
        start = self.current_line
        indent = None

        while self._peek_line() is not None:
            line = self._peek_line()

            if line == '':
                break
            if line.lstrip().startswith('- '):
                break
            if re.match(r'^\d+\.', line):
                break

            if indent is None:
                indent = self._get_indent_level(line)

            lines.append(self._consume_line())

        if not lines:
            return None

        content = '\n'.join(lines)
        return ASTNode(
            NodeType.PARAGRAPH,
            value=content,
            level=indent,
            line=start
        )

    def _parse_unordered_list(self) -> Optional[ASTNode]:
        """Parse unordered list: consecutive lines starting with '- '."""
        items = []
        start = self.current_line

        while self._peek_line() is not None:
            line = self._peek_line()

            if not line.lstrip().startswith('- '):
                break

            item_line = self._consume_line()
            indent = self._get_indent_level(item_line)
            content = item_line.strip()[2:]

            items.append(ASTNode(
                NodeType.LIST_ITEM,
                value=content,
                level=indent,
                line=self.current_line
            ))

        if not items:
            return None

        return ASTNode(
            NodeType.UNORDERED_LIST,
            children=items,
            line=start
        )

    def _parse_ordered_list(self) -> Optional[ASTNode]:
        """Parse ordered list: lines matching pattern 'N. content'."""
        items = []
        start = self.current_line

        while self._peek_line() is not None:
            line = self._peek_line()

            match = re.match(r'^(\s*)([A-Za-z0-9]+)\.\s+(.+)$', line)
            if not match:
                break

            self._consume_line()
            indent_str = match.group(1)
            number = match.group(2)
            content = match.group(3)
            indent = len(indent_str) // self.INDENT_SIZE

            items.append(ASTNode(
                NodeType.LIST_ITEM,
                value=content,
                level=indent,
                metadata={"number": number},
                line=self.current_line
            ))

        if not items:
            return None

        return ASTNode(
            NodeType.ORDERED_LIST,
            children=items,
            line=start
        )

    def _parse_footer_notes(self) -> Optional[ASTNode]:
        """Parse optional footer notes between sections and end marker."""
        start_pos = self.position

        while self._peek_line() is not None:
            line = self._peek_line()

            if line and line.startswith('|<---'):
                break

            if line.strip():
                footer_lines = []
                footer_start = self.current_line

                while self._peek_line() is not None:
                    line = self._peek_line()
                    if line and line.startswith('|<---'):
                        break
                    footer_lines.append(self._consume_line())

                while footer_lines and footer_lines[-1] == '':
                    footer_lines.pop()

                if footer_lines:
                    content = '\n'.join(footer_lines).strip()
                    return ASTNode(
                        NodeType.FOOTER_NOTES,
                        value=content,
                        line=footer_start
                    )
                break

            self._consume_line()

        return None
