"""Example FORGE plugin: a word-count tool.

Copy this file to your plugin directory (~/.forge/plugins by default)
and FORGE will pick it up on the next run - `forge tools` lists it.
"""

from forge.tools.base import Tool


class WordCountTool(Tool):
    name = "word_count"
    description = "Count the words in a piece of text."
    args_doc = "text (string): the text to count words in"

    def run(self, text: str = "", **_) -> str:
        words = text.split()
        return f"{len(words)} word(s)"


# Either expose instances in TOOLS, or just define Tool subclasses -
# FORGE discovers both. This module does both on purpose as a demo;
# duplicate class discoveries are de-duplicated automatically.
TOOLS = [WordCountTool()]
