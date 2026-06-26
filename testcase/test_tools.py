import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state import AgentState
from tools.tools import get_tool_map


def main():
    state = AgentState(
        input_query="test tools",
        repo_root=".",
    )

    tools = get_tool_map()

    result = tools["list_files"](state=state, directory=".", max_depth=2)
    print(result.to_text())

    result = tools["search_code"](state=state, query="class", include_glob="*.py")
    print(result.to_text())

    result = tools["read_file"](state=state, path="main.py")
    print(result.to_text())

    result = tools["run_tests"](state=state)
    print(result.to_text())


if __name__ == "__main__":
    main()