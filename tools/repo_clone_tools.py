# check branch
# inspect project type
# set up env
# discover test


def inspect_project_type(state: AgentState) -> ToolResult:
    root = Path(state.repo_root)

    if (root / "requirements.txt").exists() or (root / "pyproject.toml").exists():
        project_type = "python"
    elif (root / "package.json").exists():
        project_type = "node"
    elif (root / "go.mod").exists():
        project_type = "go"
    elif (root / "pom.xml").exists() or (root / "build.gradle").exists():
        project_type = "java"
    else:
        project_type = "unknown"

    state.project_type = project_type