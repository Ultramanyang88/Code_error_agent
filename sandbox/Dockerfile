# Sandbox image for run_command/run_tests/apply_patch (see docker_sandbox.py).
#
# git is baked in at BUILD time (this step needs network) specifically so
# containers built from this image can run with --network none at RUNTIME --
# apply_patch's `git apply` would otherwise have no git binary to call, and
# there's no way to `apt-get install` it after the container is already
# running with networking disabled.
FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Baked in at build time (network available then) since the container runs
# with --network none at runtime -- pip install wouldn't work later. This
# covers running a target repo's OWN test suite, not installing its
# dependencies: a repo whose tests need packages beyond the stdlib + pytest
# will still fail with ModuleNotFoundError in here. That's a deliberate
# trade-off of no network at runtime, not an oversight.
RUN pip install --no-cache-dir pytest

RUN useradd -m -u 1000 agent
USER agent
WORKDIR /workspace
