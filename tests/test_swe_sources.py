import hashlib
import io
import json
import tomllib

import pytest

from inkling_vendor_verifier.benchmarks import swe_sources as sources
from inkling_vendor_verifier.benchmarks.deepswe import DeepSWE
from inkling_vendor_verifier.benchmarks.swe_bench import SWEBench
from inkling_vendor_verifier.sandbox import phase_hosts

REVISION = "a" * 40
CONFIG = b"""[metadata]
category = "debugging"
[agent]
timeout_sec = 3000
[verifier]
timeout_sec = 3200
[environment]
cpus = 1
memory = "4G"
"""
DEEPSWE_CONFIG = b"""schema_version = "1.3"
artifacts = ["/logs/artifacts/model.patch"]
[task]
name = "datacurve/test"
[metadata]
base_commit_hash = "0000000000000000000000000000000000000000"
[agent]
network_mode = "no-network"
timeout_sec = 10800
[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 1800
[verifier.env]
[verifier.environment]
cpus = 2
memory_mb = 8192
[[verifier.collect]]
command = "cd /app && git diff --binary HEAD~ HEAD > /logs/artifacts/model.patch"
timeout_sec = 300
[environment]
docker_image = "public.ecr.aws/example/task:v1.1"
memory_mb = 8192
os = "linux"
gpus = 0
mcp_servers = []
[environment.env]
[solution.env]
"""


@pytest.fixture
def public_source(monkeypatch, request):
    subset = getattr(request, "param", "verified")
    deepswe = subset == "deepswe"
    suite = sources.SUITES[subset][0]
    repository = (
        sources.DEEPSWE_REPOSITORY
        if deepswe
        else "https://github.com/laude-institute/harbor-datasets.git"
    )
    source_dir = "tasks" if deepswe else f"datasets/{suite}"
    names = ("a_task", "z_task") if deepswe else ("z_task", "a_task")
    tasks = [
        {
            "name": name,
            "git_url": repository,
            "git_commit_id": REVISION,
            "path": f"{source_dir}/{name}",
        }
        for name in names
    ]
    monkeypatch.setitem(sources.SUITES, subset, (suite, 2, sources._digest(tasks)))
    if deepswe:
        monkeypatch.setattr(sources, "DEEPSWE_REVISION", REVISION)
    registry = json.dumps([{"name": suite, "version": "1.0", "tasks": tasks}]).encode()
    monkeypatch.setattr(sources, "REGISTRY_SHA256", hashlib.sha256(registry).hexdigest())
    tree_sha = "b" * 40
    monkeypatch.setitem(sources.SOURCE_TREES, REVISION, tree_sha)
    contents = {}
    for row in tasks:
        for path, content in {
            "task.toml": DEEPSWE_CONFIG if deepswe else CONFIG,
            "instruction.md": b"Fix the original bug.",
            "environment/Dockerfile": b"FROM upstream/image:latest\nWORKDIR /testbed\n",
            "tests/test.sh": b"echo 1 > /logs/verifier/reward.txt\n",
            "tests/config.json": b'{"original_grader": true}',
            "solution/solve.sh": b"reference solution must not be installed",
        }.items():
            if deepswe and path == "environment/Dockerfile":
                continue
            contents[f"{row['name']}/{path}"] = content
        if deepswe:
            contents[f"{row['name']}/tests/Dockerfile"] = (
                b"FROM public.ecr.aws/example/task:v1.1\nCOPY . /tests\n"
            )
    tree = {
        "sha": tree_sha,
        "truncated": False,
        "tree": [
            {
                "path": path,
                "type": "blob",
                "mode": "100755" if path.endswith(".sh") else "100644",
                "size": len(content),
                "sha": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
            }
            for path, content in contents.items()
        ]
        + [{"path": name, "type": "tree", "mode": "040000"} for name in names],
    }
    calls = []

    def fetch(url, *, timeout):
        calls.append(url)
        if url == sources.REGISTRY_URL:
            return io.BytesIO(registry)
        if url.startswith("https://api.github.com/"):
            return io.BytesIO(json.dumps(tree).encode())
        repo = repository.removeprefix("https://github.com/").removesuffix(".git")
        prefix = f"https://raw.githubusercontent.com/{repo}/{REVISION}/{source_dir}/"
        assert url.startswith(prefix)
        return io.BytesIO(contents[url.removeprefix(prefix)])

    monkeypatch.setattr(sources.urllib.request, "urlopen", fetch)
    return tasks, calls, tree, contents


@pytest.mark.parametrize(
    "public_source, subset",
    [("verified", "verified"), ("deepswe", "deepswe")],
    indirect=["public_source"],
)
def test_prepare_preserves_source_order_and_assets_and_omits_solutions(
    tmp_path, public_source, subset
):
    tasks, calls, _, _ = public_source
    benchmark = DeepSWE() if subset == "deepswe" else SWEBench(subset)
    directory = (
        benchmark.prepare(tmp_path)
        if isinstance(benchmark, DeepSWE)
        else sources.prepare_source(tmp_path, subset)
    )
    download_calls = len(calls)
    examples = benchmark.load(tmp_path)
    assert directory == tmp_path / benchmark.cache_name
    assert json.loads((directory / "manifest.json").read_text())["benchmark"] == benchmark.name
    assert len(calls) == download_calls
    assert [example.id for example in examples] == [task["name"] for task in tasks]
    assert all(example.messages[0].text == "Fix the original bug." for example in examples)
    assert examples[0].metadata["source_revision"] == REVISION
    task = directory / "tasks/z_task"
    assert not (task / "solution").exists()
    assert (task / "tests/config.json").read_bytes() == b'{"original_grader": true}'
    config = tomllib.loads((task / "task.toml").read_text())
    if subset == "deepswe":
        assert benchmark.name == "deepswe"
        assert (task / "task.toml").read_bytes() == DEEPSWE_CONFIG
        assert not (task / "environment/Dockerfile").exists()
        assert phase_hosts(config, "agent") == phase_hosts(config, "verifier") == []
        assert sources.REGISTRY_URL not in calls
    else:
        assert (
            task / "environment/Dockerfile"
        ).read_bytes() == b"FROM upstream/image:latest\nWORKDIR /testbed\n"
        assert phase_hosts(config, "agent") == phase_hosts(config, "verifier") == ["*"]
        assert config["agent"]["timeout_sec"] == 3000
        assert config["verifier"]["timeout_sec"] == 3200
        assert config["environment"]["memory"] == "4G"
    with pytest.raises(FileExistsError):
        sources.prepare_source(tmp_path, subset)


@pytest.mark.parametrize("public_source", ["deepswe"], indirect=True)
def test_deepswe_manifest_enforces_source_identity(tmp_path, public_source):
    directory = sources.prepare_source(tmp_path, "deepswe")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_revision"] = "another-revision"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="wrong benchmark or source identity"):
        sources.load_source(tmp_path, "deepswe")


@pytest.mark.parametrize("public_source", ["deepswe"], indirect=True)
@pytest.mark.parametrize("phase", ["agent", "verifier"])
def test_deepswe_rejects_network_enabled_in_either_phase(tmp_path, public_source, phase):
    directory = sources.prepare_source(tmp_path, "deepswe")
    path = directory / "tasks/a_task/task.toml"
    path.write_text(
        path.read_text().replace(
            f'[{phase}]\nnetwork_mode = "no-network"', f'[{phase}]\nnetwork_mode = "public"'
        )
    )
    with pytest.raises(ValueError, match="Unsupported DeepSWE execution configuration"):
        sources.load_source(tmp_path, "deepswe")


def test_source_manifest_rejects_changed_assets(tmp_path, public_source):
    directory = sources.prepare_source(tmp_path, "verified")
    (directory / "tasks/z_task/tests/test.sh").write_text("different grader")
    with pytest.raises(ValueError, match="files have changed"):
        sources.load_source(tmp_path, "verified")


def test_source_manifest_rejects_cross_subset_data(tmp_path, public_source):
    directory = sources.prepare_source(tmp_path, "verified")
    directory.rename(tmp_path / "swe-bench-pro")
    with pytest.raises(ValueError, match="wrong benchmark or source identity"):
        sources.load_source(tmp_path, "pro")


def test_source_manifest_rejects_reordered_selection(tmp_path, public_source):
    directory = sources.prepare_source(tmp_path, "verified")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["tasks"].reverse()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned verified source suite"):
        sources.load_source(tmp_path, "verified")


def test_source_registry_checksum_is_enforced(public_source, monkeypatch):
    monkeypatch.setattr(sources, "REGISTRY_SHA256", "not-the-checksum")
    with pytest.raises(ValueError, match="pinned checksum"):
        sources.source_selection("verified")


@pytest.mark.parametrize("policy", ['network_mode = "no-network"', "allow_internet = false"])
def test_normalization_rejects_unexpected_upstream_network_policy(tmp_path, policy):
    (tmp_path / "task.toml").write_text(CONFIG.decode() + policy + "\n")
    with pytest.raises(ValueError, match="default public network policy"):
        sources._normalize_network(tmp_path)


@pytest.mark.parametrize("mode", ["120000", "160000"])
def test_source_tree_rejects_links_and_special_assets(tmp_path, public_source, mode):
    tasks, _, tree, _ = public_source
    tree["tree"][0]["mode"] = mode
    with pytest.raises(ValueError, match="links and special assets"):
        sources._download_tasks(tmp_path, tasks, REVISION)


@pytest.mark.parametrize("path", ["/absolute", "z_task/environment/../../escape"])
def test_source_tree_rejects_unsafe_paths(tmp_path, public_source, path):
    tasks, _, tree, _ = public_source
    tree["tree"][0]["path"] = path
    with pytest.raises(ValueError, match="Unsafe source tree path"):
        sources._download_tasks(tmp_path, tasks, REVISION)


def test_source_tree_rejects_duplicate_assets(tmp_path, public_source):
    tasks, _, tree, _ = public_source
    tree["tree"].append(tree["tree"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        sources._download_tasks(tmp_path, tasks, REVISION)


def test_source_tree_must_be_complete(tmp_path, public_source):
    tasks, _, tree, _ = public_source
    tree["truncated"] = True
    with pytest.raises(ValueError, match="complete pinned"):
        sources._download_tasks(tmp_path, tasks, REVISION)


def test_downloads_validate_pinned_git_blob_hashes(tmp_path, public_source):
    tasks, _, _, contents = public_source
    contents["z_task/task.toml"] = b"modified task"
    with pytest.raises(ValueError, match="pinned git blob"):
        sources._download_tasks(tmp_path, tasks, REVISION)


def test_failed_preparation_is_atomic(tmp_path, public_source, monkeypatch):
    def failed_extract(*args):
        raise ValueError("incomplete archive")

    monkeypatch.setattr(sources, "_download_tasks", failed_extract)
    with pytest.raises(ValueError, match="incomplete archive"):
        sources.prepare_source(tmp_path, "verified")
    assert list(tmp_path.iterdir()) == []


def test_full_suite_pins_and_default_signal_are_distinct():
    assert SWEBench().name == "swe-bench-signal"
    assert {name: spec[1] for name, spec in sources.SUITES.items()} == {
        "verified": 500,
        "pro": 731,
        "multilingual": 300,
        "deepswe": 113,
    }
    with pytest.raises(ValueError, match="Unknown SWE-bench subset"):
        SWEBench("all")
    with pytest.raises(ValueError, match="Unknown SWE-bench subset"):
        SWEBench("deepswe")
