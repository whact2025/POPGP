import pytest

from popgp import engine


def test_cuda_dll_directories_are_registered_once_and_kept_alive(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    lib_dir = tmp_path / "python-lib"
    engine_root = tmp_path / "engine"
    cuda_root = tmp_path / "cuda"
    expected = [
        lib_dir,
        cuda_root / "bin",
        cuda_root / "bin" / "x64",
        engine_root / "build" / "vcpkg_installed" / "x64-windows" / "bin",
    ]
    for directory in expected:
        directory.mkdir(parents=True, exist_ok=True)

    handles: list[object] = []

    def add_dll_directory(path: str) -> object:
        assert path == str(expected[len(handles)])
        handle = object()
        handles.append(handle)
        return handle

    monkeypatch.setattr(engine, "_LIB_DIR", lib_dir)
    monkeypatch.setattr(engine, "_ENGINE_ROOT", engine_root)
    monkeypatch.setattr(engine.os, "name", "nt")
    monkeypatch.setattr(
        engine.os, "add_dll_directory", add_dll_directory, raising=False
    )
    monkeypatch.setenv("CUDA_PATH", str(cuda_root))
    engine._dll_directory_handles.clear()

    engine._add_dll_directories()
    engine._add_dll_directories()

    assert list(engine._dll_directory_handles) == expected
    assert list(engine._dll_directory_handles.values()) == handles

    engine._dll_directory_handles.clear()
