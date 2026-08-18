# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

repo_root = Path.cwd()

a = Analysis(
    ["event_reader_main.py"],
    pathex=[str(repo_root)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "engineio.async_drivers.threading",
        "flask_socketio",
        "numba",
        "socketio",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["eventlet"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="EventReader",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
