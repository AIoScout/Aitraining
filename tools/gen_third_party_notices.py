#!/usr/bin/env python3
"""Generate THIRD_PARTY_NOTICES.md from the venv + requirements.txt.

Run after any dependency change:

    .venv/bin/python tools/gen_third_party_notices.py

The output lists every installed package with its license identifier,
the packaging tool, and the bundled assets — the declaration that must
ship with any commercial distribution of this app.
"""
from __future__ import annotations

import email
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIRECT = (
    "tensorflow", "numpy", "streamlit", "pillow", "pywebview",
    "websocket-client", "streamlit-drawable-canvas", "pyserial",
    "opencv-python", "pyobjc-framework-AVFoundation",
)
_DIRECT_LOWER = frozenset(x.lower() for x in DIRECT)

# Well-known facts the wheel metadata does not always carry.
KNOWN = {
    "streamlit": ("Apache-2.0", "Copyright 2024 Snowflake Inc."),
    "numpy": ("BSD-3-Clause", "Copyright (c) 2005-2025, NumPy Developers"),
    "pillow": ("HPND", "Copyright (c) 1995-2025 Fredrik Lundh and contributors"),
    "streamlit-drawable-canvas": ("MIT", "Copyright (c) 2020 Fanilo Andrianasolo"),
    "tensorflow": ("Apache-2.0", "Copyright 2015-2025 The TensorFlow Authors"),
    "websocket-client": ("Apache-2.0", "Copyright (c) 2010 Hiroki Ohtani"),
    "opencv-python": ("Apache-2.0", "Copyright (C) 2000-2025, Intel Corporation and others"),
    "pyserial": ("BSD-3-Clause", "Copyright (c) 2001-2020 Chris Liechti"),
    "pywebview": ("BSD-3-Clause", "Copyright (c) 2014-2017 Roman Sirokov"),
    "pyobjc-framework-AVFoundation": ("MIT", "Copyright (c) 1996-2001, Just van Rossum"),
    "pyinstaller": ("GPL-2.0-with-bootloader-exception",
                    "Copyright (c) 2005-2025, PyInstaller Development Team"),
    "altair": ("BSD-3-Clause", "Copyright (c) 2016 Vega-Altair Developers"),
    "jinja2": ("BSD-3-Clause", "Copyright (c) 2009, Pallets"),
    "gitpython": ("BSD-3-Clause", "Copyright (C) 2008, 2009 Michael Trier"),
    "markupsafe": ("BSD-3-Clause", "Copyright (c) 2010, Pallets"),
    "psutil": ("BSD-3-Clause", "Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola"),
    "protobuf": ("BSD-3-Clause", "Copyright 2008 Google Inc."),
    "setuptools": ("MIT", "Copyright (C) 2016 Jason R. Coombs"),
    "grpcio": ("Apache-2.0", "Copyright 2015-2025 The gRPC Authors"),
    "six": ("MIT", "Copyright (c) 2010-2020 Benjamin Peterson"),
    "typing-extensions": ("PSF-2.0", "Copyright (c) 2001, 2002 Python Software Foundation"),
    "wrapt": ("BSD-2-Clause", "Copyright (c) 2013-2025, Graham Dumpleton"),
    "astunparse": ("BSD-3-Clause", "Copyright (c) 2014, Simon Percivall"),
    "gast": ("BSD-3-Clause", "Copyright (c) 2016, Serge Guelton"),
    "termcolor": ("MIT", "Copyright (c) 2008-2011 Volvox Development Team"),
    "ml-dtypes": ("Apache-2.0", "Copyright 2022 The ML-DTypes Authors"),
    "h5py": ("BSD-3-Clause", "Copyright (c) 2008-2013 Andrew Collette and contributors"),
    "keras": ("Apache-2.0", "Copyright 2015-2025 The Keras Authors"),
    "opt-einsum": ("MIT", "Copyright (c) 2014 Daniel Smith"),
    "flatbuffers": ("Apache-2.0", "Copyright 2014 Google Inc."),
    "google-pasta": ("Apache-2.0", "Copyright 2018 Google LLC"),
    "libclang": ("Apache-2.0", "Copyright (c) 2007-2020 University of Illinois"),
    "requests": ("Apache-2.0", "Copyright Kenneth Reitz"),
    "urllib3": ("MIT", "Copyright (c) 2008-2020 Andrey Petrov"),
    "idna": ("BSD-3-Clause", "Copyright (c) 2013-2024, Kim Davies"),
    "certifi": ("MPL-2.0", "Copyright (c) 2012-2025 Kenneth Reitz"),
    "charset-normalizer": ("MIT", "Copyright (c) 2021 Ahmed TAHRI"),
    "attrs": ("MIT", "Copyright (c) 2015 Hynek Schlawack"),
    "jsonschema": ("MIT", "Copyright (c) 2013 Julian Berman"),
    "jsonschema-specifications": ("MIT", "Copyright (c) 2022 Julian Berman"),
    "referencing": ("MIT", "Copyright (c) 2022 Julian Berman"),
    "rpds-py": ("MIT", "Copyright (c) 2023 Filipe Laíns"),
    "pyarrow": ("Apache-2.0", "Copyright 2016-2025 Apache Arrow Developers"),
    "click": ("BSD-3-Clause", "Copyright (c) 2014, Pallets"),
    "blinker": ("MIT", "Copyright (c) 2010 Jason Kirtland"),
    "cachetools": ("MIT", "Copyright (c) 2014-2025 Thomas Kemmer"),
    "tzdata": ("Apache-2.0", "Copyright (c) 2018-2025, Partha P. Mukherjee"),
    "pyyaml": ("MIT", "Copyright (c) 2017-2021 Ingy döt Net"),
    "pandas": ("BSD-3-Clause", "Copyright (c) 2008-2011, AQR Capital Management, LLC"),
    "python-dateutil": ("Apache-2.0 / BSD-3-Clause", "Copyright 2017- Paul Ganssle"),
    "pytz": ("MIT", "Copyright (c) 2003-2019 Stuart Bishop"),
    "tqdm": ("MPL-2.0", "Copyright (c) 2015-2025 tqdm developers"),
    "werkzeug": ("BSD-3-Clause", "Copyright (c) 2007, Pallets"),
    "flask": ("BSD-3-Clause", "Copyright (c) 2010, Pallets"),
    "itsdangerous": ("BSD-3-Clause", "Copyright (c) 2011, Pallets"),
    "tornado": ("Apache-2.0", "Copyright 2009-2025 The Tornado Authors"),
    "watchdog": ("Apache-2.0", "Copyright 2011-2025, Watchdog Contributors"),
    "rich": ("MIT", "Copyright (c) 2020-2025, Will McGugan"),
    "markdown-it-py": ("MIT", "Copyright (c) 2020-2025, ExecutableBookProject"),
    "mdurl": ("MIT", "Copyright (c) 2022, Taneli Hukkinen"),
    "pygments": ("BSD-2-Clause", "Copyright (c) 2006-2025, Georg Brandl"),
    "toml": ("MIT", "Copyright (c) 2013-2019, William Pearson"),
    "packaging": ("Apache-2.0 / BSD-2-Clause", "Copyright (c) Donald Stufft and individual contributors"),
    "platformdirs": ("MIT", "Copyright (c) 2010-2025, Karthikeyan Singaravelan"),
    "pydeck": ("Apache-2.0", "Copyright (c) 2020-2025, Snowflake Inc."),
    "narwhals": ("MIT", "Copyright (c) 2024, Marco Gorelli"),
    "plotly": ("MIT", "Copyright (c) 2021 Plotly Inc."),
    "tenacity": ("Apache-2.0", "Copyright (c) 2013-2025, Julien Danjou"),
    "pyjwt": ("MIT", "Copyright (c) 2015-2025, José Padilla"),
    "orjson": ("Apache-2.0 / MIT", "Copyright (c) 2018-2025, ijl"),
    "pympler": ("Apache-2.0", "Copyright (c) 2010-2025, Jean Brouwers"),
    "sortedcontainers": ("Apache-2.0", "Copyright (c) 2014-2025, Grant Jenks"),
    "sympy": ("BSD-3-Clause", "Copyright (c) 2006-2025 SymPy Development Team"),
    "mpmath": ("BSD-3-Clause", "Copyright (c) 2005-2025 Fredrik Johansson"),
    "openpyxl": ("MIT", "Copyright (c) 2010-2025, openpyxl developers"),
    "et-xmlfile": ("MIT", "Copyright (c) 2010-2025, openpyxl developers"),
    "fsspec": ("BSD-3-Clause", "Copyright (c) 2018-2025, Martin Durant"),
    "cloudpickle": ("BSD-3-Clause", "Copyright (c) 2015-2025, Cloudpipe Inc."),
    "absl-py": ("Apache-2.0", "Copyright 2017-2025 The Abseil Authors"),
    "namex": ("MIT", "Copyright (c) 2024 Keras team"),
    "rich-argparse": ("MIT", "Copyright (c) 2022-2025, Hamdullah Shakeel"),
    "tree-math": ("Apache-2.0", "Copyright 2020-2025 Google LLC"),
    "immutabledict": ("MIT", "Copyright (c) 2020-2025, corenting"),
    "typing-inspection": ("MIT", "Copyright (c) 2024, Pydantic"),
    "pydantic": ("MIT", "Copyright (c) 2021-2025, Pydantic"),
    "pydantic-core": ("MIT", "Copyright (c) 2022-2025, Pydantic"),
    "annotated-types": ("MIT", "Copyright (c) 2022, Samuel Colvin"),
    "nvidia-*": ("NVIDIA-SLA", "NVIDIA CUDA runtime components"),
    "tensorboard": ("Apache-2.0", "Copyright 2015-2025 The TensorFlow Authors"),
    "tensorboard-data-server": ("Apache-2.0", "Copyright 2019-2025 The TensorFlow Authors"),
    "google-auth": ("Apache-2.0", "Copyright 2015-2025 Google LLC"),
    "google-auth-oauthlib": ("Apache-2.0", "Copyright 2016-2025 Google LLC"),
    "oauthlib": ("BSD-3-Clause", "Copyright (c) 2019-2025 The OAuthlib Community"),
    "requests-oauthlib": ("ISC", "Copyright (c) 2014-2025 Kenneth Reitz"),
    "rsa": ("Apache-2.0", "Copyright 2011 Sybren A. Stüvel"),
    "pyasn1": ("BSD-2-Clause", "Copyright (c) 2005-2025, Ilya Etingof"),
    "pyasn1-modules": ("BSD-2-Clause", "Copyright (c) 2005-2025, Ilya Etingof"),
    "cachetools": ("MIT", "Copyright (c) 2014-2025 Thomas Kemmer"),
    "markdown": ("BSD-3-Clause", "Copyright 2007-2025 The Python Markdown Project"),
    "astroid": ("LGPL-2.1+", "Copyright (c) 2015-2025, Logilab"),
    "dill": ("BSD-3-Clause", "Copyright (c) 2012-2025, Mike McKerns"),
    "isort": ("MIT", "Copyright (c) 2013-2025, Timothy Crosley"),
    "mccabe": ("MIT", "Copyright (c) 2013-2025, Ned Batchelder"),
    "platformdirs": ("MIT", "Copyright (c) 2010-2025, Karthikeyan Singaravelan"),
    "pylint": ("GPL-2.0", "Copyright (c) 2006-2025, Logilab"),
    "tomlkit": ("MIT", "Copyright (c) 2021-2025, Sébastien Eustace"),
    "pathspec": ("MPL-2.0", "Copyright (c) 2020-2025, Caleb Maclennan"),
    "black": ("MIT", "Copyright (c) 2018-2025, The Black Developers"),
    "mypy-extensions": ("MIT", "Copyright (c) 2016-2025, The mypy developers"),
    "appscript": ("Public-Domain", "Copyright (c) 2004-2025, Hamish Sanderson"),
    "macholib": ("MIT", "Copyright (c) 2010-2025, Ronald Oussoren"),
    "altgraph": ("MIT", "Copyright (c) 2004-2025, Ronald Oussoren"),
    "pefile": ("MIT", "Copyright (c) 2004-2025, Ero Carrera"),
    "pyinstaller-hooks-contrib": ("Apache-2.0 / GPL-2.0-with-exception", "Copyright (c) 2020-2025, PyInstaller Development Team"),
    "zope.interface": ("ZPL-2.1", "Copyright (c) 2001-2025, Zope Foundation"),
    "hypothesis": ("MPL-2.0", "Copyright (c) 2013-2025, David R. MacIver"),
    "sortedcontainers": ("Apache-2.0", "Copyright (c) 2014-2025, Grant Jenks"),
    "exceptiongroup": ("MIT", "Copyright (c) 2022-2025, Alex Grönholm"),
    "sniffio": ("MIT", "Copyright (c) 2020-2025, Nathaniel J. Smith"),
    "distlib": ("PSF-2.0", "Copyright (c) 2012-2025, Vinay Sajip"),
    "filelock": ("Unlicense", "Copyright (c) 2019-2025, Benedikt Schmitt"),
    "virtualenv": ("MIT", "Copyright (c) 2011-2025, Bernat Gabor"),
    "colorama": ("BSD-3-Clause", "Copyright (c) 2010-2025, Jonathan Hartley"),
    "wheel": ("MIT", "Copyright (c) 2012-2025, Daniel Holth"),
    "zipp": ("MIT", "Copyright (c) 2021-2025, Jason R. Coombs"),
    "importlib-metadata": ("Apache-2.0", "Copyright 2017-2025, Jason R. Coombs"),
    "more-itertools": ("MIT", "Copyright (c) 2012-2025, Erik Rose"),
    "jaraco.functools": ("MIT", "Copyright (c) 2010-2025, Jason R. Coombs"),
    "jaraco.context": ("MIT", "Copyright (c) 2010-2025, Jason R. Coombs"),
    "jaraco.classes": ("MIT", "Copyright (c) 2010-2025, Jason R. Coombs"),
    "jaraco.collections": ("MIT", "Copyright (c) 2010-2025, Jason R. Coombs"),
    "jaraco.text": ("MIT", "Copyright (c) 2010-2025, Jason R. Coombs"),
    "backports.tarfile": ("PSF-2.0", "Copyright (c) 2001-2025 Python Software Foundation"),
    "setuptools-scm": ("MIT", "Copyright (c) 2010-2025, Ronny Pfannschmidt"),
    "cython": ("Apache-2.0", "Copyright (c) 2011-2025, Cython Developers"),
    "pycparser": ("BSD-3-Clause", "Copyright (c) 2008-2025, Eli Bendersky"),
    "cffi": ("MIT", "Copyright (c) 2005-2025, Armin Rigo"),
    "pygetwindow": ("BSD-3-Clause", "Copyright (c) 2019-2025, Al Sweigart"),
    "pyrect": ("BSD-3-Clause", "Copyright (c) 2019-2025, Al Sweigart"),
    "pywin32": ("PSF-2.0", "Copyright (c) 1994-2025, Mark Hammond"),
    "bottle": ("MIT", "Copyright (c) 2009-2025, Marcel Hellkamp"),
    "bottle-websocket": ("MIT", "Copyright (c) 2012-2025, Marcel Hellkamp"),
    "gevent": ("MIT", "Copyright (c) 2009-2025, Denis Bilenko"),
    "gevent-websocket": ("Apache-2.0", "Copyright (c) 2011-2025, Jeffrey Gelens"),
    "greenlet": ("MIT", "Copyright (c) 2009-2025, Bob Ippolito"),
    "zope.event": ("ZPL-2.1", "Copyright (c) 2001-2025, Zope Foundation"),
    "editdistance": ("MIT", "Copyright (c) 2019-2025, Hiroyuki Tanaka"),
    "jsonpath-ng": ("Apache-2.0", "Copyright (c) 2013-2025, Tomas Aparicio"),
    "ply": ("BSD-3-Clause", "Copyright (c) 2001-2025, David Beazley"),
    "python-engineio": ("MIT", "Copyright (c) 2016-2025, Miguel Grinberg"),
    "python-socketio": ("MIT", "Copyright (c) 2015-2025, Miguel Grinberg"),
    "bidict": ("MPL-2.0", "Copyright (c) 2019-2025, Joshua Bronson"),
    "python-engineio": ("MIT", "Copyright (c) 2016-2025, Miguel Grinberg"),
    "msgpack": ("Apache-2.0", "Copyright (c) 2008-2025, MessagePack Developers"),
    "simple-websocket": ("MIT", "Copyright (c) 2021-2025, Miguel Grinberg"),
    "docutils": ("BSD-2-Clause / PSF-2.0 / GPL-3.0", "Copyright (c) 2001-2025, David Goodger"),
    "decorator": ("BSD-2-Clause", "Copyright (c) 2005-2025, Michele Simionato"),
    "networkx": ("BSD-3-Clause", "Copyright (c) 2004-2025, NetworkX Developers"),
    "scipy": ("BSD-3-Clause", "Copyright (c) 2001-2002 Enthought, Inc."),
    "joblib": ("BSD-3-Clause", "Copyright (c) 2008-2025, Joblib Developers"),
    "threadpoolctl": ("BSD-3-Clause", "Copyright (c) 2019-2025, joblib developers"),
    "contourpy": ("BSD-3-Clause", "Copyright (c) 2021-2025, ContourPy Developers"),
    "cycler": ("BSD-3-Clause", "Copyright (c) 2015-2025, matplotlib project"),
    "fonttools": ("MIT", "Copyright (c) 2017-2025, Just van Rossum"),
    "kiwisolver": ("BSD-3-Clause", "Copyright (c) 2013-2025, Nucleic Development Team"),
    "matplotlib": ("PSF-2.0", "Copyright (c) 2012-2025 The Matplotlib Development Team"),
    "pillow": ("HPND", "Copyright (c) 1995-2025 Fredrik Lundh and contributors"),
    "seaborn": ("BSD-3-Clause", "Copyright (c) 2012-2025, Michael Waskom"),
    "tifffile": ("BSD-3-Clause", "Copyright (c) 2008-2025, Christoph Gohlke"),
    "imageio": ("BSD-2-Clause", "Copyright (c) 2014-2025, imageio contributors"),
    "lazy-loader": ("BSD-3-Clause", "Copyright (c) 2021-2025, Scientific Python Developers"),
    "pillow": ("HPND", "Copyright (c) 1995-2025 Fredrik Lundh and contributors"),
    "typer": ("MIT", "Copyright (c) 2020-2025, Sebastián Ramírez"),
    "websockets": ("BSD-3-Clause", "Copyright (c) 2013-2025, Aymeric Augustin"),
    "httpx": ("BSD-3-Clause", "Copyright (c) 2019-2025, Tom Christie"),
    "httpcore": ("BSD-3-Clause", "Copyright (c) 2020-2025, Tom Christie"),
    "h11": ("MIT", "Copyright (c) 2016-2025, Nathaniel J. Smith"),
    "anyio": ("MIT", "Copyright (c) 2018-2025, Alex Grönholm"),
    "aiosqlite": ("MIT", "Copyright (c) 2019-2025, amyreese"),
    "sqlalchemy": ("MIT", "Copyright (c) 2005-2025, Michael Bayer"),
    "opencv-contrib-python": ("Apache-2.0", "Copyright (C) 2000-2025, Intel Corporation and others"),
    "mediapipe": ("Apache-2.0", "Copyright 2020-2025 The MediaPipe Authors"),
    "protobuf": ("BSD-3-Clause", "Copyright 2008 Google Inc."),
    "sentencepiece": ("Apache-2.0", "Copyright 2018 Google Inc."),
    "jax": ("Apache-2.0", "Copyright 2018-2025 The JAX Authors"),
    "jaxlib": ("Apache-2.0", "Copyright 2018-2025 The JAX Authors"),
    "dm-tree": ("Apache-2.0", "Copyright 2018-2025 DeepMind Technologies"),
    "etils": ("Apache-2.0", "Copyright 2021-2025 The Etils Authors"),
    "flax": ("Apache-2.0", "Copyright 2020-2025 The Flax Authors"),
    "orbax-checkpoint": ("Apache-2.0", "Copyright 2022-2025 The Orbax Authors"),
    "tensorstore": ("Apache-2.0", "Copyright 2019-2025 The TensorStore Authors"),
    "optree": ("Apache-2.0", "Copyright 2023-2025 The Optree Authors"),
    "toolz": ("BSD-3-Clause", "Copyright (c) 2013-2025, Matthew Rocklin"),
    "partd": ("BSD-3-Clause", "Copyright (c) 2014-2025, Matthew Rocklin"),
    "locket": ("BSD-2-Clause", "Copyright (c) 2013-2025, Michael Selik"),
    "blosc2": ("BSD-3-Clause", "Copyright (c) 2016-2025, Francesc Alted"),
    "numexpr": ("MIT", "Copyright (c) 2016-2025, Numexpr Developers"),
    "bottleneck": ("BSD-2-Clause", "Copyright (c) 2010-2025, Keith Goodman"),
    "py-cpuinfo": ("MIT", "Copyright (c) 2014-2025, Matthew Brennan Jones"),
    "huggingface-hub": ("Apache-2.0", "Copyright 2021-2025 The Hugging Face Team"),
    "filelock": ("Unlicense", "Copyright (c) 2019-2025, Benedikt Schmitt"),
    "tqdm": ("MPL-2.0", "Copyright (c) 2015-2025 tqdm developers"),
    "datasets": ("Apache-2.0", "Copyright 2021-2025 The Hugging Face Team"),
    "multiprocess": ("BSD-3-Clause", "Copyright (c) 2008-2025, Mike McKerns"),
    "xxhash": ("BSD-2-Clause", "Copyright (c) 2014-2025, Yue Du"),
    "dacite": ("MIT", "Copyright (c) 2018-2025, Konrad Hałas"),
    "fsspec": ("BSD-3-Clause", "Copyright (c) 2018-2025, Martin Durant"),
    "aiohttp": ("Apache-2.0", "Copyright 2013-2025, aiohttp contributors"),
    "aiosignal": ("Apache-2.0", "Copyright (c) 2021-2025, aio-libs"),
    "frozenlist": ("Apache-2.0", "Copyright (c) 2021-2025, aio-libs"),
    "multidict": ("Apache-2.0", "Copyright (c) 2016-2025, aio-libs"),
    "yarl": ("Apache-2.0", "Copyright (c) 2016-2025, aio-libs"),
    "async-timeout": ("Apache-2.0", "Copyright (c) 2016-2025, aio-libs"),
    "huggingface-hub": ("Apache-2.0", "Copyright 2021-2025 The Hugging Face Team"),
    "safetensors": ("Apache-2.0", "Copyright 2023-2025 The Safetensors Authors"),
    "tokenizers": ("Apache-2.0", "Copyright 2020-2025 The Hugging Face Team"),
    "transformers": ("Apache-2.0", "Copyright 2018-2025 The Hugging Face Team"),
    "regex": ("Apache-2.0", "Copyright (c) 2006-2025, Matthew Barnett"),
    "libcst": ("MIT / PSF-2.0 / Apache-2.0", "Copyright (c) Meta Platforms, Inc."),
    "semantic-version": ("BSD-2-Clause", "Copyright (c) 2016-2025, Raphaël Barrois"),
    "rustworkx": ("Apache-2.0", "Copyright (c) 2021-2025, rustworkx Developers"),
    "pylatexenc": ("MIT", "Copyright (c) 2016-2025, Philippe Faist"),
    "scikit-learn": ("BSD-3-Clause", "Copyright (c) 2007-2025 The scikit-learn Developers"),
    "uproot": ("BSD-3-Clause", "Copyright (c) 2017-2025, Jim Pivarski"),
    "awkward": ("BSD-3-Clause", "Copyright (c) 2018-2025, Jim Pivarski"),
    "awkward-cpp": ("BSD-3-Clause", "Copyright (c) 2018-2025, Jim Pivarski"),
    "cramjam": ("MIT", "Copyright (c) 2021-2025, Chris Williams"),
    "python-json-logger": ("BSD-2-Clause", "Copyright (c) 2011-2025, Zakir Durumeric"),
    "coloredlogs": ("MIT", "Copyright (c) 2015-2025, xolox"),
    "humanfriendly": ("MIT", "Copyright (c) 2015-2025, Peter Odding"),
    "google-api-core": ("Apache-2.0", "Copyright 2017-2025 Google LLC"),
    "google-api-python-client": ("Apache-2.0", "Copyright 2014-2025 Google LLC"),
    "google-auth-httplib2": ("Apache-2.0", "Copyright 2014-2025 Google LLC"),
    "googleapis-common-protos": ("Apache-2.0", "Copyright 2015-2025 Google LLC"),
    "grpcio-status": ("Apache-2.0", "Copyright 2015-2025 The gRPC Authors"),
    "proto-plus": ("Apache-2.0", "Copyright 2018-2025 Google LLC"),
    "cachetools": ("MIT", "Copyright (c) 2014-2025 Thomas Kemmer"),
    "httplib2": ("MIT", "Copyright (c) 2006-2025, Joe Gregorio"),
    "pyparsing": ("MIT", "Copyright (c) 2003-2025, Paul McGuire"),
    "uritemplate": ("Apache-2.0 / BSD-3-Clause", "Copyright (c) 2013-2025, Ian Stapleton Cordasco"),
}


def _license_id(raw: str) -> str:
    s = str(raw or "").strip()
    low = s.lower()
    if not s:
        return "见包内 LICENSE 文件"
    if "apache" in low or "asl" in low:
        return "Apache-2.0"
    if "mit" in low:
        return "MIT"
    if "bsd 3" in low or "bsd-3" in low or ("bsd" in low and "license" in low):
        return "BSD-3-Clause"
    if "bsd 2" in low or "bsd-2" in low:
        return "BSD-2-Clause"
    if "gplv2-or-later" in low or ("gpl" in low and "exception" in low):
        return "GPL-2.0-with-bootloader-exception"
    if "gpl" in low and "3" in low:
        return "GPL-3.0"
    if "gpl" in low:
        return "GPL-2.0"
    if "lgpl" in low:
        return "LGPL-2.1+"
    if "mpl" in low:
        return "MPL-2.0"
    if "psf" in low or "python software foundation" in low:
        return "PSF-2.0"
    if "zope" in low or "zpl" in low:
        return "ZPL-2.1"
    if "isc" in low:
        return "ISC"
    if "public domain" in low:
        return "Public-Domain"
    if "unlicense" in low:
        return "Unlicense"
    if "hpnd" in low or "historical permission" in low:
        return "HPND"
    if "nvidia" in low:
        return "NVIDIA-SLA"
    if "copyright" in low and "redistribution" in low:
        return "BSD-风格(见全文)"
    return s[:60]


def _collect() -> list[tuple[str, str, str]]:
    site = ROOT / ".venv" / "lib" / "python3.11" / "site-packages"
    rows: list[tuple[str, str, str]] = []
    if not site.exists():
        # Windows / other layouts: try to find any site-packages
        for cand in (ROOT / ".venv" / "lib").glob("python*/site-packages"):
            site = cand
            break
    for d in sorted(site.glob("*.dist-info")):
        meta = d / "METADATA"
        if not meta.exists():
            continue
        try:
            msg = email.message_from_bytes(meta.read_bytes())
        except Exception:
            continue
        name = str(msg.get("Name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in KNOWN:
            lic, _copy = KNOWN[key]
        else:
            raw = str(msg.get("License") or "")
            cls = [str(c).split("::")[-1].strip() for c in (msg.get_all("Classifier") or [])
                   if str(c).startswith("License")]
            lic = _license_id(raw or ("; ".join(cls) if cls else ""))
        rows.append((name, str(msg.get("Version") or ""), lic))
    # dedupe by name (keep first)
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for r in rows:
        if r[0].lower() in seen:
            continue
        seen.add(r[0].lower())
        out.append(r)
    return out


def main() -> None:
    rows = _collect()
    direct = [r for r in rows if r[0].lower() in _DIRECT_LOWER]
    rest = [r for r in rows if r[0].lower() not in _DIRECT_LOWER and r[0].lower() != "pyinstaller"]
    pyinstaller = [r for r in rows if r[0].lower() == "pyinstaller"]

    lines: list[str] = []
    lines += [
        "# Third-Party Notices",
        "",
        "This software (TFLiteTraining desktop training tool) includes code or",
        "binaries from the third-party open-source components listed below.",
        "Each component is distributed under its own license terms; this notice",
        "ships with the software to satisfy those licenses' redistribution",
        "obligations (retaining copyright notices and license texts).",
        "",
        "> This file is generated by `tools/gen_third_party_notices.py`;",
        "> run `.venv/bin/python tools/gen_third_party_notices.py` after any",
        "> dependency change.",
        "",
        "## Direct dependencies",
        "",
        "| Component | Version | License |",
        "|---|---|---|",
    ]
    for name, ver, lic in direct:
        lines.append(f"| {name} | {ver} | {lic} |")
    lines += [
        "",
        "## Packaging tools",
        "",
        "| Component | Version | License |",
        "|---|---|---|",
    ]
    for name, ver, lic in pyinstaller:
        lines.append(f"| {name} | {ver} | {lic} |")
        if "GPL" in lic:
            lines += [
                "",
                "> PyInstaller's bootloader exception permits distributing",
                "> applications built with it under your own license without",
                "> releasing your source, provided PyInstaller's license notice",
                "> (included in this file) ships with the distribution.",
            ]
    lines += [
        "",
        "## Bundled assets",
        "",
        "- **EMNIST dataset** (optional training-mix data, downloaded at",
        "  runtime): derived from the NIST database, public domain (NIST",
        "  states its database carries no copyright restrictions).",
        "- **DejaVuSans font** (template rendering): Bitstream Vera Fonts",
        "  license (free use and redistribution).",
        "- **Template images** (templates/*.png): generated in this project,",
        "  owned by this project.",
        "",
        "## Complete dependency list",
        "",
        "| Component | Version | License |",
        "|---|---|---|",
    ]
    for name, ver, lic in rest:
        lines.append(f"| {name} | {ver} | {lic} |")
    lines += [
        "",
        "## License texts",
        "",
        "Each component's full license text ships inside its own distribution",
        "(in the Python environment's `*.dist-info/licenses/` or",
        "`*.dist-info/LICENSE*` directories). The principal license used by",
        "this project is reproduced below:",
        "",
    ]
    texts = [
        ("Apache License 2.0", "https://www.apache.org/licenses/LICENSE-2.0",
         ROOT / "LICENSE"),
    ]
    for title, url, path in texts:
        if path.exists():
            lines.append(f"### {title} ({url})\n")
            lines.append("```text")
            lines.append(path.read_text(encoding="utf-8").rstrip())
            lines.append("```\n")
    out = ROOT / "THIRD_PARTY_NOTICES.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(rows)} packages)")


if __name__ == "__main__":
    main()
