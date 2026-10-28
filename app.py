from __future__ import annotations

import base64
import dataclasses
import json
import shutil
import time
import uuid
import os
import sys
import re
import subprocess
import tempfile
import urllib.request
import zipfile
from html import escape as html_escape
from pathlib import Path

# ── Dev-mode hot reload of core modules ────────────────────────────────
# Streamlit reruns only re-execute app.py — imported modules stay cached in
# sys.modules, so code changes in trainer/record_controller/image_preprocess
# etc. silently never take effect (the "parameters changed but nothing
# happened" trap).  Module mtimes are tracked in a FILE (module-level
# globals do not reliably survive streamlit reruns); when any core module
# changes, all of them are dropped from sys.modules before the imports
# below re-execute, and the cached RecordController is replaced (see the
# guard after _get_record_controller — the old HTTP server is shut down).
_HOT_MODULES = (
    ("trainer", "trainer.py"),
    ("record_controller", "record_controller.py"),
    ("image_preprocess", "image_preprocess.py"),
    ("serial_device", "serial_device.py"),
    ("dataset_io", "dataset_io.py"),
)


def _hot_state_path() -> Path:
    try:
        env = os.environ.get("TFLITE_TRAINING_DATA_DIR")
        base = Path(env).expanduser().resolve() if env else (
            Path.home() / "Library" / "Application Support" / "TFLiteTraining"
        )
        base.mkdir(parents=True, exist_ok=True)
        return base / ".hot_reload_state.json"
    except Exception:
        return Path(tempfile.gettempdir()) / "tflite_training_hot_reload.json"


def _hot_module_mtimes() -> Dict[str, float]:
    base = Path(__file__).resolve().parent
    out: Dict[str, float] = {}
    for name, fname in _HOT_MODULES:
        try:
            out[name] = (base / fname).stat().st_mtime
        except OSError:
            pass
    return out


def _hot_reload_core_modules() -> bool:
    current = _hot_module_mtimes()
    state: Dict[str, float] = {}
    sp = _hot_state_path()
    try:
        if sp.exists():
            state = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    changed = bool(state) and [n for n, m in current.items() if state.get(n) != m]
    try:
        sp.write_text(json.dumps(current), encoding="utf-8")
    except Exception:
        pass
    if changed:
        # Drop every core module so the imports below bind fresh objects.
        for name, _fname in _HOT_MODULES:
            sys.modules.pop(name, None)
    return bool(changed)


_hot_reloaded_now = _hot_reload_core_modules()
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from camera_permission import ensure_camera_access, get_camera_access_status
from trainer import TrainConfig, new_run_dir, recommend_train_params, train_and_export
from ui_styles import inject_teachable_style

from dataset_io import (
    IMAGE_EXTS,
    KNOWN_PROJECT_TYPES,
    PROJECT_FORMAT_LEGACY,
    PROJECT_MAGIC,
    PROJECT_TYPE_CLASSIFIED_IMPORT,
    PROJECT_TYPE_IMAGE,
    PROJECT_TYPE_OCR,
    PROJECT_TYPE_SIGN_OCR,
    ImportedData,
    app_data_dir,
    default_img_size,
    export_classified_dataset,
    export_from_assignments,
    infer_imported_data,
    read_project_header,
    sanitize_class_name,
    materialize_files,
    materialize_zip_bytes,
    new_output_dir,
)
from image_preprocess import normalize_class_preprocess_map, normalize_manual_roi
from ocr_dataset import BACKGROUND_CLASS_NAME, DEFAULT_EMNIST_SPLIT, EMNIST_SPLITS
from ocr_preprocess import GLYPH_SIZE
from serial_device import list_serial_ports
from record_controller import RecordController, SessionConfig, make_hold_button_html


APP_NAME = "TFLiteTraining"

# The project types an "Open Project" card will accept from disk — every type
# main() routes to a workspace, which is the only test that matters here: a
# type named in a manifest but with no page behind it would open into
# nothing.  `image_classified_import` is deliberately absent for the same
# reason it always was: it is a way to START a project, not a workspace a
# project lives in, and an archive claiming it is really an image project.
_TM_OPENABLE_PROJECT_TYPES = (
    PROJECT_TYPE_IMAGE,
    PROJECT_TYPE_OCR,
    PROJECT_TYPE_SIGN_OCR,
)


# AIoScouts logo (200x200 PNG) embedded as base64 for the topnav menu
_AIO_SCOUTS_LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAoZUlEQVR4nOydC1xTV7b/10btpaFNTiEQKsk5VhR5JbZaTRihykPtFezjAp12fEwfdzqdmU7RGb137FSt2tcdbavTdv7T6WNsO9pOgZnpCLRVEFuxBBVHEt6gJQ+UQJAEJWpV9v8TBActCefkQU7C/n4+qSVZe+8Vwi/r7H32XmsijDEUJQuSyZQKAPiBXJ5zF0J4GsZIihBEAIBwrP0h8I7LAGAGAAsAbjGZGppMpvrDQUFQWVNTYBxrZ5C3B2CYpHCRSDpHJJqcKpOplAghuzhE3h6XEHhgDAar1Xi4t9eg1unU+3p7jfUWS/slb47pNYHI5TkJM2fmbsAYP+StMQjjHotOV7m1tXXvax0dDRe8MYBHBSIQhAqio1Nzp01Le1QgCFvgyb4JBCf0dXU15zc2Fv9Br6864smOPSYQuTx7vlyesxMhNMVTfRIIXLFYDP84cuT9n3V2NnR4oj+3BcIwqvhZs1a8KBCEPeAJhwgED2DR6yu3a7UFL7k7R3FLICkpeT+l6aQ/utMHgeAtMIa2Q4dez9Tpqupd7WOCK40YRiVNSVn1jkSSsHYsVsIIBFdACCiZLGk5RcnOmc3N1Zcuncec++DaQCKJj8rIWP8VAIrm2pZA8BUWi3FncfGax7i24xRBaFqVkpr6v2UASMZ1IALBlwQHC++kadWdPT1tX9ls3efYtmMdQRYt2jQvPHzGlwAQ4rKXBILPQeqyss0pHR11l9lYB7ExskeO8PAZpUQcBP8Hq9LTNxwUi2PC2FiPKhD7nCMlZVWBPUp5xD8CwedglVL55NtsLJ3OQRhGJVuw4H/32f/XY74RCDwgOFgYzzDKW43GY/suXTrv0M6pQJKTV70THCyc7w0HCQRfExws+sGlS+ebOzvrtY5sHAokJSXv5xJJwhpyn4MQyERExKt6e9s/sVqNZ0d6fcQ5CMMo42k66S22k3gCwV9BCKKSk58ppCjZiH/rIz45a9bKF73uGYHAG5Bq6tT5K0Z65XsCkcuzF5ONh4TxRlxc5rMjPX/dHEQgCBXMn7/2c4QQNWaeEQi8AIXRtOpKS8ver4c/e10EiY5OzSXnOQjjFYqSbmQYlWT4czcIJO0nY+4VgcAfJoaFTV82/IlrAlEosmeGhITN84lbBAJPiI1d8qPhP18TiEym+o1PPCIQeARC6C6FInfO0M8DAmEYZThFyR72qWcEAj8Ikstznrn2g/0/IhGt9KlLBAKvwFlDNw4HMisKhVHJvnaJ4D9gDM3Hjn341KlTNRW9vd9PikDTqkiRKCpNoch5FQBF+sZLt6CEwqgki8VwaOA+SHJy3gaEENmxS2CFRpOfXF//z6MXL57tH+l1q9V4rrOzQSsSyYIoSrpw7D30BLhOr6+qDLKHEoTQnb52h+AfYAzttbWFjWxs9Xp1tfc98g4CQdgs+79BND1XAQC3+tohgn+AEHDJM3XFi654FbF4ul0XEIQxSvK1MwQC30AIRduvroLk8pxZvnaGQOAhAplMOTUIITzN154QCDzljiAAJPW1FwQCT5kaZJ+P+NoLAoGnhNoFQs5+EAgjEBmZICBnzgkEB4jFsbcQgRAITiACIRCcQARCIDiBCIRAcAIRCIHgBCIQAsEJRCAEghOIQAgEJxCBEAhOIAIhEJxABEIgOGGirx0IAHQYQ6ter25FCOv7+9Fpg0FtBoAuANwLAOcB0GUAfAEATUAIB2MMAgB0CwCIwsJiIkNCQiMBgJbJkqYFBWH7/88AgJt9/cYIRCCu0I0x/lyrLSzq70dldXX5Znc60+vVw37aPvBfkUgaTNOqO2WypKzbbovKBoBYd50muAYRCDt6r1xBn9bXf1qk06n3Wa3tNm8OZrUaL2i1BWr7AwCeS0jIjb/99tgVEREJjyIE/phnym8hAnEI7ujsbC08fVpTcvp0zdfd3c3nfOVJXV1+fV0drBMIQjeIxTEqmlYtZhjVAwCQ4CufxgtEIN+nTaer3FBRseMjXztyIzbbmUt6vfqg/VFRAc/J5dnzFYqcVwCQyte+BSpEIFe5bLEYvjx1SvPn1tYDfz971jBixkC+odUWfqXVFibNmLFk7rRpqT+jKGkuAArxtV+BBBEI4NbjxwuX19UVVPnaE1dpaio5bH/IZMpNKSmrXkcIkRqTHmIcCwQ3NzQUv9DS8tUuf4kYo2EwVLXt3v3Ig7GxS5Kjo9N+S1HSjPH9GbvPuPzlWSyGV6uqdv7GbK677GtfvEFjY0lFY2PJfyYn5/2QYZJ2AkCwr33yV8bVnXSbrfuwRpO/oLh47ZpAFcdwKip2/LWo6Nd0V1fzu772xV8ZNxHEZKr/4+HD7z0zUj0LbyIURk2iadW1ibPVavzOYKjy6n2U4Vit7V179274SXr6+qLIyPidAKTENxcCXiAYQ3djY9HTx4795RNv9C8WT79NIAhTMIwqAWOIDQ+PkQkEt8kAkHgwKd9Iq0pXMIYehMAMAEadTq0DwCd7e0/VWSyGNrO5ud5mO+NRIZeVbfmMppX3JCev/hNCQJaFWRLQAsEYGktLNy/s7Kw3erLfmTNzpbfeOnkZwyTlAMBdADCBYxcTEIIhAcUyzPf+Xi0YQ5FWm/+xVltY4im/9foq7e7dDydlZKz/rUSSsMkFv8cdASsQi8X4RU1NwU89IQ6BIPQ/pk1LnycURi1gGFUmAHg7Iz6FECxXKHLtj+6uruYvTp+uKTl9WlNkNrf0utt5aemWF5OTV7UwjOod+1WgZ1wOTAJSIBaL4dPi4rU/dLcfipIFzZnz+Irw8NhnEUIxnvGOM2Hh4THL7A+FItdiMtVvO3Lkvbes1naLO51WVGz/tKUl/lhGxvpDACjCc+4GFgG3imWxGHaWl7/8mDt9iMUxopSU1T/PzNx6PCIibqcPxXEjlEQS/0Jm5ram5OS834nF06Pd6cxkqm89eHD7/QC403MuBhYBFUHs4iguXuuWOBYu3Hh/RETcHwH4u2sWIRTBMElrGSYpz2Sq33z06Hu/s1hcW53T66vUpaWblRkZG6pIJPk+ASMQi8Wwq7z85V+42l6hyJkVHZ26WSAIy3Sxi8sA0NrX133CbG45CQAd9kdXV0vv+fPdw5d1/0MqTbolKAjbJ+hRAkEoHR4eMxUA4l04JHXT1Yjy6g81mvwtJ06U/81mO8O5LqDJ1NBmjyQpKas+IyK5noAQSEdH3QtlZVvWu9o+K2vbGpFIutWFppf7+9FntbX5+T097V8ajZWs5gXXH5K6ytVDUkqVRJKQLpHEPwIAXC6f5ApF7qc0rdpVXf3+f3d0NFzg8iZgMJLs27d5/sKFG486WJoel/i9QM6d697pqjgYRhktl+e+JhJJ72PfClt6ek7lnzxZtuf06ZoyTx2eunpIynhAqy08EBwctiEiYnoiTSszaVr1Xwihu9h8VhQlW5aevmFOdfVHv2hsLCnl6kNnZ0OjTqd+nGFUu8kS8FX8XSCtBw68/DNXGioUuQlyeXYp17lGTU3BgtrawhpXxmTLhQvdWK/v1ur1ai3AjldoWjknJWXV7wDQgtFbo5jZs1cWR0Xd/auyss1vcR27omL7p8HB6++QSBJecdH9gMJvV7Ewxm1FRWtS7N+8XNumpOT9WC7PPspVHBjDP7wtjpHQ66uO7Nr1SOrRox/OtVoNuwbnO864KTIy/s3Fi7e8SVFRk7iOV1q65f/6+rr3u+5x4OC3AtFqC35ptRo7uLbLyNjwvzTt6g5X7FaCBndpaio5UlS0dnlNTf58ABj1BqhYPP0XWVnb/mmf33Ady2xurnTZ0QDCLwVisRi2a7WFRVzbJSev2iiRxLt86YAQShUIQv/D1faeora28Js9e349VaerfB4AvnNmizG6NzV13W6uIhEIwkh5cH8UCMb4UHHx2tVc22VkbNjIMKrn3Rw+Ojo6dambfXiE3t72SxUVOzaVlm6aAQAnnNmGhIQ9mJm5bR9bkSxatCEzPDzmQY8568f4nUCqqz/6Ldc2KSl5T0kk8e6KY4Do6LSVnujHU5hMDW1ffLH+HovF4HS3MkKQnJb2m50UJXP4mQuF0lsXL978Znh4vD063+QVh/0MvxJIR0fdH5uaSr7i0mbRoo3pNJ20w1M+hISELVEocnl1+dHd3XKquHjtIz09hi3O7AQC8Q/vvvuxEX8XCkXOrKVLt9WLxTEu32wNRPxJIJba2kJO9ztEIml4eHjcTg9/G06Qy3M4R7GxoKJi+xaE0D+c2Ugk8U+np2/49fDnFIpspVyeUwwAUq876Wf4zX2Qzs6mN0ymek6rSGlpv3nTOx86XhEXt+S1hoYS7fBnGSbpVoEgdG5Y2LSZACiGYZImA+BIACy4mo4H2yfUNgDUbTa3nLq6LaW5zmY7c0yvV5901yv7vKSoaE12auq6dwSC0Mcd2UVGxm978MG35pvNzdU33xwWHx4ek01uDI6MvwjkwuHDf3qNS4OFC9f/VCAQP+QlfybMmrXi44kTb37Caj1VIxJFpSsUuT8DwIuv/53iwX/RDf8OLMEOPIYOS2GMm7u66t83mRp3aTT5Lp9hsVgM/eXlL/8kK2tbKMbYYfofgSBsKU0n8WLBgc/4hUB0usrfcDn/EBeXeXdERMLvvesVSlAocodtqsLu9YZQTEREwisREQkvTp26oOzEif27T5woz7fZznDeymIXyT//+auH0tLWfRoSIiY5styA93MQjHFbRcUOTpPsqVNTX/TjVZgJISFhixSK3J0PPPBWnUKRPdOVTuyXW+XlrzzB5oYiwTG8F0hjYwmn/UQpKXmPU5R0kfc8GjsQQlPk8tzjixZt/rNIJA3n2t5qNZ75+uvXM+xTOO94GPjwXiBdXc2ss5HY/4hoOonzBj2+Ex4e82hW1ja9QpGdzLWtwVDVpNdXbfKOZ4EPrwVisRh2GgxVrC8RVKon1wRwFsHgxMTcopSU1U8LBKGcVpyqqz94DwAd8J5rgQuvBaLTqVl/80VGxgWLxTFPetcj34IQiGha+UZ6+rrnuLSz2c5cLC19/kEA7Faih/EIbwXS19d9oLa2sI2t/d13P/4yAIyLrIFCoey5pUu3bhQIwhDbNiZTg8VsbuW0VE7gsUDM5uZ9bG0VilypSCR7yrse8YqJQqHs+ejo1Ce4NFKr395uv3L1nluBB28FoterP2drK5PN3RLAcw+HyOU5WxWK7Dls7a1W41m9vnKVd70KLPgqkDa9vupfbAwZRhlOUbJHve8S/0AIKLk8d69QKA1l2+bgwR0fYIy/nzWCMCK8FIjV2r6Lra1YHMMh4UJAQqlUT27m0sBoPMzLzZZ8hJcC0eu/KWBrGxk58z+96w3/CQ+PeYrLFvyvv359PwCu865XgQEPBYI7NJrC42wsKSpqEkVJ53nfJ94zQS7/r11cjgPrdOp/etelwICHAkGsxAEDk3PVD/icInRsQXOnTUtNZ2ut11d96V1/AgPeCUSnU2vZ2g4W0ycMwuU4sNncrAbA3d71yP/hnUAQgka2tiKRLMu73vgXAkFYVkRE/C1sbG22Mxcxhj3e98q/4Z1AdDp1Mxs7mlaFA4DDianFYiiurv7wgYMHty9oa1P/ymIxfIIxZn1n3k8JmT59IesootUWEIGMAg8PTGEdS8OpDnvA+HBx8dpr0UWvV19L9JCYmD2FomQ5DKN8AgDFuust35gyJemxQ4d2/IGNrcnUsBcArpDjto7hWwT5rqur5RQbw/Dw6Q6jh7O78LW1hW0VFdu37dmzRmGzdf/VVUd5zF00rWJVwqCzs/4cADZ43yX/hVcCwRj058+zq28hEIQ6TMZgtbY7TaQGgyfuqqs/ep2rj36APRoo2Bpj7Dzp3HiHVwJBiPXllf2Dvd3Ra1arkeUZksDc/k3Tyji2tnp9FYkgTuCVQACAw4c1UIfcAWyTTKMo9uP5E0jG3hZzTgA+nuCVQPr70Tm2tiJRlLMqSKzW9ylKxnqTn58hYWuIEJz1riv+Da8EYjRWnWFr6ywRM8bQx6YPoXByQApELJ4m4mDO6nc1XuGVQCwWQz9bW4Scnv9gVfEVIb9NDeQUgSCMSzFQl6rjjhd4JRAf4PNaH94gUIXvC/xWIBgD53LHI3DeA33wDraXmITR4ZVAaFrF4dgscpiSEyEQsumhr+9MQC7zms2trBc7MAaBd73xb3glEKFQymqjHVy9W97r5OVb2fRhNrcEZMZBm63bxNaW7ZfJeIVXAgkKwg5v/n0fZzf5EMszIljPfjx/gv2mTIwR55Sm4wleCQQAONy4Q12OXhEKoyaz6cFiMX7LoqSy36HXVzWxtRWJpAF6s9Qz8EogGKMpbG1ttm6H3/4UJWN1Pru3t/0SAA64euAYYw1bW4qKcrgrmsAzgSAEkQyTxGrSaDa3ONu3xTqBgUZT+De2tv4AxtDINp9xZGTCRAAU7X2v/BdeCQQGPuB+mqVlq6NXaFrFejdrb2+7XSAX2NrzHb2+8j22tuHhsXPHY8I9LvBOIACIVcjX66vsl1gjbkpECGZGRMSx+uB1usouABQoidS+a2nZ+y5b44iIBFKCbRR4JxCaVs7gYF7i4PkJM2bcm8O2k4aGIqeVYf2Fvj5zkcnUwOrejkAQhiIj43O975V/wzuBcDkGq9XmO5w/0PTcZWz76e5u+SQQVrNaW8s/ZGsrFk+LAwAy/xgF3gmEppWsa/LpdOp9jnejooU0rWJ1LkKnU5usVuML7L3kHxjj4ydPln/B1l4sjlnsXY8CA94JBCE08+rqyuhYre02AHA0f5jAMElL2I579OifXwSAdtaO8gyNpuBRm+3MRbb2cXFLxntOY1bwTiAAEBweHse6CKdery519FpY2DTWddI7OuouNzQUc0oCzRe6upp31tYW1rC1p2nldADEud7heISPAgG5PPsRtrY6XaXDHLMhIWFpXJI6nzx54F0AvysNcKG+/nNO2drvuWfV7/iZ8ol/8FIgCKH7xOLprJZpzebWBgB82NHrcnn2H9mWKrNYDP1FRWsfxNjxPRY+gTGcranJv89orGSVKgkGxJGXgTFifek53uGlQABAKBCE/YCNoc3WjbXagv9xYpI+bVrqPWwHtlqNHVpt4c/Z2vuSnh7j72prC1mXqqOoqEkyWdIbAORAFVv4KhCIi8u8l62tRlP4FQCUOXo9OnrBWi5ja7UF+zSa/IcBMG8PHlksxu0HDrz0Mpc2U6em/QgAAi6bpDfhrUDE4umcElPX1hY+NZhG83sIBOLMhQs3ZnLpT6st/KtGU/ALLm3GCovF8E5x8ZrVbJPsDREXl7nee14FJrwViP3zVChy5rM1rqnJb7XZugsdvR4eHvu+WDyd1Tb4IbTawg80mvyHAIB1thUv811Pj2FLefnLnIWblbX1TXJjkDt8Foh9gv08F/sTJw686eg1hFCEUsmtlh9cFUl+TU1+mqN9X2PI5W+/rVpZUrJ2g812hlMmEoUiO0MkkvEyGvIdXgsEAC1gGCXrb32NJv+gs2VaipKtUChyuez1GqC2trCmtHTTHIzxAa5tPQHG0N3e3pDxzTevu5RsWyZL+o3nvRof8FwgALGxWT/hYn/w4PaVALjHwcs3JSZml9G08g6ufphMDW27dz+S2tBQvHIM64z09fQY39q7d/2sAwc2fcW1sUAQhrKytm6jKCnr0myE6+G9QMTi6Y9zsdfrq1p0uqodjl5HCKLk8tx3XPXn2LGPPtq9+5E7dLqBgvxePEeCDxw8uH12Scmap83mFpfOzkdHpy4TiWS/9rxv4wfeCwQA6OTkvF9xaXDs2IevY4wdFgO1f6NmZm77WCAIdTnlTUXFjh1FRb++vaGh+CcWi6HYAyk8L2CMD+l06leOHPlo5q5dj6Tq9WrWZ8tvRKHISVEochzOyQjs8IvtBgyT9H8GQ9UunU7NKp2NzXamt7R0y9KFCzfUAYyc1oaipA+npq7rLS5e+1NX/bJa2y3Hjn30LgC8KxJJb6Vp5X1yee79CEGqPfix6OICxlBWW1uwp6fH8JnBUOWRTOszZ2bPSkzM2T/K59uHMexBCCYBQCY5WTgyfiEQu5+xsZm/1OnUz7Ft0NlZb9Tp1KsZRuXwCCpFyZ7MzNx20/79L/3y/PkzrJOtjYTVajyr1Rp3abWFu+Dqsd9IAJgMAGGDebqGypydBQD7HKm9p6f91Nmz7PMRs2HmzBxlYmL2J84+W4yxvqJie4b9chQGJvHKKbNnr9waEhLG+pDZeMFfBGKfi6ySSOK2sT0xBwOXQdvfDw7ecJdEEv+0IxuKkj6alvasoLh4zQ895uzVXcb2aDCmtTfk8pysxMScv4/yuZpLSzfP7+xsuLbQYDBUtRkMVbnp6et/HBmZ8CeyFeXf+MMcZIiQjIyNBQJB6CQujY4e/XNeX1/3p85sKEr60LJlf61gGOV0t730AQJB6ISUlNV5CkXOX0aJHLrjxwvShotjOGVlWz44ePD1u61WA+v9XYGOPwnETnp0dBqnb3qLxdBfXv7SYxjjUeqv43nz5q2uUCiy57np45gSGZkwMS3t2Q9oWrkdAJzWBdFqCx6vqyvQOrPR66u0RUVrF1ksho887qwf4m8CAYUie5NQKOVU+MZqbbcVF6+dZ7OZK53ZIQQRcnnu/sWLN78qEkVRbjvrZWJjl6Smp6+vFomkTs/fY4ytGk3+fVptIeskeVptwasecdLP8TuBAKCpWVnbPqMoGSffrVbjmf37X8kaPZLATWJxzK+ysl5tUCiyM9zz1TuIRFGC5cs/fWf27JX72FS01WoLfqTVFu7hOAypXeifAhn4pk9OTMzmdG8EBkVSXLxm9mhzkkEi5fLczzMztxbFxi5hvWnSm4jFMaKUlNVPZ2Vtq8O4/7+HrYyNCMa4TaMpmKfVFjpKj+RsLJIzy18FAgP3RlQvKhS5rDOgDGG/3Nq//6XlJlPddhbmEylKljl79soDy5Z9XC6XZ9/vmrfuYY8YCxdu/MXixZsbaVr5BsDoOYwxBuPXX/8+Xast+IbreBJJfFRcXOaLLjscQPjNMu8I3JSYmL3XYjGk6vXqei4Ne3vbL5WWblk9b15e3ZQpqq0AiMV8Ay1QKHLtjzqdTp2v11f+U6+v+pcb/jtFKIyaNHnyzNRp09IfFImilnLJfH/u3JmPDxx46Wl7xOQ6LsOoZMnJeZ8DQARnpwMQfxbIwKQ6JSVvV1nZ2TkdHXWcE78dOrTj3ZaWfV9kZKwvRgixzeebwDAq++N5AGizWtt36XTffKLVFtZyfwfXExERFyyRJMyjadUjFCXNBgCuCwUXTKb6p0tLN7POz3sjiYk5vwdACa62DzTQsmWfYF874S4Wi7GsomL7MqvVyLqy0nAEglDhrFkrVjOMahW7aDIS2IRQ0L/a2ipPAOCTZnPLKZvtTOfgzuKLg/UQ7V9INwOgWwAglKZV9qggpSjZNJEoKt4uvtHmFQ7ot1oNZS0tB37b1FR8xBXvBYJQQVras++JRNKHRzFt27XrYVa7oWlaNT8lZZVPjgh4gv5+vN2vI8gQg5sP1cXFa+ZarUaHhXUcYbOd6a2o2LFJp1N/cM89q/MB4G7uXiAJxvhehlEN/MQwSdy7cI0LOl3lExUVO3a700l6+rMbhcJRxTHu8NtJ+o0gBFNSU9d9JhJJXS4pZjBU2b8d5zQ0FP0XxvigZz30NLjDZGp9/quvdtzhjjgEgtBJS5du+51QKF3jWf8CAz8SCO44darhKa02f7ZGU3AvxvjQjRYhIWFJmZnbDrsjEhjYLv+Xv+/e/cg9bW2VC51tm/cRl3t6jM9/+eULstLS5zYZjZVu3a9IS1v3e6FQuta//hbGDr+4xLL/kR48uP0Bg6HqWlWp1tb9pamp65677TbZc8Pfx2AkKfv22/JlNTXOt1WMxqFDO0oPHdpxV2zskvli8fRchlEtBEAx7r4fF7jQ348ONTUVfdLZ2fo3o7HS7SQSAkHobWlpz/6BxZxjXMN7gdjFUVq6Oamzs+G603vnz5+5UlKydtOSJVvhtttk1yV3CAkJkycm5hzGGKdrNIWc7wPcSGNjyVcA8FVFxQ6Qy7PnyOU5P0cI5dqHcrfvUWg+darhTy0tez80Gis5z60cERkZF5ye/vwXAHiup/oMVHgtkKHIcaM4hlNe/vILqanr7CJZf8MKUHBiYm6JSEQ/V139wf+z2bjlkHKEVlt4RKstfEwgCH1KLI6ZLZOp5iAEd9K0KgEhmOHogNYofAeA2yyW9nqr1Xiir+/Mse7uJnVnZ+u3Fy50e2yVUSAIQ9HRqSvl8pznATDrgqnjGd4KxFHkuJGhSJKRsSFSIol/avhrCIGIppVvCIVR86qr/7zClXsljrDZzlzU69Xf2B/Dn09MzJ2MUL8cAE0RixOkt98eJ0YIB//7fcGlK1eCrPX1+acBwH7J2HD6dEOz2ew53xyRlrbuDZFIStL/cICXAmETOW7EZKp77UaBDEFR0ofT05+TV1d/lNfYWOIwRaknqK3NPwUAg8mkC7w5FGtiYzPnzp69/A8AaLavffE3eCcQtpHjRkymhtPOLVDC7Nkrv4yKuvuN6ur3/sdiaeeUfM0foShZ0N13P7pOIknYQE4JugbflvYuaDQFj3IVB1zdYMdm4+KEyMj4VfPmrS6KiIiLdM1F/0AiiaOSk/MKJJKEF4g4XIc3EWRwa/YDXColDSESScMViuzX2NpTlHTRwoUbG8zm5jfU6rdfs1rbWZ9z5zsiURSlUv3sl2LxtLzBhBEEN+BLBLngjjiysrYdBkBclywpsThmfVbWq6eTk/PyuI7LR5KT835sfz9i8bTNRByewecRxN3IkZa27jMAcGfJMphhkrbTtGqVwaB+W69Xf6zT/fuGJN9hmCQJTSufoGnVijGq/cFhMyX2yDwPY3xSr696DyE8YqTHGIXRtPInCCFWVY254GuBeCByuCWOayCEptB00sv2R2Ki8UODoXKLRlPI21JsCsVD0TQ9Z6NIJFsxxkM7TQxxA544tnvh669/n2Y0Vjr90pLLcwoVipw6D4x3HT4TCA8ih0MoSrqSonJXyuW5h3U69R69vnJvb++poxaLZ5O8cfNJFiQURs2kadUihlHdDwBjtl34Bm6RyVS3Ggzqs6MZms2t3wLgdgDE+rDXjfT0GN4aTRxw9dx9vUKRvRcAsa6QzAZfCYQ3kWMU5jKMyv7YAgAWjOELrTb/k66u+i87OrivtHHFLgqZTDmHYZIeFomilrNMZ+ptgigqaqbBABWjGdps3dhkqn9HIkngVOdlOHq9mnV+YYzhDGJVrpU9Yy4QPkeOUaAQgocVityHr24NgUadTn0MAJp6e9tPWCwG++WYzmxu6bH/YXDp2P6NjBAwAHAHTStlACieppUJg6ccOaU4GgsYJildqy0cVSB2jhx5f2tm5quPDb4/TvT1mT+rrS1kXWoCIcSq8CsXxlog/hI5RuMm+zSAYVQjHdPtAwADADYAIBPGcA5jfC3aIASTEEIiACwGQJEAMNXF/Vs+QyiMegAANrGxtVrbbVpt/kaFIncnx2EuVFd/9CxbY5pW2n+PNMcxRmXMBOLHkYMrIVdXk9DAipI95KMR476HrwXGEITQnTStitbr1SfY2Le2lv9FJlNNufFogiMwBoNWW/CowVDFOhlHeHhMKltbLozVfRC3I4dAEOarSSlhBGhayToT/NCGUr1e/SMWRYcaKypeT9dqC1hngYSBLfwz7+NizxavRxCM8XGNpuDRcRA5xhUMk7SstvZvW7ms7B08uD1fJIraFx2d9qPIyDtTbrttcjQABGOMuvV6tdZsbt7b3q4t4VoSgmGUDEVJF7r0RkbBq1lNXN14CPybcxBGoLOz4dF9+zZ94Gs/li//+O8Yowc83W9/P97utUssV7asD0Eih38QERH3qlg8XepLH2Jjlyz1hjjgaqra7+wC8fhBnaHIMfwMOVvInMOvCFMqf7rBV4NHRiZMnD17xcve6t9gqLLZBeJ2AoAbaCeRY/xAUdLHUlJW/3isxxWJpKLZsx/L93IWyN4gANzpyR41moLfksgxrphI08qdiYk5WWM1IEVJg5cu3VZEUVKvXFoNo9MeQU56tk9cxbUFiRz+z8yZOR/I5Tlez35vjxypqes+xRiSvT0WAP42qKOjnlNmdBZw2i9EIkfAEKpQ5PwjJSXP5bLao8EwyulZWduOCwRhY1K7JCgoqHnCLbeEiyWSBI+V/5VIEqUnTuz/+NKl86MuHw9FjkmTBHJPjU/wLSKR7F6aVspvvlnYbjI1GDzRp0AQeotK9dSTcnn2+wCeP/MxEhhDZ2np5s32S6zvpfB0s+tF6enrRq1nTiJHwDKRomS5cnnuoaysrX+RSOLcqvUol2fPf/DBt9SDhYNu95ybo4GbYGhD0LJlH+s9rMzLvb2GF8rKXn5hpIRtQ5GDiGNc8J3NduYfra1lRVZru1qvV7c4M2aYpFsFgtA7xeJpi2k66T67RsbO1X+j01W+UVGx45mBrSYWS3s1RUk9KZCJQqHs+fT0dbBnz9rrdn2SO+TjjpsEgtCHFIrchwZ/NgPAUYxxm1ZbOHDikGGSgkWiKCkAJA4+fH3SFaxWYzUMOdLba6zwxpKZUCh7LjNz651Wa/v+7u4WU1jY9DsYRvVjIo5xjRgA7kUIgULhsamvx7FYjAPnXQYusWhaNSslZVW1r50iEHjC8V27Hr4Lhra79/a2H7eLxtdeEQh8oKOj7lrO2AGBWCyG/t5e46s+9YpA4Ad9zc17/zT0w7XdvIcPv/fm4HFRAmHcYrEY9hkMVddqsVwTiMnUYOnqav6rzzwjEHhAa2v5dSW0rzsP0tGh+XDMPSIQeALGuPH06Zovhz93nUBaW8u/tlgM/xhzzwgE33NZoyl4srf3+rIY1wnEZuvGR4++/xiZixDGGxaL4cva2sLvlf7+3pFb+1zEbG7eOmaeEQi+57uWlgPrR3phxDPpavXbdoEYve4WgcADOjrqXmpuLv7XSK+NmMr+4sWzlxBC+yWS+IcA4Gave0gg+Ii+vu69hw79/mcXL54dMdWQw1oPJlO9iaJkF0Ui6b1e9ZBA8B0X1Oq3s7q6GsyODJwWQ+nqaj46ebKCDg4W3ekV9wgEH4Extmo0Bfe3tu5zugfRqUAuXz6Pm5v3fUbTytnBwaIZHveSQPARWm3hM7W1hYWj2bFKHFdb+7enAJDaI54RCL7lck+PYX1LS/m7bIxZpxiPjEyYmJ6+4SAAVrnlHoHgQ+ziKClZ+wJbe9YFGc+d6+o/cWL/3yZPVkwNDhZ5M1kXgeBx7HMOrbbgmerqv+y4fPk863YcKpYCXLp0/kJz8758mlaGBAeL5rniKIHgAy5otYX3a7WFhVzEAVwFMoTRWL3/0iVbU0RE/Nyr1ZIIBH7S12curap6O6ulxflqlSPcKnPEMEmTk5OfKQRAZF5C4BvfmUx1z1ZV7Xyda72R4XikDtisWStWxMYu2YwQIskYCL7mssViKGptPfBSU1PxEXc782ihvOTkvDyGSXoFAII92S+BwA7cevLkkZ9WVr7GqXybMzxeSVIkiqKio9OeiItbshwAkTvwBG/TZ7UaStrbNX9tbT3wd3cup0bCq6VW5fLsOXJ5zjMIoSwAcCsFJYFwPfh4R0f9J01Npe8bjZVd3hplTGoRC4VRkyhKNpthlCk33xx2V3h4TBwAzCA7hQks6QSAJr2+sqa/P+iowfDNN3p9ldMUpp7CZ8W6b71VFnTHHcqpGEMMQnA7xiABgEmJidkk0oxTMEZX6uoKzg6eaDUA4FNXrkzQ1td/6rOcbf8/AAD//xCk4tUI5ZI0AAAAAElFTkSuQmCC"


#region debug-point capture-webcam-source-helper
def _dbg_capture_webcam_source(hypothesis_id: str, run_id: str, location: str, msg: str, data: dict) -> None:
    _paths = [".dbg/packaged-webcam-bounce.env", ".dbg/capture-webcam-source.env"]
    _u = "http://127.0.0.1:7777/event"
    _s = "packaged-webcam-bounce"
    try:
        for _p in _paths:
            if not Path(_p).exists():
                continue
            with open(_p, "r", encoding="utf-8") as f:
                for _line in f.read().splitlines():
                    if _line.startswith("DEBUG_SERVER_URL="):
                        _u = _line.split("=", 1)[1].strip() or _u
                    elif _line.startswith("DEBUG_SESSION_ID="):
                        _s = _line.split("=", 1)[1].strip() or _s
            break
        urllib.request.urlopen(
            urllib.request.Request(
                _u,
                data=json.dumps(
                    {
                        "sessionId": _s,
                        "runId": run_id,
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": msg,
                        "data": data,
                        "ts": int(time.time() * 1000),
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.5,
        ).read()
    except Exception:
        pass
#endregion


#region debug-point open-project-layout-helper
def _dbg_open_project_layout(hypothesis_id: str, run_id: str, location: str, msg: str, data: dict) -> None:
    _paths = [".dbg/open-project-layout.env"]
    _u = "http://127.0.0.1:7777/event"
    _s = "open-project-layout"
    try:
        for _p in _paths:
            if not Path(_p).exists():
                continue
            with open(_p, "r", encoding="utf-8") as f:
                for _line in f.read().splitlines():
                    if _line.startswith("DEBUG_SERVER_URL="):
                        _u = _line.split("=", 1)[1].strip() or _u
                    elif _line.startswith("DEBUG_SESSION_ID="):
                        _s = _line.split("=", 1)[1].strip() or _s
            break
        urllib.request.urlopen(
            urllib.request.Request(
                _u,
                data=json.dumps(
                    {
                        "sessionId": _s,
                        "runId": run_id,
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": msg,
                        "data": data,
                        "ts": int(time.time() * 1000),
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.5,
        ).read()
    except Exception:
        pass
#endregion


def _app_data_dir() -> Path:
    # Delegates to dataset_io: the training worker and the EMNIST cache
    # need the same directory but cannot import streamlit, so the rule has
    # one home there rather than a copy per entry point.
    return app_data_dir()


APP_DATA_DIR = _app_data_dir()
WORKSPACE_DIR = APP_DATA_DIR / "workspace"
DATASETS_DIR = APP_DATA_DIR / "datasets"


def _documents_dir() -> Path:
    return (Path.home() / "Documents").resolve()


def _default_export_dir() -> Path:
    return (_documents_dir() / APP_NAME / "exports").resolve()


def _pick_directory_dialog(initial_dir: Optional[str] = None) -> Optional[str]:
    if sys.platform == "darwin":
        try:
            start_dir = Path(initial_dir).expanduser().resolve() if initial_dir else _documents_dir()
        except Exception:
            start_dir = _documents_dir()
        try:
            start_posix = str(start_dir).replace("\\", "\\\\").replace('"', '\\"')
            script = (
                'POSIX path of (choose folder with prompt "Choose Folder" '
                f'default location (POSIX file "{start_posix}"))'
            )
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                picked = (proc.stdout or "").strip()
                if picked:
                    return picked
        except Exception:
            pass

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        path = filedialog.askdirectory(initialdir=initial_dir or str(_documents_dir()))
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    return path or None


def _pick_tmproj_file_dialog() -> Optional[Path]:
    picked: Optional[str] = None
    if sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose file with prompt "Open Project")'],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                out = (proc.stdout or "").strip()
                if out:
                    picked = out
        except Exception:
            picked = None

    if not picked:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            return None
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            path = filedialog.askopenfilename(
                title="Open Project",
                filetypes=[("Teachable Machine Project", "*.tmproj")],
                initialdir=str(_documents_dir()),
            )
        finally:
            try:
                root.destroy()
            except Exception:
                pass
        if path:
            picked = path

    if not picked:
        return None
    p = Path(picked).expanduser()
    if p.suffix.lower() != ".tmproj":
        return None
    return p


def _tmproj_read_manifest(path: Path) -> Dict[str, Any]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            raw = zf.read("manifest.json")
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tmproj_detect_project_type(path: Path, manifest: Dict[str, Any]) -> str:
    """Which workspace a .tmproj belongs to.

    The MANIFEST is authoritative — it is written by the one manifest writer
    (_project_save) from the session's project_type.  The filename is only a
    fallback for archives written before `project_type` existed, and it must
    stay a *prefix* test so `ocr_*.tmproj` never matches the "image" test.
    """
    ptype = str(manifest.get("project_type") or "").strip().lower()
    name = str(path.name or "").lower()
    if "pose" in ptype or "pose" in name:
        return "pose"
    if ptype in KNOWN_PROJECT_TYPES:
        return ptype
    # "sign" is tested BEFORE "ocr" because a road-sign archive written by an
    # older build would have been named "sign_ocr_*.tmproj" — which contains
    # both words, and the OCR test would win by document order alone.  The
    # narrower type has to be asked about first.
    if "sign" in name:
        return PROJECT_TYPE_SIGN_OCR
    if "ocr" in name or "glyph" in name or "char" in name:
        return PROJECT_TYPE_OCR
    if "image" in ptype or "image" in name:
        return PROJECT_TYPE_IMAGE
    return PROJECT_TYPE_IMAGE


def _tmproj_manifest_status(manifest: Dict[str, Any]) -> str:
    """How much of a picked .tmproj we actually recognise.

    `current` — carries the format magic, so every field in it is one this
    build wrote.  `legacy` — no magic, but it is ours: our v1 writer stamped
    "format": "tmproj" plus the dataset/state path keys, and an archive with
    those is readable field-by-field with the defaults filling the gaps.
    `unreadable`/`foreign` — a ZIP that was never one of ours, which is worth
    saying at the picker instead of failing deep inside the import with
    "missing dataset".
    """
    if not manifest:
        return "unreadable"
    if read_project_header(manifest)["known"]:
        return "current"
    if str(manifest.get("format") or "").strip() == "tmproj" or manifest.get("dataset_dir"):
        return "legacy"
    return "foreign"


def _tmproj_pick_and_classify() -> Tuple[Optional[Tuple[Path, str]], str]:
    """Pick a .tmproj, decide which workspace it belongs to, or say why not.

    The one gate every "Open Project" card shares: choose a file, read its
    manifest, check the format header, resolve the project type.  A file that
    is not one of ours is refused HERE, where the message can name the file,
    instead of surfacing three layers down as "missing dataset".

    Returns `(picked, error)` — the message is handed BACK rather than drawn
    here, so the caller can print it at full width instead of inside one
    narrow card column:

      · picker cancelled   → `(None, "")`      a silent no-op, no banner;
      · unopenable file    → `(None, why)`     the caller reports `why`;
      · a good project     → `((path, type), "")`.
    """
    p = _pick_tmproj_file_dialog()
    if not p:
        # Cancelling the native dialog is not an error and not worth a
        # warning box: the user simply changed their mind.
        return None, ""
    manifest = _tmproj_read_manifest(p)
    status = _tmproj_manifest_status(manifest)
    if status == "unreadable":
        return None, f"'{p.name}' has no readable manifest.json — it is not a project file this app wrote."
    if status == "foreign":
        return None, (
            f"'{p.name}' is not a project file this app wrote: its manifest has neither the "
            f"{PROJECT_MAGIC} header nor the older format marker."
        )
    ptype = _tmproj_detect_project_type(p, manifest)
    # The same whitelist gate, but written against the registry constant rather
    # than a literal list: a type this build can route to (dataset_io knows it
    # and main() has a page for it) must not be refused here, or the card and
    # the gate disagree about what the app supports.
    if ptype not in _TM_OPENABLE_PROJECT_TYPES:
        return None, f"Unsupported project type: {ptype}"
    return (p, ptype), ""


def _validate_export_inputs(export_dir: Path, model_name: str, array_name: str, tflite_path: Path) -> List[str]:
    errors: List[str] = []
    if not model_name.strip():
        errors.append("Model file prefix cannot be empty.")
    if any(sep in model_name for sep in ("/", "\\", os.sep)):
        errors.append("Model file prefix must not include path separators.")
    if not re.match(r"^[A-Za-z0-9_.-]+$", model_name):
        errors.append("Model file prefix may only contain letters, numbers, '.', '_' and '-'.")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", array_name):
        errors.append("C array name must be a valid identifier (letters/numbers/underscore, not starting with a number).")
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Failed to create export directory: {export_dir} ({e})")
        return errors
    if not export_dir.is_dir():
        errors.append(f"Export path is not a directory: {export_dir}")
    if not tflite_path.exists():
        errors.append("Missing .tflite file. Train the model first.")
    try:
        test_file = export_dir / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except Exception as e:
        errors.append(f"Export directory is not writable: {export_dir} ({e})")
    return errors


def _init_session() -> None:
    if "session_id" not in st.session_state:
        resumed_session_id = ""
        try:
            resumed_session_id = str(st.query_params.get("tm_session") or "")
        except Exception:
            try:
                resumed_session_id = str(st.experimental_get_query_params().get("tm_session", [""])[0] or "")
            except Exception:
                resumed_session_id = ""
        st.session_state.session_id = resumed_session_id.strip() or _restore_last_session_id()
    if "imported" not in st.session_state:
        st.session_state.imported = None
    if "project_type" not in st.session_state:
        resumed_project_type = ""
        try:
            resumed_project_type = str(st.query_params.get("tm_project") or "")
        except Exception:
            try:
                resumed_project_type = str(st.experimental_get_query_params().get("tm_project", [""])[0] or "")
            except Exception:
                resumed_project_type = ""
        st.session_state.project_type = resumed_project_type.strip() or None
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "dataset_dir" not in st.session_state:
        st.session_state.dataset_dir = None
    if "assignments" not in st.session_state:
        st.session_state.assignments = {}
    if "class_names" not in st.session_state:
        st.session_state.class_names = ["Class 1", "Class 2"]
    if "class_rename" not in st.session_state:
        st.session_state.class_rename = {}
    if "last_export_dir" not in st.session_state:
        st.session_state.last_export_dir = str(_default_export_dir())
    if "train_cfg" not in st.session_state:
        st.session_state.train_cfg = TrainConfig()
    if "train_result" not in st.session_state:
        st.session_state.train_result = None
    if "local_import_path" not in st.session_state:
        st.session_state.local_import_path = ""
    if "export_validated_token" not in st.session_state:
        st.session_state.export_validated_token = ""
    if "tm_classes" not in st.session_state:
        st.session_state.tm_classes = ["Class 1", "Class 2"]
    if "tm_device_url" not in st.session_state:
        st.session_state.tm_device_url = ""
    if "tm_device_enabled" not in st.session_state:
        st.session_state.tm_device_enabled = False
    if "tm_serial_port" not in st.session_state:
        st.session_state.tm_serial_port = ""
    if "tm_serial_baud" not in st.session_state:
        st.session_state.tm_serial_baud = 115200
    if "tm_serial_sync" not in st.session_state:
        st.session_state.tm_serial_sync = "AA 55 AA"
    if "tm_serial_frame_side" not in st.session_state:
        st.session_state.tm_serial_frame_side = 96
    if "tm_serial_channels" not in st.session_state:
        st.session_state.tm_serial_channels = 1
    if "tm_preview_dark_thresh" not in st.session_state:
        st.session_state.tm_preview_dark_thresh = None
    if "tm_preview_lum_thresh" not in st.session_state:
        st.session_state.tm_preview_lum_thresh = None
    if "tm_last_device_frame" not in st.session_state:
        st.session_state.tm_last_device_frame = None
    if "tm_capture_open" not in st.session_state:
        st.session_state.tm_capture_open = False
    if "tm_capture_source" not in st.session_state:
        st.session_state.tm_capture_source = ""
    if "tm_capture_class" not in st.session_state:
        st.session_state.tm_capture_class = ""
    if "tm_record_fps" not in st.session_state:
        st.session_state.tm_record_fps = 8.0
    if "tm_webcam_index" not in st.session_state:
        st.session_state.tm_webcam_index = 0
    if "tm_webcam_id" not in st.session_state:
        st.session_state.tm_webcam_id = ""
    if "tm_webcam_user_selected" not in st.session_state:
        st.session_state.tm_webcam_user_selected = False
    if "tm_record_crop_box" not in st.session_state:
        st.session_state.tm_record_crop_box = None
    if "tm_crop_mode" not in st.session_state:
        st.session_state.tm_crop_mode = "full"
    if "tm_pending_image" not in st.session_state:
        st.session_state.tm_pending_image = None
    if "tm_pending_class" not in st.session_state:
        st.session_state.tm_pending_class = None
    if "tm_camera_permission_status" not in st.session_state:
        st.session_state.tm_camera_permission_status = ""
    if "tm_camera_permission_note" not in st.session_state:
        st.session_state.tm_camera_permission_note = ""
    if "tm_camera_permission_class" not in st.session_state:
        st.session_state.tm_camera_permission_class = ""
    if "tm_open_source_class" not in st.session_state:
        st.session_state.tm_open_source_class = ""
    if "tm_open_source_kind" not in st.session_state:
        st.session_state.tm_open_source_kind = ""
    if "tm_frontend_notice" not in st.session_state:
        st.session_state.tm_frontend_notice = ""
    if "tm_return_target" not in st.session_state:
        st.session_state.tm_return_target = "home"
    if "tm_current_project_path" not in st.session_state:
        st.session_state.tm_current_project_path = None


def _restore_last_session_id() -> str:
    """Restore the last-used session id so app restarts keep the SAME
    workspace (the trained model lives there — a fresh uuid on every
    restart made it look like the model never updated)."""
    try:
        p = APP_DATA_DIR / "last_session_id.txt"
        if p.exists():
            sid = p.read_text(encoding="utf-8").strip()
            if sid and len(sid) == 32 and (WORKSPACE_DIR / sid).exists():
                return sid
    except Exception:
        pass
    return uuid.uuid4().hex


def _persist_session_id() -> None:
    try:
        sid = str(st.session_state.get("session_id") or "").strip()
        if sid:
            APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
            (APP_DATA_DIR / "last_session_id.txt").write_text(sid, encoding="utf-8")
    except Exception:
        pass


def _session_workspace() -> Path:
    global WORKSPACE_DIR
    p = WORKSPACE_DIR / st.session_state.session_id
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except PermissionError:
        fallback = (Path(tempfile.gettempdir()) / APP_NAME / "workspace").resolve()
        fallback.mkdir(parents=True, exist_ok=True)
        WORKSPACE_DIR = fallback
        p = WORKSPACE_DIR / st.session_state.session_id
        p.mkdir(parents=True, exist_ok=True)
        return p


def _reset_session_workspace() -> None:
    p = WORKSPACE_DIR / st.session_state.session_id
    if p.exists():
        shutil.rmtree(p)
    st.session_state.imported = None
    st.session_state.project_type = None
    st.session_state.step = 0
    st.session_state.dataset_dir = None
    st.session_state.assignments = {}
    st.session_state.class_names = ["Class 1", "Class 2"]
    st.session_state.class_rename = {}
    st.session_state.train_cfg = TrainConfig()
    st.session_state.train_result = None
    st.session_state.export_validated_token = ""
    st.session_state.tm_classes = ["Class 1", "Class 2"]
    st.session_state.tm_device_enabled = False
    st.session_state.tm_crop_mode = "full"
    st.session_state.tm_pending_image = None
    st.session_state.tm_pending_class = None
    st.session_state.tm_last_device_frame = None
    st.session_state.tm_capture_open = False
    st.session_state.tm_capture_source = ""
    st.session_state.tm_capture_class = ""
    st.session_state.tm_record_crop_box = None
    st.session_state.tm_camera_permission_status = ""
    st.session_state.tm_camera_permission_note = ""
    st.session_state.tm_camera_permission_class = ""
    st.session_state.tm_open_source_class = ""
    st.session_state.tm_open_source_kind = ""
    st.session_state.tm_frontend_notice = ""
    st.session_state.tm_webcam_index = 0
    st.session_state.tm_webcam_id = ""
    st.session_state.tm_webcam_user_selected = False
    st.session_state.tm_current_project_path = None


def _begin_fresh_tm_session() -> str:
    # #region debug-point B:begin-fresh-session
    _dbg_open_project_layout("B", "pre-fix", "app.py:_begin_fresh_tm_session", "[DEBUG] begin fresh session", {"old_session": str(st.session_state.get("session_id", ""))})
    # #endregion
    _reset_session_workspace()
    st.session_state.session_id = uuid.uuid4().hex
    # #region debug-point B:begin-fresh-session-new
    _dbg_open_project_layout("B", "pre-fix", "app.py:_begin_fresh_tm_session", "[DEBUG] fresh session created", {"new_session": str(st.session_state.session_id)})
    # #endregion
    return st.session_state.session_id


def _tm_teardown_controller_session(controller: "RecordController", session_id: str) -> None:
    """Stop live workers + record threads and drop all per-session state
    (configs, conditions, live/record threads, train/preview entries) so the
    singleton controller releases camera/serial handles. Safe to call multiple
    times; every step is guarded."""
    if not session_id:
        return
    try:
        controller._stop_record(session_id=session_id, wait=True, timeout_s=2.5)
    except Exception:
        pass
    try:
        controller._stop_live(session_id=session_id, source=None, wait=True, timeout_s=2.0)
    except Exception:
        pass
    try:
        controller.drop_session(session_id)
    except Exception:
        pass


def _tm_handle_return_home(controller: "RecordController") -> None:
    """Return-to-home handler (called from the SPA 'Return' button via a
    hidden st.button click, and from the tm_action=home query-param path):
    stop live preview (/live/close for ALL sources) + any hold recording
    (/stop), drop the session from the record controller, reset Streamlit
    session state so the home page renders fresh, and rewrite the parent URL.
    st.query_params is applied by Streamlit's own (non-sandboxed) frontend
    JS, so the address bar resets even though the components.html iframe
    cannot navigate the parent (no allow-top-navigation in its sandbox)."""
    session_id = str(st.session_state.get("session_id", ""))
    _tm_teardown_controller_session(controller, session_id)
    target = str(st.session_state.get("tm_return_target", "home") or "home")
    if target == "classified-import":
        # Return to the classified-import page: keep the workspace and URL,
        # reset only the SPA-side state.
        st.session_state.tm_open_source_class = ""
        st.session_state.tm_open_source_kind = ""
        st.session_state.tm_frontend_notice = ""
        st.session_state.tm_capture_open = False
        st.session_state.tm_capture_source = ""
        st.session_state.tm_capture_class = ""
        st.session_state.tm_pending_image = None
        st.session_state.tm_pending_class = None
        st.session_state.tm_last_device_frame = None
        st.session_state.project_type = "image_classified_import"
        st.session_state.step = 0
        _tm_set_query_params(tm_project="image_classified_import", tm_session=session_id)
    else:
        _reset_session_workspace()
        _tm_clear_query_params()
    st.rerun()


def _read_uploaded_images(uploaded_files) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    for f in uploaded_files:
        out.append((f.name, f.getvalue()))
    return out


def _render_import_panel() -> None:
    st.subheader("Import images")
    method = st.radio("Import method", ["ZIP file", "Multiple images", "Local folder path"], horizontal=True)
    ws = _session_workspace()

    if method == "ZIP file":
        zip_file = st.file_uploader("Drop or select a .zip file", type=["zip"])
        if zip_file is not None and st.button("Import", type="primary"):
            dest = ws / "import"
            materialize_zip_bytes(zip_file.getvalue(), dest)
            st.session_state.imported = infer_imported_data(dest)
            st.session_state.assignments = {}
            st.session_state.class_rename = {}

    elif method == "Multiple images":
        exts = sorted([e.lstrip(".") for e in IMAGE_EXTS])
        files = st.file_uploader("Select or drop multiple images", type=exts, accept_multiple_files=True)
        if files and st.button("Import", type="primary"):
            dest = ws / "import"
            materialize_files(_read_uploaded_images(files), dest)
            st.session_state.imported = infer_imported_data(dest)
            st.session_state.assignments = {}
            st.session_state.class_rename = {}

    else:
        left, right = st.columns([4, 1])
        with left:
            path_str = st.text_input(
                "Local folder path (classified or unclassified)",
                value=st.session_state.local_import_path,
            )
            st.session_state.local_import_path = path_str
        with right:
            if st.button("Browse...", key="browse_import_dir"):
                picked = _pick_directory_dialog(initial_dir=st.session_state.local_import_path)
                if picked:
                    st.session_state.local_import_path = picked
                    st.rerun()

        if st.button("Load folder", type="primary"):
            path_str = st.session_state.local_import_path
            p = Path(path_str).expanduser()
            if not p.exists() or not p.is_dir():
                st.error("Path does not exist or is not a folder.")
                return
            st.session_state.imported = infer_imported_data(p)
            st.session_state.assignments = {}
            st.session_state.class_rename = {}


def _render_overview(imported: ImportedData) -> None:
    st.subheader("Preview")
    st.markdown(
        f'''
<div class="tm-kv">
  <b>Source</b>: {imported.root_dir}<br/>
  <b>Images</b>: {len(imported.images)}<br/>
  <b>Classified folders</b>: {imported.classified}
</div>
        ''',
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    sample = imported.images[: min(len(imported.images), 6)]
    for c, p in zip(cols, sample):
        with c:
            st.image(str(p), use_container_width=True)
            st.caption(p.name)


def _render_classified_flow(imported: ImportedData) -> Optional[Path]:
    st.subheader("Rename classes (classified folders)")
    class_names = list(imported.class_to_images.keys())
    st.write({"Detected classes": class_names})

    rename: Dict[str, str] = {}
    for c in class_names:
        rename[c] = st.text_input(f"{c} →", value=st.session_state.class_rename.get(c, c), key=f"rename_{c}")
    st.session_state.class_rename = rename

    st.write({c: len(imported.class_to_images[c]) for c in class_names})

    if st.button("Save as dataset", type="primary"):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = new_output_dir(DATASETS_DIR, prefix="classified")
        try:
            export_classified_dataset(imported.class_to_images, rename, out_dir)
        except Exception as e:
            st.error(str(e))
            return None
        st.success(f"Dataset created: {out_dir}")
        return out_dir
    return None


def _render_unclassified_flow(imported: ImportedData) -> Optional[Path]:
    st.subheader("Assign classes (unclassified images)")
    st.session_state.class_names = _render_class_editor(st.session_state.class_names)
    class_names = st.session_state.class_names

    if not class_names:
        st.warning("Create at least 1 class first.")
        return None

    page_size = st.slider("Images per page", min_value=6, max_value=60, value=18, step=6)
    total = len(imported.images)
    max_page = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)
    start = (page - 1) * page_size
    end = min(total, start + page_size)

    for idx in range(start, end):
        img = imported.images[idx]
        left, right = st.columns([1, 2])
        with left:
            st.image(str(img), use_container_width=True)
        with right:
            default = st.session_state.assignments.get(idx)
            options = ["Unassigned"] + class_names
            try:
                default_index = options.index(default) if default in options else 0
            except Exception:
                default_index = 0
            choice = st.selectbox(
                f"[{idx}] {img.name}",
                options=options,
                index=default_index,
                key=f"assign_{idx}",
            )
            if choice == "Unassigned":
                st.session_state.assignments.pop(idx, None)
            else:
                st.session_state.assignments[idx] = choice

    require_all = st.checkbox("Require all images to be assigned before saving", value=True)
    assigned_count = len(st.session_state.assignments)
    st.write({"Assigned": assigned_count, "Unassigned": total - assigned_count})

    if st.button("Save as dataset", type="primary"):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = new_output_dir(DATASETS_DIR, prefix="labeled")
        assignments: List[Optional[str]] = []
        for i in range(total):
            assignments.append(st.session_state.assignments.get(i))
        try:
            export_from_assignments(imported.images, assignments, out_dir, require_all_assigned=require_all)
        except Exception as e:
            st.error(str(e))
            return None
        st.success(f"Dataset created: {out_dir}")
        return out_dir

    return None


def _render_class_editor(class_names: List[str]) -> List[str]:
    st.write("Classes (one per line)")
    txt = st.text_area(" ", value="\n".join(class_names), height=140, key="class_editor")
    names = [x.strip() for x in txt.splitlines()]
    names = [x for x in names if x]
    return names


def _render_steps() -> None:
    steps = ["Import", "Label", "Train", "Export"]
    s = int(st.session_state.step)
    parts: List[str] = []
    for i, name in enumerate(steps):
        if i == s:
            parts.append(f'<span class="tm-step-on">{i+1}. {name}</span>')
        else:
            parts.append(f"{i+1}. {name}")
    st.markdown(f'<div class="tm-steps">{"  ·  ".join(parts)}</div>', unsafe_allow_html=True)


def _next_class_name(existing: List[str]) -> str:
    idx = 1
    existing_set = {sanitize_class_name(name) for name in existing}
    while True:
        candidate = f"Class {idx}"
        if sanitize_class_name(candidate) not in existing_set:
            return candidate
        idx += 1


def _rename_tm_class_dir(old_name: str, new_name: str) -> None:
    old_safe = sanitize_class_name(old_name)
    new_safe = sanitize_class_name(new_name)
    if old_safe == new_safe:
        return
    root = _tm_dataset_dir()
    old_dir = root / old_safe
    new_dir = root / new_safe
    if not old_dir.exists():
        return
    if new_dir.exists():
        raise ValueError(f"Class already exists: {new_safe}")
    old_dir.rename(new_dir)


def _apply_tm_class_names(current_names: List[str], edited_names: List[str]) -> List[str]:
    normalized: List[str] = []
    used: set[str] = set()
    for idx, raw in enumerate(edited_names):
        candidate = sanitize_class_name(raw) if raw.strip() else f"Class {idx + 1}"
        if candidate in used:
            raise ValueError(f"Duplicate class name: {candidate}")
        used.add(candidate)
        normalized.append(candidate)

    for old_name, new_name in zip(current_names, normalized):
        _rename_tm_class_dir(old_name, new_name)

    return normalized


def _remove_tm_class(class_name: str) -> None:
    class_dir = _tm_dataset_dir() / sanitize_class_name(class_name)
    if class_dir.exists():
        shutil.rmtree(class_dir)
    if st.session_state.tm_capture_class == class_name:
        st.session_state.tm_capture_open = False
        st.session_state.tm_capture_source = ""
        st.session_state.tm_capture_class = ""
    if st.session_state.tm_camera_permission_class == class_name:
        st.session_state.tm_camera_permission_class = ""
        st.session_state.tm_camera_permission_note = ""
        st.session_state.tm_camera_permission_status = ""
    if st.session_state.tm_pending_class == class_name:
        st.session_state.tm_pending_class = None
        st.session_state.tm_pending_image = None


def _import_classified_into_workspace(imported: ImportedData, rename: Dict[str, str]) -> List[str]:
    dataset_root = _tm_dataset_dir()
    export_classified_dataset(imported.class_to_images, rename, dataset_root)
    labels: List[str] = []
    for class_name in imported.class_to_images.keys():
        labels.append(sanitize_class_name(rename.get(class_name, class_name)))
    if labels:
        _tm_save_classes_meta(labels)
    return labels


def _tm_dataset_stats() -> Tuple[int, int]:
    dataset_dir = _tm_dataset_dir()
    if not dataset_dir.exists():
        return 0, 0
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    sample_count = 0
    for class_dir in class_dirs:
        sample_count += len(_tm_class_image_files(class_dir))
    return len(class_dirs), sample_count


def _tm_class_image_files(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    return sorted(
        [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _tm_train_latest_path() -> Path:
    return _tm_dataset_dir().parent / "tm_train_latest.json"


def _tm_load_train_latest() -> Optional[Dict[str, Any]]:
    p = _tm_train_latest_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tm_classes_meta_path() -> Path:
    return _tm_dataset_dir().parent / "tm_classes.json"


def _tm_load_classes_meta_full() -> Optional[Dict[str, Any]]:
    p = _tm_classes_meta_path()
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    classes = raw.get("classes")
    if not (isinstance(classes, list) and all(isinstance(x, str) for x in classes) and classes):
        return None
    from image_preprocess import normalize_class_preprocess_map, normalize_sample_preprocess_map

    class_preprocess = normalize_class_preprocess_map(raw.get("class_preprocess"))
    sample_preprocess = normalize_sample_preprocess_map(raw.get("sample_preprocess"))
    return {
        "classes": [str(x) for x in classes],
        "class_preprocess": class_preprocess,
        "sample_preprocess": sample_preprocess,
    }


def _tm_load_classes_meta() -> Optional[List[str]]:
    meta = _tm_load_classes_meta_full()
    if not meta:
        return None
    classes = meta.get("classes")
    if isinstance(classes, list) and all(isinstance(x, str) for x in classes) and classes:
        return [str(x) for x in classes]
    return None


def _tm_save_classes_meta(classes: List[str]) -> None:
    p = _tm_classes_meta_path()
    # Merge with existing meta — preserve class_preprocess / sample_preprocess
    # written by record_controller._classes_save; only update the classes list.
    existing: Dict[str, Any] = {}
    try:
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
    except Exception:
        existing = {}
    payload: Dict[str, Any] = dict(existing)
    payload["classes"] = list(classes)
    # A brand-new session routes here before anything has created the dataset
    # directory (set_config does not touch the filesystem), so creating it is
    # this writer's job — without it the first render of a fresh workspace
    # raised FileNotFoundError on tm_classes.json.
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _status_style(status: str) -> str:
    if status in {"granted", "not_required", "trained", "ready"}:
        return "tm-status tm-status-ok"
    if status in {"denied", "restricted", "open_failed", "timeout"}:
        return "tm-status tm-status-bad"
    if status in {"not_determined"}:
        return "tm-status tm-status-warn"
    return "tm-status tm-status-idle"


def _render_licenses_expander() -> None:
    """In-product third-party license declaration.

    The commercial distribution obligation: every bundled open-source
    component's license declaration must ship WITH the app and stay
    reachable from the UI.  The notices file is generated from the venv
    (tools/gen_third_party_notices.py).  Called from the Start Page and
    from every workspace header so the declaration is always findable.
    """
    try:
        _notices_path = os.path.join(
            os.path.dirname(__file__) or ".", "THIRD_PARTY_NOTICES.md"
        )
        if os.path.exists(_notices_path):
            with open(_notices_path, encoding="utf-8") as _nfh:
                _notices_text = _nfh.read()
            with st.expander("📄 Open Source Licenses"):
                st.markdown(_notices_text)
                _license_path = os.path.join(os.path.dirname(__file__) or ".", "LICENSE")
                if os.path.exists(_license_path):
                    with open(_license_path, encoding="utf-8") as _lfh:
                        st.markdown("### This software's license (Apache License 2.0)\n")
                        st.code(_lfh.read(), language=None)
    except Exception:
        pass


def _render_tm_workspace_header() -> None:
    class_count, sample_count = _tm_dataset_stats()
    meta = _tm_load_train_latest()
    metrics = dict(meta.get("metrics") or {}) if isinstance(meta, dict) else {}
    trained_label = f"{float(metrics.get('val_accuracy', 0.0)):.1%}" if metrics else "Not trained"
    trained_status = "trained" if metrics else "idle"
    capture_label = st.session_state.tm_capture_source.upper() if st.session_state.tm_capture_open else "Idle"
    camera_status = get_camera_access_status()
    st.markdown(
        f'''
<div class="tm-hero">
  <div class="tm-hero-compact">
    <div class="tm-hero-copy">
      <div class="tm-eyebrow">AI Training Platform</div>
      <h2>Image Training Workspace</h2>
      <p>Collect samples, train, preview, and export in a single workspace.</p>
    </div>
    <div class="tm-hero-stats">
      <div class="tm-stat">
        <div class="tm-stat-label">Classes</div>
        <div class="tm-stat-value">{class_count}</div>
      </div>
      <div class="tm-stat">
        <div class="tm-stat-label">Samples</div>
        <div class="tm-stat-value">{sample_count}</div>
      </div>
      <div class="tm-stat">
        <div class="tm-stat-label">Model</div>
        <div class="tm-stat-value">{html_escape(trained_label)}</div>
        <div class="tm-stat-note"><span class="{_status_style(trained_status)}">{'Ready' if metrics else 'Waiting'}</span></div>
      </div>
      <div class="tm-stat">
        <div class="tm-stat-label">Capture</div>
        <div class="tm-stat-value">{html_escape(capture_label)}</div>
        <div class="tm-stat-note"><span class="{_status_style(camera_status.status)}">{html_escape(camera_status.status.replace('_', ' ').title())}</span></div>
      </div>
    </div>
  </div>
</div>
        ''',
        unsafe_allow_html=True,
    )
    _render_licenses_expander()


def _render_tm_flow_lane() -> None:
    c1, a1, c2, a2, c3 = st.columns([2.2, 0.22, 1.0, 0.22, 1.75], gap="small")
    with c1:
        st.markdown('<div class="tm-flow-step"><span>1</span><strong>Classes</strong><small>Add samples and rename classes</small></div>', unsafe_allow_html=True)
    with a1:
        st.markdown('<div class="tm-flow-arrow">→</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="tm-flow-step"><span>2</span><strong>Training</strong><small>Train a lightweight model</small></div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="tm-flow-arrow">→</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="tm-flow-step"><span>3</span><strong>Preview / Export</strong><small>Check inputs and export files</small></div>', unsafe_allow_html=True)


def _render_camera_permission_card() -> None:
    status = get_camera_access_status()
    left, right = st.columns([3, 1.2])
    with left:
        st.markdown(
            f'''
<div class="tm-inline-note">
  <strong>Camera</strong>
  <div style="margin-top:6px;"><span class="{_status_style(status.status)}">{html_escape(status.message)}</span></div>
</div>
            ''',
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Request", key="tm_request_camera_access", use_container_width=True):
            result = ensure_camera_access(int(st.session_state.tm_webcam_index or 0), probe_open=False, unique_id=str(st.session_state.get("tm_webcam_id") or ""))
            st.session_state.tm_camera_permission_status = result.status
            st.session_state.tm_camera_permission_note = result.message
            st.rerun()

    note = st.session_state.tm_camera_permission_note
    if note:
        level = st.success if st.session_state.tm_camera_permission_status in {"granted", "not_required"} else st.info
        level(note)


def _tm_session_config(**overrides: Any) -> SessionConfig:
    """The ONE SessionConfig builder for the SPA.

    Every workspace page used to construct its own copy of this literal, so
    each new session-level field (project_type, img_size, ...) had to be
    added in three places or it silently defaulted on two of them.  Build it
    here and override per call site instead.
    """
    values: Dict[str, Any] = {
        "dataset_root": _tm_dataset_dir(),
        "serial_port": st.session_state.tm_serial_port,
        "serial_baud": int(st.session_state.tm_serial_baud),
        "serial_sync": str(st.session_state.tm_serial_sync),
        "serial_frame_side": int(st.session_state.tm_serial_frame_side),
        "serial_channels": int(st.session_state.tm_serial_channels),
        "webcam_index": int(st.session_state.tm_webcam_index),
        "webcam_id": str(st.session_state.get("tm_webcam_id") or ""),
        "img_size": int(getattr(st.session_state.get("train_cfg"), "img_size", 96) or 96),
        "fps": float(st.session_state.tm_record_fps),
        "crop_box": st.session_state.tm_record_crop_box,
        "project_type": str(st.session_state.get("project_type") or PROJECT_TYPE_IMAGE),
    }
    values.update(overrides)
    return SessionConfig(**values)


# ── TrainConfig plumbing ────────────────────────────────────────────────
# TrainConfig is frozen and every field has a default, so `TrainConfig(...)`
# silently resets whatever the call site forgets to pass.  Three places here
# used to construct it from a hand-picked list of ~14 fields, which meant
# switching project type, changing the model-input size, or reopening a
# saved project all reset crop_mode (the model-input geometry!) and all 13
# ocr_* mixing / two-phase knobs to their dataclass defaults — the settings
# the archive was saved with were read back, then overwritten.
#
# These two helpers are the only sanctioned ways to build one, and both
# derive the field list from the dataclass, so a field added to TrainConfig
# is carried through without an edit here.

# Not part of the SPA's train_cfg, with the reason each time.  The first two
# are published as their own top-level payload keys and normalised by their
# own code paths — a second copy inside train_cfg would be a second thing to
# keep in sync; the other two are decided by routing / derived at train time.
_TRAIN_CFG_SESSION_KEYS = (
    "class_preprocess",
    "sample_preprocess",
    "project_type",
    "class_names_order",
)


def _tm_train_cfg_state_keys() -> Tuple[str, ...]:
    """TrainConfig fields the SPA's train_cfg carries — and therefore saves.

    The save payload is `STATE.train_cfg` sent verbatim, so this list IS the
    set of training parameters a project file can hold.
    """
    return tuple(f.name for f in dataclasses.fields(TrainConfig) if f.name not in _TRAIN_CFG_SESSION_KEYS)


def _tm_train_cfg_replace(cfg: TrainConfig, **overrides: Any) -> TrainConfig:
    """Rebuild `cfg` with `overrides` applied, preserving every other field."""
    values = {f.name: getattr(cfg, f.name, f.default) for f in dataclasses.fields(TrainConfig)}
    values.update(overrides)
    return TrainConfig(**values)


def _tm_train_cfg_from_state(state_cfg: Any, base: TrainConfig, **overrides: Any) -> TrainConfig:
    """Layer an archived train_cfg over `base`, then apply `overrides`.

    The archive is what the user actually chose when they saved; `base` fills
    only the gaps.  Values are taken as stored rather than re-validated — the
    record controller's pick_train_cfg is the one place they were coerced and
    checked, on the way into the archive.
    """
    src = state_cfg if isinstance(state_cfg, dict) else {}
    values = {f.name: getattr(base, f.name, f.default) for f in dataclasses.fields(TrainConfig)}
    for name in list(values):
        if name in src and src[name] is not None:
            values[name] = src[name]
    values.update(overrides)
    try:
        return TrainConfig(**values)
    except TypeError:
        # A file written by a NEWER build can name a field this one does not
        # have.  Keep the fields both sides know rather than refusing to open
        # the project over an unknown extra.
        known = {f.name for f in dataclasses.fields(TrainConfig)}
        return TrainConfig(**{k: v for k, v in values.items() if k in known})


def _tm_train_cfg_payload(cfg: TrainConfig) -> Dict[str, Any]:
    """The SPA's train_cfg: every round-trippable field, JSON-safe.

    Derived from the dataclass for the same reason as the helpers above — v1
    hand-listed 19 of the 35 fields here, so the other 16 were not merely
    unsaved, they were unreachable: the Advanced panel renders from this
    dict, and a field that never arrives is a field no user can set.
    """
    out: Dict[str, Any] = {}
    for name in _tm_train_cfg_state_keys():
        value = getattr(cfg, name, None)
        if name == "manual_roi":
            roi = normalize_manual_roi(value)
            out[name] = list(roi) if roi is not None else None
        elif value is None or isinstance(value, (str, bool, int, float)):
            out[name] = value
        elif isinstance(value, (list, tuple)):
            out[name] = list(value)
        else:
            out[name] = None
    return out


def _tm_set_img_size(size: int) -> None:
    """Rebuild the frozen TrainConfig with a new model-input side."""
    prev_cfg = st.session_state.train_cfg
    if int(getattr(prev_cfg, "img_size", 0) or 0) == int(size):
        return
    st.session_state.train_cfg = _tm_train_cfg_replace(prev_cfg, img_size=int(size))


def _tm_apply_project_defaults(project_type: str) -> None:
    """Seed the model-input size from the project type.

    This is the ONLY difference a new project type needs on the Start Page:
    the size comes from DEFAULT_IMG_SIZE_BY_PROJECT rather than a literal
    sprinkled through the cards, and everything downstream (SessionConfig,
    processed cache, model_settings.h kNumCols/kNumRows, .tmproj manifest)
    already reads img_size.
    """
    _tm_set_img_size(int(default_img_size(project_type)))
    st.session_state.tm_classes = ["Class 1", "Class 2"]


def _tm_notice_opened_project(path: Path, state: Any, project_type: str) -> None:
    """Queue the one-shot toast for the page the open lands on.

    A v1 archive predates the complete train_cfg snapshot, so part of what the
    reopened workspace is now configured with came from THIS build's defaults
    rather than from the file.  Which settings, and that it happened at all,
    is the user's business: silently retraining with a different EMNIST mix or
    a different crop geometry than the file implies is exactly the kind of
    difference nobody notices until the model behaves differently.
    """
    fmt = state.get("project_format") if isinstance(state, dict) else None
    fmt = fmt if isinstance(fmt, dict) else {}
    raw_missing = fmt.get("missing_train_cfg_fields")
    missing = [str(m) for m in raw_missing] if isinstance(raw_missing, list) else []
    if not missing:
        if fmt.get("legacy"):
            st.session_state.tm_frontend_notice = (
                f"Opened “{path.name}” — a project saved before this version. "
                "Its settings were restored where it recorded them."
            )
        return
    # Only the ones that change what gets TRAINED are worth naming; the rest
    # are recorded in the file either way.
    named = [n for n in ("crop_mode", "ocr_emnist_per_class", "ocr_emnist_enabled", "ocr_transfer_learning") if n in missing]
    detail = f" ({', '.join(named)})" if named else ""
    st.session_state.tm_frontend_notice = (
        f"Opened “{path.name}” — a v1.0 project. It predates {len(missing)} training "
        f"settings{detail}, which came back at this version's defaults; check Advanced "
        "settings before retraining."
    )


_TM_PROJECT_TYPE_LABELS = {
    "image": "Image Project",
    "ocr": "OCR / Character Project",
    "sign_ocr": "🚗 Car License Project",
    "image_classified_import": "Classified Import",
}


def _tm_project_type_label(ptype: str) -> str:
    key = str(ptype or "").strip().lower()
    return _TM_PROJECT_TYPE_LABELS.get(key, key or "Project")


def _tm_notice_project_type_switch(path: Path, card_type: str, effective_type: str) -> None:
    """Announce a cross-type open instead of letting it happen silently.

    The card a user clicked promises one workspace; an archive whose manifest
    says otherwise is still opened (the file is the authority on what it is),
    but the app has just switched project type under them — and that changes
    the input geometry, the sample transform and the training labels.  Silent
    is how a user ends up training a 28x28 model at 96x96 without noticing.
    """
    want = str(card_type or "").strip().lower()
    got = str(effective_type or "").strip().lower()
    if not want or not got or want == got:
        return
    note = (
        f"Opened “{path.name}” — that file is a {_tm_project_type_label(got)}, "
        f"not a {_tm_project_type_label(want)}. The workspace has switched to "
        f"{_tm_project_type_label(got)}."
    )
    existing = str(st.session_state.get("tm_frontend_notice") or "").strip()
    st.session_state.tm_frontend_notice = f"{note} {existing}".strip()


def _tm_open_project_into_session(p: Path, ptype: str, card_type: str = "") -> None:
    """Open a .tmproj into a FRESH workspace and route to its project type.

    Shared by every Start-Page card — an OCR .tmproj restores exactly the
    same way an image one does, because everything that differs (the sample
    transform, the model input size, the training labels) is already keyed
    off `project_type` server-side.  Adding a card must never mean copying
    this block.

    `card_type` is the type of the CARD the button belonged to.  A file
    whose manifest says otherwise (an Image .tmproj picked from the OCR
    card) is still opened — the archive is the authority on what it is, not
    the button that reached it — but the switch is announced instead of
    happening silently behind a card that promised a different workspace.
    """
    _begin_fresh_tm_session()
    st.session_state.tm_return_target = "home"
    controller = _ensure_fresh_record_controller()
    # Seed the session config from the ARCHIVE's type, not from whatever the
    # previous session was: a legacy file carries no img_size, and building
    # the session at the image default would leave the controller at 96x96
    # while the SPA ran at 28x28 — every later capture would then overwrite
    # the restored cache at the wrong geometry.  _project_open adopts the
    # file's own recorded size on top of this when it has one.
    controller.set_config(
        st.session_state.session_id,
        _tm_session_config(
            project_type=str(ptype),
            img_size=int(default_img_size(str(ptype))),
        ),
    )
    state = controller._project_open(st.session_state.session_id, p)
    st.session_state.tm_classes = list(state.get("classes") or ["Class 1", "Class 2"])
    st.session_state.tm_preview_dark_thresh = 0
    st.session_state.tm_preview_lum_thresh = 100
    # _project_open adopts the manifest's project_type + img_size into the
    # session BEFORE restoring anything, so read them back from the live
    # config rather than assuming the type this card was rendered as.
    opened_cfg = controller.get_config(st.session_state.session_id)
    effective_type = str(getattr(opened_cfg, "project_type", ptype) or ptype)
    effective_size = int(getattr(opened_cfg, "img_size", 0) or 0) or default_img_size(effective_type)
    # Adopt the geometry the archive was built at FIRST (a 28x28 OCR project
    # must not come back as 96x96, or the next capture would overwrite the
    # restored cache at the wrong size), then layer the archived training
    # hyperparameters on top.
    _tm_set_img_size(int(effective_size))
    train_cfg_state = state.get("train_cfg") if isinstance(state, dict) else None
    st.session_state.train_cfg = _tm_train_cfg_from_state(
        train_cfg_state, st.session_state.train_cfg, img_size=int(effective_size)
    )
    st.session_state.project_type = effective_type
    st.session_state.tm_current_project_path = str(p)
    # Tell the user what the file could NOT tell us.  A v1 archive carries a
    # 12-field train_cfg, so its OCR mixing / two-phase settings are not "the
    # project's" — they are this build's defaults, and a retrain would use
    # them.  Saying so is the difference between a restore and a quiet reset.
    _tm_notice_opened_project(p, state, effective_type)
    _tm_notice_project_type_switch(p, card_type, effective_type)
    _tm_set_query_params(tm_project=effective_type, tm_session=st.session_state.session_id)
    st.rerun()


def _render_new_project() -> None:
    # #region debug-point B:render-new-project
    _dbg_open_project_layout("B", "pre-fix", "app.py:_render_new_project", "[DEBUG] render new project page", {"session": str(st.session_state.get("session_id", "")), "project_type": str(st.session_state.get("project_type", "")), "query": dict(st.query_params) if hasattr(st, "query_params") else {}})
    # #endregion
    import textwrap as _tw_dedent_helper
    import base64 as _b64_helper
    inject_teachable_style()
    _icon_b64 = ""
    try:
        with open(os.path.join(os.path.dirname(__file__) or ".", "AIoScouts.png"), "rb") as _ifh:
            _icon_b64 = _b64_helper.b64encode(_ifh.read()).decode()
    except Exception:
        pass
    _hero_icon_html = (
        f'<img src="data:image/png;base64,{_icon_b64}" style="width:100%;height:100%;object-fit:contain;" alt="AIoScouts" />'
        if _icon_b64
        else '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="100%" height="100%"><circle cx="12" cy="12" r="2"/></svg>'
    )
    st.markdown('<div class="tm-image-project-hero-marker"></div>', unsafe_allow_html=True)
    st.markdown(
        _tw_dedent_helper.dedent('''
        <div class="tm-start-hero" style="background-color:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:40px;margin-bottom:2rem;display:flex;flex-direction:row;justify-content:space-between;align-items:center;gap:40px;position:relative;overflow:hidden;">
        <div style="flex:1;min-width:500px;z-index:2;">
        <h2 class="tm-start-hero-eyebrow" style="margin:0 0 8px 0;font-size:14px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;">Desktop AI Training</h2>
        <h1 class="tm-start-hero-title" style="margin:0 0 16px 0;font-size:2.8rem;font-weight:800;color:#0F172A;letter-spacing:-1px;line-height:1.1;">TFLite Training</h1>
        <p class="tm-start-hero-sub" style="margin:0 0 28px 0;max-width:540px;line-height:1.6;color:#475569;font-size:15px;">Build classroom-friendly image models without touching code. Inspired by Google Teachable Machine, with a desktop-first workflow for importing data, training, and exporting TFLite + C sources.</p>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:40px;">
        <span class="tm-chip">Import local images</span>
        <span class="tm-chip">Capture from webcam</span>
        <span class="tm-chip">Train lightweight CNN</span>
        <span class="tm-chip">Export .tflite + model.cpp</span>
        </div>
        <div class="tm-start-hero-workflow" style="border-top:1px solid #E2E8F0;padding-top:32px;display:flex;flex-direction:column;align-items:flex-start;">
        <div class="tm-start-hero-workflow-label" style="font-size:11px !important;text-transform:uppercase;letter-spacing:1.5px;color:#64748B;font-weight:700;margin-bottom:16px;line-height:1;">Recommended Workflow</div>
        <div style="display:flex;flex-direction:row;align-items:center;gap:14px;">
        <div style="display:flex;align-items:center;gap:8px;"><span class="tm-start-hero-step-num" style="background:#E0E7FF;color:#3169FF;border-radius:50%;width:24px;height:24px;display:flex;justify-content:center;align-items:center;font-size:12px;font-weight:800;line-height:1;flex-shrink:0;text-align:center;padding:0;margin:0;letter-spacing:normal;">1</span><span class="tm-start-hero-step-label" style="font-weight:600;font-size:14px;color:#1E293B;line-height:1;">Collect</span></div>
        <span class="tm-start-hero-arrow" style="display:flex;align-items:center;color:#CBD5E1;font-size:14px;line-height:1;margin-top:-1px;">➔</span>
        <div style="display:flex;align-items:center;gap:8px;"><span class="tm-start-hero-step-num" style="background:#E0E7FF;color:#3169FF;border-radius:50%;width:24px;height:24px;display:flex;justify-content:center;align-items:center;font-size:12px;font-weight:800;line-height:1;flex-shrink:0;text-align:center;padding:0;margin:0;letter-spacing:normal;">2</span><span class="tm-start-hero-step-label" style="font-weight:600;font-size:14px;color:#1E293B;line-height:1;">Label</span></div>
        <span class="tm-start-hero-arrow" style="display:flex;align-items:center;color:#CBD5E1;font-size:14px;line-height:1;margin-top:-1px;">➔</span>
        <div style="display:flex;align-items:center;gap:8px;"><span class="tm-start-hero-step-num" style="background:#E0E7FF;color:#3169FF;border-radius:50%;width:24px;height:24px;display:flex;justify-content:center;align-items:center;font-size:12px;font-weight:800;line-height:1;flex-shrink:0;text-align:center;padding:0;margin:0;letter-spacing:normal;">3</span><span class="tm-start-hero-step-label" style="font-weight:600;font-size:14px;color:#1E293B;line-height:1;">Train</span></div>
        <span class="tm-start-hero-arrow" style="display:flex;align-items:center;color:#CBD5E1;font-size:14px;line-height:1;margin-top:-1px;">➔</span>
        <div style="display:flex;align-items:center;gap:8px;"><span class="tm-start-hero-step-num" style="background:#E0E7FF;color:#3169FF;border-radius:50%;width:24px;height:24px;display:flex;justify-content:center;align-items:center;font-size:12px;font-weight:800;line-height:1;flex-shrink:0;text-align:center;padding:0;margin:0;letter-spacing:normal;">4</span><span class="tm-start-hero-step-label" style="font-weight:600;font-size:14px;color:#1E293B;line-height:1;">Export</span></div>
        </div>
        </div>
        </div>
        <div class="tm-start-hero-watermark" style="position:absolute;right:20px;top:40px;width:280px;height:280px;opacity:0.04;z-index:1;pointer-events:none;">
        __HERO_ICON__
        </div>
        </div>
        ''').strip().replace("__HERO_ICON__", _hero_icon_html),
        unsafe_allow_html=True,
    )

    # Section header bridging the top panel and the project card grid
    st.markdown(
        '''
        <div style="margin-top:1rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:8px;">
            <h3 class="tm-start-section-title" style="margin:0;font-size:20px;font-weight:700;color:#1E293B;letter-spacing:-0.5px;">
                ✨ Choose a project type
            </h3>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # Which card's "Open Project (.tmproj)" was clicked, and where a real
    # failure to open that file gets reported.  Both are handled AFTER the
    # grid below: the file picker is a modal native dialog that blocks this
    # script, so opening it mid-grid left the OCR and sign cards unrendered
    # behind it — and cancelling used to `return` straight out of the column,
    # dropping them for good.  The three buttons now only RECORD the request.
    open_request = ""
    open_error_slot = st.empty()
    card_col1, card_col2, card_col3 = st.columns([1, 1, 1], gap="large")
    with card_col1:
        st.markdown(
            _tw_dedent_helper.dedent('''
            <div class="tm-start-card">
            <div class="tm-start-card-head">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <h3 class="tm-start-card-title" style="margin:0;padding:0;font-size:22px;font-weight:800;">Image Project</h3>
            <span class="tm-start-ready-badge" style="background:#E8F5E9;color:#2E7D32;padding:2px 6px;border-radius:8px;font-size:9px;font-weight:800;letter-spacing:0.5px;">READY NOW</span>
            </div>
            <p>Build an image classification model from images, webcam, or ESP32-P4 device stream. Great for classroom demos and local training for tinyML MCU deployment.</p>
            </div>
            </div>
            ''').strip(),
            unsafe_allow_html=True,
        )
        if st.button("Get Started", key="tm_get_started", type="primary", use_container_width=True):
            _begin_fresh_tm_session()
            st.session_state.tm_return_target = "home"
            st.session_state.project_type = "image"
            st.session_state.step = 0
            _tm_set_query_params(tm_project="image", tm_session=st.session_state.session_id)
            st.rerun()
        if st.button("Open Project (.tmproj)", key="tm_open_tmproj", type="secondary", use_container_width=True):
            open_request = "image"
        if st.button("Import classified class", key="tm_start_classified", type="secondary", use_container_width=True):
            _begin_fresh_tm_session()
            st.session_state.tm_return_target = "classified-import"
            st.session_state.project_type = "image_classified_import"
            st.session_state.step = 0
            _tm_set_query_params(tm_project="image_classified_import", tm_session=st.session_state.session_id)
            st.rerun()
    with card_col2:
        st.markdown(
            _tw_dedent_helper.dedent('''
            <div class="tm-start-card">
            <div class="tm-start-card-head">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <h3 class="tm-start-card-title" style="margin:0;padding:0;font-size:22px;font-weight:800;">🔠 OCR / Character Project</h3>
            <span class="tm-start-ready-badge" style="background:#E8F5E9;color:#2E7D32;padding:2px 6px;border-radius:8px;font-size:9px;font-weight:800;letter-spacing:0.5px;">READY NOW</span>
            </div>
            <p>Train a highly optimized INT8 character recognition model for ESP32. Features EMNIST dataset mixing, bit-exact Vertical Projection, and OOD rejection.</p>
            </div>
            </div>
            ''').strip(),
            unsafe_allow_html=True,
        )
        if st.button("Get Started", key="tm_get_started_ocr", type="primary", use_container_width=True):
            _begin_fresh_tm_session()
            st.session_state.tm_return_target = "home"
            st.session_state.project_type = PROJECT_TYPE_OCR
            st.session_state.step = 0
            _tm_apply_project_defaults(PROJECT_TYPE_OCR)
            _tm_set_query_params(tm_project=PROJECT_TYPE_OCR, tm_session=st.session_state.session_id)
            st.rerun()
        if st.button("Open Project (.tmproj)", key="tm_open_tmproj_ocr", type="secondary", use_container_width=True):
            open_request = PROJECT_TYPE_OCR
    with card_col3:
        st.markdown(
            _tw_dedent_helper.dedent('''
            <div class="tm-start-card">
            <div class="tm-start-card-head">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <h3 class="tm-start-card-title" style="margin:0;padding:0;font-size:22px;font-weight:800;">🚗 Car License Project</h3>
            <span class="tm-start-ready-badge" style="background:#E8F5E9;color:#2E7D32;padding:2px 6px;border-radius:8px;font-size:9px;font-weight:800;letter-spacing:0.5px;">READY NOW</span>
            </div>
            <p>Train a 3-character car-license reader. The same INT8 character model as the OCR project, plus a target sequence, automatic 3-way glyph splitting and a MATCH / NOT MATCH preview against the plate in view.</p>
            </div>
            </div>
            ''').strip(),
            unsafe_allow_html=True,
        )
        if st.button("Get Started", key="tm_get_started_sign", type="primary", use_container_width=True):
            _begin_fresh_tm_session()
            st.session_state.tm_return_target = "home"
            st.session_state.project_type = PROJECT_TYPE_SIGN_OCR
            st.session_state.step = 0
            _tm_apply_project_defaults(PROJECT_TYPE_SIGN_OCR)
            _tm_set_query_params(tm_project=PROJECT_TYPE_SIGN_OCR, tm_session=st.session_state.session_id)
            st.rerun()
        if st.button("Open Project (.tmproj)", key="tm_open_tmproj_sign", type="secondary", use_container_width=True):
            open_request = PROJECT_TYPE_SIGN_OCR
        if st.button("Import classified class", key="tm_start_classified_sign", type="secondary", use_container_width=True):
            # Deliberately the IMAGE classified-import flow, not a third one:
            # the importer copies folders of images and knows nothing about
            # characters, so claiming it here would import 28x28 glyph crops
            # through a path that has never seen one.  What the user gets is
            # the ordinary importer; they then build the sign project around
            # the classes it produces.
            _begin_fresh_tm_session()
            st.session_state.tm_return_target = "classified-import"
            st.session_state.project_type = PROJECT_TYPE_CLASSIFIED_IMPORT
            st.session_state.step = 0
            _tm_set_query_params(tm_project=PROJECT_TYPE_CLASSIFIED_IMPORT, tm_session=st.session_state.session_id)
            st.rerun()

    # ── Open Source Licenses ────────────────────────────────────────────
    _render_licenses_expander()

    # All three cards are on screen before any of them acts.  Cancelling the
    # picker comes back as (None, "") — a silent no-op, not a warning box in
    # one narrow column.  A file that really cannot be opened comes back as a
    # message, which goes into `open_error_slot` above the grid so the whole
    # width carries it and the cards stay where they were.
    if open_request:
        picked, why = _tmproj_pick_and_classify()
        if why:
            open_error_slot.error(why)
        elif picked:
            p, ptype = picked
            # #region debug-point A:home-open-project-click
            _dbg_open_project_layout("A", "pre-fix", "app.py:_render_new_project", "[DEBUG] home open project selected file", {"picked_path": str(p), "project_type": str(ptype), "session_before_fresh": str(st.session_state.get("session_id", ""))})
            # #endregion
            _tm_open_project_into_session(p, ptype, open_request)


def _tm_train_recommendations(cfg: TrainConfig) -> Dict[str, Any]:
    """Advisory training recommendations for the SPA Advanced panel."""
    total = _tm_dataset_stats()[1]
    try:
        rec_kwargs, reasons = recommend_train_params(total, cfg)
    except Exception:
        return {}
    return {
        "total_samples": int(total),
        "batch_size": int(rec_kwargs["batch_size"]),
        "epochs": int(rec_kwargs["epochs"]),
        "learning_rate": float(rec_kwargs["learning_rate"]),
        "validation_split": float(rec_kwargs["validation_split"]),
        "conv1_filters": int(rec_kwargs.get("conv1_filters", 32)),
        "conv2_filters": int(rec_kwargs.get("conv2_filters", 64)),
        "dense_units": int(rec_kwargs.get("dense_units", 128)),
        "reasons": {
            "batch": str(reasons[0]),
            "epochs": str(reasons[1]),
            "lr": str(reasons[2]),
            "val_split": str(reasons[3]),
            "capacity": str(reasons[4]) if len(reasons) > 4 else "",
        },
    }


def _render_train_config(cfg: TrainConfig) -> TrainConfig:
    st.info("Recommendations are advisory — you can keep your own values.")
    total_samples = _tm_dataset_stats()[1]
    rec_kwargs, reasons = recommend_train_params(total_samples, cfg)
    batch_opt = [8, 16, 32, 64]

    # Snap recommendations to widget constraints (selectbox options / input caps).
    rec_batch = int(rec_kwargs["batch_size"])
    if rec_batch not in batch_opt:
        rec_batch = min(batch_opt, key=lambda b: abs(b - rec_batch))
    rec_epochs = min(int(rec_kwargs["epochs"]), 200)
    rec_lr = float(rec_kwargs["learning_rate"])
    rec_val_split = float(rec_kwargs["validation_split"])
    rec_conv1 = int(rec_kwargs.get("conv1_filters", 32))
    rec_conv2 = int(rec_kwargs.get("conv2_filters", 64))
    rec_dense = int(rec_kwargs.get("dense_units", 128))
    capacity_reason = str(reasons[4]) if len(reasons) > 4 else ""

    # Guard against saved values outside the widget ranges.
    cfg_batch = int(cfg.batch_size)
    if cfg_batch not in batch_opt:
        cfg_batch = min(batch_opt, key=lambda b: abs(b - cfg_batch))
    cfg_epochs = min(int(cfg.epochs), 200)

    left, right = st.columns(2)
    with left:
        # Range matches the server-side /live/config guard (8..512) so the
        # OCR project's 28x28 input is representable here too.
        img_size = st.number_input("Image size", min_value=8, max_value=512, value=int(cfg.img_size), step=4)
        _proj_type = str(st.session_state.get("project_type") or PROJECT_TYPE_IMAGE)
        _expected_size = int(default_img_size(_proj_type))
        if int(cfg.img_size) != _expected_size:
            st.caption(f"💡 Note: model input should match preprocessing ({_expected_size}×{_expected_size})")
        color_mode = st.selectbox("Color mode", options=["rgb", "grayscale"], index=0 if cfg.color_mode == "rgb" else 1)
        batch_size = st.selectbox("Batch size", options=batch_opt, index=batch_opt.index(cfg_batch))
        if cfg_batch != rec_batch:
            st.caption(f"💡 Recommended: {rec_batch} — {reasons[0]}")
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=cfg_epochs, step=1)
        if cfg_epochs != rec_epochs:
            st.caption(f"💡 Recommended: {rec_epochs} — {reasons[1]}")
        validation_split = st.slider("Validation split", min_value=0.05, max_value=0.5, value=float(cfg.validation_split), step=0.05)
        if abs(float(cfg.validation_split) - rec_val_split) > 1e-9:
            st.caption(f"💡 Recommended: {rec_val_split} — {reasons[3]}")
    with right:
        optimizer = st.selectbox("Optimizer", options=["adam", "sgd", "rmsprop"], index=["adam", "sgd", "rmsprop"].index(cfg.optimizer.lower()))
        learning_rate = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=float(cfg.learning_rate), format="%.5f")
        if abs(float(cfg.learning_rate) - rec_lr) > 1e-9:
            st.caption(f"💡 Recommended: {rec_lr} — {reasons[2]}")
        conv1_filters = st.selectbox("Conv1 filters", options=[4, 8, 16, 32], index=[4, 8, 16, 32].index(cfg.conv1_filters))
        conv2_filters = st.selectbox("Conv2 filters", options=[8, 16, 32, 64], index=[8, 16, 32, 64].index(cfg.conv2_filters))
        dense_units = st.selectbox("Dense units", options=[16, 32, 64, 128], index=[16, 32, 64, 128].index(cfg.dense_units))
        if int(conv1_filters) != rec_conv1 or int(conv2_filters) != rec_conv2 or int(dense_units) != rec_dense:
            st.caption(f"💡 Recommended: conv {rec_conv1}/{rec_conv2}, dense {rec_dense} — {capacity_reason}")

    if st.button("Use recommended settings", key="tm_use_recommended", use_container_width=True):
        st.session_state.train_cfg = TrainConfig(
            img_size=int(img_size),
            color_mode=str(color_mode),
            batch_size=rec_batch,
            epochs=rec_epochs,
            validation_split=rec_val_split,
            seed=int(cfg.seed),
            optimizer=str(optimizer),
            learning_rate=rec_lr,
            conv1_filters=int(rec_conv1),
            conv2_filters=int(rec_conv2),
            dense_units=int(rec_dense),
            representative_samples=int(cfg.representative_samples),
            preprocess_mode=str(getattr(cfg, "preprocess_mode", "auto_by_label")),
            manual_roi=getattr(cfg, "manual_roi", None),
        )
        st.rerun()

    return TrainConfig(
        img_size=int(img_size),
        color_mode=str(color_mode),
        batch_size=int(batch_size),
        epochs=int(epochs),
        validation_split=float(validation_split),
        seed=int(cfg.seed),
        optimizer=str(optimizer),
        learning_rate=float(learning_rate),
        conv1_filters=int(conv1_filters),
        conv2_filters=int(conv2_filters),
        dense_units=int(dense_units),
        representative_samples=int(cfg.representative_samples),
        preprocess_mode=str(getattr(cfg, "preprocess_mode", "auto_by_label")),
        manual_roi=getattr(cfg, "manual_roi", None),
    )


def _tm_get_query_param(key: str) -> str:
    try:
        v = st.query_params.get(key)
        if v is None:
            return ""
        if isinstance(v, list):
            return str(v[0]) if v else ""
        return str(v)
    except Exception:
        v = st.experimental_get_query_params().get(key, [""])
        return str(v[0]) if v else ""


def _tm_set_query_params(**params: str) -> None:
    try:
        st.query_params.clear()
        for k, v in params.items():
            if v is not None and str(v) != "":
                st.query_params[k] = str(v)
    except Exception:
        st.experimental_set_query_params(**{k: v for k, v in params.items() if v is not None and str(v) != ""})


def _tm_clear_query_params() -> None:
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()


def _tm_render_page_scroll_reset() -> None:
    components.html(
        '''
        <script>
        function resetFrameBox(node) {
          try {
            if (!node || !node.style) return;
            node.style.height = '';
            node.style.minHeight = '';
            node.style.maxHeight = '';
            node.style.overflow = '';
            node.style.overflowY = '';
            node.style.overflowX = '';
          } catch (e) {}
        }
        try {
          window.scrollTo(0, 0);
          document.documentElement.scrollTop = 0;
          document.body.scrollTop = 0;
        } catch (e) {}
        try {
          if (window.parent) {
            window.parent.scrollTo(0, 0);
            if (window.parent.document) {
              window.parent.document.documentElement.scrollTop = 0;
              window.parent.document.body.scrollTop = 0;
              try {
                const frames = window.parent.document.querySelectorAll('iframe');
                frames.forEach((frame) => {
                  resetFrameBox(frame);
                  resetFrameBox(frame.parentElement);
                });
              } catch (e) {}
            }
          }
        } catch (e) {}
        </script>
        ''',
        height=0,
    )


def _tm_render_shell_reflow_ping(reason: str = "image-project-mount") -> None:
    safe_reason = html_escape(str(reason or "image-project-mount"))
    components.html(
        f'''
        <script>
        (function() {{
          const reason = "{safe_reason}";
          function ping(tag) {{
            try {{
              const targets = [window, window.parent, window.top];
              for (const target of targets) {{
                try {{
                  if (target) target.dispatchEvent(new Event('resize'));
                }} catch (e) {{}}
                try {{
                  if (target) target.dispatchEvent(new Event('orientationchange'));
                }} catch (e) {{}}
                try {{
                  if (target && target.pywebview && target.pywebview.api && typeof target.pywebview.api.request_reflow === 'function') {{
                    target.pywebview.api.request_reflow(String(reason + ':' + tag));
                  }}
                }} catch (e) {{}}
              }}
            }} catch (e) {{}}
          }}
          ping('now');
          window.setTimeout(() => ping('t120'), 120);
          window.setTimeout(() => ping('t360'), 360);
          window.setTimeout(() => ping('t900'), 900);
        }})();
        </script>
        ''',
        height=0,
    )


def _list_camera_options(max_count: int = 6) -> List[Dict[str, str]]:
    # Fresh enumeration at call time (delegated to the shared helper so the
    # HTTP /webcam/options route and the streamlit render use the same
    # logic).  AVCaptureDevice order flips when virtual cameras connect or
    # disconnect — never cache this list.
    try:
        from record_controller import list_webcam_options
        result = list_webcam_options(max_count=max_count)
        # Normalise: every entry must carry a unique_id key (even if empty)
        # so the frontend <option data-uid=""> render never gets
        # `undefined` → empty string, and Apply never sends an undefined
        # uid to /live/config.  A missing `unique_id` key on any entry
        # cascades to the UI showing the correct label but opening the
        # wrong camera (silent positional fallback).
        for item in result:
            if "unique_id" not in item:
                item["unique_id"] = str(
                    item.get("uid") or item.get("id") or ""
                )
        return result
    except Exception:
        # Last-resort fallback: produce entries with empty unique_id and
        # generic labels.  In this degraded mode the camera open falls
        # back to positional index, which on macOS with Iriun/OBS installed
        # will prefer the virtual feed.  The unique_id key is still present
        # (albeit empty) so the frontend's JS property access does not
        # yield `undefined`.
        return [
            {"index": i, "label": f"Camera {i}", "unique_id": ""}
            for i in range(max_count)
        ]


def _tm_sample_previews(classes: List[str], limit_per_class: Optional[int] = None) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    root = _tm_dataset_dir()
    for name in classes:
        class_dir = root / sanitize_class_name(name)
        previews: List[Dict[str, str]] = []
        if class_dir.exists():
            files = _tm_class_image_files(class_dir)
            if limit_per_class is not None:
                files = files[:limit_per_class]
            for p in files:
                try:
                    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                    previews.append(
                        {
                            "src": f"data:image/png;base64,{b64}",
                            "filename": str(p.name),
                        }
                    )
                except Exception:
                    continue
        out[name] = previews
    return out


def _is_likely_user_serial_port(device: str, description: str) -> bool:
    d = (device or "").strip().lower()
    desc = (description or "").strip().lower()
    blocked = {
        "/dev/cu.bluetooth-incoming-port",
        "/dev/cu.debug-console",
        "/dev/tty.bluetooth-incoming-port",
        "/dev/tty.debug-console",
    }
    if d in blocked:
        return False
    if "bluetooth" in desc and "serial" not in desc:
        return False
    return bool(d)


def _is_virtual_camera_label(label: str) -> bool:
    s = (label or "").strip().lower()
    markers = ("iriun", "virtual", "obs", "camo", "epoccam", "ndi", "droidcam")
    return any(m in s for m in markers)


def _preferred_webcam(options: List[Dict[str, str]]) -> Tuple[int, str]:
    """Return (preferred_index, preferred_unique_id) tuple.

    Always returns the unique_id alongside the index so callers do not need
    to do a second lookup.  The previous helper returned only the index and
    every call site had to re-derive the id (or forgot it and fell back to
    positional index 0 which opened Iriun/OBS instead of the built-in
    camera on macOS).
    """
    for item in options:
        label = str(item.get("label", ""))
        if not _is_virtual_camera_label(label):
            try:
                idx = int(item.get("index", 0))
                uid = str(item.get("unique_id") or item.get("uid") or item.get("id") or "")
                return idx, uid
            except Exception:
                continue
    if options:
        try:
            idx = int(options[0].get("index", 0))
            uid = str(options[0].get("unique_id") or options[0].get("uid") or options[0].get("id") or "")
            return idx, uid
        except Exception:
            return 0, ""
    return 0, ""


# Backwards-compatible alias — a few call sites were already written to
# consume just the index; kept so nothing breaks if we missed one.
def _preferred_webcam_index(options: List[Dict[str, str]]) -> int:
    return _preferred_webcam(options)[0]


def _load_persisted_webcam() -> Optional[Dict[str, Any]]:
    """Last-used camera (uniqueID + index) persisted by the controller."""
    try:
        p = _tm_dataset_dir().parent / "webcam.json"
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass
    return None


def _render_tm_old_frontend_html(
    *,
    port: int,
    session_id: str,
    classes: List[str],
    counts: Dict[str, int],
    train_enabled: bool,
    export_enabled: bool,
    notice: str,
    train_cfg: TrainConfig,
    serial_ports: List[Dict[str, str]],
    current_serial_port: str,
    current_serial_baud: int,
    current_serial_sync: str,
    current_serial_frame_side: int,
    current_serial_channels: int,
    webcam_options: List[Dict[str, str]],
    current_webcam_index: int,
    current_webcam_id: str,
    sample_previews: Dict[str, List[Dict[str, str]]],
    initial_open_source_class: str,
    initial_open_source_kind: str,
) -> None:
    import json as _json

    # Prefer the backend-persisted config (e.g. applied recommendations) over
    # Streamlit session memory, so frontend changes survive page reruns.
    try:
        saved_cfg_path = _tm_dataset_dir().parent / "runs" / "latest" / "train_config.json"
        if saved_cfg_path.exists():
            saved_raw = json.loads(saved_cfg_path.read_text(encoding="utf-8"))
            # The run's own train_config.json is a full TrainConfig dump, so
            # every key it carries is restorable — the field list comes from
            # the dataclass rather than from a list here that has to be kept
            # in step with it.
            train_cfg = _tm_train_cfg_from_state(saved_raw, train_cfg)
    except Exception:
        pass

    meta = _tm_load_classes_meta_full() or {}
    # Read once: the project type decides both the reset baseline's img_size
    # and the OCR panel's visibility, and those must not disagree inside one
    # payload.
    payload_project_type = str(st.session_state.get("project_type") or PROJECT_TYPE_IMAGE)
    payload = {
        "port": int(port),
        "session": str(session_id),
        "return_target": str(st.session_state.get("tm_return_target", "home") or "home"),
        "classes": classes,
        "counts": {k: int(v) for k, v in counts.items()},
        "train_enabled": bool(train_enabled),
        "export_enabled": bool(export_enabled),
        "notice": str(notice or ""),
        "serial_ports": serial_ports,
        "current_serial_port": str(current_serial_port or ""),
        "current_serial_baud": int(current_serial_baud),
        "current_serial_sync": str(current_serial_sync or "AA 55 AA"),
        "current_serial_frame_side": int(current_serial_frame_side),
        "current_serial_channels": int(current_serial_channels),
        "preview_dark_thresh": int(st.session_state.get("tm_preview_dark_thresh") if st.session_state.get("tm_preview_dark_thresh") is not None else 0),
        "preview_lum_thresh": int(st.session_state.get("tm_preview_lum_thresh") if st.session_state.get("tm_preview_lum_thresh") is not None else 100),
        "webcam_options": webcam_options,
        "current_webcam_index": int(current_webcam_index),
        "current_webcam_id": str(current_webcam_id or ""),
        "sample_previews": sample_previews,
        "initial_open_source_class": str(initial_open_source_class or ""),
        "initial_open_source_kind": str(initial_open_source_kind or ""),
        "class_preprocess": dict(meta.get("class_preprocess") or {}),
        "sample_preprocess": dict(meta.get("sample_preprocess") or {}),
        "train_cfg": _tm_train_cfg_payload(train_cfg),
        # The dataclass's own defaults, sent so "Reset to defaults" in the
        # Advanced panel resets to what the TRAINER calls default.  The button
        # used to write its own literal list, which had drifted to the old
        # 8/16/32 architecture TrainConfig's comment records as collapsing to
        # a single class.
        # ...except img_size, which is the WORKSPACE's geometry rather than a
        # training knob: the dataclass default is the image project's 96, and
        # resetting an OCR workspace to it would point the next capture at the
        # wrong input size.  The project's own default is the right baseline.
        "train_cfg_defaults": _tm_train_cfg_payload(
            _tm_train_cfg_replace(
                TrainConfig(), img_size=int(default_img_size(payload_project_type))
            )
        ),
        # The OCR blocks in the Advanced panel are shown only for an OCR
        # workspace — its knobs are inert for a sign project and vice versa.
        "project_type": payload_project_type,
        "train_rec": _tm_train_recommendations(train_cfg),
        "current_project_path": str(st.session_state.get("tm_current_project_path") or "") or None,
    }
    debug_server_url = None
    debug_session_id = None
    try:
        for env_path in (Path(".dbg/open-project-layout.env"), Path(".dbg/open-project-page-stuck.env"), Path(".dbg/packaged-webcam-bounce.env"), Path(".dbg/capture-webcam-source.env")):
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("DEBUG_SERVER_URL="):
                        debug_server_url = line.split("=", 1)[1].strip() or debug_server_url
                    elif line.startswith("DEBUG_SESSION_ID="):
                        debug_session_id = line.split("=", 1)[1].strip() or debug_session_id
                break
    except Exception:
        pass
    payload["debug_server_url"] = debug_server_url
    payload["debug_session_id"] = debug_session_id
    data = _json.dumps(payload)
    # Built here rather than inline so the OCR split list has exactly one
    # source (ocr_dataset.EMNIST_SPLITS) — a hand-written <option> list would
    # silently drift the day a split is added.
    emnist_split_options = "".join(
        f'<option value="{name}">{name}</option>' for name in EMNIST_SPLITS
    )
    components.html(
        f'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --card: #ffffff;
      --surface: #ffffff;
      --surface-soft: #f4f7fb;
      --surface-elevated: #ffffff;
      --surface-frost: rgba(255,255,255,0.72);
      --shadow: 0 2px 8px rgba(0,0,0,0.08);
      --blue: #1a73e8;
      --blue-bg: #e8f0fe;
      --blue-bg-hover: #d2e3fc;
      --text: #202124;
      --text-inverse: #ffffff;
      --muted: #5f6368;
      --line: #c5c8cc;
      --input-bg: #ffffff;
      --input-border: rgba(0,0,0,0.12);
      --overlay: rgba(32,33,36,0.56);
      --danger: #d93025;
      --danger-soft: #fff1f3;
      --radius: 12px;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        color-scheme: dark;
        --bg: #0f141a;
        --card: #1a212b;
        --surface: #18202a;
        --surface-soft: #111821;
        --surface-elevated: #202833;
        --surface-frost: rgba(24,32,42,0.88);
        --shadow: 0 10px 30px rgba(0,0,0,0.32);
        --blue: #8ab4f8;
        --blue-bg: rgba(138,180,248,0.16);
        --blue-bg-hover: rgba(138,180,248,0.24);
        --text: #e8eaed;
        --text-inverse: #0f141a;
        --muted: #a8b0ba;
        --line: rgba(255,255,255,0.12);
        --input-bg: #111821;
        --input-border: rgba(255,255,255,0.12);
        --overlay: rgba(0,0,0,0.62);
        --danger: #f28b82;
        --danger-soft: rgba(242,139,130,0.16);
      }}
    }}
    html, body {{
      min-height: 100vh;
      width: 100%;
      height: 100%;
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      color: var(--text);
      background: var(--bg);
    }}
    .wrap {{
      position: relative;
      background: var(--bg);
      border-radius: 12px;
      padding: 72px 16px 92px 16px;
      min-height: 620px;
      overflow: visible;
      /* Fluid, window-responsive canvas: fills the full-bleed iframe on
         any resolution but caps at 1680px so the three columns never
         stretch into an unreadable ultra-wide ribbon. */
      width: 100%;
      max-width: min(1680px, calc(100vw - 48px));
      margin: 0 auto;
      box-sizing: border-box;
    }}
    .topnav {{
      position: absolute;
      top: 12px;
      left: 18px;
      background: var(--card);
      box-shadow: var(--shadow);
      border-radius: 8px;
      padding: 12px 18px;
      color: var(--blue);
      font-weight: 700;
      font-size: 18px;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
      user-select: none;
      z-index: 1000;
    }}
    .topnav img {{
      width: 26px;
      height: 26px;
      object-fit: contain;
      display: block;
      border-radius: 4px;
    }}
    .navmenu {{
      position: absolute;
      top: 56px;
      left: 18px;
      width: 220px;
      background: var(--card);
      box-shadow: var(--shadow);
      border-radius: 10px;
      border: 1px solid rgba(0,0,0,0.06);
      padding: 8px;
      z-index: 1001;
      display: none;
    }}
    .navmenu.open {{ display: block; }}
    .navmenu button {{
      width: 100%;
      border: 0;
      background: transparent;
      text-align: left;
      padding: 10px 10px;
      border-radius: 8px;
      cursor: pointer;
      color: var(--text);
      font-weight: 600;
      font-size: 13px;
    }}
    .navmenu button:hover {{ background: rgba(0,0,0,0.05); }}
    .navmenu .danger {{ color: #b42318; }}
    .navmenu .divider {{
      height: 1px;
      background: rgba(0,0,0,0.08);
      margin: 6px 0;
    }}
    /* ── Problem 2+3.2: persistent segmented Toolbar (replaces topnav card) ── */
    .wrap {{
      padding-top: 86px;
    }}
    .topnav {{
      position: absolute;
      top: 10px;
      left: 16px;
      right: 16px;
      width: calc(100% - 32px);
      /* content-box (the default) made the 10px×2 padding + 1px×2 border
         push the toolbar 6px past the right edge, which showed up as a
         hairline horizontal overflow once the workspace went full-bleed. */
      box-sizing: border-box;
      border-radius: 12px;
      padding: 6px 10px;
      background: var(--card);
      border: 1px solid rgba(0,0,0,0.06);
      display: flex;
      align-items: center;
      gap: 6px;
      z-index: 1000;
      cursor: default;
      box-shadow: var(--shadow);
    }}
    .tm-toolbar-brand {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px 6px 6px;
      border-right: 1px solid rgba(0,0,0,0.08);
      margin-right: 2px;
      color: var(--text);
      font-weight: 800;
      font-size: 14px;
    }}
    .tm-toolbar-brand img {{
      width: 22px; height: 22px; object-fit: contain; border-radius: 4px; display:block;
    }}
    .tm-toolbar-menu {{
      position: relative;
      display: inline-flex;
      align-items: stretch;
    }}
    .tm-toolbar-btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 12px;
      border: 0;
      background: transparent;
      border-radius: 8px;
      color: var(--text);
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      white-space: nowrap;
    }}
    .tm-toolbar-btn:hover {{ background: rgba(0,0,0,0.05); }}
    .tm-toolbar-btn.danger {{ color: #b42318; }}
    .tm-toolbar-btn.primary {{ color: var(--blue); font-weight: 700; }}
    .tm-toolbar-btn .kbd {{
      font-size: 10.5px;
      padding: 1px 5px;
      background: rgba(0,0,0,0.06);
      border-radius: 4px;
      color: var(--muted);
      margin-left: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-weight: 600;
    }}
    .tm-toolbar-dropdown {{
      position: absolute;
      top: calc(100% + 4px);
      left: 0;
      min-width: 280px;
      background: var(--card);
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 10px;
      padding: 6px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.14);
      z-index: 1002;
      display: none;
      flex-direction: column;
    }}
    .tm-toolbar-dropdown.open {{ display: flex; }}
    .tm-toolbar-dropdown button {{
      width: 100%;
      display: flex;
      align-items: center;
      gap: 8px;
      border: 0;
      background: transparent;
      padding: 9px 10px;
      border-radius: 7px;
      cursor: pointer;
      color: var(--text);
      text-align: left;
      font-weight: 600;
      font-size: 13px;
    }}
    .tm-toolbar-dropdown button:hover {{ background: rgba(0,0,0,0.05); }}
    .tm-toolbar-dropdown button.danger {{ color: #b42318; }}
    .tm-toolbar-dropdown button span {{ display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; flex: 0 0 auto; }}
    .tm-toolbar-dropdown button span svg {{ width: 16px; height: 16px; display: block; }}
    .tm-toolbar-btn .tm-toolbar-icon {{ display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; margin-right: 4px; vertical-align: middle; }}
    .tm-toolbar-btn .tm-toolbar-icon svg {{ width: 15px; height: 15px; display: block; }}
    .tm-toolbar-dropdown .dd-sep {{
      height:1px; background: rgba(0,0,0,0.08); margin: 4px 4px;
    }}
    .tm-toolbar-sep {{
      width: 1px;
      background: rgba(0,0,0,0.08);
      align-self: stretch;
      margin: 6px 4px;
    }}
    .tm-toolbar-spacer {{ flex: 1 1 auto; }}
    .tm-toolbar-return {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 12px;
      border: 1px solid rgba(0,0,0,0.1);
      background: rgba(0,0,0,0.02);
      border-radius: 8px;
      color: var(--text);
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
    }}
    .tm-toolbar-return:hover {{ background: rgba(0,0,0,0.06); }}
    /* Legacy navmenu (the old floating dropdown) is visually hidden but kept
       so JS bindings fail-safe (it will never open because topnav.onclick no
       longer calls toggleNavMenu — but its buttons still have the IDs). */
    .navmenu {{ display: none !important; }}
    /* ── Settings SSOT Modal (Problem 3.2) ── */
    .tm-settings-modal {{
      position: fixed;
      inset: 0;
      background: rgba(10, 14, 22, 0.45);
      z-index: 2000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}
    .tm-settings-modal.open {{ display: flex; }}
    .tm-settings-card {{
      width: min(720px, 100%);
      max-height: calc(100vh - 40px);
      overflow: auto;
      background: var(--card);
      border-radius: 14px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.28);
      padding: 22px 24px 18px;
    }}
    .tm-settings-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 16px;
    }}
    .tm-settings-head h3 {{
      margin: 0;
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }}
    .tm-settings-head p {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
    .tm-settings-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .tm-settings-group {{
      background: #F8FAFC;
      border: 1px solid #E2E8F0;
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .tm-settings-empty {{
      font-size: 13px;
      color: var(--muted);
      line-height: 1.5;
      padding: 8px 4px;
    }}
    .tm-settings-group-head {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding-bottom: 8px;
      border-bottom: 1px solid #E2E8F0;
      margin-bottom: 4px;
    }}
    .tm-settings-group-head h4 {{
      margin: 0;
      font-size: 13px;
      font-weight: 700;
      color: #0F172A;
      letter-spacing: 0.02em;
    }}
    .tm-settings-group-icon {{
      font-size: 15px;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: var(--blue);
    }}
    .tm-settings-group-icon svg {{ width: 18px; height: 18px; display: block; }}
    .card-head-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      vertical-align: middle;
      margin-right: 6px;
      color: inherit;
    }}
    .card-head-icon svg {{ width: 18px; height: 18px; display: block; }}
    .tm-settings-field label {{
      display: block;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      margin: 0 0 6px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .tm-settings-field select,
    .tm-settings-field input[type="number"],
    .tm-settings-field input[type="text"] {{
      width: 100%;
      box-sizing: border-box;
      padding: 9px 10px;
      border-radius: 8px;
      border: 1px solid rgba(0,0,0,0.12);
      background: var(--surface-soft);
      color: var(--text);
      font-size: 13.5px;
    }}
    .tm-settings-foot {{
      margin-top: 20px;
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      border-top: 1px solid rgba(0,0,0,0.08);
      padding-top: 14px;
    }}
    .tm-settings-advanced {{
      margin-top: 16px;
      border: 1px solid #E2E8F0;
      border-radius: 12px;
      background: #F8FAFC;
      overflow: hidden;
    }}
    .tm-settings-advanced-toggle {{
      width: 100%;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px 16px;
      border: 0;
      background: transparent;
      color: var(--text);
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      text-align: left;
    }}
    .tm-settings-advanced-toggle:hover {{ background: rgba(0,0,0,0.03); }}
    .tm-settings-advanced-chevron {{
      display: inline-block;
      transition: transform 0.15s ease;
      font-size: 12px;
      color: var(--muted);
    }}
    .tm-settings-advanced-toggle[aria-expanded="true"] .tm-settings-advanced-chevron {{ transform: rotate(90deg); }}
    .tm-settings-advanced-body {{
      padding: 0 16px 16px 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    /* Developer/demo shortcut: a checkbox row that reads as sentence case,
       not as another uppercase field label, plus the action button row. */
    .tm-settings-check {{
      display: flex;
      align-items: flex-start;
      gap: 8px;
      font-size: 12.5px;
      font-weight: 600;
      line-height: 1.35;
      color: var(--text);
      cursor: pointer;
      margin: 0 0 6px;
    }}
    .tm-settings-check input[type="checkbox"] {{ margin: 2px 0 0; }}
    .tm-settings-seed-row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .tm-settings-field small {{
      display: block;
      margin-top: 4px;
      font-size: 11.5px;
      color: var(--muted);
      line-height: 1.4;
    }}
    .tm-btn {{
      padding: 9px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 700;
      border: 1px solid rgba(0,0,0,0.1);
      background: var(--surface);
      color: var(--text);
      cursor: pointer;
    }}
    .tm-btn:hover {{ background: rgba(0,0,0,0.04); }}
    .tm-btn.primary {{
      background: var(--blue);
      color: #fff;
      border-color: transparent;
    }}
    .tm-btn.primary:hover {{ filter: brightness(0.94); }}
    /* ── Source chip (Problem 3.2: class card compact source indicator) ── */
    .tm-source-chip-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 4px 4px;
    }}
    .tm-source-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: 999px;
      background: rgba(0,0,0,0.04);
      color: var(--text);
      font-size: 11.5px;
      font-weight: 600;
    }}
    .tm-source-chip .chip-icon,
    .tm-source-chip .tm-chip-icon {{ font-size: 13px; line-height: 1; display: inline-flex; align-items: center; justify-content: center; }}
    .tm-source-chip .tm-chip-icon svg {{ width: 14px; height: 14px; display: block; }}
    .tm-source-chip .tm-chip-text {{ line-height: 1; }}
    .tm-source-chip.settings-hint,
    .tm-source-chip.tm-source-chip-settings {{
      display: none;
    }}
    .tm-source-chip.settings-hint:hover,
    .tm-source-chip.tm-source-chip-settings:hover {{ background: color-mix(in srgb, var(--blue) 18%, transparent); }}
    .tm-rename-btn {{
      width: 32px !important;
      height: 32px !important;
      padding: 0;
      border-radius: 6px;
      font-size: 14px;
      display: inline-flex !important;
      align-items: center;
      justify-content: center;
      background: transparent;
      color: var(--muted);
      border: none;
    }}
    .tm-rename-btn:hover {{
      background: rgba(0,0,0,0.05);
      color: var(--text);
    }}
    .layout {{
      position: relative;
      display: flex;
      flex-direction: row;
      flex-wrap: nowrap;
      width: 100%;
      box-sizing: border-box;
      justify-content: flex-start;
      align-items: flex-start;
      /* 60px, not 28px: the class→training and training→preview connector
         wires (svg.flow) are drawn between the column edges, so a 28px gap
         left them a hairline sliver.  The wire control points are a fixed
         44px offset, which reads as a proper S-curve once the span is this
         wide rather than as two wires crossing. */
      gap: 60px;
      z-index: 2;
    }}
    .col-classes {{ flex: 1 1 500px; min-width: 360px; display: flex; flex-direction: column; gap: 24px; }}
    .col-train {{
      width: 270px;
      flex: 0 0 270px;
      min-width: 270px;
      align-self: stretch;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}
    .col-preview {{
      flex: 1 1 320px;
      min-width: 260px;
      align-self: stretch;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}
    @media (max-width: 1320px) {{
      .wrap {{
        padding: 72px 16px 92px 16px;
      }}
      .layout {{
        /* One responsive step down from the desktop 60px, still inside the
           56-64px band the workspace was designed around, so the connector
           wires keep their span on a laptop that has lost a few columns. */
        gap: 56px;
      }}
      .col-classes {{
        flex-basis: 440px;
        min-width: 300px;
      }}
      .col-train {{
        width: 230px;
        flex: 0 0 230px;
        min-width: 230px;
        align-self: stretch;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }}
      .col-preview {{
        flex-basis: 260px;
        min-width: 200px;
        align-self: stretch;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }}
    }}
    @media (max-width: 860px) {{
      .wrap {{
        padding: 72px 12px 92px 12px;
      }}
      /* Three nowrap columns already out-run this viewport; a 56px gutter on
         top of that would push the preview card off the edge, so the gap
         steps back down with everything else. */
      .layout {{
        gap: 24px;
      }}
      .source-panel {{
        grid-template-columns: 1fr;
      }}
      .source-right {{
        border-top: 1px solid rgba(0,0,0,0.08);
      }}
    }}
    .card {{
      background: #FFFFFF;
      border-radius: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      border: 1px solid #E2E8F0;
    }}
    .class-card {{ padding: 0; position: relative; overflow: hidden; min-width: 0; }}
    .class-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      /* 22px, not 16px: the card's own body rows (.summary-title, .summary-row,
         .tm-action-toolbar) use the same inset, so the head, the "N Image
         Samples" line and the Webcam/Upload/Device row share one left edge. */
      padding: 12px 22px;
      border-bottom: 1px solid rgba(0,0,0,0.08);
    }}
    .class-drag-handle {{
      cursor: grab;
      user-select: none;
      touch-action: none;
      font-size: 15px;
      color: var(--text);
      opacity: 0.45;
      padding: 2px 6px 2px 2px;
      flex: 0 0 auto;
    }}
    .class-drag-handle:hover {{
      opacity: 0.9;
    }}
    .card.class-card {{
      transition: transform 0.16s ease;
    }}
    .card.class-card.drag-live {{
      transition: none;
    }}
    body.class-dragging {{
      /* Headroom while dragging: the translated card must never be clipped
         by the iframe bottom — the extra padding grows the SPA content and
         the frame-height sync follows it. */
      padding-bottom: 360px;
    }}
    .class-title {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      flex: 1 1 auto;
      min-width: 0;
      font-size: 16px;
      font-weight: 600;
    }}
    .iconbtn {{
      border: 0;
      background: transparent;
      color: var(--muted);
      width: 32px;
      height: 32px;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }}
    .iconbtn:hover {{ background: rgba(0,0,0,0.05); color: var(--text); }}
    .iconbtn svg {{ width: 18px; height: 18px; display: block; }}
    .tm-delete-class {{ font-size: 14px; }}
    .tm-delete-class:hover {{ background: rgba(239,68,68,0.08); color: #EF4444; }}
    .clear-samples {{ font-size: 14px; }}
    .clear-samples:hover {{ background: rgba(245,158,11,0.08); color: #F59E0B; }}
    .more {{
      flex: 0 0 auto;
    }}
    .divider {{ height: 1px; background: rgba(0,0,0,0.08); margin: 0; }}
    .subhead {{ font-size: 12px; color: var(--muted); margin: 0 0 10px 0; }}
    .tm-action-toolbar {{
      display: flex;
      gap: 12px;
      width: 100%;
      justify-content: flex-start;
    }}
    /* Open source drawer: the toolbar is appended straight to .class-card
       (which is padding:0) instead of into the .summary-row wrapper it uses
       when the drawer is closed.  Without this the Webcam/Upload/Device row
       slid flush against the card border.  Same inset as .summary-row so the
       row does not move when the drawer opens. */
    .class-card > .tm-action-toolbar {{
      padding: 10px 22px 14px 22px;
    }}
    .tm-action-toolbar .sample {{
      flex: 0 0 auto;
      min-width: 0;
    }}
    .sample {{
      width: 80px;
      height: 80px;
      border: 0;
      border-radius: 8px;
      background: var(--blue-bg);
      color: var(--blue);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      user-select: none;
      transition: background 0.15s ease;
    }}
    .sample:hover {{ background: var(--blue-bg-hover); }}
    /* Active capture source: its drawer is open and clicking the tile again
       folds it back up.  An outer ring (rather than a border) keeps the 80px
       tile from reflowing the row when the state flips. */
    .sample.active {{
      background: var(--blue-bg-hover);
      box-shadow: 0 0 0 2px #3b82f6;
    }}
    .sample svg {{ width: 22px; height: 22px; stroke: var(--blue); fill: none; stroke-width: 2; }}
    .note {{ margin-top: 12px; font-size: 12px; color: var(--muted); line-height: 1.35; }}
    .preview-pane {{
      margin-top: 14px;
      min-height: 180px;
      border-radius: 16px;
      background: rgba(26,115,232,0.07);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .preview-controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: 6px;
    }}
    /* Context bar: exactly ONE input-source control, chosen in Settings.
       The Preview card never offers a source switcher of its own — showing
       a webcam picker, a serial picker and an upload button together would
       ask the user to choose the same thing twice. */
    #previewContextBar {{ margin-top: 10px; }}
    #previewContextBar:empty {{ display: none; }}
    .preview-context-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .preview-context-row select {{
      flex: 1 1 160px;
      min-width: 0;
      padding: 7px 10px;
      border-radius: 9px;
      border: 1px solid var(--line, #d7dbe2);
      background: var(--surface, #fff);
      color: var(--text, #202124);
      font-size: 12px;
      font-weight: 600;
    }}
    .preview-context-row .upload-pick {{
      flex: 0 0 auto;
    }}
    .preview-upload-name {{
      flex: 1 1 120px;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 12px;
      color: var(--muted, #5f6368);
    }}
    .preview-context-row .tm-btn {{
      flex: 0 0 auto;
      padding: 7px 12px;
      font-size: 12px;
    }}
    /* Sign-only one-click plates: the project's own samples combined into
       a 3-character sign, so the Preview card is testable with no local
       road-sign photo.  Pills, not buttons, because they are three variants
       of ONE action and must not read as three separate features. */
    .preview-synth-group {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .preview-synth-pill {{
      flex: 0 0 auto;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line, #d7dbe2);
      background: var(--surface, #fff);
      color: var(--text, #202124);
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }}
    .preview-synth-pill:hover:not(:disabled) {{
      border-color: #1a73e8;
      color: #1a73e8;
    }}
    .preview-synth-pill:disabled {{
      opacity: 0.5;
      cursor: not-allowed;
    }}
    #previewPane.preview-drop-target {{
      outline: 2px dashed #1a73e8;
      outline-offset: -4px;
    }}
    .preview-controls-right {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .preview-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      user-select: none;
    }}
    .preview-toggle input {{
      width: 16px;
      height: 16px;
      margin: 0;
      accent-color: var(--blue);
    }}
    .preview-select {{
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 700;
      color: var(--text);
      background: var(--input-bg);
    }}
    .preview-pane img {{
      width: 100%;
      height: 100%;
      max-height: 260px;
      object-fit: contain;
      display: block;
      background: #f4f7fb;
    }}
    .preview-empty {{
      padding: 18px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .preview-thresh-bar {{
      display: flex;
      gap: 12px;
      margin-top: 8px;
      padding: 6px 8px;
      background: var(--surface-soft);
      border-radius: 8px;
      align-items: center;
      flex-wrap: wrap;
      font-size: 12px;
      color: var(--muted);
    }}
    .preview-thresh-bar label {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      cursor: default;
    }}
    .preview-thresh-bar input[type="range"] {{
      width: 72px;
      accent-color: var(--blue);
    }}
    .preview-thresh-bar input[type="number"] {{
      width: 48px;
      padding: 2px 4px;
      border: 1px solid var(--input-border);
      border-radius: 4px;
      background: var(--input-bg);
      color: var(--text);
      font-size: 12px;
    }}
    .preview-output {{
      margin-top: 12px;
      display: grid;
      gap: 10px;
    }}
    .out-row {{
      display: grid;
      grid-template-columns: 1fr 140px;
      gap: 10px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
    }}
    .out-bar {{
      height: 10px;
      border-radius: 999px;
      background: rgba(0,0,0,0.08);
      overflow: hidden;
      position: relative;
    }}
    .out-fill {{
      height: 100%;
      width: 0%;
      background: rgba(26,115,232,0.55);
      border-radius: 999px;
      transition: width 120ms linear;
    }}
    .out-toggle {{
      display: block;
      margin: 8px auto 0;
      padding: 4px 12px;
      border: 1px solid var(--input-border);
      border-radius: 8px;
      background: transparent;
      color: var(--muted);
      font-size: 11px;
      cursor: pointer;
    }}
    .out-toggle:hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    .out-pct {{
      position: absolute;
      right: 6px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 10px;
      color: rgba(0,0,0,0.55);
      font-weight: 800;
      pointer-events: none;
    }}
    .addclass {{
      border: 2px dashed #CBD5E1;
      background: transparent;
      border-radius: 12px;
      padding: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      color: #475569;
      font-weight: 600;
      cursor: pointer;
      user-select: none;
      transition: all 0.2s ease;
    }}
    .addclass:hover {{
      border-color: #3169FF;
      color: #3169FF;
      background: #F0F4FF;
    }}
    .addclass .addclass-icon {{ display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px; }}
    .addclass .addclass-icon svg {{ width: 18px; height: 18px; display: block; }}
    /* ── OCR character selector ──────────────────────────────────────────
       Occupies the same slot as .addclass, and only in OCR projects.  Its job
       is to be the *entry point* to the class list, not a second class list:
       every chip writes through the normal class state and the normal card
       renderer, so this file owns no capture, preview or card styling. */
    .charsel {{
      border: 1px solid #CBD5E1;
      background: #F8FAFC;
      border-radius: 12px;
      padding: 14px 16px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .charsel-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }}
    .charsel-title {{ font-family: inherit; font-size: 14px; font-weight: 700; color: #0F172A; }}
    .charsel-count {{ font-family: inherit; font-size: 12px; font-weight: 600; color: #475569; }}
    .charsel-help {{ font-family: inherit; font-size: 12px; line-height: 1.45; color: #475569; }}
    /* 9 columns fits A-Z then 0-9 as four tidy rows of a 500px column. */
    .charsel-grid {{ display: grid; grid-template-columns: repeat(9, minmax(0, 1fr)); gap: 6px; }}
    .charchip {{
      appearance: none;
      -webkit-appearance: none;
      border: 1px solid #CBD5E1;
      background: #fff;
      color: #0F172A;
      border-radius: 8px;
      height: 34px;
      padding: 0;
      font-family: inherit;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease;
    }}
    .charchip:hover:not(:disabled) {{ border-color: #3169FF; color: #3169FF; background: #F0F4FF; }}
    /* A chip that is already a class renders as the selected state — :disabled
       rather than a JS flag, so the "cannot add twice" rule and the visual are
       the same fact. */
    .charchip:disabled {{ background: #E8F0FE; border-color: #3169FF; color: #1a73e8; cursor: default; }}
    .charchip:disabled::after {{ content: '✓'; margin-left: 3px; font-size: 11px; }}
    .charsel-row {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .charsel-btn {{
      appearance: none;
      -webkit-appearance: none;
      border: 1px solid #CBD5E1;
      background: #fff;
      color: #475569;
      border-radius: 8px;
      padding: 7px 10px;
      font-family: inherit;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease;
    }}
    .charsel-btn:hover:not(:disabled) {{ border-color: #3169FF; color: #3169FF; background: #F0F4FF; }}
    .charsel-btn:disabled {{ opacity: 0.45; cursor: default; }}
    .charsel-input {{
      flex: 1 1 120px;
      min-width: 110px;
      border: 1px solid #CBD5E1;
      border-radius: 8px;
      padding: 7px 10px;
      font-family: inherit;
      font-size: 13px;
      color: #0F172A;
      background: #fff;
    }}
    .charsel-input:focus {{ outline: none; border-color: #3169FF; }}
    .charsel-input::placeholder {{ color: #94A3B8; }}
    /* ── OCR empty state ─────────────────────────────────────────────────
       Replaces the card column entirely while an OCR project holds nothing but
       untouched "Class N" placeholders.  It occupies the same slot as the
       cards, so the workspace keeps its shape when the first character lands
       and the hero is swapped for real cards. */
    .ocr-empty {{
      border: 1px dashed #CBD5E1;
      background: #F8FAFC;
      border-radius: 14px;
      padding: 42px 28px 38px;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      gap: 10px;
    }}
    .ocr-empty-icon {{
      width: 46px;
      height: 46px;
      border-radius: 12px;
      background: #E8F0FE;
      color: #1a73e8;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 2px;
    }}
    .ocr-empty-icon svg {{ width: 24px; height: 24px; display: block; }}
    .ocr-empty-title {{ font-family: inherit; font-size: 17px; font-weight: 700; color: #0F172A; }}
    .ocr-empty-sub {{ font-family: inherit; font-size: 13px; line-height: 1.5; color: #475569; max-width: 32em; }}
    .ocr-empty-cta {{
      appearance: none;
      -webkit-appearance: none;
      border: 0;
      border-radius: 10px;
      background: #1a73e8;
      color: #fff;
      font-family: inherit;
      font-size: 14px;
      font-weight: 700;
      padding: 12px 20px;
      margin-top: 8px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: background 0.15s ease, box-shadow 0.15s ease;
    }}
    .ocr-empty-cta:hover {{ background: #1765cc; box-shadow: 0 2px 10px rgba(26,115,232,0.32); }}
    .ocr-empty-cta svg {{ width: 16px; height: 16px; display: block; }}
    /* The character modal reuses .modal; these three rules only strip the
       panel's own card chrome (the modal already provides a surface) and give
       the grid room to breathe at the modal's width. */
    .ocr-char-modal .modal-body {{ padding: 16px 18px 18px; }}
    .ocr-char-modal .charsel {{ border: 0; background: transparent; padding: 0; gap: 12px; }}
    .ocr-char-modal .charsel-grid {{ gap: 8px; }}
    .ocr-char-modal .charchip {{ height: 40px; font-size: 15px; }}
    /* ── OCR glyph settings modal ────────────────────────────────────────
       One row per class.  This is what used to be 3 Streamlit widgets per
       class in the main document flow; at 36 characters that was 108
       controls stacked down the page.  Four fixed columns keep the rows
       aligned as a table so the eye can scan a single column of values. */
    /* Compound, not descendant: the SPA sets the card's className to
       "modal ocr-glyph-modal", so `.ocr-glyph-modal .modal` would need a
       SECOND .modal inside the card and would never match — the width would
       silently fall back to the base 820px. */
    .modal.ocr-glyph-modal {{ width: min(920px, 94vw); }}
    .glyph-note {{ font-family: inherit; font-size: 12px; line-height: 1.5; color: #475569; margin: 0 0 12px; }}
    .glyph-scroll {{ max-height: 46vh; overflow-y: auto; padding-right: 4px; }}
    .glyph-row {{
      display: grid;
      grid-template-columns: minmax(84px, 1fr) 214px 132px 236px;
      gap: 10px;
      align-items: center;
      padding: 7px 0;
      border-top: 1px solid #EEF2F6;
    }}
    .glyph-row:first-child {{ border-top: 0; }}
    .glyph-row.is-head {{
      position: sticky;
      top: 0;
      background: #fff;
      border-top: 0;
      border-bottom: 1px solid #E2E8F0;
      padding-bottom: 6px;
    }}
    .glyph-row.is-head span {{
      font-family: inherit;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #64748B;
    }}
    .glyph-name {{
      font-family: inherit;
      font-size: 13px;
      font-weight: 600;
      color: #0F172A;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .glyph-select {{
      font-family: inherit;
      font-size: 12px;
      padding: 6px 8px;
      border: 1px solid #CBD5E1;
      border-radius: 8px;
      background: #fff;
      color: #0F172A;
      width: 100%;
    }}
    .glyph-check {{ display: flex; align-items: center; gap: 6px; font-family: inherit; font-size: 12px; color: #475569; cursor: pointer; }}
    .glyph-empty {{ font-family: inherit; font-size: 13px; color: #475569; padding: 10px 0; }}
    .train-card, .preview-card {{ padding: 16px 18px 18px 18px; position: relative; }}
    .card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
      min-width: 0;
    }}
    .card-head h3 {{ margin: 0; font-size: 14px; font-weight: 700; min-width: 0; flex-shrink: 1; }}
    .train-status {{
      width: 100%;
      border-radius: 6px;
      background: rgba(0,0,0,0.06);
      color: rgba(0,0,0,0.55);
      padding: 10px 10px;
      font-weight: 700;
      text-align: center;
      margin-bottom: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      box-sizing: border-box;
      max-width: 100%;
      font-size: 13px;
    }}
    .train-adv-panel {{
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px solid rgba(0,0,0,0.08);
      display: block;
    }}
    .train-adv-grid {{
      display: grid;
      gap: 10px;
    }}
    .adv-rec-hint {{
      grid-column: 2;
      font-size: 10px;
      color: var(--muted);
      margin-top: 2px;
      line-height: 1.3;
    }}
    .train-adv-sub {{
      margin: 14px 0 8px 0;
      padding-top: 10px;
      border-top: 1px solid rgba(0,0,0,0.08);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .train-adv-sub:first-child {{
      margin-top: 0;
      padding-top: 0;
      border-top: 0;
    }}
    .adv-help {{
      grid-column: 1 / -1;
      font-size: 10px;
      color: var(--muted);
      line-height: 1.35;
      margin: -4px 0 2px 0;
    }}
    /* Checkboxes inherit the number-input styling above; undo the
       right-aligned 15px bold blue box so they read as toggles. */
    .train-adv-row input[type="checkbox"],
    .train-adv-row input[type="radio"] {{
      width: 18px;
      height: 18px;
      min-width: 0;
      justify-self: end;
      padding: 0;
      margin: 0;
      border: 0;
      background: transparent;
      accent-color: var(--blue);
    }}
    .adv-use-rec {{
      background: transparent;
      border: 1px solid var(--accent, #4caf50);
      color: var(--accent, #4caf50);
      border-radius: 8px;
      padding: 4px 10px;
      font-size: 11px;
      cursor: pointer;
    }}
    .adv-use-rec:hover {{
      background: rgba(76, 175, 80, 0.1);
    }}
    .train-adv-row {{
      display: grid;
      grid-template-columns: 1fr 128px;
      gap: 12px;
      align-items: center;
      font-size: 13px;
      color: rgba(0,0,0,0.62);
      font-weight: 600;
      min-width: 0;
    }}
    .train-adv-row span {{
      line-height: 1.2;
      word-break: normal;
      overflow-wrap: normal;
      white-space: normal;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 4px;
      min-width: 0;
    }}
    /* Inline help tooltip: a small "?" circle whose hover reveals data-tooltip. */
    .tm-help {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 15px;
      height: 15px;
      border-radius: 50%;
      background: rgba(0,0,0,0.12);
      color: rgba(0,0,0,0.55);
      font-size: 10px;
      font-weight: 700;
      cursor: help;
      position: relative;
      flex: 0 0 auto;
      line-height: 1;
      vertical-align: middle;
    }}
    .tm-help:hover {{ background: var(--blue); color: #fff; }}
    .tm-help[data-tooltip]::after {{
      content: attr(data-tooltip);
      position: absolute;
      left: 50%;
      bottom: calc(100% + 8px);
      transform: translateX(-50%);
      background: #1e293b;
      color: #fff;
      font-size: 11.5px;
      font-weight: 500;
      line-height: 1.4;
      padding: 7px 10px;
      border-radius: 6px;
      white-space: normal;
      width: max-content;
      max-width: 220px;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.15s ease;
      z-index: 9999;
      box-shadow: 0 4px 14px rgba(0,0,0,0.22);
    }}
    .tm-help[data-tooltip]:hover::after {{ opacity: 1; }}
    .train-adv-row input,
    .train-adv-row select {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 15px;
      font-weight: 700;
      color: var(--blue);
      background: var(--input-bg);
      text-align: right;
      min-width: 128px;
    }}
    .trainbtn {{
      width: 100%;
      border: 0;
      border-radius: 6px;
      background: rgba(0,0,0,0.08);
      color: rgba(0,0,0,0.38);
      padding: 12px 10px;
      font-weight: 700;
      cursor: not-allowed;
      white-space: normal;
      line-height: 1.2;
    }}
    .trainbtn.enabled {{
      background: rgba(26,115,232,0.12);
      color: var(--blue);
      cursor: pointer;
    }}
    .train-progress {{
      margin-top: 10px;
      display: none;
    }}
    .train-progress-bar {{
      width: 100%;
      height: 8px;
      border-radius: 999px;
      background: rgba(0,0,0,0.08);
      overflow: hidden;
    }}
    .train-progress-fill {{
      height: 100%;
      width: 0%;
      background: rgba(26,115,232,0.55);
      border-radius: 999px;
      transition: width 120ms linear;
    }}
    .train-progress-text {{
      margin-top: 6px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.25;
    }}
    .advanced {{
      /* This rule predates the toggle being a <button>; on a span the two
         UA defaults below never applied, on a button they draw the native
         gray 3D chrome. Same reset .adv-reset already carries. */
      border: 0;
      background: transparent;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      user-select: none;
      padding: 6px 8px;
      border-radius: 6px;
    }}
    .advanced:hover {{ background: rgba(0,0,0,0.04); }}
    .advanced-row {{
      margin-top: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .tm-train-adv-actions {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .adv-reset {{
      border: 0;
      background: transparent;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      padding: 6px 8px;
      border-radius: 6px;
      user-select: none;
    }}
    .adv-reset:hover {{ background: rgba(0,0,0,0.04); color: var(--text); }}
    .exportbtn {{
      border: 0;
      border-radius: 4px;
      background: #f1f3f4;
      padding: 8px 12px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-weight: 700;
      cursor: pointer;
      user-select: none;
      flex-shrink: 0;
      white-space: nowrap;
      max-width: 100%;
      overflow: hidden;
    }}
    .exportbtn:disabled {{
      opacity: 0.55;
      cursor: not-allowed;
    }}
    .exportbtn svg {{ width: 16px; height: 16px; stroke: var(--muted); fill: none; stroke-width: 2; }}
    .footer {{
      position: fixed;
      right: 18px;
      bottom: 12px;
      font-size: 12px;
      color: rgba(0,0,0,0.45);
      pointer-events: none;
      z-index: 1000;
    }}
    .flow {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 0;
      overflow: visible;
      z-index: 1;
      pointer-events: none;
    }}
    .flow path {{ stroke: var(--line); stroke-width: 2; fill: none; }}
    .toast {{
      position: fixed;
      left: 50%;
      top: 18px;
      transform: translateX(-50%);
      background: rgba(32,33,36,0.92);
      color: white;
      padding: 10px 14px;
      border-radius: 10px;
      font-size: 12px;
      z-index: 2001;
      max-width: 720px;
      display: none;
    }}
    .modal-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(32,33,36,0.56);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 2200;
      padding: 20px;
    }}
    .modal {{
      width: min(820px, 92vw);
      background: var(--card);
      border-radius: 14px;
      box-shadow: 0 14px 40px rgba(0,0,0,0.28);
      overflow: hidden;
    }}
    .modal-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(0,0,0,0.08);
    }}
    .modal-head strong {{
      font-size: 14px;
      font-weight: 700;
    }}
    .modal-body {{
      padding: 12px 14px 14px 14px;
    }}
    .modal-actions {{
      display: flex;
      gap: 10px;
      justify-content: flex-end;
      margin-top: 12px;
    }}
    .btn {{
      border: 0;
      border-radius: 8px;
      padding: 10px 12px;
      font-weight: 700;
      cursor: pointer;
      user-select: none;
    }}
    .btn-secondary {{
      background: rgba(0,0,0,0.06);
      color: var(--muted);
    }}
    .btn-secondary:hover {{
      background: rgba(0,0,0,0.09);
      color: var(--text);
    }}
    .btn-primary {{
      background: var(--blue);
      color: white;
    }}
    .btn-primary:hover {{
      background: #1669d4;
    }}
    .preprocess-chip {{
      border: 0;
      border-radius: 999px;
      background: rgba(26,115,232,0.10);
      color: var(--blue);
      padding: 6px 10px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
    }}
    .preprocess-chip:hover {{
      background: rgba(26,115,232,0.16);
    }}
    #classPreprocessModal {{
      z-index: 2400;
      align-items: center;
      justify-content: center;
      padding: 18px;
    }}
    .class-preprocess-shell {{
      width: min(1240px, calc(100vw - 36px));
      height: min(840px, calc(100vh - 36px));
      background: var(--card);
      border-radius: 18px;
      box-shadow: 0 18px 44px rgba(0,0,0,0.28);
      overflow: hidden;
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.9fr);
    }}
    .class-preprocess-left {{
      background: var(--blue-bg);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-width: 0;
      min-height: 0;
    }}
    .class-preprocess-right {{
      background: var(--surface-elevated);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-width: 0;
      min-height: 0;
      overflow: auto;
    }}
    .class-preprocess-visuals {{
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .class-preprocess-visual-pane {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-height: 0;
    }}
    .class-preprocess-visual-title {{
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }}
    .class-preprocess-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--blue);
      font-size: 15px;
      font-weight: 700;
    }}
    .class-preprocess-stage {{
      width: 100%;
      aspect-ratio: 1 / 1;
      height: auto;
      min-height: 0;
      border-radius: 16px;
      background: var(--surface-frost);
      position: relative;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      touch-action: none;
      cursor: crosshair;
    }}
    .class-preprocess-stage img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      image-rendering: pixelated;
      user-select: none;
      -webkit-user-drag: none;
    }}
    .roi-overlay {{
      position: absolute;
      border: 2px solid #1a73e8;
      background: rgba(26,115,232,0.14);
      box-shadow: 0 0 0 1px rgba(255,255,255,0.35) inset;
      pointer-events: none;
    }}
    .preview-stage {{
      position: relative;
      width: 100%;
      max-width: 420px;
      aspect-ratio: 1 / 1;
      height: auto;
      margin: 0 auto;
      overflow: hidden;
      background: var(--surface-soft);
    }}
    .preview-stage img {{
      width: 100%;
      height: 100%;
      max-height: none;
      object-fit: contain;
      display: block;
      /* The device streams 96x96 / 160x160 frames.  Do not let the browser's
         smooth upscale hide individual pixels while debugging a sign. */
      image-rendering: pixelated;
    }}
    .class-preprocess-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(72px, 1fr));
      gap: 10px;
    }}
    .class-preprocess-window-note {{
      margin: 0 0 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    .class-preprocess-thumb {{
      border: 0;
      border-radius: 10px;
      padding: 4px;
      background: rgba(26,115,232,0.08);
      cursor: pointer;
    }}
    .class-preprocess-thumb.active {{
      background: rgba(26,115,232,0.20);
      box-shadow: 0 0 0 2px rgba(26,115,232,0.22) inset;
    }}
    .class-preprocess-thumb.defined {{
      background: rgba(30, 142, 62, 0.14);
      box-shadow: 0 0 0 2px rgba(30, 142, 62, 0.28) inset;
    }}
    .class-preprocess-thumb img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;  /* preserve the original aspect ratio (letterbox) */
      border-radius: 8px;
      display: block;
      background: #eef3f9;
    }}
    .class-preprocess-thumb-state {{
      margin-top: 4px;
      font-size: 11px;
      font-weight: 700;
      text-align: center;
      color: var(--muted);
    }}
    .class-preprocess-thumb.defined .class-preprocess-thumb-state {{
      color: #1e8e3e;
    }}
    .preview-mode-tabs {{
      display: inline-flex;
      gap: 6px;
      padding: 4px;
      border-radius: 999px;
      background: rgba(26,115,232,0.08);
    }}
    .preview-mode-tab {{
      border: 0;
      background: transparent;
      color: var(--muted);
      border-radius: 999px;
      padding: 6px 8px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }}
    .preview-mode-tab.active {{
      background: var(--surface-elevated);
      color: var(--blue);
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .class-preprocess-fields {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .class-preprocess-fields label,
    .class-preprocess-side label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }}
    .class-preprocess-fields input,
    .class-preprocess-side input,
    .class-preprocess-side select {{
      border: 1px solid var(--input-border);
      border-radius: 10px;
      padding: 9px 10px;
      font-size: 13px;
      color: var(--text);
      background: var(--input-bg);
      width: 100%;
      box-sizing: border-box;
    }}
    .class-preprocess-info {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .class-preprocess-current {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .class-preprocess-current .class-preprocess-info {{
      flex: 1;
      min-width: 0;
    }}
    .class-preprocess-result {{
      width: 100%;
      aspect-ratio: 1 / 1;
      height: auto;
      min-height: 0;
      border-radius: 14px;
      background: var(--surface-soft);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .class-preprocess-mode-row {{
      display: flex;
      align-items: end;
      gap: 10px;
    }}
    .class-preprocess-mode-row label {{
      flex: 1;
      min-width: 0;
    }}
    .class-preprocess-mode-save {{
      white-space: nowrap;
      align-self: end;
    }}
    .class-preprocess-status-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .class-preprocess-status-chip {{
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 700;
      background: rgba(26,115,232,0.12);
      color: var(--blue);
    }}
    .class-preprocess-status-chip.dirty {{
      background: rgba(217,119,6,0.14);
      color: #b45309;
    }}
    .class-preprocess-result img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      image-rendering: pixelated;
    }}
    .class-preprocess-actions {{
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: auto;
    }}
    @media (max-width: 600px) {{
      .class-preprocess-shell {{
        grid-template-columns: minmax(0, 1fr);
      }}
      .class-preprocess-visuals {{
        grid-template-columns: minmax(0, 1fr);
      }}
      .class-preprocess-left {{
        min-height: 440px;
      }}
    }}
    .source-panel {{
      margin-top: 16px;
      margin-left: 0;
      margin-right: 0;
      border-top: 1px solid rgba(0,0,0,0.08);
      display: grid;
      grid-template-columns: 1.05fr 1fr;
      min-height: 360px;
      overflow: hidden;
      border-radius: 0 0 12px 12px;
      min-width: 0;
    }}
    .source-left {{
      background: var(--blue-bg);
      /* Matches the 22px body inset above so the "Webcam" pane's chips and
         buttons line up with the toolbar row that opened it. */
      padding: 20px 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-width: 0;
    }}
    .source-right {{
      background: var(--surface-elevated);
      padding: 22px;
      display: flex;
      flex-direction: column;
      max-height: none;
      overflow: hidden;
      min-width: 0;
    }}
    .source-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: var(--blue);
      font-size: 14px;
      font-weight: 700;
    }}
    .source-tools {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    /* The drawer's own dismiss control.  A chevron-up pill reads as "fold
       this panel back up", which is the opposite of the trash can that now
       deletes the whole class a few pixels above it. */
    .source-close-pill {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      border: 1px solid color-mix(in srgb, var(--blue) 35%, transparent);
      background: color-mix(in srgb, var(--blue) 8%, transparent);
      color: var(--blue);
      font: inherit;
      font-size: 12px;
      font-weight: 700;
      line-height: 1;
      padding: 5px 10px 5px 8px;
      border-radius: 999px;
      cursor: pointer;
      user-select: none;
      transition: background 0.15s ease, border-color 0.15s ease;
    }}
    .source-close-pill:hover {{
      background: color-mix(in srgb, var(--blue) 16%, transparent);
      border-color: var(--blue);
    }}
    .source-close-pill svg {{ width: 14px; height: 14px; display: block; }}
    .source-preview-wrap {{
      flex: 1;
      border-radius: 12px;
      overflow: hidden;
      background: color-mix(in srgb, var(--surface-elevated) 55%, transparent);
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .preview-frame {{
      width: 100%;
      height: 100%;
      display: block;
      background: var(--surface-soft);
      object-fit: contain;
      image-rendering: auto;
    }}
    .source-note {{
      min-height: 18px;
      font-size: 12px;
      color: var(--muted);
    }}
    .source-actions {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: auto;
    }}
    .source-settings {{
      color: var(--blue);
      font-size: 26px;
      line-height: 1;
      cursor: pointer;
    }}
    .source-settings-panel {{
      margin-top: 12px;
      padding: 12px;
      border-radius: 12px;
      background: color-mix(in srgb, var(--surface-elevated) 84%, transparent);
      border: 1px solid rgba(26,115,232,0.16);
      box-shadow: 0 6px 18px rgba(0,0,0,0.05);
      box-sizing: border-box;
      max-width: 100%;
    }}
    .source-settings-grid {{
      display: grid;
      gap: 10px;
      width: 100%;
    }}
    .source-settings-grid label {{
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
      width: 100%;
      min-width: 0;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .source-settings-grid input,
    .source-settings-grid select {{
      width: 100%;
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 13px;
      color: var(--text);
      background: var(--input-bg);
      box-sizing: border-box;
      min-width: 0;
    }}
    .source-settings-actions {{
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 10px;
    }}
    .source-settings-actions button {{
      border: 0;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .source-settings-save {{
      background: rgba(26,115,232,0.14);
      color: var(--blue);
    }}
    .source-settings-cancel {{
      background: rgba(0,0,0,0.06);
      color: var(--text);
    }}
    .source-right h4 {{
      margin: 0;
      font-size: 14px;
      font-weight: 700;
      color: var(--text);
    }}
    .source-count {{
      margin-top: 6px;
      font-size: 28px;
      font-weight: 700;
      color: var(--text);
    }}
    .source-count small {{
      font-size: 14px;
      font-weight: 600;
      color: var(--muted);
      margin-left: 4px;
    }}
    .device-select {{
      margin-top: 12px;
      width: 100%;
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      color: var(--text);
      background: var(--input-bg);
    }}
    .device-help {{
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}
    .upload-pick {{
      width: 100%;
      min-height: 144px;
      border: 0;
      border-radius: 12px;
      background: #d6e4f8;
      color: var(--blue);
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      padding: 18px;
      text-align: center;
    }}
    .upload-hint {{
      margin-top: 12px;
      color: #7e93b0;
      font-size: 13px;
      line-height: 1.45;
    }}
    .samples-grid {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      align-items: start;
      min-height: 180px;
      min-width: 0;
    }}
    .samples-grid .sample-item {{
      width: 100%;
    }}
    .samples-grid .sample-thumb {{
      width: 100%;
      aspect-ratio: 1 / 1;
    }}
    .sample-item {{
      position: relative;
    }}
    .sample-thumb {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;  /* preserve the original aspect ratio (letterbox) */
      border-radius: 10px;
      background: #eef2f6;
      border: 1px solid rgba(0,0,0,0.06);
    }}
    .sample-delete {{
      position: absolute;
      top: 6px;
      right: 6px;
      width: 22px;
      height: 22px;
      border: 0;
      border-radius: 999px;
      background: rgba(32, 33, 36, 0.82);
      color: var(--text-inverse);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
      line-height: 1;
      cursor: pointer;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.15s ease;
    }}
    .sample-item:hover .sample-delete,
    .sample-item:focus-within .sample-delete {{
      opacity: 1;
      pointer-events: auto;
    }}
    .sample-delete:hover {{
      background: rgba(180, 35, 24, 0.92);
    }}
    .samples-empty {{
      margin-top: 14px;
      border: 1px dashed rgba(0,0,0,0.12);
      border-radius: 12px;
      min-height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 13px;
    }}
    .summary-title {{
      padding: 10px 22px;
      font-size: 13px;
      font-weight: 700;
      color: rgba(0,0,0,0.7);
    }}
    .summary-row {{
      padding: 10px 22px 14px 22px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      align-items: flex-start;
      min-width: 0;
    }}
    .summary-samples {{
      min-height: 62px;
      display: flex;
      align-items: center;
      overflow: hidden;
      min-width: 0;
      width: 100%;
    }}
    .samples-strip {{
      display: flex;
      gap: 10px;
      align-items: center;
      overflow-x: auto;
      overflow-y: hidden;
      min-width: 0;
      width: 100%;
      min-height: 62px;
      padding-bottom: 8px;
    }}
    .samples-strip .sample-thumb {{
      width: 54px;
      min-width: 54px;
      aspect-ratio: 1 / 1;
    }}
    .samples-strip .sample-item {{
      width: 54px;
      min-width: 54px;
    }}
    .samples-strip .sample-delete {{
      width: 18px;
      height: 18px;
      top: 4px;
      right: 4px;
      font-size: 11px;
    }}
    .samples-strip-empty {{
      width: 100%;
      min-height: 62px;
      border: 1px dashed rgba(0,0,0,0.12);
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 13px;
    }}
    .samples-grid-more {{
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 10px;
      background: #eef2f6;
      border: 1px solid rgba(0,0,0,0.06);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      font-weight: 700;
      color: var(--muted);
      cursor: default;
    }}
    .samples-strip-more {{
      flex: 0 0 auto;
      min-width: 54px;
      width: 54px;
      height: 54px;
      border-radius: 10px;
      background: #eef2f6;
      border: 1px solid rgba(0,0,0,0.06);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      font-weight: 700;
      color: var(--muted);
      cursor: default;
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .field {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .field label {{
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
    }}
    .field input {{
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      color: var(--text);
      background: var(--input-bg);
    }}
    /* Dark mode: hardcoded light-mode colors are unreadable on dark surfaces. */
    @media (prefers-color-scheme: dark) {{
      .navmenu .danger {{ color: #f28b82; }}
      .out-pct {{ color: var(--muted); }}
      .train-status {{ color: var(--muted); background: rgba(255,255,255,0.06); }}
      .train-adv-row {{ color: var(--muted); }}
      .trainbtn {{ color: rgba(255,255,255,0.4); background: rgba(255,255,255,0.06); }}
      .footer {{ color: rgba(255,255,255,0.5); }}
      .summary-title {{ color: var(--text); }}
      .class-preprocess-thumb.defined .class-preprocess-thumb-state {{ color: #34a853; }}
      .class-preprocess-status-chip.dirty {{ color: #f0b429; }}
      .preview-pane img {{ background: #111821; }}
      .out-bar {{ background: rgba(255,255,255,0.12); }}
      .navmenu {{ border-color: rgba(255,255,255,0.1); }}
      .navmenu button:hover {{ background: rgba(255,255,255,0.06); }}
      .navmenu .divider {{ background: rgba(255,255,255,0.1); }}
      .iconbtn:hover {{ background: rgba(255,255,255,0.06); }}
      .divider {{ background: rgba(255,255,255,0.1); }}
      .addclass {{ border-color: #475569; color: var(--muted); }}
      .addclass:hover {{ border-color: #8ab4f8; color: #8ab4f8; background: rgba(138,180,248,0.12); }}
      .charsel {{ border-color: rgba(255,255,255,0.14); background: rgba(255,255,255,0.04); }}
      .charsel-title {{ color: #e8eaed; }}
      .charsel-count, .charsel-help {{ color: #9aa0a6; }}
      .charchip {{ border-color: rgba(255,255,255,0.18); background: rgba(255,255,255,0.06); color: #e8eaed; }}
      .charchip:hover:not(:disabled) {{ border-color: #8ab4f8; color: #8ab4f8; background: rgba(138,180,248,0.12); }}
      .charchip:disabled {{ border-color: #8ab4f8; color: #8ab4f8; background: rgba(138,180,248,0.18); }}
      .charsel-btn {{ border-color: rgba(255,255,255,0.18); background: rgba(255,255,255,0.06); color: #9aa0a6; }}
      .charsel-btn:hover:not(:disabled) {{ border-color: #8ab4f8; color: #8ab4f8; background: rgba(138,180,248,0.12); }}
      .charsel-input {{ border-color: rgba(255,255,255,0.18); background: rgba(255,255,255,0.06); color: #e8eaed; }}
      .charsel-input::placeholder {{ color: #80868b; }}
      .ocr-empty {{ border-color: rgba(255,255,255,0.18); background: rgba(255,255,255,0.03); }}
      .ocr-empty-icon {{ background: rgba(138,180,248,0.16); color: #8ab4f8; }}
      .ocr-empty-title {{ color: #e8eaed; }}
      .ocr-empty-sub {{ color: #9aa0a6; }}
      .ocr-empty-cta {{ background: #8ab4f8; color: #202124; }}
      .ocr-empty-cta:hover {{ background: #aecbfa; box-shadow: 0 2px 10px rgba(0,0,0,0.4); }}
      .glyph-note, .glyph-check, .glyph-empty {{ color: #9aa0a6; }}
      .glyph-name {{ color: #e8eaed; }}
      .glyph-row {{ border-color: rgba(255,255,255,0.1); }}
      .glyph-row.is-head {{ background: #202124; border-bottom-color: rgba(255,255,255,0.18); }}
      .glyph-row.is-head span {{ color: #80868b; }}
      .glyph-select {{ border-color: rgba(255,255,255,0.18); background: rgba(255,255,255,0.06); color: #e8eaed; }}
      /* Toolbar + settings modal dark overrides (Problem 2+3.2) */
      .topnav {{ border-color: rgba(255,255,255,0.08); }}
      .tm-toolbar-brand {{ border-color: rgba(255,255,255,0.1); }}
      .tm-toolbar-btn:hover {{ background: rgba(255,255,255,0.06); }}
      .tm-toolbar-btn.danger {{ color: #f28b82; }}
      .tm-toolbar-btn .kbd {{ background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.6); }}
      .tm-toolbar-dropdown {{ border-color: rgba(255,255,255,0.1); }}
      .tm-toolbar-dropdown button:hover {{ background: rgba(255,255,255,0.06); }}
      .tm-toolbar-dropdown button.danger {{ color: #f28b82; }}
      .tm-toolbar-dropdown .dd-sep {{ background: rgba(255,255,255,0.1); }}
      .tm-toolbar-sep {{ background: rgba(255,255,255,0.1); }}
      .tm-toolbar-return {{ border-color: rgba(255,255,255,0.14); background: rgba(255,255,255,0.03); }}
      .tm-toolbar-return:hover {{ background: rgba(255,255,255,0.07); }}
      .tm-settings-modal {{ background: rgba(0,0,0,0.6); }}
      .tm-settings-card {{ box-shadow: 0 20px 60px rgba(0,0,0,0.55); }}
      .tm-settings-field select,
      .tm-settings-field input[type="number"],
      .tm-settings-field input[type="text"] {{ border-color: rgba(255,255,255,0.14); }}
      .tm-btn {{ border-color: rgba(255,255,255,0.14); background: rgba(255,255,255,0.03); }}
      .tm-btn:hover {{ background: rgba(255,255,255,0.08); }}
      .tm-btn.primary {{ background: var(--blue); color: #0f141a; }}
      .tm-source-chip {{ background: rgba(255,255,255,0.06); }}
      .tm-source-chip.settings-hint,
      .tm-source-chip.tm-source-chip-settings {{ background: rgba(138,180,248,0.14); color: var(--blue); border-color: transparent; }}
      .tm-source-chip.settings-hint:hover,
      .tm-source-chip.tm-source-chip-settings:hover {{ background: rgba(138,180,248,0.24); }}
      .tm-rename-btn {{
        background: transparent;
        color: #9aa4b2;
        border: none;
      }}
      .tm-rename-btn:hover {{
        background: rgba(255,255,255,0.08);
        color: #e8ecf1;
      }}

      /* ── 1. Main workspace cards & containers ─────────────────────────
         .card hardcodes #FFFFFF / #E2E8F0, so every card (class, train,
         preview) stayed light while the text turned white. Route them to
         the dark-aware variables instead. */
      .card,
      .class-card,
      .train-card,
      .preview-card {{
        background: var(--card);
        border-color: var(--line);
        box-shadow: 0 2px 10px rgba(0,0,0,0.35);
      }}
      .class-head,
      .train-adv-panel,
      .train-adv-sub,
      .source-panel,
      .modal-head,
      .tm-settings-foot {{
        border-color: rgba(255,255,255,0.10);
      }}

      /* ── 2. Class card inner elements ──────────────────────────────── */
      .preprocess-chip,
      .class-preprocess-chip {{
        background: rgba(138,180,248,0.16);
        color: var(--blue);
      }}
      .preprocess-chip:hover,
      .class-preprocess-chip:hover {{ background: rgba(138,180,248,0.26); }}
      .samples-empty,
      .samples-strip-empty {{
        background: var(--surface-soft);
        border-color: var(--line);
        color: var(--muted);
      }}
      .sample-thumb,
      .samples-grid-more,
      .samples-strip-more,
      .class-preprocess-thumb img {{
        background: var(--surface-soft);
        border-color: rgba(255,255,255,0.10);
      }}
      .upload-pick {{ background: rgba(138,180,248,0.16); color: var(--blue); }}
      .source-settings-panel {{
        background: rgba(255,255,255,0.04);
        border-color: rgba(138,180,248,0.22);
        box-shadow: 0 6px 18px rgba(0,0,0,0.40);
      }}
      .source-settings-cancel {{ background: rgba(255,255,255,0.08); color: var(--text); }}

      /* ── 3. Training card ──────────────────────────────────────────── */
      .train-status {{ color: var(--muted); background: rgba(255,255,255,0.06); }}
      .trainbtn {{ background: #222b38; color: #64748b; }}
      .trainbtn.enabled {{ background: #2563eb; color: #ffffff; cursor: pointer; }}
      .trainbtn.enabled:hover {{ background: #1d4ed8; }}
      .train-progress-bar {{ background: rgba(255,255,255,0.12); }}
      .train-adv-row {{ color: var(--muted); }}
      .tm-help {{ background: rgba(255,255,255,0.16); color: rgba(255,255,255,0.72); }}
      .tm-help:hover {{ background: var(--blue); color: #0f141a; }}
      .advanced:hover,
      .adv-reset:hover,
      .tm-settings-advanced-toggle:hover {{ background: rgba(255,255,255,0.06); }}
      .adv-use-rec {{ border-color: rgba(94,222,128,0.5); color: #7ee2a0; }}
      .adv-use-rec:hover {{ background: rgba(76,175,80,0.16); }}

      /* ── 4. Preview card ───────────────────────────────────────────── */
      .preview-pane {{ background: rgba(138,180,248,0.08); }}
      .preview-empty {{ color: var(--muted); }}
      .exportbtn {{
        background: var(--surface-elevated);
        color: var(--text);
        border: 1px solid var(--line);
      }}
      .exportbtn:disabled {{ background: var(--surface-soft); color: #64748b; border-color: #2f3b4c; }}
      .exportbtn svg {{ stroke: currentColor; }}
      .preview-mode-tabs {{ background: rgba(138,180,248,0.12); }}
      .preview-mode-tab.active {{ box-shadow: 0 1px 4px rgba(0,0,0,0.40); }}

      /* ── 5. Modals (settings, ROI/edit, OCR char, glyph) ───────────── */
      .modal,
      .class-preprocess-shell,
      .tm-settings-card {{ background: var(--card); }}
      .btn-secondary {{ background: rgba(255,255,255,0.08); color: var(--text); }}
      .btn-secondary:hover {{ background: rgba(255,255,255,0.14); color: var(--text); }}
      .tm-settings-group {{ background: rgba(255,255,255,0.04); border-color: var(--line); }}
      .tm-settings-group-head {{ border-bottom-color: var(--line); }}
      .tm-settings-group-head h4 {{ color: var(--text); }}
      .tm-settings-advanced {{ background: rgba(255,255,255,0.04); border-color: var(--line); }}
      .class-preprocess-left {{ background: rgba(138,180,248,0.10); }}
    }}

    /* ── Built-in Interactive Coach-Mark Tour ─────────────────────────
       Adobe-style product tour, fully self-contained: a spotlight ring
       punched out of a full-screen dim layer plus a floating lesson
       card.  Every colour is drawn from the app's own variables, so the
       tour renders natively in both OS Light and OS Dark mode. */
    #tmTourSpotlight {{
      position: fixed;
      display: none;
      z-index: 4000;
      border-radius: 10px;
      pointer-events: none;
      box-shadow:
        0 0 0 9999px rgba(15, 20, 26, 0.68),
        0 0 0 2px #3b82f6,
        0 0 22px 5px rgba(59, 130, 246, 0.55);
      transition: all 0.28s cubic-bezier(0.22, 1, 0.36, 1);
    }}
    #tmTourSpotlight.on {{ display: block; }}
    /* Steps with target: null (welcome / finish) have nothing to cut out.
       A zero-area box pinned to the top-left corner lets the 9999px
       box-shadow dim the whole viewport uniformly, so the card floats on
       a true full-screen backdrop instead of empty air.  The extra class
       raises specificity above both the base rule and the dark-mode
       override, so the ring never leaks into the dim-only mode. */
    #tmTourSpotlight.dim {{
      top: 0;
      left: 0;
      width: 0;
      height: 0;
      border-radius: 0;
      box-shadow: 0 0 0 9999px rgba(15, 20, 26, 0.68);
    }}
    #tmTourCard {{
      position: fixed;
      display: none;
      z-index: 4001;
      width: 392px;
      max-width: calc(100vw - 28px);
      max-height: calc(100vh - 28px);
      overflow: auto;
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 24px 60px rgba(10, 15, 25, 0.42);
      font-size: 13px;
      line-height: 1.55;
      transition: top 0.28s cubic-bezier(0.22, 1, 0.36, 1), left 0.28s cubic-bezier(0.22, 1, 0.36, 1);
    }}
    #tmTourCard.on {{ display: block; }}
    /* Directional pointer.  A step names a control, and a ring around it
       still leaves the eye to work out which of two boxes the card is
       talking about — so the arrow sits in the GAP between the card and
       the ring and its tip faces the ring.  It is anchored at its own
       centre (translate -50%/-50% in the inline transform) and rotated
       to the side the card is on; the inner SVG then bobs along the
       arrow's own axis, which is why the animation lives on the child. */
    #tmTourArrow {{
      position: fixed;
      display: none;
      z-index: 4002;
      pointer-events: none;
      width: 34px;
      height: 34px;
      color: #3b82f6;
      filter: drop-shadow(0 4px 12px rgba(10, 15, 25, 0.5));
      transition: top 0.28s cubic-bezier(0.22, 1, 0.36, 1), left 0.28s cubic-bezier(0.22, 1, 0.36, 1);
    }}
    #tmTourArrow.on {{ display: block; }}
    #tmTourArrow svg {{ display: block; width: 34px; height: 34px; animation: tmTourArrowBob 1.5s ease-in-out infinite; }}
    @keyframes tmTourArrowBob {{
      0%, 100% {{ transform: translateY(0); }}
      50% {{ transform: translateY(-5px); }}
    }}
    /* Click beacon: a pulsing ring with a pointing hand, pinned to the
       edge of the spotlight beside the control the user is being asked
       to use, so "click this" is anchored to a place, not only words. */
    #tmTourBeacon {{
      position: fixed;
      display: none;
      z-index: 4003;
      pointer-events: none;
      width: 26px;
      height: 26px;
      align-items: center;
      justify-content: center;
    }}
    #tmTourBeacon.on {{ display: flex; }}
    .tm-tour-beacon-ring {{
      position: absolute;
      inset: 0;
      border-radius: 50%;
      border: 2px solid #3b82f6;
      background: color-mix(in srgb, #3b82f6 24%, transparent);
      animation: tmTourBeaconPulse 1.5s ease-out infinite;
    }}
    .tm-tour-beacon-hand {{
      position: relative;
      font-size: 14px;
      line-height: 1;
      filter: drop-shadow(0 2px 5px rgba(10, 15, 25, 0.55));
    }}
    @keyframes tmTourBeaconPulse {{
      0% {{ transform: scale(1); opacity: 0.95; }}
      70% {{ transform: scale(2.05); opacity: 0; }}
      100% {{ transform: scale(2.05); opacity: 0; }}
    }}
    .tm-tour-head {{ padding: 13px 15px 11px; border-bottom: 1px solid var(--line); }}
    .tm-tour-head-top {{ display: flex; align-items: center; gap: 8px; }}
    .tm-tour-phase {{
      display: inline-flex; align-items: center;
      padding: 3px 9px; border-radius: 999px;
      background: color-mix(in srgb, var(--blue) 14%, transparent);
      color: var(--blue);
      font-size: 10px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
      white-space: nowrap;
    }}
    .tm-tour-stepcount {{ color: var(--muted); font-size: 11px; font-weight: 700; white-space: nowrap; }}
    .tm-tour-close {{
      margin-left: auto; width: 24px; height: 24px; flex: 0 0 auto;
      display: inline-flex; align-items: center; justify-content: center;
      border: 0; border-radius: 7px; background: transparent;
      color: var(--muted); cursor: pointer; font-size: 16px; line-height: 1;
    }}
    .tm-tour-close:hover {{ background: color-mix(in srgb, var(--text) 9%, transparent); color: var(--text); }}
    .tm-tour-title {{ margin-top: 9px; font-size: 15px; font-weight: 800; color: var(--text); }}
    .tm-tour-body {{ padding: 12px 15px 4px; }}
    .tm-tour-body p {{ margin: 0 0 9px; }}
    .tm-tour-body ul {{ margin: 0 0 9px; padding-left: 18px; }}
    .tm-tour-body li {{ margin-bottom: 5px; }}
    .tm-tour-body b, .tm-tour-body strong {{ color: var(--text); font-weight: 700; }}
    .tm-tour-muted {{ color: var(--muted); }}
    #tmTourCard code {{
      padding: 1px 4px; border-radius: 4px;
      background: color-mix(in srgb, var(--text) 8%, transparent);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px;
    }}
    #tmTourCard kbd {{
      display: inline-block; padding: 1px 5px; margin: 0 1px;
      border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 5px;
      background: var(--surface-soft); color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px; font-weight: 700;
    }}
    .tm-tour-tip {{
      display: flex; gap: 8px; align-items: flex-start;
      margin: 0 0 9px; padding: 9px 11px; border-radius: 9px;
      background: color-mix(in srgb, var(--blue) 10%, transparent);
    }}
    .tm-tour-tip-ico {{ flex: 0 0 auto; }}
    .tm-tour-details {{
      margin: 0 0 9px; border: 1px solid var(--line); border-radius: 9px;
      background: var(--surface-soft);
    }}
    .tm-tour-details > summary {{
      display: flex; align-items: center; gap: 6px;
      padding: 8px 11px; cursor: pointer;
      color: var(--text); font-weight: 700; list-style: none;
    }}
    .tm-tour-details > summary::-webkit-details-marker {{ display: none; }}
    .tm-tour-details > summary::before {{ content: '▸'; color: var(--muted); font-size: 11px; }}
    .tm-tour-details[open] > summary::before {{ content: '▾'; }}
    .tm-tour-details-body {{ padding: 0 11px 10px; }}
    .tm-tour-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .tm-tour-table th, .tm-tour-table td {{
      padding: 5px 6px; text-align: left; vertical-align: top;
      border-bottom: 1px solid var(--line); color: var(--text);
    }}
    .tm-tour-table th {{ color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .tm-tour-table tr:last-child td, .tm-tour-table tr:last-child th {{ border-bottom: 0; }}
    .tm-tour-foot {{ padding: 10px 15px 13px; border-top: 1px solid var(--line); }}
    .tm-tour-chapters {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 9px; }}
    .tm-tour-chapter {{
      padding: 3px 9px; border-radius: 999px;
      border: 1px solid var(--line); background: var(--surface-soft);
      color: var(--muted); font-size: 10.5px; font-weight: 700; cursor: pointer;
    }}
    .tm-tour-chapter:hover {{ color: var(--text); border-color: var(--blue); }}
    .tm-tour-chapter.active {{ background: var(--blue); border-color: var(--blue); color: var(--text-inverse); }}
    .tm-tour-autoskip {{
      display: flex; align-items: center; gap: 7px;
      margin-bottom: 10px; color: var(--muted); font-size: 11.5px; cursor: pointer;
    }}
    /* Click-to-advance prompt.  Solid tint + a pointing hand when the
       spotlighted control really will advance the tour; a distinct amber
       "hint" variant when the control is disabled and cannot, so the user is
       told which of the two they are looking at instead of clicking a dead
       button wondering why nothing happened. */
    .tm-tour-cue {{
      display: flex; align-items: flex-start; gap: 7px;
      margin: 0 0 10px; padding: 7px 10px; border-radius: 9px;
      border: 1px dashed color-mix(in srgb, var(--blue) 55%, transparent);
      background: color-mix(in srgb, var(--blue) 9%, transparent);
      color: var(--text); font-size: 11.5px; font-weight: 600; line-height: 1.45;
    }}
    .tm-tour-cue-ico {{ flex: 0 0 auto; }}
    .tm-tour-cue.hint {{
      border-style: solid;
      border-color: color-mix(in srgb, #f59e0b 60%, transparent);
      background: color-mix(in srgb, #f59e0b 15%, transparent);
    }}
    /* Trigger phase: the step's screen is still shut and the card is
       asking the user to open it, which is a different instruction from
       the generic "click the highlighted control" — so it gets its own
       solid, unmistakable prompt. */
    .tm-tour-cue.phase {{
      border-style: solid;
      border-color: var(--blue);
      background: color-mix(in srgb, var(--blue) 16%, transparent);
      font-weight: 700;
    }}
    /* The checkpoint was satisfied.  The tour no longer advances on its own
       — the user does — so this green pass state lives on the floating orb
       (`#tmTourOrb.ready`) and in `flashOrb`, not on the card. */
    .tm-tour-actions {{ display: flex; align-items: center; gap: 8px; }}
    .tm-tour-actions + .tm-tour-actions {{ margin-top: 8px; }}
    .tm-tour-spacer {{ flex: 1 1 auto; }}
    .tm-tour-btn {{
      padding: 7px 12px; border-radius: 8px;
      border: 1px solid var(--line); background: var(--surface-soft);
      color: var(--text); font-size: 12.5px; font-weight: 700; cursor: pointer;
    }}
    /* The one button that swaps the card for the floating orb spans the
       whole footer: it is the difference between reading and doing, so it
       gets the row to itself rather than competing with Skip / Back. */
    .tm-tour-btn.wide {{ flex: 1 1 100%; text-align: center; }}
    .tm-tour-btn:hover:not(:disabled) {{ border-color: var(--blue); color: var(--blue); }}
    .tm-tour-btn:disabled {{ opacity: 0.45; cursor: default; }}
    .tm-tour-btn.primary {{ background: var(--blue); border-color: var(--blue); color: var(--text-inverse); }}
    .tm-tour-btn.primary:hover:not(:disabled) {{ background: var(--blue); color: var(--text-inverse); filter: brightness(1.06); }}
    .tm-tour-btn.ghost {{ background: transparent; border-color: transparent; color: var(--muted); }}
    .tm-toolbar-tour {{ color: var(--blue); }}
    /* ── Workflow overview (step 0) ───────────────────────────────────
       Step 0 is the map rather than a lesson, so instead of prose it lists
       the whole trip as numbered stages: the user sees where they are
       going before the guide starts pointing at controls. */
    .tm-tour-stages {{ list-style: none; margin: 0 0 10px; padding: 0; }}
    .tm-tour-stages li {{
      display: flex; gap: 9px; align-items: flex-start;
      padding: 7px 0; border-top: 1px solid var(--line);
    }}
    .tm-tour-stages li:first-child {{ border-top: 0; }}
    .tm-tour-stage-num {{
      flex: 0 0 auto; width: 20px; height: 20px; border-radius: 50%;
      display: inline-flex; align-items: center; justify-content: center;
      background: var(--blue); color: var(--text-inverse);
      font-size: 11px; font-weight: 800;
    }}
    .tm-tour-stage-txt {{ display: flex; flex-direction: column; gap: 1px; min-width: 0; }}
    .tm-tour-stage-t {{ font-weight: 700; color: var(--text); }}
    .tm-tour-stage-d {{ color: var(--muted); font-size: 11.5px; line-height: 1.45; }}
    /* ── Floating orb · Phase B "hands-on" mode ───────────────────────
       The moment the user starts working, the dim layer and the big card
       are pushed out of the way and this pill takes over: docked in the
       bottom-right corner, draggable, gently pulsing, and carrying the
       LIVE state of whatever they are doing (sample counts, thresholds,
       training).  It is deliberately small and corner-parked so it never
       covers the control being used, and it never advances on its own —
       the user clicks it when they are ready. */
    #tmTourOrb {{
      position: fixed;
      display: none;
      right: 18px;
      bottom: 18px;
      z-index: 4004;
      max-width: min(420px, calc(100vw - 36px));
      align-items: center;
      gap: 9px;
      padding: 9px 12px;
      border-radius: 999px;
      border: 1px solid color-mix(in srgb, var(--blue) 55%, var(--line));
      background: var(--card);
      color: var(--text);
      box-shadow: 0 12px 32px rgba(10, 15, 25, 0.38);
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      user-select: none;
    }}
    #tmTourOrb.on {{ display: inline-flex; }}
    /* The gentle "I am still here" pulse lives on a pseudo-element so the
       dark-mode override can keep owning the real drop shadow. */
    #tmTourOrb::before {{
      content: '';
      position: absolute;
      inset: 0;
      border-radius: inherit;
      pointer-events: none;
      animation: tmTourOrbPulse 2.4s ease-out infinite;
    }}
    @keyframes tmTourOrbPulse {{
      0% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--blue) 55%, transparent); }}
      70%, 100% {{ box-shadow: 0 0 0 10px color-mix(in srgb, var(--blue) 0%, transparent); }}
    }}
    /* The step's own job is done — the badge turns green and stops
       pulsing, so "you may move on now" is a colour change away from
       "keep working". */
    #tmTourOrb.ready {{
      border-color: #16a34a;
      background: color-mix(in srgb, #16a34a 14%, var(--card));
      box-shadow: 0 12px 32px rgba(10, 15, 25, 0.38), 0 0 0 3px color-mix(in srgb, #16a34a 28%, transparent);
    }}
    #tmTourOrb.ready::before {{ display: none; }}
    #tmTourOrb.dragging {{ cursor: grabbing; }}
    #tmTourOrb.dragging::before {{ display: none; }}
    .tm-tour-orb-dot {{
      flex: 0 0 auto; width: 9px; height: 9px; border-radius: 50%;
      background: #16a34a;
    }}
    .tm-tour-orb-text {{ flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .tm-tour-orb-text b {{ color: var(--text); font-weight: 800; }}
    .tm-tour-orb-go {{
      flex: 0 0 auto; padding: 4px 10px; border-radius: 999px;
      border: 1px solid transparent; background: var(--blue); color: var(--text-inverse);
      font-size: 11.5px; font-weight: 800; white-space: nowrap; cursor: pointer;
    }}
    #tmTourOrb.ready .tm-tour-orb-go {{ background: #16a34a; }}
    .tm-tour-orb-help {{
      flex: 0 0 auto; width: 22px; height: 22px; border-radius: 50%;
      display: inline-flex; align-items: center; justify-content: center;
      border: 1px solid var(--line); background: var(--surface-soft);
      color: var(--muted); font-size: 12px; font-weight: 800; cursor: pointer;
    }}
    .tm-tour-orb-help:hover {{ color: var(--text); border-color: var(--blue); }}
    @media (prefers-color-scheme: dark) {{
      #tmTourCard {{ box-shadow: 0 24px 60px rgba(0, 0, 0, 0.6); }}
      #tmTourOrb {{ box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6); }}
      #tmTourOrb.ready {{ box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6), 0 0 0 3px color-mix(in srgb, #16a34a 32%, transparent); }}
      #tmTourArrow {{ color: #8ab4f8; }}
      .tm-tour-beacon-ring {{
        border-color: #8ab4f8;
        background: color-mix(in srgb, #8ab4f8 26%, transparent);
      }}
      #tmTourSpotlight {{
        box-shadow:
          0 0 0 9999px rgba(0, 0, 0, 0.72),
          0 0 0 2px #8ab4f8,
          0 0 22px 5px rgba(138, 180, 248, 0.5);
      }}
    }}
  </style>
</head>
<body>
<div class="toast" id="toast"></div>
<div class="topnav" id="goHome">
  <div class="tm-toolbar-brand">
    <img src="data:image/png;base64,{_AIO_SCOUTS_LOGO_B64}" alt="AIoScout"/>
    <span>AIoScout</span>
  </div>
  <div class="tm-toolbar-menu" id="menuFile">
    <button type="button" class="tm-toolbar-btn primary" id="menuFileBtn">File ▾</button>
    <div class="tm-toolbar-dropdown" id="menuFileDropdown">
      <button type="button" id="navNewProject"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg></span>New project<span class="kbd">⌘N</span></button>
      <button type="button" id="navOpenProject"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg></span>Open project<span class="kbd">⌘O</span></button>
      <div class="dd-sep"></div>
      <button type="button" id="navSaveProject"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path><polyline points="17 21 17 13 7 13 7 21"></polyline><polyline points="7 3 7 8 15 8"></polyline></svg></span>Save<span class="kbd">⌘S</span></button>
      <button type="button" id="navSaveAsProject"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg></span>Save as<span class="kbd">⇧⌘S</span></button>
      <div class="dd-sep"></div>
      <button type="button" id="navExportDataset"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg></span>Export dataset</button>
    </div>
  </div>
  <div class="tm-toolbar-menu" id="menuEdit">
    <button type="button" class="tm-toolbar-btn" id="menuEditBtn">Edit ▾</button>
    <div class="tm-toolbar-dropdown" id="menuEditDropdown">
      <button type="button" id="navEditCharacters" style="display:none"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 7 4 4 20 4 20 7"></polyline><line x1="9" y1="20" x2="15" y2="20"></line><line x1="12" y1="4" x2="12" y2="20"></line></svg></span>Edit characters</button>
      <button type="button" id="navGlyphSettings" style="display:none"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="6" x2="20" y2="6"></line><line x1="4" y1="12" x2="20" y2="12"></line><line x1="4" y1="18" x2="20" y2="18"></line><circle cx="9" cy="6" r="2"></circle><circle cx="15" cy="12" r="2"></circle><circle cx="11" cy="18" r="2"></circle></svg></span>Glyph settings…</button>
      <div class="dd-sep" id="navEditCharactersSep" style="display:none"></div>
      <button type="button" class="danger" id="navResetProject"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg></span>Reset project</button>
    </div>
  </div>
  <button type="button" class="tm-toolbar-btn" id="navSettings" title="Global input settings (camera, serial, FPS, frame)"><span class="tm-toolbar-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.19.42.58.74 1.06.91.48.17.99.22 1.51.09"></path></svg></span>Settings</button>
  <button type="button" class="tm-toolbar-btn" id="navLicenseList" style="display:none" title="Manage allowed 3-character car license plates (1 or more valid targets)"><span class="tm-toolbar-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"></rect><line x1="7" y1="9" x2="17" y2="9"></line><line x1="7" y1="13" x2="14" y2="13"></line></svg></span>🪪 License List <span id="navLicenseCount" style="display:inline-block;min-width:18px;padding:1px 6px;margin-left:6px;border-radius:9px;background:#DBEAFE;color:#1D4ED8;font-size:11px;font-weight:800;text-align:center;">1</span></button>
  <button type="button" class="tm-toolbar-btn" id="navPreferences" title="Preferences (display limits, caching)"><span class="tm-toolbar-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg></span>Preferences</button>
  <button type="button" class="tm-toolbar-btn tm-toolbar-tour" id="navTutorial" title="Interactive tutorial &amp; built-in manual">📖 Tutorial</button>
  <div class="tm-toolbar-sep"></div>
  <div class="tm-toolbar-spacer"></div>
  <button type="button" class="tm-toolbar-return" id="navReturn">← Return Home</button>
</div>
<div class="navmenu" id="navMenu">
  <button type="button">Open project</button>
  <button type="button">Save project</button>
  <button type="button">Export dataset</button>
  <button type="button">Return</button>
  <div class="divider"></div>
  <button type="button" class="danger">Reset project</button>
</div>
<div class="modal-backdrop" id="advancedModal">
  <div class="modal">
    <div class="modal-head">
      <strong>Advanced</strong>
      <button class="iconbtn" id="advancedClose" type="button">✕</button>
    </div>
    <div class="modal-body">
      <div class="form-grid">
        <div class="field"><label>Batch Size</label><input id="advBatch" type="number" min="1" step="1"/></div>
        <div class="field"><label>Epochs</label><input id="advEpochs" type="number" min="1" step="1"/></div>
        <div class="field"><label>Validation Split</label><input id="advVal" type="number" min="0.05" max="0.5" step="0.05"/></div>
        <div class="field"><label>Learning Rate</label><input id="advLr" type="number" min="0.00001" step="0.0001"/></div>
        <div class="field"><label>Conv1 Filters</label><input id="advConv1" type="number" min="1" step="1"/></div>
        <div class="field"><label>Conv2 Filters</label><input id="advConv2" type="number" min="1" step="1"/></div>
        <div class="field"><label>Dense Units</label><input id="advDense" type="number" min="1" step="1"/></div>
      </div>
      <div class="modal-actions">
        <button class="btn btn-secondary" id="advancedCancel" type="button">Close</button>
        <button class="btn btn-primary" id="advancedSave" type="button">Save</button>
      </div>
    </div>
  </div>
</div>
<div class="modal-backdrop" id="classPreprocessModal"></div>
<div class="modal-backdrop" id="ocrCharModal"></div>
<div class="modal-backdrop" id="ocrGlyphModal"></div>
<!-- Road-sign target sequence.  Its rules ship in their own small sheet next
     to their own markup rather than being appended to the workspace's: that
     sheet is where the theme layer lives, and a component whose styles travel
     with it can be read, moved or deleted in one piece. -->
<style>
  .signseq-bar {{
    border: 1px solid #BFDBFE; background: #EFF6FF; border-radius: 12px;
    padding: 10px 12px; margin-bottom: 12px; display: flex; flex-direction: column; gap: 8px;
  }}
  .signseq-head {{ display: flex; align-items: center; gap: 8px; }}
  .signseq-title {{
    font-size: 12px; font-weight: 700; color: #1E3A8A;
    text-transform: uppercase; letter-spacing: 0.6px; flex: 1 1 auto;
  }}
  .signseq-row {{ display: flex; align-items: flex-end; gap: 14px; flex-wrap: wrap; }}
  .signseq-slot-wrap {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
  /* The slot is an <input>, not a div: the characters are edited in place, so
     there is no second editing surface (and no modal) to open. */
  .signseq-slot {{
    width: 56px; height: 56px; border-radius: 12px; border: 2px solid #2563EB;
    background: #FFFFFF; color: #0F172A; text-align: center; padding: 0;
    font-size: 26px; font-weight: 800; line-height: 1; text-transform: uppercase;
    -webkit-appearance: none; appearance: none;
  }}
  .signseq-slot:focus {{ outline: none; border-color: #1D4ED8; box-shadow: 0 0 0 3px #DBEAFE; }}
  /* An unset slot is dashed and greyed, so it never reads as the character "?". */
  .signseq-slot.empty {{ border-style: dashed; border-color: #94A3B8; color: #94A3B8; }}
  .signseq-move {{ display: flex; gap: 4px; }}
  .signseq-move button {{
    width: 26px; height: 22px; border: 1px solid #CBD5E1; border-radius: 6px;
    background: #FFFFFF; color: #334155; font-size: 11px; cursor: pointer; line-height: 1;
  }}
  .signseq-move button:hover:not(:disabled) {{ background: #E2E8F0; }}
  .signseq-move button:disabled {{ opacity: 0.35; cursor: default; }}
  .signseq-note {{ font-size: 12px; color: #475569; line-height: 1.45; }}
  /* "Slots 1 & 3": which positions of the sign this one class is matched in. */
  .signslot-badge {{
    margin-left: 8px; padding: 2px 7px; border-radius: 999px;
    background: #E0E7FF; color: #3730A3; font-size: 10px; font-weight: 700;
    letter-spacing: 0.3px; vertical-align: middle;
  }}

  /* The MATCH banner.  Sized to be read from across a desk — the answer is the
     whole point of the project, so it is the loudest thing in the pane. */
  .signmatch {{
    border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
    border: 2px solid #CBD5E1; background: #F8FAFC;
  }}
  .signmatch.yes {{ border-color: #16A34A; background: #F0FDF4; }}
  .signmatch.no {{ border-color: #DC2626; background: #FEF2F2; }}
  .signmatch.idle {{ border-color: #CBD5E1; background: #F8FAFC; }}
  .signmatch-main {{ font-size: 20px; font-weight: 800; letter-spacing: 0.3px; }}
  .signmatch.yes .signmatch-main {{ color: #15803D; }}
  .signmatch.no .signmatch-main {{ color: #B91C1C; }}
  .signmatch.idle .signmatch-main {{ color: #475569; }}
  .signmatch-slots {{ display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }}
  .signmatch-slot {{
    border: 1px solid #CBD5E1; border-radius: 10px; background: #FFFFFF;
    padding: 6px 8px; min-width: 62px; text-align: center;
  }}
  .signmatch-slot .want {{ font-size: 10px; color: #64748B; font-weight: 700; letter-spacing: 0.4px; }}
  .signmatch-slot .got {{ font-size: 20px; font-weight: 800; color: #0F172A; line-height: 1.1; }}
  .signmatch-slot .prob {{ font-size: 11px; color: #475569; }}
  .signmatch-slot.ok {{ border-color: #16A34A; background: #F0FDF4; }}
  .signmatch-slot.bad {{ border-color: #DC2626; background: #FEF2F2; }}
  .signmatch-slot.ok .got {{ color: #15803D; }}
  .signmatch-slot.bad .got {{ color: #B91C1C; }}
  /* The 28x28 tensor the model was actually handed, nearest-neighbour upscaled.
     pixelated, not smooth: anti-aliasing a glyph at this magnification invents
     the intermediate greys the model never saw. */
  .signmatch-slot img {{
    width: 34px; height: 34px; display: block; margin: 2px auto 0;
    image-rendering: pixelated; background: #FFFFFF; border-radius: 4px;
  }}
  .signmatch-hint {{ font-size: 12px; color: #475569; margin-top: 8px; line-height: 1.45; }}
</style>
<!-- Problem 3.2: Global Input Settings (Single Source of Truth) -->
<div class="tm-settings-modal" id="globalSettingsModal" role="dialog" aria-modal="true">
  <div class="tm-settings-card">
    <div class="tm-settings-head">
      <div>
        <h3>Input Settings</h3>
        <p>Configure webcam, serial device, and capture parameters. Applied globally to all class cards.</p>
      </div>
    </div>
    <div class="tm-settings-grid">
      <div class="tm-settings-group" id="settingsCameraGroup">
        <div class="tm-settings-group-head">
          <span class="tm-settings-group-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="14" height="12" rx="2"></rect><path d="M22 8l-6 4 6 4V8z"></path></svg></span>
          <h4>Webcam Settings</h4>
        </div>
        <div class="tm-settings-field">
          <label>Webcam</label>
          <select id="settingsWebcam"></select>
        </div>
        <div class="tm-settings-field">
          <label>Capture FPS</label>
          <input id="settingsFps" type="number" min="1" max="60" step="0.5"/>
        </div>
      </div>
      <div class="tm-settings-group" id="settingsMcuGroup">
        <div class="tm-settings-group-head">
          <span class="tm-settings-group-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"></rect><line x1="9" y1="9" x2="15" y2="9"></line><line x1="9" y1="15" x2="15" y2="15"></line></svg></span>
          <h4>MCU Device Settings</h4>
        </div>
        <div class="tm-settings-field">
          <label>Serial Device (port)</label>
          <select id="settingsSerialPort">
            <option value="">— Not connected —</option>
          </select>
        </div>
        <div class="tm-settings-field">
          <label>Baud rate</label>
          <select id="settingsSerialBaud">
            <option value="9600">9600</option>
            <option value="57600">57600</option>
            <option value="115200">115200 (default)</option>
            <option value="230400">230400</option>
            <option value="460800">460800</option>
            <option value="921600">921600</option>
          </select>
        </div>
        <div class="tm-settings-field">
          <label>Frame side (pixels)</label>
          <input id="settingsFrameSide" type="number" min="32" max="512" step="16"/>
        </div>
        <div class="tm-settings-field">
          <label>Channels</label>
          <select id="settingsChannels">
            <option value="1">1 — Grayscale</option>
            <option value="3">3 — RGB Color</option>
          </select>
        </div>
      </div>
      <div class="tm-settings-group" id="settingsOodGroup">
        <div class="tm-settings-group-head">
          <span class="tm-settings-group-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg></span>
          <h4>Preview</h4>
        </div>
        <div class="tm-settings-field">
          <label>Preview Image Input Source</label>
          <select id="settingsPreviewSource">
            <option value="webcam">Webcam</option>
            <option value="device">Device (MCU)</option>
            <option value="upload">Local Image File</option>
          </select>
        </div>
        <div class="tm-settings-field">
          <label title="Minimum sign_pct before Preview treats the frame as empty / invalid.">OOD sign min %</label>
          <input id="settingsOodSignPctMin" type="number" min="0" max="100" step="0.1"/>
        </div>
        <div class="tm-settings-field">
          <label title="Maximum sign_pct before Preview treats the frame as too full / shadowed.">OOD sign max %</label>
          <input id="settingsOodSignPctMax" type="number" min="0" max="100" step="0.1"/>
        </div>
        <div class="tm-settings-field">
          <label title="Minimum softmax confidence required before Preview accepts a class result.">OOD max prob</label>
          <input id="settingsOodMaxProbMin" type="number" min="0" max="1" step="0.01"/>
        </div>
        <div class="tm-settings-field">
          <label title="Maximum entropy ratio allowed before Preview suppresses to No Sign.">OOD entropy max</label>
          <input id="settingsOodEntropyRatioMax" type="number" min="0" max="1" step="0.01"/>
        </div>
      </div>
      <div class="tm-settings-group">
        <div class="tm-settings-group-head">
          <span class="tm-settings-group-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg></span>
          <h4>Training Hyperparameters</h4>
        </div>
        <div id="modalOcrEmpty" class="tm-settings-empty" style="display:none">
          Standard hyperparameters live in the <b>Training</b> card's <b>Advanced</b> panel. No extra settings for image projects.
        </div>
        <div id="modalOcrSection" style="display:none">
          <div class="train-adv-grid" id="advOcrPanel">
            <div class="train-adv-sub">Dataset mixing</div>
            <div class="train-adv-row"><span>Mix EMNIST into every class <span class="tm-help" data-tooltip="Augments your captured samples with EMNIST glyphs so the model learns printed characters too.">?</span></span><input id="advOcrEmnist" type="checkbox"/></div>
            <div class="adv-help" id="advOcrMixHelp">EMNIST glyphs are already isolated and centred, so they bypass segmentation and are polarity-aligned to the capture convention before scaling.</div>
            <div class="train-adv-row" id="advOcrSplitRow"><span>EMNIST split <span class="tm-help" data-tooltip="Which EMNIST character set to sample from (ByClass, ByMerge, Letters, Digits, MNIST).">?</span></span><select id="advOcrSplit">{emnist_split_options}</select></div>
            <div class="train-adv-row" id="advOcrPerClassRow"><span>Synthetic per class <span class="tm-help" data-tooltip="Extra EMNIST glyphs added per class to balance and enlarge the training set.">?</span></span><input id="advOcrPerClass" type="number" min="0" max="5000" step="10"/></div>
            <div class="train-adv-row" id="advOcrBackgroundRow"><span>Use <b>_background</b> noise class to replace OOD rejection <span class="tm-help" data-tooltip="OFF (the Car License default): train only the target character classes and reject non-sign crops with the standard OOD gates (sign %, max prob, entropy) plus the match-probability floor. ON: train an extra synthetic _background class and let it be the only per-slot rejector (kSignBackgroundIsTheOnlyRejector = 1 on export).">?</span></span><input id="settingsUseBackgroundOod" type="checkbox"/></div>
            <div class="train-adv-row" id="advOcrBackgroundNRow"><span>Background crops <span class="tm-help" data-tooltip="Number of random background/noise crops used to train the rejector class. Used only when the _background toggle above is ON.">?</span></span><input id="advOcrBackgroundN" type="number" min="0" max="5000" step="10"/></div>
            <div class="adv-help">0 mirrors the per-class count. With the toggle OFF a Car License model has exactly the target characters as classes (e.g. A, 1, 2) and no <b>{BACKGROUND_CLASS_NAME}</b> output; with it ON the extra N+1 class named <b>{BACKGROUND_CLASS_NAME}</b> teaches the model to reject OOD crops directly.</div>
            <div class="train-adv-sub">Transfer learning</div>
            <div class="train-adv-row"><span>Fine-tune pre-trained backbone <span class="tm-help" data-tooltip="Uses a pre-trained EMNIST backbone and fine-tunes it on your data instead of training from scratch.">?</span></span><input id="advOcrTransfer" type="checkbox"/></div>
            <div class="train-adv-row"><span>Pre-train samples / class <span class="tm-help" data-tooltip="Samples per class used to warm up the backbone before the head is attached.">?</span></span><input id="advOcrPretrainN" type="number" min="100" max="6000" step="100"/></div>
            <div class="train-adv-row"><span>Pre-train epochs <span class="tm-help" data-tooltip="Epochs spent training the backbone on the warm-up data.">?</span></span><input id="advOcrPretrainEpochs" type="number" min="1" max="100" step="1"/></div>
            <div class="train-adv-row"><span>Head epochs <span class="tm-help" data-tooltip="Epochs spent training only the new classification head with the backbone frozen.">?</span></span><input id="advOcrHeadEpochs" type="number" min="1" max="200" step="1"/></div>
            <div class="train-adv-row"><span>Head learning rate <span class="tm-help" data-tooltip="Learning rate for the classification head during the frozen-backbone phase.">?</span></span><input id="advOcrHeadLr" type="number" min="0" step="0.0001"/></div>
            <div class="train-adv-row"><span>Fine-tune epochs <span class="tm-help" data-tooltip="Epochs spent unfreezing the backbone's last conv block(s) at the lower fine-tune rate.">?</span></span><input id="advOcrFinetuneEpochs" type="number" min="0" max="200" step="1"/></div>
            <div class="train-adv-row"><span>Fine-tune learning rate <span class="tm-help" data-tooltip="Lower learning rate used while fine-tuning the unfrozen backbone layers.">?</span></span><input id="advOcrFinetuneLr" type="number" min="0" step="0.0001"/></div>
            <div class="train-adv-row"><span>Conv blocks to unfreeze <span class="tm-help" data-tooltip="How many of the backbone's last convolution blocks are unfrozen during fine-tuning.">?</span></span><input id="advOcrUnfreeze" type="number" min="0" max="16" step="1"/></div>
            <div class="adv-help">Phase 1 trains only the new N+1 head with the backbone frozen; phase 2 unfreezes the last conv block(s) at the lower fine-tune rate.</div>
          </div>
        </div>
      </div>
    </div>
    <div class="tm-settings-advanced">
      <button type="button" class="tm-settings-advanced-toggle" id="settingsAdvancedToggle" aria-expanded="false">
        <span class="tm-settings-advanced-chevron" id="settingsAdvancedChevron">▸</span>
        Advanced Settings
      </button>
      <div class="tm-settings-advanced-body" id="settingsAdvancedBody" style="display:none">
        <div class="tm-settings-field">
          <label>Frame sync word (hex, space separated)</label>
          <input id="settingsSerialSync" type="text" placeholder="AA 55 AA"/>
          <small>Sync header used to align incoming MCU frames. Applied to both class capture and preview device source.</small>
        </div>
        <div class="tm-settings-field" id="settingsEmnistGroup" style="display:none">
          <label>Developer: EMNIST template samples</label>
          <label class="tm-settings-check">
            <input id="settingsAutoSeedEmnist" type="checkbox"/>
            <span>Auto-fill empty classes with EMNIST template samples (24 / class)</span>
          </label>
          <div class="tm-settings-seed-row">
            <button type="button" class="tm-btn" id="settingsSeedEmnistNowBtn">Fill Empty Classes with EMNIST Samples Now</button>
          </div>
          <small id="settingsSeedEmnistHint">Draws 24 dark-ink-on-light EMNIST captures into every EMPTY character class in this project, so the whole capture → preprocess → train → preview pipeline can be exercised before a camera is wired up. Classes with real camera samples are kept untouched.</small>
        </div>
      </div>
    </div>
    <div class="tm-settings-foot">
      <button type="button" class="tm-btn" id="settingsCancel">Cancel</button>
      <button type="button" class="tm-btn primary" id="settingsSave">Save changes</button>
    </div>
  </div>
</div>
<!-- Allowed License Plates (Car License whitelist) -->
<div class="tm-settings-modal" id="licenseListModal" role="dialog" aria-modal="true">
  <div class="tm-settings-card">
    <div class="tm-settings-head" style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;">
      <div>
        <h3>🪪 Allowed License Plates</h3>
        <p>One or more valid 3-character plates. Classes are the union of their characters; Preview says YES when the plate in view matches any entry.</p>
      </div>
      <button type="button" id="licenseListClose" class="tm-btn" style="min-width:34px;padding:6px 10px;" title="Close">✕</button>
    </div>
    <div id="licenseListRows" style="display:flex;flex-direction:column;gap:10px;margin:14px 0;"></div>
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <button type="button" id="licenseListAdd" class="tm-btn">+ Add Allowed License Plate</button>
      <span id="licenseListHint" style="font-size:12px;color:#475569;"></span>
    </div>
    <div style="display:flex;justify-content:flex-end;gap:10px;margin-top:20px;">
      <button type="button" id="licenseListCancel" class="tm-btn">Cancel</button>
      <button type="button" id="licenseListSave" class="tm-btn" style="background:#1D4ED8;color:#fff;border-color:#1D4ED8;">Save &amp; Sync Classes</button>
    </div>
  </div>
</div>
<div class="tm-settings-modal" id="preferencesModal" role="dialog" aria-modal="true">
  <div class="tm-settings-card">
    <div class="tm-settings-head">
      <div>
        <h3>Preferences</h3>
        <p>Adjust display and caching behavior.</p>
      </div>
    </div>
    <div class="tm-settings-grid">
      <div class="tm-settings-group">
        <div class="tm-settings-group-head">
          <span class="tm-settings-group-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"></rect><line x1="9" y1="9" x2="15" y2="9"></line><line x1="9" y1="15" x2="15" y2="15"></line></svg></span>
          <h4>Display</h4>
        </div>
        <div class="tm-settings-field">
          <label>Max samples displayed per class</label>
          <input id="prefMaxSamples" type="number" min="1" max="100" step="1" value="12"/>
          <small>Controls how many sample thumbnails are cached and shown in the class card (default: 12). Lower values reduce memory usage.</small>
        </div>
      </div>
    </div>
    <div class="tm-settings-foot">
      <button type="button" class="tm-btn" id="prefCancel">Cancel</button>
      <button type="button" class="tm-btn primary" id="prefSave">Save changes</button>
    </div>
  </div>
</div>
<div class="wrap">
  <svg class="flow" viewBox="0 0 1200 700" preserveAspectRatio="none" aria-hidden="true">
    <path id="flowClass1" d="" />
    <path id="flowClass2" d="" />
    <path id="flowTrainPreview" d="" />
  </svg>
  <div class="layout">
    <div class="col-classes" id="classes"></div>
    <div class="col-train">
      <div class="card train-card" id="trainCard">
        <div class="card-head"><h3><span class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg></span> Training</h3></div>
        <div class="train-status" id="trainStatus">Not trained</div>
        <button class="trainbtn" id="trainBtn">Train Embedded Model</button>
        <div class="train-progress" id="trainProgress">
          <div class="train-progress-bar"><div class="train-progress-fill" id="trainProgressFill"></div></div>
          <div class="train-progress-text" id="trainProgressText"></div>
        </div>
        <div class="advanced-row">
          <button class="advanced" id="advBtn" type="button">
            <span id="advChevron" class="advanced-chevron">▸</span> Advanced
          </button>
        </div>
        <div class="train-adv-panel" id="trainAdvPanel" style="display:none">
          <div class="tm-train-adv-actions">
            <button class="adv-reset" id="advReset" type="button">Reset</button>
            <button class="adv-use-rec" id="advUseRec" type="button" title="Apply recommended hyperparameters">💡 Use recommended</button>
          </div>
          <div class="train-adv-grid">
            <div id="advInkPolarityRow" class="train-adv-row" style="display:none">
              <span>Ink Polarity <span class="tm-help" data-tooltip="Global override applied to every class at train time. Pick the one that matches your physical camera setup — dark ink on a bright page, or bright ink/EMNIST on a dark background.">?</span></span>
              <select id="advInkPolarity">
                <option value="0">Dark ink on bright bg</option>
                <option value="1">Bright ink on dark bg</option>
              </select>
            </div>
            <div id="advReadingOrderRow" class="train-adv-row" style="display:none">
              <span>Sign Reading Direction <span class="tm-help" data-tooltip="Which way the characters on a license plate are read. Left to Right for plates like &quot;L 50&quot;; Right to Left for plates whose glyphs run the other way, such as Arabic or Hebrew signage.">?</span></span>
              <select id="advReadingOrder">
                <option value="ltr">Left to Right (L ➔ R)</option>
                <option value="rtl">Right to Left (R ➔ L)</option>
              </select>
            </div>
            <div class="train-adv-row"><span>Image Size <span class="tm-help" data-tooltip="Width &amp; height of the square input the model is trained on; larger sizes capture more detail but cost memory and latency on device.">?</span></span><input id="advImgSizeInline" type="number" min="8" max="512" step="4"/></div>
            <div class="train-adv-row"><span>Batch Size <span class="tm-help" data-tooltip="How many samples the model sees per gradient step; bigger batches train more stably but use more RAM.">?</span></span><input id="advBatchInline" type="number" min="1" step="1"/><div class="adv-rec-hint" id="recBatchHint"></div></div>
            <div class="train-adv-row"><span>Epochs <span class="tm-help" data-tooltip="Number of full passes over the training data; more epochs learn better but can overfit if too high.">?</span></span><input id="advEpochsInline" type="number" min="1" step="1"/><div class="adv-rec-hint" id="recEpochsHint"></div></div>
            <div class="train-adv-row"><span>Validation Split <span class="tm-help" data-tooltip="Fraction of samples held out to measure accuracy; a larger split gives a more honest score but leaves less data to train on.">?</span></span><input id="advValInline" type="number" min="0" max="0.5" step="0.05"/><div class="adv-rec-hint" id="recValHint"></div></div>
            <div class="train-adv-row"><span>Learning Rate <span class="tm-help" data-tooltip="Step size for each weight update; too large causes instability, too small slows or stalls learning.">?</span></span><input id="advLrInline" type="number" min="0" step="0.0001"/><div class="adv-rec-hint" id="recLrHint"></div></div>
            <div id="advArchRows" style="display:contents">
              <div class="train-adv-row"><span>Conv1 Filters <span class="tm-help" data-tooltip="Number of feature detectors in the first convolution layer; more filters learn richer features but increase model size.">?</span></span><input id="advConv1Inline" type="number" min="1" step="1"/></div>
              <div class="train-adv-row"><span>Conv2 Filters <span class="tm-help" data-tooltip="Number of feature detectors in the second convolution layer; more filters increase capacity and flash footprint.">?</span></span><input id="advConv2Inline" type="number" min="1" step="1"/></div>
              <div class="train-adv-row"><span>Dense Units <span class="tm-help" data-tooltip="Width of the final fully-connected layer; larger heads that map features to class logits.">?</span></span><input id="advDenseInline" type="number" min="1" step="1"/></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="col-preview">
      <div class="card preview-card" id="previewCard">
        <div class="card-head">
          <h3><span class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg></span> Preview</h3>
          <button class="exportbtn" id="exportBtn"><svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg><span>Export Model</span></button>
        </div>
        <div class="preview-controls">
          <label class="preview-toggle"><input id="previewInputToggle" type="checkbox"/><span>Input</span></label>
          <label class="preview-toggle"><input id="previewRoiToggle" type="checkbox"/><span>ROI</span></label>
          <label class="preview-toggle"><input id="previewRawToggle" type="checkbox"/><span>Orig</span></label>
        </div>
        <div id="previewContextBar"></div>
        <div id="previewSettingsHost"></div>
        <div class="preview-pane" id="previewPane"></div>
        <div class="preview-thresh-bar" id="previewThreshBar">
          <label title="Min G brightness for sign (0-255). Darker pixels → shadow → white.">Dark <input id="previewDarkSlider" type="range" min="0" max="100" value="0" step="1"/><input id="previewDarkNum" type="number" min="0" max="255" value="0"/></label>
          <label title="Max G brightness for sign (50-255). Brighter pixels → background → white.">Lum <input id="previewLumSlider" type="range" min="50" max="255" value="100" step="1"/><input id="previewLumNum" type="number" min="50" max="255" value="100"/></label>
        </div>
        <div class="preview-output" id="previewOutput"></div>
        <div class="note" id="previewNote"></div>
      </div>
    </div>
  </div>
</div>
<div class="footer">English | release-2-4-14 - 2.4.14</div>
<!-- Interactive coach-mark tour: spotlight ring + floating lesson card. -->
<div id="tmTourSpotlight" aria-hidden="true"></div>
<div id="tmTourCard" role="dialog" aria-label="Interactive tutorial"></div>
<!-- Directional arrow between the card and the ring, and a pulsing click
     beacon on the ring's edge: together they say WHERE the card is
     pointing, not only what it says.  Both are hidden for steps with no
     target and whenever the tour is closed. -->
<div id="tmTourArrow" aria-hidden="true">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21V5"/><path d="M5 12l7-7 7 7"/></svg>
</div>
<div id="tmTourBeacon" aria-hidden="true">
  <span class="tm-tour-beacon-ring"></span>
  <span class="tm-tour-beacon-hand">👆</span>
</div>
<!-- Phase B of every step.  The card explains the step and points at the
     control; the moment the user starts working, the dim layer and the
     card step aside and this pill takes over, carrying the live state of
     whatever they are doing.  The whole pill advances the tour; `?` brings
     the instruction card back.  It is filled in by the tour engine, so an
     unstarted tour leaves nothing behind. -->
<div id="tmTourOrb" role="status" aria-live="polite" title="Click to move to the next step · drag to move me">
  <span class="tm-tour-orb-dot"></span>
  <span class="tm-tour-orb-text" id="tmTourOrbText"></span>
  <button type="button" class="tm-tour-orb-go" id="tmTourOrbGo">Next Step &rarr;</button>
  <button type="button" class="tm-tour-orb-help" id="tmTourOrbHelp" title="Re-open the instruction card">?</button>
</div>

<script>
(function() {{
  function setStageLabel(text) {{
    try {{
      var id = 'tmBootStageBadge';
      var box = document.getElementById(id);
      if (!box) {{
        box = document.createElement('div');
        box.id = id;
        box.style.position = 'fixed';
        box.style.right = '16px';
        box.style.top = '16px';
        box.style.zIndex = '99998';
        box.style.display = 'none';
        box.style.padding = '8px 10px';
        box.style.borderRadius = '999px';
        box.style.background = 'rgba(32,33,36,0.88)';
          box.style.color = themeVar('--text-inverse', '#fff');
        box.style.font = '11px/1.2 system-ui, -apple-system, Segoe UI, sans-serif';
        box.style.whiteSpace = 'pre-wrap';
        document.body.appendChild(box);
      }}
      box.textContent = 'stage: ' + String(text || 'unknown');
    }} catch (e) {{}}
  }}
  function showBootError(message) {{
    try {{
      var id = 'tmBootErrorBanner';
      var box = document.getElementById(id);
      if (!box) {{
        box = document.createElement('div');
        box.id = id;
        box.style.position = 'fixed';
        box.style.left = '16px';
        box.style.right = '16px';
        box.style.bottom = '16px';
        box.style.zIndex = '99999';
        box.style.padding = '12px 14px';
        box.style.borderRadius = '10px';
          box.style.background = themeVar('--danger-soft', '#fff1f3');
          box.style.border = `1px solid ${{themeVar('--danger', '#f28b82')}}`;
          box.style.color = themeVar('--text', '#202124');
        box.style.font = '12px/1.45 system-ui, -apple-system, Segoe UI, sans-serif';
        box.style.whiteSpace = 'pre-wrap';
        document.body.appendChild(box);
      }}
      box.textContent = 'Frontend boot error: ' + String(message || 'unknown error');
    }} catch (e) {{}}
  }}
  window.__tmStageMark = function(stage) {{
    setStageLabel(stage);
  }};
  setStageLabel('boot-probe-ready');
  window.addEventListener('error', function(event) {{
    showBootError(event && event.message ? event.message : event);
  }});
  window.addEventListener('unhandledrejection', function(event) {{
    var reason = event && event.reason ? event.reason : 'Unhandled promise rejection';
    if (reason && reason.message) reason = reason.message;
    showBootError(reason);
  }});
}})();
</script>

<script>
const STATE = {data};
STATE.trainClassAccuracies = null;
STATE.currentProjectPath = STATE.current_project_path || null;
const baseUrl = `http://127.0.0.1:${{STATE.port}}`;
if (window.__tmStageMark) window.__tmStageMark('script-start');
function dbgEvent(hypothesisId, location, msg, data) {{
  const url = String(STATE.debug_server_url || '').trim();
  if (!url) return;
  try {{
    fetch(url, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        sessionId: String(STATE.debug_session_id || 'open-project-page-stuck'),
        runId: 'pre-fix',
        hypothesisId,
        location,
        msg,
        data,
        ts: Date.now()
      }})
    }}).catch(() => null);
  }} catch (e) {{}}
}}
function logScrollLayers(tag) {{
  try {{
    const frame = window.frameElement || null;
    const frameParent = frame && frame.parentElement ? frame.parentElement : null;
    const frameGrand = frameParent && frameParent.parentElement ? frameParent.parentElement : null;
    let parentIframeCount = -1;
    try {{
      parentIframeCount = window.parent && window.parent.document ? window.parent.document.querySelectorAll('iframe').length : -1;
    }} catch (e) {{}}
    dbgEvent('H6', 'app.py:logScrollLayers', '[DEBUG] scroll layer snapshot', {{
      tag: String(tag || ''),
      parentIframeCount,
      docEl: {{
        clientWidth: document.documentElement ? document.documentElement.clientWidth : 0,
        clientHeight: document.documentElement ? document.documentElement.clientHeight : 0,
        scrollWidth: document.documentElement ? document.documentElement.scrollWidth : 0,
        scrollHeight: document.documentElement ? document.documentElement.scrollHeight : 0,
        overflow: document.documentElement && document.documentElement.style ? String(document.documentElement.style.overflow || '') : '',
        overflowX: document.documentElement && document.documentElement.style ? String(document.documentElement.style.overflowX || '') : '',
        overflowY: document.documentElement && document.documentElement.style ? String(document.documentElement.style.overflowY || '') : '',
      }},
      body: {{
        clientWidth: document.body ? document.body.clientWidth : 0,
        clientHeight: document.body ? document.body.clientHeight : 0,
        scrollWidth: document.body ? document.body.scrollWidth : 0,
        scrollHeight: document.body ? document.body.scrollHeight : 0,
        overflow: document.body && document.body.style ? String(document.body.style.overflow || '') : '',
        overflowX: document.body && document.body.style ? String(document.body.style.overflowX || '') : '',
        overflowY: document.body && document.body.style ? String(document.body.style.overflowY || '') : '',
      }},
      frame: frame ? {{
        clientWidth: frame.clientWidth || 0,
        clientHeight: frame.clientHeight || 0,
        scrollWidth: frame.scrollWidth || 0,
        scrollHeight: frame.scrollHeight || 0,
        overflow: frame.style ? String(frame.style.overflow || '') : '',
        overflowX: frame.style ? String(frame.style.overflowX || '') : '',
        overflowY: frame.style ? String(frame.style.overflowY || '') : '',
      }} : null,
      frameParent: frameParent ? {{
        clientWidth: frameParent.clientWidth || 0,
        clientHeight: frameParent.clientHeight || 0,
        scrollWidth: frameParent.scrollWidth || 0,
        scrollHeight: frameParent.scrollHeight || 0,
        overflow: frameParent.style ? String(frameParent.style.overflow || '') : '',
        overflowX: frameParent.style ? String(frameParent.style.overflowX || '') : '',
        overflowY: frameParent.style ? String(frameParent.style.overflowY || '') : '',
      }} : null,
      frameGrand: frameGrand ? {{
        clientWidth: frameGrand.clientWidth || 0,
        clientHeight: frameGrand.clientHeight || 0,
        scrollWidth: frameGrand.scrollWidth || 0,
        scrollHeight: frameGrand.scrollHeight || 0,
        overflow: frameGrand.style ? String(frameGrand.style.overflow || '') : '',
        overflowX: frameGrand.style ? String(frameGrand.style.overflowX || '') : '',
        overflowY: frameGrand.style ? String(frameGrand.style.overflowY || '') : '',
      }} : null,
    }});
  }} catch (e) {{}}
}}
function sendStreamlitMessage(type, data) {{
  if (window.__tmNavigatingAway) return;
  try {{
    window.parent.postMessage(Object.assign({{
      isStreamlitMessage: true,
      type: type,
    }}, data || {{}}), '*');
  }} catch (e) {{}}
}}
function initStreamlitFrame() {{
  if (window.__tmNavigatingAway) return;
  if (window.__tmStreamlitReady) return;
  window.__tmStreamlitReady = true;
  sendStreamlitMessage('streamlit:componentReady', {{apiVersion: 1}});
}}
let frameHeightRaf = 0;
let layoutResyncRaf = 0;
let layoutResyncTimers = [];
let mountReflowTimers = [];
let homeNavigateTimer = 0;
let _lastFrameHeight = 0;
let _lastFrameHeightAt = 0;
function syncFrameHeight() {{
  if (window.__tmNavigatingAway) return;
  initStreamlitFrame();
  // Save outer scroll positions BEFORE touching any heights.
  // The previous cleanse step set iframe.height=0 which collapsed the whole
  // streamlit wrapper, forcing the browser to snap scrollTop back to 0 and
  // making the scrollbar "jump to top" every time we synced.
  let savedScrollTop = 0;
  let savedOuterScrollTop = 0;
  let oldOuterScrollH = 0;
  try {{
    const main = document.querySelector('section.stMain, main');
    if (main) savedOuterScrollTop = main.scrollTop || 0;
    const scrollEl = document.scrollingElement || document.documentElement;
    if (scrollEl) {{
      savedScrollTop = scrollEl.scrollTop || 0;
      oldOuterScrollH = scrollEl.scrollHeight || 0;
    }}
  }} catch (e) {{}}
  // --- Pre-cleanse: unlock flex locks (don't touch heights to avoid jump) -
  // The real lock is flex: 0 0 <old-px> on stElementContainer. Setting
  // iframe.height to 0 here caused the scroll-jump bug, so we only unlock
  // flex/flexBasis/overflow/maxHeight now.
  try {{
    const frame = window.frameElement;
    if (frame && frame.style) {{
      // Do NOT set frame.style.height = '0px' here — it causes scroll reset.
      frame.style.minHeight = '0px';
      frame.style.maxHeight = 'none';
      frame.style.flex = '0 1 auto';
      frame.style.flexBasis = 'auto';
      frame.style.display = 'block';
      if (frame.parentElement && frame.parentElement.style) {{
        const el1 = frame.parentElement;
        el1.style.height = 'auto';
        el1.style.minHeight = '0px';
        el1.style.maxHeight = 'none';
        el1.style.overflow = 'visible';
        el1.style.overflowY = 'visible';
        el1.style.flex = '0 1 auto';
        el1.style.flexBasis = 'auto';
        el1.style.flexGrow = '0';
        el1.style.flexShrink = '1';
        const el2 = el1.parentElement;
        if (el2 && el2.style) {{
          el2.style.height = 'auto';
          el2.style.minHeight = '0px';
          el2.style.maxHeight = 'none';
          el2.style.overflow = 'visible';
          el2.style.overflowY = 'visible';
          el2.style.flex = '1 1 auto';
          el2.style.flexBasis = 'auto';
          const el3 = el2.parentElement;
          if (el3 && el3.style && !String(el3.className || '').includes('stMain')) {{
            el3.style.height = 'auto';
            el3.style.minHeight = '0px';
            el3.style.maxHeight = 'none';
            el3.style.overflow = 'visible';
            el3.style.overflowY = 'visible';
            el3.style.flex = '0 1 auto';
            el3.style.flexBasis = 'auto';
          }}
        }}
      }}
    }}
    const wrap = document.querySelector('.wrap');
    if (wrap && wrap.style) {{
      wrap.style.overflow = 'visible';
      wrap.style.overflowY = 'visible';
      wrap.style.overflowX = 'visible';
    }}
  }} catch (e) {{}}
  // -----------------------------------------------------------------------
  let nextHeight = 0;
  let nextWidth = 0;
  let metrics = {{}};
  try {{
    const body = document.body;
    const doc = document.documentElement;
    const wrap = document.querySelector('.wrap');
    const layout = document.querySelector('.layout');
    const WRAP_PAD_TOP = 72;
    const WRAP_PAD_BOT = 92;
    const WRAP_MIN = 620;
    let naturalH = 0;
    if (layout) {{
      const layoutH = Math.max(
        Number(layout.scrollHeight) || 0,
        Number(layout.offsetHeight) || 0,
        Number(layout.getBoundingClientRect?.().height) || 0,
      );
      naturalH = Math.max(layoutH + WRAP_PAD_TOP + WRAP_PAD_BOT, WRAP_MIN);
    }}
    const legacyH = Math.max(
      wrap ? Math.ceil(wrap.scrollHeight) : 0,
      wrap ? Math.ceil(wrap.offsetHeight) : 0,
      body ? Math.ceil(body.scrollHeight) : 0,
      body ? Math.ceil(body.offsetHeight) : 0,
      doc ? Math.ceil(doc.scrollHeight) : 0,
      0,
    );
    if (naturalH > 0) {{
      nextHeight = naturalH;
      if (legacyH > naturalH && (legacyH - naturalH) <= 24) {{
        nextHeight = legacyH;
      }}
    }} else {{
      nextHeight = legacyH || WRAP_MIN;
    }}
    nextWidth = Math.max(
      Number(doc && doc.clientWidth) || 0,
      Number(body && body.clientWidth) || 0,
      wrap ? Math.round(wrap.getBoundingClientRect().width) : 0,
      0,
    );
    metrics = {{
      nextHeight,
      nextWidth,
      naturalH,
      legacyH,
      wrapScroll: wrap ? Math.ceil(wrap.scrollHeight) : 0,
      wrapOffset: wrap ? Math.ceil(wrap.offsetHeight) : 0,
      bodyScroll: body ? Math.ceil(body.scrollHeight) : 0,
      bodyOffset: body ? Math.ceil(body.offsetHeight) : 0,
      docScroll: doc ? Math.ceil(doc.scrollHeight) : 0,
      wrapWidth: wrap ? Math.round(wrap.getBoundingClientRect().width) : 0,
      lastSent: _lastFrameHeight,
      windowInnerHeight: window.innerHeight || 0,
      windowInnerWidth: window.innerWidth || 0,
      classCount: Array.isArray(STATE.classes) ? STATE.classes.length : 0,
    }};
  }} catch (e) {{}}
  if (!nextHeight || !Number.isFinite(nextHeight)) return;
  const now = Date.now();
  const clampedH = Math.round(nextHeight);
  let sendH = clampedH;
  // Never let the workspace iframe be shorter than the real browser
  // viewport, otherwise the dim tutorial backdrop (position: fixed) and
  // the page background stop short and expose the grey Streamlit shell
  // below the fold. window.parent is the host document; fall back to our
  // own window.innerHeight inside unusual embedding contexts.
  try {{
    const parentH = (window.parent && window.parent !== window && window.parent.innerHeight)
      ? window.parent.innerHeight
      : (window.innerHeight || 0);
    if (parentH > 0) sendH = Math.max(sendH, parentH);
  }} catch (e) {{}}
  if (_lastFrameHeight > 0) {{
    const diff = sendH - _lastFrameHeight;
    if (diff > 0 && diff < 4) return; // ignore tiny growth jitter
  }}
  try {{
    const frame = window.frameElement;
    if (frame && frame.style) {{
      frame.style.height = `${{sendH}}px`;
      frame.style.minHeight = '0px';
      frame.style.maxHeight = 'none';
      frame.style.flex = '0 1 auto';
      frame.style.flexBasis = 'auto';
      frame.style.width = '100%';
      frame.style.maxWidth = '100%';
      if (frame.parentElement && frame.parentElement.style) {{
        const el1 = frame.parentElement;
        el1.style.width = '100%';
        el1.style.maxWidth = '100%';
        el1.style.height = `${{sendH}}px`;
        el1.style.minHeight = '0px';
        el1.style.maxHeight = 'none';
        el1.style.overflow = 'visible';
        el1.style.overflowY = 'visible';
        el1.style.flex = '0 1 auto';
        el1.style.flexBasis = 'auto';
        el1.style.flexGrow = '0';
        el1.style.flexShrink = '1';
        const el2 = el1.parentElement;
        if (el2 && el2.style) {{
          el2.style.height = 'auto';
          el2.style.minHeight = '0px';
          el2.style.maxHeight = 'none';
          el2.style.overflow = 'visible';
          el2.style.overflowY = 'visible';
          el2.style.flex = '1 1 auto';
          el2.style.flexBasis = 'auto';
          const el3 = el2.parentElement;
          if (el3 && el3.style && !String(el3.className || '').includes('stMain')) {{
            el3.style.height = 'auto';
            el3.style.minHeight = '0px';
            el3.style.maxHeight = 'none';
            el3.style.overflow = 'visible';
            el3.style.overflowY = 'visible';
            el3.style.flex = '0 1 auto';
            el3.style.flexBasis = 'auto';
          }}
        }}
      }}
    }}
    if (nextWidth && Number.isFinite(nextWidth) && document.body && document.body.style) {{
      document.body.style.width = '100%';
      document.body.style.maxWidth = '100%';
    }}
  }} catch (e) {{}}
  // --- Restore outer scroll positions after the resize --------------------
  // Because we changed iframe/container size, the browser may have clamped
  // scrollTop. Restore what the user had, clamped to the new max.
  try {{
    const main = document.querySelector('section.stMain, main');
    if (main) {{
      const maxMain = Math.max(0, (main.scrollHeight || 0) - (main.clientHeight || 0));
      if (savedOuterScrollTop > 0 && main.scrollTop !== savedOuterScrollTop) {{
        main.scrollTop = Math.min(savedOuterScrollTop, Math.max(0, maxMain));
      }}
    }}
    const scrollEl = document.scrollingElement || document.documentElement;
    if (scrollEl) {{
      const newScrollH = scrollEl.scrollHeight || 0;
      // If the page got shorter, scale the scroll position proportionally so
      // the user stays at roughly the same relative spot instead of snapping
      // to an invalid (too-large) saved position.
      let wantTop = savedScrollTop;
      if (oldOuterScrollH > 0 && newScrollH > 0 && newScrollH !== oldOuterScrollH) {{
        const rel = savedScrollTop / Math.max(1, oldOuterScrollH);
        wantTop = Math.round(rel * newScrollH);
      }}
      const maxTop = Math.max(0, newScrollH - (scrollEl.clientHeight || 0));
      if (savedScrollTop > 0 || (scrollEl.scrollTop > 0 && scrollEl.scrollTop !== wantTop)) {{
        scrollEl.scrollTop = Math.min(wantTop, Math.max(0, maxTop));
      }}
    }}
    // Notify Streamlit AFTER we've restored scroll so it doesn't override.
    sendStreamlitMessage('streamlit:setFrameHeight', {{height: sendH}});
  }} catch (e) {{}}
  // #region debug-point A:sync-frame-height
  dbgEvent('A', 'app.py:syncFrameHeight', '[DEBUG] syncFrameHeight posting iframe height', metrics);
  // #endregion
  _lastFrameHeight = sendH;
  _lastFrameHeightAt = now;
}}
function queueFrameHeightSync() {{
  if (window.__tmNavigatingAway) return;
  if (frameHeightRaf) {{
    try {{ cancelAnimationFrame(frameHeightRaf); }} catch (e) {{}}
  }}
  frameHeightRaf = requestAnimationFrame(() => {{
    frameHeightRaf = 0;
    syncFrameHeight();
  }});
}}
function scheduleLayoutResync() {{
  if (window.__tmNavigatingAway) return;
  // #region debug-point B:schedule-layout-resync
  dbgEvent('B', 'app.py:scheduleLayoutResync', '[DEBUG] scheduleLayoutResync queued', {{
    classCount: Array.isArray(STATE.classes) ? STATE.classes.length : 0,
    sampleCounts: Object.assign({{}}, STATE.counts || {{}}),
    previewReady: !!document.getElementById('previewImage'),
    windowInnerHeight: window.innerHeight || 0,
  }});
  // #endregion
  if (layoutResyncRaf) {{
    try {{ cancelAnimationFrame(layoutResyncRaf); }} catch (e) {{}}
    layoutResyncRaf = 0;
  }}
  layoutResyncTimers.forEach((t) => {{
    try {{ clearTimeout(t); }} catch (e) {{}}
  }});
  layoutResyncTimers = [];
  queueFrameHeightSync();
  try {{
    layoutResyncRaf = window.requestAnimationFrame(() => {{
      layoutResyncRaf = 0;
      if (window.__tmNavigatingAway) return;
      updateFlow();
      queueFrameHeightSync();
      const t120 = window.setTimeout(() => {{
        if (window.__tmNavigatingAway) return;
        updateFlow();
        queueFrameHeightSync();
      }}, 120);
      const t320 = window.setTimeout(() => {{
        if (window.__tmNavigatingAway) return;
        updateFlow();
        queueFrameHeightSync();
      }}, 320);
      layoutResyncTimers.push(t120, t320);
    }});
  }} catch (e) {{}}
}}
function bindLayoutImageObservers(scope) {{
  const root = scope || document;
  if (!root || !root.querySelectorAll) return;
  root.querySelectorAll('#previewImage, #classPreprocessImage').forEach((img) => {{
    if (!img || img.dataset.layoutBound === '1') return;
    img.dataset.layoutBound = '1';
    const onDone = () => scheduleLayoutResync();
    img.addEventListener('load', onDone, {{once: true}});
    img.addEventListener('error', onDone, {{once: true}});
  }});
}}
function requestShellLayoutRefresh(reason) {{
  if (window.__tmNavigatingAway) return;
  // #region debug-point C:request-shell-layout-refresh
  dbgEvent('C', 'app.py:requestShellLayoutRefresh', '[DEBUG] requestShellLayoutRefresh called', {{
    reason: String(reason || ''),
    hasPywebviewWindow: !!(window && window.pywebview && window.pywebview.api),
    hasPywebviewParent: !!(window.parent && window.parent.pywebview && window.parent.pywebview.api),
    hasPywebviewTop: !!(window.top && window.top.pywebview && window.top.pywebview.api),
    innerWidth: window.innerWidth || 0,
    innerHeight: window.innerHeight || 0,
  }});
  // #endregion
  try {{
    const candidates = [window, window.parent, window.top];
    for (const target of candidates) {{
      if (target && target.pywebview && target.pywebview.api && typeof target.pywebview.api.request_reflow === 'function') {{
        target.pywebview.api.request_reflow(String(reason || 'ui-refresh'));
        return;
      }}
    }}
  }} catch (e) {{}}
}}
let openSourceClass = STATE.initial_open_source_class || '';
let openSourceKind = STATE.initial_open_source_kind || '';
// #region debug-point B:initial-open-source
dbgEvent('B', 'app.py:initialOpenSource', '[DEBUG] initial open source from payload', {{
  session: String(STATE.session || ''),
  openSourceClass,
  openSourceKind,
}});
// #endregion
// Class-panel source live loop: updates sourcePreview-${{className}} <img> from /live/frame.
let sourceLiveTimer = null;
// Class-panel source preview uses its own blob URL so it does NOT get
// revoked when the Preview-card visual loop updates `previewBlobUrl`.
// Previously the two loops shared one `previewBlobUrl` and each side
// called `clearPreviewBlob()` → `URL.revokeObjectURL(previewBlobUrl)` for
// the other loop's active <img src>, so pressing Hold froze the class
// panel camera view on every single frame refresh.
let classSourceBlobUrl = '';
let classSourceRequestInFlight = false;
let classSourceAbortController = null;
let classSourceInFlightAt = 0;
// Preview-card visual-only loop: updates the Preview pane's <img> WITHOUT running
// inference — used so the preview stays "live" even when we pause the prediction
// timer during hold-to-capture to save CPU.
let previewVisualTimer = null;
let previewVisualToken = 0;
let previewVisualInFlight = false;
// Preview prediction loop: runs /preview/predict inference, updates bars & roi/crop toggles.
let previewPredictTimer = null;
let previewBlobUrl = '';
let previewRequestInFlight = false;
let previewFrameController = null;
let captureInFlight = false;
let holdRecording = false;
let holdRecordClass = '';
let holdRecordSource = '';
let holdResumePreviewPredict = false;
let holdResumePreviewVisual = false;
let holdPreviewSourceBeforeCapture = 'webcam';
let holdSyncTimer = null;
let holdPollController = null;
let holdSeq = 0;
let holdNextToken = 0;
let PREF_MAX_SAMPLES_DISPLAY = 12;
// Developer/demo shortcut state (Settings -> Advanced).  Declared up here,
// with the other boot-restored flags, because the top-level
// restoreAutoSeedEmnist() call runs long before the helper block below.
const AUTO_SEED_EMNIST_STORAGE_KEY = 'auto_seed_emnist_samples';
let autoSeedEmnistSamples = false;
let autoSeedEmnistInFlight = false;
// Classes this project has already been offered to the auto-seeder.  Without
// it, "Clear Samples" on an auto-filled class would silently repopulate, and
// a failed request would retry on every render.
let autoSeedEmnistAttempted = new Set();
const CLASS_PREPROCESS_WINDOW = 15;
const UPLOAD_CONCURRENCY = 3;
const UPLOAD_MAX_MB = 25;
let sourceSwitchInFlight = false;
let sourceSwitchClass = '';
let sourceSwitchKind = '';
let trainInFlight = false;
let trainPollToken = 0;
let sourceSettingsOpen = false;
let previewIntervalMs = 80;
let currentSerialPort = STATE.current_serial_port || '';
let currentWebcamIndex = Number(STATE.current_webcam_index || 0);
let currentWebcamUid = String(STATE.current_webcam_id || '');
let currentSerialBaud = Number(STATE.current_serial_baud || 115200);
let currentSerialSync = String(STATE.current_serial_sync || 'AA 55 AA');
let currentSerialFrameSide = Number(STATE.current_serial_frame_side || 96);
let currentSerialChannels = Number(STATE.current_serial_channels || 1);
let directTensorStream = false;  // device streams the preprocessed tensor (mode gray)
let previewInputOn = false;
let previewSource = 'webcam';
let previewPreprocessMode = 'auto_by_label';
let previewShowRoi = false;
let previewShowRaw = false;  // toggle to show original (unprocessed) image
let previewShowScore = false;  // toggle: percentage (%) vs raw score (0.000-1.000)
let previewDarkThresh = 0;
let previewLumThresh = 100;
let previewThreshUserModified = false;  // true only after user moves a slider
let previewOodSignPctMin = 0.3;
let previewOodSignPctMax = 70.0;
let previewOodMaxProbMin = 0.40;
let previewOodEntropyRatioMax = 0.85;
let previewUploadImageSrc = '';
let previewUploadImageB64 = '';
let previewUploadFilename = '';
let previewPredictInFlight = false;
let previewSynthInFlight = false;
let previewSynthPlateIndex = 0;
let previewPredictController = null;
let previewPredictToken = 0;
let previewLastPredictData = null;
let previewLastStaticImageSrc = '';
let previewSettingsOpen = false;
let classPreprocessOpen = false;
let classPreprocessClass = '';
let classPreprocessDraft = null;
let classPreprocessSampleIndex = 0;
let classPreprocessSamples = [];
let classPreprocessProcessedSrc = '';
let classPreprocessBusy = false;
let classPreprocessScrollTop = 0;
let classPreprocessDirty = false;
let classPreprocessAutoCrop = null;  // auto-detected crop box (normalized 0-1)
let classPreprocessPreviewController = null;
let classPreprocessPreviewToken = 0;
let classPreprocessThresholdTimer = null;
let navMenuOpen = false;
let navMenuBound = false;
let resizeBound = false;
STATE.counts = STATE.counts || {{}};
STATE.sample_previews = STATE.sample_previews || {{}};
STATE.class_preprocess = STATE.class_preprocess || {{}};
STATE.sample_preprocess = STATE.sample_preprocess || {{}};
STATE.processed_previews = STATE.processed_previews || {{}};
const openSourceStorageKey = `tm-open-source-${{STATE.session}}`;
const trainCfgStorageKey = `tm-train-cfg-${{STATE.session}}`;
const previewStorageKey = `tm-preview-${{STATE.session}}`;
// v2: OOD MAX PROB became the single probability threshold (0.40 default,
// was 0.60) and ENTROPY MAX default became 0.85 (was 0.70).  Cached v1
// values are migrated once in restorePreviewState().
const PREVIEW_STORAGE_VERSION = 2;
const exportDirStorageKey = `tm-export-dir-${{STATE.session}}`;
const exportNameStorageKey = `tm-export-name-${{STATE.session}}`;
const exportArrayStorageKey = `tm-export-array-${{STATE.session}}`;
const preferencesStorageKey = `tm-preferences`;
let exportDir = '';
let exportModelName = 'tm';
let exportArrayName = '';
function readOpenSourceStorage() {{
  try {{
    const raw = window.localStorage.getItem(openSourceStorageKey);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data || !data.className || !data.kind) return null;
    return {{
      className: String(data.className || ''),
      kind: String(data.kind || '')
    }};
  }} catch (e) {{
    return null;
  }}
}}
function persistOpenSourceState() {{
  try {{
    if (!openSourceClass || !openSourceKind) {{
      window.localStorage.removeItem(openSourceStorageKey);
      return;
    }}
    window.localStorage.setItem(openSourceStorageKey, JSON.stringify({{
      className: openSourceClass,
      kind: openSourceKind
    }}));
  }} catch (e) {{}}
}}
function clearOpenSourceState() {{
  try {{
    window.localStorage.removeItem(openSourceStorageKey);
  }} catch (e) {{}}
}}
if ((!openSourceClass || !openSourceKind) && typeof window !== 'undefined' && window.localStorage) {{
  const restoredOpenSource = readOpenSourceStorage();
  if (restoredOpenSource) {{
    openSourceClass = restoredOpenSource.className;
    openSourceKind = restoredOpenSource.kind;
    // #region debug-point B:restore-open-source-storage
    dbgEvent('B', 'app.py:restoreOpenSourceStorage', '[DEBUG] restored open source from localStorage', {{
      session: String(STATE.session || ''),
      openSourceClass,
      openSourceKind,
    }});
    // #endregion
  }}
}}
if (window.__tmStageMark) window.__tmStageMark('open-source-restored');
function restoreTrainCfgFromStorage() {{
  try {{
    const raw = window.localStorage.getItem(trainCfgStorageKey);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (!data || typeof data !== 'object') return;
    const prev = Object.assign({{}}, STATE.train_cfg || {{}});
    const next = Object.assign({{}}, prev);
    const clampInt = (v, lo, hi, fallback) => {{
      const n = parseInt(String(v), 10);
      if (!isFinite(n)) return fallback;
      return Math.max(lo, Math.min(hi, n));
    }};
    const clampFloat = (v, lo, hi, fallback) => {{
      const n = Number(v);
      if (!isFinite(n)) return fallback;
      return Math.max(lo, Math.min(hi, n));
    }};
    // Lower bound is 8, not 32: the OCR project's glyph model input is 28x28
    // and the old 32 floor silently coerced it on every client-config merge.
    // (The server-side /live/config guard has always been 8..512.)
    if ('img_size' in data) next.img_size = clampInt(data.img_size, 8, 512, nullish(prev.img_size, 96));
    if ('batch_size' in data) next.batch_size = clampInt(data.batch_size, 1, 512, nullish(prev.batch_size, 32));
    if ('epochs' in data) next.epochs = clampInt(data.epochs, 1, 1000, nullish(prev.epochs, 20));
    if ('validation_split' in data) next.validation_split = clampFloat(data.validation_split, 0, 0.5, nullish(prev.validation_split, 0.25));
    if ('learning_rate' in data) next.learning_rate = clampFloat(data.learning_rate, 0.000001, 1.0, nullish(prev.learning_rate, 0.0016));
    if ('conv1_filters' in data) next.conv1_filters = clampInt(data.conv1_filters, 1, 256, nullish(prev.conv1_filters, 8));
    if ('conv2_filters' in data) next.conv2_filters = clampInt(data.conv2_filters, 1, 512, nullish(prev.conv2_filters, 16));
    if ('dense_units' in data) next.dense_units = clampInt(data.dense_units, 1, 2048, nullish(prev.dense_units, 32));
    if (data.preprocess_mode === 'auto_by_label' || data.preprocess_mode === 'manual_roi') {{
      next.preprocess_mode = String(data.preprocess_mode);
    }}
    if (Array.isArray(data.manual_roi) && data.manual_roi.length === 4) {{
      next.manual_roi = data.manual_roi.map((v, idx) => clampFloat(v, 0, 1, idx < 2 ? 0 : 1));
    }}
    // OCR mixing / transfer learning.  localStorage is user-writable, so
    // everything is range-checked here rather than trusted — these values go
    // straight into a training run.
    if ('ocr_emnist_enabled' in data) next.ocr_emnist_enabled = !!data.ocr_emnist_enabled;
    if (typeof data.ocr_emnist_split === 'string' && data.ocr_emnist_split) {{
      next.ocr_emnist_split = String(data.ocr_emnist_split);
    }}
    if ('ocr_emnist_per_class' in data) next.ocr_emnist_per_class = clampInt(data.ocr_emnist_per_class, 0, 5000, nullish(prev.ocr_emnist_per_class, 200));
    if ('ocr_background_enabled' in data) next.ocr_background_enabled = !!data.ocr_background_enabled;
    if ('sign_background_replace_ood' in data) next.sign_background_replace_ood = !!data.sign_background_replace_ood;
    if ('ocr_background_samples' in data) next.ocr_background_samples = clampInt(data.ocr_background_samples, 0, 5000, nullish(prev.ocr_background_samples, 0));
    if ('ocr_transfer_learning' in data) next.ocr_transfer_learning = !!data.ocr_transfer_learning;
    if ('ocr_pretrain_per_class' in data) next.ocr_pretrain_per_class = clampInt(data.ocr_pretrain_per_class, 100, 6000, nullish(prev.ocr_pretrain_per_class, 1200));
    if ('ocr_pretrain_epochs' in data) next.ocr_pretrain_epochs = clampInt(data.ocr_pretrain_epochs, 1, 100, nullish(prev.ocr_pretrain_epochs, 4));
    if ('ocr_head_epochs' in data) next.ocr_head_epochs = clampInt(data.ocr_head_epochs, 1, 200, nullish(prev.ocr_head_epochs, 12));
    if ('ocr_head_lr' in data) next.ocr_head_lr = clampFloat(data.ocr_head_lr, 0.000001, 1.0, nullish(prev.ocr_head_lr, 0.0015));
    if ('ocr_finetune_epochs' in data) next.ocr_finetune_epochs = clampInt(data.ocr_finetune_epochs, 0, 200, nullish(prev.ocr_finetune_epochs, 25));
    if ('ocr_finetune_lr' in data) next.ocr_finetune_lr = clampFloat(data.ocr_finetune_lr, 0.000001, 1.0, nullish(prev.ocr_finetune_lr, 0.0002));
    if ('ocr_unfreeze_blocks' in data) next.ocr_unfreeze_blocks = clampInt(data.ocr_unfreeze_blocks, 0, 16, nullish(prev.ocr_unfreeze_blocks, 1));
    STATE.train_cfg = next;
  }} catch (e) {{}}
}}
function persistTrainCfgStorage() {{
  try {{
    window.localStorage.setItem(trainCfgStorageKey, JSON.stringify(STATE.train_cfg || {{}}));
  }} catch (e) {{}}
}}
// Mirrors ocr_preprocess.normalize_ink_polarity: only the bright-on-dark
// spellings reach BRIGHT (1); everything else — including absent — is the
// camera default, dark ink on a bright page (0).
function normalizeInkPolarity(value) {{
  if (typeof value === 'boolean') return value ? 1 : 0;
  if (typeof value === 'number') return isFinite(value) && Math.trunc(value) === 1 ? 1 : 0;
  const key = String(value == null ? '' : value).trim().toLowerCase();
  return ['bright', 'bright_on_dark', '1', 'light', 'inverted', 'emnist'].indexOf(key) >= 0 ? 1 : 0;
}}
// Mirrors ocr_preprocess.normalize_norm_convention.
function normalizeNormConvention(value) {{
  const key = String(value == null ? '' : value).trim().toLowerCase();
  return ['canon', 'canonical', 'mnist', 'emnist', 'centre', 'center'].indexOf(key) >= 0 ? 'canon' : 'stretch';
}}
function normalizeClassPreprocessConfig(raw) {{
  const src = raw && typeof raw === 'object' ? raw : {{}};
  const mode = ['auto_by_label', 'manual_roi'].includes(String(src.mode || ''))
    ? String(src.mode)
    : 'auto_by_label';
  let manual = null;
  if (Array.isArray(src.manual_roi) && src.manual_roi.length === 4) {{
    manual = src.manual_roi.map((v, idx) => {{
      const n = Number(v);
      if (!isFinite(n)) return idx < 2 ? 0 : 1;
      return Math.max(0, Math.min(1, n));
    }});
    manual[2] = Math.max(manual[2], manual[0] + 0.01);
    manual[3] = Math.max(manual[3], manual[1] + 0.01);
    manual[0] = Math.min(manual[0], manual[2] - 0.01);
    manual[1] = Math.min(manual[1], manual[3] - 0.01);
  }}
  let bg_lum = Number(src.bg_lum_thresh);
  if (!isFinite(bg_lum) || bg_lum < 50) bg_lum = 100; else if (bg_lum > 255) bg_lum = 255;
  let bg_dark = Number(src.bg_dark_thresh);
  if (!isFinite(bg_dark) || bg_dark < 0) bg_dark = 0; else if (bg_dark > 100) bg_dark = 100;
  const out = {{ mode, manual_roi: manual, bg_dark_thresh: bg_dark, bg_lum_thresh: bg_lum }};
  // Glyph (OCR) knobs, carried in this same per-class record rather than a
  // parallel OCR schema — see normalize_class_preprocess, which is the Python
  // half of this function.
  //
  // Guarded on presence, matching that half exactly: it re-emits a glyph key
  // only when it saw one, so an image project's record keeps its precise
  // shape.  This is not cosmetic.  /classes/save_config REPLACES the whole
  // class_preprocess map, so a save that dropped these keys would not "leave
  // the defaults alone" — it would silently reset every class's polarity,
  // segmentation and normalisation the first time anything else on the page
  // saved a class config.
  if ('ink_polarity' in src || 'glyph_segment' in src || 'glyph_convention' in src) {{
    out.ink_polarity = normalizeInkPolarity(src.ink_polarity);
    out.glyph_segment = src.glyph_segment == null ? true : !!src.glyph_segment;
    out.glyph_convention = normalizeNormConvention(src.glyph_convention);
  }}
  return out;
}}
function getClassPreprocessConfig(className) {{
  const all = STATE.class_preprocess && typeof STATE.class_preprocess === 'object' ? STATE.class_preprocess : {{}};
  return normalizeClassPreprocessConfig(all[className]);
}}
// Master ink-polarity sync: the single global Ink Polarity dropdown (in the
// Training card's Advanced panel) is the source of truth for every class.
// This writes ink_polarity into every existing entry of STATE.class_preprocess
// and persists the whole map to /classes/save_config, so live preview / live
// inference and newly added classes all see the same polarity immediately —
// not just at train time. (startTrain() still re-applies it as a safety net.)
function applyGlobalInkPolarityToAllClasses(polarity) {{
  const globalPol = normalizeInkPolarity(polarity);
  const all = Object.assign({{}}, STATE.class_preprocess || {{}});
  const names = Object.keys(all);
  if (!names.length) return;
  let changed = false;
  for (const name of names) {{
    const rec = (all[name] && typeof all[name] === 'object') ? Object.assign({{}}, all[name]) : {{}};
    if (normalizeInkPolarity(rec.ink_polarity) !== globalPol) {{
      rec.ink_polarity = globalPol;
      all[name] = rec;
      changed = true;
    }}
  }}
  if (!changed) return;
  STATE.class_preprocess = all;
  fetch(`${{baseUrl}}/classes/save_config`, {{
    method: 'POST',
    headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify({{session: STATE.session, class_preprocess: all}})
  }}).catch(() => {{}});
}}
function setClassPreprocessConfig(className, cfg) {{
  const next = Object.assign({{}}, STATE.class_preprocess || {{}});
  // Merged over what is already stored, not written straight in.  Callers
  // edit one slice of the record (mode / ROI / thresholds) and have no
  // opinion about the glyph keys, so normalising their cfg on its own would
  // drop ink_polarity / glyph_segment / glyph_convention for this class —
  // and because save_config replaces the whole map, that loss would be
  // persisted, not just local.
  next[className] = normalizeClassPreprocessConfig(Object.assign({{}}, next[className] || {{}}, cfg));
  STATE.class_preprocess = next;
  // Persist to backend so per-class thresholds survive page reload
  fetch(`${{baseUrl}}/classes/save_config`, {{
    method: 'POST',
    headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify({{session: STATE.session, class_preprocess: next}})
  }}).catch(() => {{}});
}}
function normalizeSamplePreprocessMap(raw) {{
  const src = raw && typeof raw === 'object' ? raw : {{}};
  const out = {{}};
  for (const [className, value] of Object.entries(src)) {{
    if (!value || typeof value !== 'object') continue;
    const classKey = String(className || '').trim();
    if (!classKey) continue;
    const classMap = {{}};
    for (const [filename, cfg] of Object.entries(value)) {{
      const fileKey = String(filename || '').trim();
      if (!fileKey) continue;
      classMap[fileKey] = normalizeClassPreprocessConfig(cfg);
    }}
    if (Object.keys(classMap).length) out[classKey] = classMap;
  }}
  return out;
}}
function getSampleOwnPreprocessConfig(className, filename) {{
  const map = normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}});
  const classMap = map[String(className || '')] || {{}};
  return normalizeClassPreprocessConfig(classMap[String(filename || '')] || null);
}}
function sampleHasOwnPreprocessConfig(className, filename) {{
  const map = STATE.sample_preprocess && typeof STATE.sample_preprocess === 'object' ? STATE.sample_preprocess : {{}};
  const classMap = map[String(className || '')];
  if (!classMap || typeof classMap !== 'object') return false;
  return Object.prototype.hasOwnProperty.call(classMap, String(filename || ''));
}}
function getSampleEffectivePreprocessConfig(className, filename) {{
  if (sampleHasOwnPreprocessConfig(className, filename)) {{
    return getSampleOwnPreprocessConfig(className, filename);
  }}
  return getClassPreprocessConfig(className);
}}
function samePreprocessConfig(a, b) {{
  const left = normalizeClassPreprocessConfig(a);
  const right = normalizeClassPreprocessConfig(b);
  if (String(left.mode || '') !== String(right.mode || '')) return false;
  const lroi = Array.isArray(left.manual_roi) ? left.manual_roi : null;
  const rroi = Array.isArray(right.manual_roi) ? right.manual_roi : null;
  if (!lroi && !rroi) return true;
  if (!lroi || !rroi || lroi.length !== 4 || rroi.length !== 4) return false;
  for (let i = 0; i < 4; i += 1) {{
    if (Math.abs(Number(lroi[i] || 0) - Number(rroi[i] || 0)) > 1e-6) return false;
  }}
  return true;
}}
// Written straight through, exactly like setClassPreprocessConfig does for the
// class map.  Before this the per-sample ROI existed only in this tab: the Edit
// modal's "Save Sample" flipped its chip to Saved, so the UI reported a write
// that had never happened, and the next reload — or anything that reads
// tm_classes.json instead of the page — was back to the old crop.  The whole
// map is sent, not the one entry, because the endpoint REPLACES it: a sample
// returning to its class default is a DELETE, which travels as a missing key.
function persistSamplePreprocess(next) {{
  fetch(`${{baseUrl}}/classes/save_config`, {{
    method: 'POST',
    headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify({{session: STATE.session, sample_preprocess: next}})
  }}).catch(() => {{}});
}}
function setSamplePreprocessConfig(className, filename, cfg) {{
  const name = String(className || '').trim();
  const file = String(filename || '').trim();
  if (!name || !file) return;
  const next = normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}});
  const classDefault = getClassPreprocessConfig(name);
  const normalized = normalizeClassPreprocessConfig(cfg);
  if (samePreprocessConfig(normalized, classDefault)) {{
    if (next[name] && typeof next[name] === 'object') {{
      delete next[name][file];
      if (!Object.keys(next[name]).length) delete next[name];
    }}
  }} else {{
    if (!next[name] || typeof next[name] !== 'object') next[name] = {{}};
    next[name][file] = normalized;
  }}
  STATE.sample_preprocess = next;
  persistSamplePreprocess(next);
}}
function deleteSamplePreprocessConfig(className, filename) {{
  const next = normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}});
  const name = String(className || '').trim();
  const file = String(filename || '').trim();
  if (next[name] && typeof next[name] === 'object') {{
    delete next[name][file];
    if (!Object.keys(next[name]).length) delete next[name];
  }}
  STATE.sample_preprocess = next;
  persistSamplePreprocess(next);
}}
function selectedClassSample(className) {{
  const items = getClassPreprocessSamples(className);
  if (!items.length) return null;
  const idx = Math.max(0, Math.min(items.length - 1, Number(classPreprocessSampleIndex || 0)));
  classPreprocessSampleIndex = idx;
  return items[idx];
}}
function getClassPreprocessSamples(className) {{
  if (classPreprocessOpen && classPreprocessClass === className && Array.isArray(classPreprocessSamples) && classPreprocessSamples.length) {{
    return normalizePreviewList(classPreprocessSamples);
  }}
  return normalizePreviewList((STATE.sample_previews && STATE.sample_previews[className]) || []);
}}
function selectedClassSampleFilename(className) {{
  const sample = selectedClassSample(className);
  return sample ? String(previewFilename(sample) || '') : '';
}}
function classPreprocessModeLabel(mode) {{
  if (mode === 'manual_roi') return 'Manual ROI';
  return 'Auto';
}}
function samplePreprocessStatus(className, filename) {{
  return sampleHasOwnPreprocessConfig(className, filename) ? 'Edited' : 'Auto';
}}
function currentClassPreprocessDirty() {{
  if (!classPreprocessClass) return false;
  const filename = selectedClassSampleFilename(classPreprocessClass);
  if (!filename) return false;
  return !samePreprocessConfig(
    classPreprocessDraft,
    getSampleEffectivePreprocessConfig(classPreprocessClass, filename)
  );
}}
function maybeDiscardClassPreprocessDraft() {{
  classPreprocessDirty = currentClassPreprocessDirty();
  if (!classPreprocessDirty) return true;
  return window.confirm('This sample has unsaved ROI changes. Discard them?');
}}
function saveCurrentClassPreprocessSample(closeAfterSave = false) {{
  const filename = selectedClassSampleFilename(classPreprocessClass);
  if (!classPreprocessClass || !filename) return;
  setSamplePreprocessConfig(classPreprocessClass, filename, classPreprocessDraft);
  classPreprocessDirty = false;
  render();
  if (closeAfterSave) {{
    closeClassPreprocessEditor(true);
    return;
  }}
  renderClassPreprocessModal();
  refreshClassProcessedPreview();
}}
function rememberClassPreprocessScroll() {{
  const pane = document.getElementById('classPreprocessRightPane');
  if (!pane) return;
  classPreprocessScrollTop = Number(pane.scrollTop || 0);
}}
function restoreClassPreprocessScroll() {{
  const pane = document.getElementById('classPreprocessRightPane');
  if (!pane) return;
  pane.scrollTop = Math.max(0, Number(classPreprocessScrollTop || 0));
}}
function getClassPreprocessGridColumnCount() {{
  const grid = document.querySelector('#classPreprocessModal .class-preprocess-grid');
  if (!grid) return 1;
  try {{
    const cols = String(window.getComputedStyle(grid).gridTemplateColumns || '')
      .split(' ')
      .map((part) => String(part || '').trim())
      .filter(Boolean);
    return Math.max(1, cols.length || 1);
  }} catch (e) {{
    return 1;
  }}
}}
function getClassPreprocessWindow(items) {{
  const list = Array.isArray(items) ? items : [];
  const total = list.length;
  if (total <= CLASS_PREPROCESS_WINDOW) {{
    return {{
      start: 0,
      end: total,
      items: list,
    }};
  }}
  const current = Math.max(0, Math.min(total - 1, Number(classPreprocessSampleIndex || 0)));
  const half = Math.floor(CLASS_PREPROCESS_WINDOW / 2);
  let start = Math.max(0, current - half);
  let end = Math.min(total, start + CLASS_PREPROCESS_WINDOW);
  start = Math.max(0, end - CLASS_PREPROCESS_WINDOW);
  return {{
    start,
    end,
    items: list.slice(start, end),
  }};
}}
function moveClassPreprocessSample(delta) {{
  if (!classPreprocessOpen || !classPreprocessClass) return;
  const items = getClassPreprocessSamples(classPreprocessClass);
  if (!items.length) return;
  if (!maybeDiscardClassPreprocessDraft()) return;
  rememberClassPreprocessScroll();
  const nextIndex = Math.max(0, Math.min(items.length - 1, Number(classPreprocessSampleIndex || 0) + Number(delta || 0)));
  if (nextIndex === Number(classPreprocessSampleIndex || 0)) return;
  classPreprocessSampleIndex = nextIndex;
  classPreprocessDraft = normalizeClassPreprocessConfig(
    getSampleEffectivePreprocessConfig(classPreprocessClass, selectedClassSampleFilename(classPreprocessClass))
  );
  classPreprocessDirty = false;
  renderClassPreprocessModal();
  refreshClassProcessedPreview();
}}
async function refreshClassProcessedPreview(opts) {{
  // keepPrevious: for a live control (the Dark/Lum Thr sliders), where
  // dropping the pane to "Rendering..." twenty-five times a second reads as a
  // broken preview.  The last frame stays up until its replacement arrives;
  // the token guard below still means only the newest one can win.
  const keepPrevious = !!(opts && opts.keepPrevious);
  classPreprocessPreviewToken += 1;
  const token = classPreprocessPreviewToken;
  if (classPreprocessPreviewController) {{
    try {{ classPreprocessPreviewController.abort(); }} catch (e) {{}}
    classPreprocessPreviewController = null;
  }}
  if (!classPreprocessOpen || !classPreprocessClass) return;
  const sample = selectedClassSample(classPreprocessClass);
  if (!sample) {{
    classPreprocessProcessedSrc = '';
    classPreprocessBusy = false;
    classPreprocessAutoCrop = null;
    renderClassPreprocessModal();
    return;
  }}
  const targetClass = String(classPreprocessClass || '');
  const targetFilename = String(previewFilename(sample) || '');
  const targetClassConfig = getClassPreprocessConfig(targetClass);
  const targetSampleConfig = normalizeClassPreprocessConfig(classPreprocessDraft);
  classPreprocessBusy = true;
  if (!keepPrevious) {{
    classPreprocessProcessedSrc = '';
    classPreprocessAutoCrop = null;
    refreshClassPreprocessResultPane();
  }}
  classPreprocessPreviewController = new AbortController();
  try {{
    const res = await fetch(`${{baseUrl}}/preprocess/preview`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      signal: classPreprocessPreviewController.signal,
      body: JSON.stringify({{
        session: STATE.session,
        class: targetClass,
        filename: targetFilename,
        class_config: targetClassConfig,
        sample_config: targetSampleConfig,
        // sign_ocr Auto needs the target sequence to extract this class's
        // slot from a whole-board capture (the run may not have a saved
        // train_config.json before the first training).
        sign_sequence: isSignOcrProject() ? signSequence().join('') : '',
        sign_sequences: isSignOcrProject() ? signSequences() : [],
        reading_order: String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr'))
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to render preprocess preview.');
    if (
      token !== classPreprocessPreviewToken ||
      !classPreprocessOpen ||
      String(classPreprocessClass || '') !== targetClass ||
      String(selectedClassSampleFilename(targetClass) || '') !== targetFilename
    ) {{
      return;
    }}
    const procB64 = data.processed_image_b64 || data.image_b64;
    // `#t=<token>` is the cache-buster for the LIVE case, and it is not
    // decoration: the pane is an <img> whose src IS the payload, so two
    // responses with equal bytes (a drag inside one Otsu step, say) would
    // otherwise assign an identical string — a no-op the browser is free to
    // skip — and an aborted race could leave the older frame on screen.
    // The fragment is ignored when the image is decoded, so the pixels are
    // exactly the payload's.
    classPreprocessProcessedSrc = procB64 ? `data:image/png;base64,${{procB64}}#t=${{token}}` : '';
    // Save auto-crop ROI (normalized 0-1) for overlay display
    classPreprocessAutoCrop = Array.isArray(data.crop) && data.crop.length === 4 ? data.crop : null;
  }} catch (err) {{
    if (err && (err.name === 'AbortError' || String(err).includes('aborted'))) return;
    if (
      token !== classPreprocessPreviewToken ||
      !classPreprocessOpen ||
      String(classPreprocessClass || '') !== targetClass
    ) {{
      return;
    }}
    classPreprocessProcessedSrc = '';
    classPreprocessAutoCrop = null;
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    if (token !== classPreprocessPreviewToken) return;
    classPreprocessPreviewController = null;
    classPreprocessBusy = false;
    if (!classPreprocessOpen || String(classPreprocessClass || '') !== targetClass) return;
    refreshClassPreprocessResultPane();
    // Re-assert the overlay from the DRAFT, not from the auto crop.  This
    // runs on every refresh — including the one pointerup triggers — so
    // passing null here replaced the box the user had just dragged with the
    // green auto-detected one.  That was the "the ROI immediately snaps back
    // to the whole frame" half of the bug: null fails the isManual test, so
    // the box fell through to classPreprocessAutoCrop.
    updateClassPreprocessOverlay(classPreprocessDraft && classPreprocessDraft.manual_roi);
  }}
}}
async function loadClassPreprocessSamples(className) {{
  if (!className) return;
  try {{
    const res = await fetch(`${{baseUrl}}/class_filenames?session=${{encodeURIComponent(STATE.session)}}&class=${{encodeURIComponent(className)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to load class samples.');
    const filenames = Array.isArray(data.filenames) ? data.filenames.map((v) => String(v || '').trim()).filter(Boolean) : [];
    classPreprocessSamples = filenames.map((filename) => ({{
      filename,
      src: `${{baseUrl}}/sample_image?session=${{encodeURIComponent(STATE.session)}}&class=${{encodeURIComponent(className)}}&filename=${{encodeURIComponent(filename)}}`
    }}));
    if (classPreprocessOpen && classPreprocessClass === className) {{
      const maxIndex = Math.max(0, classPreprocessSamples.length - 1);
      classPreprocessSampleIndex = Math.max(0, Math.min(Number(classPreprocessSampleIndex || 0), maxIndex));
      renderClassPreprocessModal();
      refreshClassProcessedPreview();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function openClassPreprocessEditor(className) {{
  stopClassPreprocessPreviewWork();
  classPreprocessClass = String(className || '');
  classPreprocessSampleIndex = 0;
  classPreprocessSamples = normalizePreviewList((STATE.sample_previews && STATE.sample_previews[classPreprocessClass]) || []);
  classPreprocessDraft = normalizeClassPreprocessConfig(
    getSampleEffectivePreprocessConfig(classPreprocessClass, selectedClassSampleFilename(classPreprocessClass))
  );
  classPreprocessProcessedSrc = '';
  classPreprocessDirty = false;
  classPreprocessOpen = true;
  renderClassPreprocessModal();
  refreshClassProcessedPreview();
  await loadClassPreprocessSamples(classPreprocessClass);
}}
function stopClassPreprocessPreviewWork() {{
  classPreprocessPreviewToken += 1;
  if (classPreprocessPreviewController) {{
    try {{ classPreprocessPreviewController.abort(); }} catch (e) {{}}
    classPreprocessPreviewController = null;
  }}
  if (classPreprocessThresholdTimer) {{
    try {{ clearTimeout(classPreprocessThresholdTimer); }} catch (e) {{}}
    classPreprocessThresholdTimer = null;
  }}
}}
function refreshClassPreprocessResultPane() {{
  if (!classPreprocessOpen) return;
  const pane = document.querySelector('#classPreprocessModal .class-preprocess-result');
  if (!pane) return;
  pane.innerHTML = classPreprocessProcessedSrc
    ? `<img src="${{escapeHtml(classPreprocessProcessedSrc)}}" alt="Processed preview"/>`
    : `<div class="preview-empty">${{classPreprocessBusy ? 'Rendering...' : 'Choose a sample to preview preprocessing.'}}</div>`;
  scheduleLayoutResync();
}}
function closeClassPreprocessEditor(force = false) {{
  if (!force && !maybeDiscardClassPreprocessDraft()) return;
  stopClassPreprocessPreviewWork();
  classPreprocessOpen = false;
  classPreprocessClass = '';
  classPreprocessSamples = [];
  classPreprocessDraft = null;
  classPreprocessProcessedSrc = '';
  classPreprocessBusy = false;
  classPreprocessDirty = false;
  classPreprocessAutoCrop = null;
  renderClassPreprocessModal();
}}
  function getClassPreprocessImageContentRect() {{
    const stage = document.getElementById('classPreprocessStage');
    const img = document.getElementById('classPreprocessImage');
    if (!stage || !img) return null;
    const stageRect = stage.getBoundingClientRect();
    if (!stageRect.width || !stageRect.height) return null;
    const naturalW = Number(img.naturalWidth || 0);
    const naturalH = Number(img.naturalHeight || 0);
    if (naturalW <= 0 || naturalH <= 0) {{
      return {{
        stageLeft: stageRect.left,
        stageTop: stageRect.top,
        stageWidth: stageRect.width,
        stageHeight: stageRect.height,
        left: stageRect.left,
        top: stageRect.top,
        width: stageRect.width,
        height: stageRect.height,
        offsetX: 0,
        offsetY: 0,
      }};
    }}
    const scale = Math.min(stageRect.width / naturalW, stageRect.height / naturalH);
    const width = naturalW * scale;
    const height = naturalH * scale;
    const offsetX = (stageRect.width - width) * 0.5;
    const offsetY = (stageRect.height - height) * 0.5;
    return {{
      stageLeft: stageRect.left,
      stageTop: stageRect.top,
      stageWidth: stageRect.width,
      stageHeight: stageRect.height,
      left: stageRect.left + offsetX,
      top: stageRect.top + offsetY,
      width,
      height,
      offsetX,
      offsetY,
    }};
  }}
  function normalizeClassPreprocessClientPoint(clientX, clientY, allowClamp = true) {{
    const rect = getClassPreprocessImageContentRect();
    if (!rect || !rect.width || !rect.height) return null;
    const inside = clientX >= rect.left && clientX <= (rect.left + rect.width)
      && clientY >= rect.top && clientY <= (rect.top + rect.height);
    if (!inside && !allowClamp) return null;
    const px = allowClamp ? Math.max(rect.left, Math.min(rect.left + rect.width, clientX)) : clientX;
    const py = allowClamp ? Math.max(rect.top, Math.min(rect.top + rect.height, clientY)) : clientY;
    return {{
      x: Math.max(0, Math.min(1, (px - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (py - rect.top) / rect.height)),
    }};
  }}
  function applyClassPreprocessMode(mode) {{
    const nextMode = ['auto_by_label', 'manual_roi'].includes(String(mode || ''))
      ? String(mode)
      : 'auto_by_label';
    rememberClassPreprocessScroll();
    const next = normalizeClassPreprocessConfig(Object.assign({{}}, classPreprocessDraft || {{}}, {{mode: nextMode}}));
    // Seed a first-time Manual ROI from the auto-detected box instead of the
    // whole frame: switching the mode used to hand the user a full-image
    // selection they then had to shrink by hand before it meant anything.
    if (!next.manual_roi) {{
      next.manual_roi = (Array.isArray(classPreprocessAutoCrop) && classPreprocessAutoCrop.length === 4)
        ? classPreprocessAutoCrop.slice()
        : [0, 0, 1, 1];
    }}
    classPreprocessDraft = next;
    if (nextMode === 'auto_by_label') {{
      const filename = selectedClassSampleFilename(classPreprocessClass);
      if (filename) setSamplePreprocessConfig(classPreprocessClass, filename, next);
    }}
    classPreprocessDirty = (nextMode !== 'auto_by_label');
    renderClassPreprocessModal();
    refreshClassProcessedPreview();
  }}
  function nudgeClassPreprocessRoi(dx, dy) {{
    if (!classPreprocessOpen || !classPreprocessDraft || String(classPreprocessDraft.mode || 'auto_by_label') !== 'manual_roi') return;
    rememberClassPreprocessScroll();
    const draft = normalizeClassPreprocessConfig(classPreprocessDraft);
    const roi = Array.isArray(draft.manual_roi) && draft.manual_roi.length === 4 ? draft.manual_roi.slice() : [0, 0, 1, 1];
    const width = Math.max(0.01, Math.min(1, Number(roi[2] || 1) - Number(roi[0] || 0)));
    const height = Math.max(0.01, Math.min(1, Number(roi[3] || 1) - Number(roi[1] || 0)));
    const nextX1 = Math.max(0, Math.min(1 - width, Number(roi[0] || 0) + Number(dx || 0)));
    const nextY1 = Math.max(0, Math.min(1 - height, Number(roi[1] || 0) + Number(dy || 0)));
    draft.manual_roi = [nextX1, nextY1, nextX1 + width, nextY1 + height];
    classPreprocessDraft = normalizeClassPreprocessConfig(draft);
    classPreprocessDirty = true;
    renderClassPreprocessModal();
    refreshClassProcessedPreview();
  }}
function updateClassPreprocessOverlay(roi) {{
  const overlay = document.getElementById('classPreprocessOverlay');
  if (!overlay) return;
  // Show crop box for manual ROI OR auto-detected crop
  const isManual = Array.isArray(roi) && roi.length === 4 && classPreprocessDraft && String(classPreprocessDraft.mode || 'auto_by_label') === 'manual_roi';
  const effectiveRoi = isManual ? roi : classPreprocessAutoCrop;
  const valid = Array.isArray(effectiveRoi) && effectiveRoi.length === 4;
  if (!valid) {{
    overlay.style.display = 'none';
    return;
  }}
    const rect = getClassPreprocessImageContentRect();
    if (!rect || !rect.width || !rect.height) {{
      overlay.style.display = 'none';
      return;
    }}
  const x1 = Math.max(0, Math.min(1, Number(effectiveRoi[0] || 0)));
  const y1 = Math.max(0, Math.min(1, Number(effectiveRoi[1] || 0)));
  const x2 = Math.max(x1, Math.min(1, Number(effectiveRoi[2] || x1)));
  const y2 = Math.max(y1, Math.min(1, Number(effectiveRoi[3] || y1)));
  overlay.style.display = 'block';
    overlay.style.left = `${{rect.offsetX + x1 * rect.width}}px`;
    overlay.style.top = `${{rect.offsetY + y1 * rect.height}}px`;
    overlay.style.width = `${{Math.max(0, (x2 - x1) * rect.width)}}px`;
    overlay.style.height = `${{Math.max(0, (y2 - y1) * rect.height)}}px`;
    // Green border = auto-detected, blue border = manual ROI.  --accent is
    // not defined anywhere in this sheet, and without a fallback the manual
    // branch resolved to currentColor — so a manual box and an auto box were
    // drawn the same colour and the user could not tell which one they were
    // looking at.  #1a73e8 is .roi-overlay's own blue.
    overlay.style.borderColor = isManual ? 'var(--accent, #1a73e8)' : '#4caf50';
}}
function syncClassPreprocessRoiInputs(roi) {{
  if (!Array.isArray(roi) || roi.length !== 4) return;
  const ids = ['classPreprocessX1', 'classPreprocessY1', 'classPreprocessX2', 'classPreprocessY2'];
  ids.forEach((id, idx) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.value = Number(roi[idx] || 0).toFixed(2);
  }});
}}
function bindClassPreprocessStage() {{
  const stage = document.getElementById('classPreprocessStage');
  const img = document.getElementById('classPreprocessImage');
  if (!stage || !img) return;
  let dragStart = null;
  const applyPoint = (clientX, clientY, finalize = false) => {{
    if (!dragStart || !classPreprocessDraft || String(classPreprocessDraft.mode || 'auto_by_label') !== 'manual_roi') return;
      const start = normalizeClassPreprocessClientPoint(dragStart.x, dragStart.y, true);
      const end = normalizeClassPreprocessClientPoint(clientX, clientY, true);
      if (!start || !end) return;
      const x1 = start.x;
      const y1 = start.y;
      const x2 = end.x;
      const y2 = end.y;
    const nextRoi = [
      Math.min(x1, x2),
      Math.min(y1, y2),
      Math.max(x1, x2),
      Math.max(y1, y2),
    ];
    classPreprocessDraft.manual_roi = nextRoi;
    classPreprocessDraft = normalizeClassPreprocessConfig(classPreprocessDraft);
    classPreprocessDirty = true;
    updateClassPreprocessOverlay(classPreprocessDraft.manual_roi);
    syncClassPreprocessRoiInputs(classPreprocessDraft.manual_roi);
    if (finalize) {{
      renderClassPreprocessModal();
      refreshClassProcessedPreview();
    }}
  }};
  stage.onpointerdown = (e) => {{
    if (!classPreprocessDraft || String(classPreprocessDraft.mode || 'auto_by_label') !== 'manual_roi') return;
      const start = normalizeClassPreprocessClientPoint(e.clientX, e.clientY, false);
      if (!start) return;
    dragStart = {{x: e.clientX, y: e.clientY}};
    try {{ stage.setPointerCapture(e.pointerId); }} catch (err) {{}}
    applyPoint(e.clientX, e.clientY, false);
    e.preventDefault();
  }};
  stage.onpointermove = (e) => {{
    if (!dragStart) return;
    applyPoint(e.clientX, e.clientY, false);
  }};
  stage.onpointerup = (e) => {{
    if (!dragStart) return;
    applyPoint(e.clientX, e.clientY, true);
    dragStart = null;
  }};
  stage.onpointercancel = () => {{
    dragStart = null;
  }};
    const syncOverlay = () => updateClassPreprocessOverlay(classPreprocessDraft && classPreprocessDraft.manual_roi);
    if (img.complete) syncOverlay();
    img.onload = syncOverlay;
}}
function renderClassPreprocessModal() {{
  const host = document.getElementById('classPreprocessModal');
  if (!host) return;
  if (!classPreprocessOpen || !classPreprocessClass) {{
    host.style.display = 'none';
    host.innerHTML = '';
    return;
  }}
  const sample = selectedClassSample(classPreprocessClass);
  const sampleFilename = sample ? String(previewFilename(sample) || '') : '';
  const cfg = normalizeClassPreprocessConfig(
    classPreprocessDraft || getSampleEffectivePreprocessConfig(classPreprocessClass, sampleFilename)
  );
  classPreprocessDraft = cfg;
  // Seeded from the DRAFT — the same record the preview renders from — not
  // from the saved class row.  Reading the class left the two out of step the
  // moment a sample carried its own override: the boxes showed the class's
  // thresholds while the preview binarised at the sample's.  (The 150/20
  // fallbacks that used to live here disagreed with normalizeClassPreprocess
  // Config's own 100/0, so an unset class rendered a third, phantom value.)
  const classPreprocessThresholds = {{
    bg_lum_thresh: cfg.bg_lum_thresh != null ? cfg.bg_lum_thresh : 100,
    bg_dark_thresh: cfg.bg_dark_thresh != null ? cfg.bg_dark_thresh : 0,
}};
  const roi = Array.isArray(cfg.manual_roi) && cfg.manual_roi.length === 4 ? cfg.manual_roi : null;
  const left = roi ? roi[0] * 100 : 0;
  const top = roi ? roi[1] * 100 : 0;
  const width = roi ? Math.max(0, (roi[2] - roi[0]) * 100) : 0;
  const height = roi ? Math.max(0, (roi[3] - roi[1]) * 100) : 0;
  const samples = getClassPreprocessSamples(classPreprocessClass);
  const windowed = getClassPreprocessWindow(samples);
  const dirty = currentClassPreprocessDirty();
  classPreprocessDirty = dirty;
  const thumbs = windowed.items.map((item, localIdx) => {{
    const idx = windowed.start + localIdx;
    return `
    <button class="class-preprocess-thumb${{idx === classPreprocessSampleIndex ? ' active' : ''}}${{sampleHasOwnPreprocessConfig(classPreprocessClass, String(previewFilename(item) || '')) ? ' defined' : ''}}" type="button" data-preprocess-sample="${{idx}}">
      <img src="${{escapeHtml(previewSrc(item))}}" loading="lazy" decoding="async" alt="Sample ${{idx + 1}}"/>
      <div class="class-preprocess-thumb-state">${{samplePreprocessStatus(classPreprocessClass, String(previewFilename(item) || ''))}}</div>
    </button>
  `;
  }}).join('');
  const thumbsMeta = samples.length > windowed.items.length
    ? `<div class="class-preprocess-window-note">Showing ${{windowed.start + 1}}-${{windowed.end}} of ${{samples.length}} samples</div>`
    : '';
  host.style.display = 'flex';
  host.innerHTML = `
    <div class="class-preprocess-shell">
      <div class="class-preprocess-left">
        <div class="class-preprocess-head">
          <span>${{escapeHtml(classPreprocessClass)}} · ${{classPreprocessModeLabel(cfg.mode)}}</span>
          <button class="iconbtn" type="button" id="classPreprocessCloseTop">✕</button>
        </div>
        <div class="class-preprocess-visuals">
          <div class="class-preprocess-visual-pane">
            <div class="class-preprocess-visual-title">Original Sample</div>
            <div class="class-preprocess-stage" id="classPreprocessStage">
              ${{
                sample
                  ? `<img id="classPreprocessImage" src="${{escapeHtml(previewSrc(sample))}}" alt="Raw sample"/>`
                  : `<div class="preview-empty">This class has no samples yet.</div>`
              }}
                <div class="roi-overlay" id="classPreprocessOverlay" style="display:${{(roi && cfg.mode === 'manual_roi') || classPreprocessAutoCrop ? 'block' : 'none'}};"></div>
            </div>
          </div>
          <div class="class-preprocess-visual-pane">
            <div class="class-preprocess-visual-title">Processed Preview</div>
            <div class="class-preprocess-result">${{classPreprocessProcessedSrc ? `<img src="${{escapeHtml(classPreprocessProcessedSrc)}}" alt="Processed preview"/>` : `<div class="preview-empty">${{classPreprocessBusy ? 'Rendering...' : 'Choose a sample to preview preprocessing.'}}</div>`}}</div>
          </div>
        </div>
          <div class="class-preprocess-info">In Manual ROI mode, drag directly on the original image and save the current sample. Shortcuts: F1 Auto, F2 Manual ROI, S Save, D Delete Sample, Esc Close, ←→↑↓ Navigate, Shift+Arrows Move ROI.</div>
      </div>
      <div class="class-preprocess-right class-preprocess-side" id="classPreprocessRightPane">
        <div class="class-preprocess-current class-preprocess-status-row">
          <div class="class-preprocess-info">Current Sample: ${{sampleFilename ? escapeHtml(sampleFilename) : 'No sample selected'}} · ${{samplePreprocessStatus(classPreprocessClass, sampleFilename)}}</div>
          <div class="class-preprocess-status-chip${{dirty ? ' dirty' : ''}}">${{cfg.mode === 'auto_by_label' ? 'Auto' : (dirty ? 'Unsaved' : 'Saved')}}</div>
          <button class="btn btn-secondary" type="button" id="classPreprocessDeleteSample"${{sampleFilename ? '' : ' disabled'}}>Delete Sample</button>
        </div>
        <div class="class-preprocess-mode-row">
          <label>
            Preprocess Mode
            <select id="classPreprocessMode">
              <option value="auto_by_label"${{cfg.mode === 'auto_by_label' ? ' selected' : ''}}>Auto (find sign)</option>
              <option value="manual_roi"${{cfg.mode === 'manual_roi' ? ' selected' : ''}}>Manual ROI</option>
            </select>
          </label>
          <button class="btn btn-primary class-preprocess-mode-save" type="button" id="classPreprocessSave"${{sampleFilename && dirty ? '' : ' disabled'}}>${{dirty ? 'Save Sample' : 'Saved'}}</button>
        </div>
        <div class="class-preprocess-fields">
        </div>
        <div class="class-preprocess-fields">
          <label title="Min brightness for sign (0-100). Pixels darker than this become white.">Dark Thr <input id="classPreprocessDarkNum" type="number" min="0" max="100" value="${{classPreprocessThresholds.bg_dark_thresh || 0}}" style="width:55px"/><input id="classPreprocessDark" type="range" min="0" max="100" value="${{classPreprocessThresholds.bg_dark_thresh || 0}}" step="1"/></label>
          <label title="Max brightness to consider as sign (50-255). Lower = stricter.">Lum Thr <input id="classPreprocessLumNum" type="number" min="50" max="255" value="${{classPreprocessThresholds.bg_lum_thresh || 100}}" style="width:55px"/><input id="classPreprocessLum" type="range" min="50" max="255" value="${{classPreprocessThresholds.bg_lum_thresh || 100}}" step="1"/></label>
        </div>
        <div class="class-preprocess-fields">
          <label>ROI X1<input id="classPreprocessX1" type="number" min="0" max="1" step="0.01" value="${{roi ? roi[0].toFixed(2) : '0.00'}}" ${{cfg.mode === 'manual_roi' ? '' : 'disabled'}}/></label>
          <label>ROI Y1<input id="classPreprocessY1" type="number" min="0" max="1" step="0.01" value="${{roi ? roi[1].toFixed(2) : '0.00'}}" ${{cfg.mode === 'manual_roi' ? '' : 'disabled'}}/></label>
          <label>ROI X2<input id="classPreprocessX2" type="number" min="0" max="1" step="0.01" value="${{roi ? roi[2].toFixed(2) : '1.00'}}" ${{cfg.mode === 'manual_roi' ? '' : 'disabled'}}/></label>
          <label>ROI Y2<input id="classPreprocessY2" type="number" min="0" max="1" step="0.01" value="${{roi ? roi[3].toFixed(2) : '1.00'}}" ${{cfg.mode === 'manual_roi' ? '' : 'disabled'}}/></label>
        </div>
        <div>
          <div class="class-preprocess-info">Samples</div>
          ${{thumbsMeta}}
          <div class="class-preprocess-grid">${{thumbs || '<div class="preview-empty">No samples yet.</div>'}}</div>
        </div>
        <div class="class-preprocess-actions">
          <button class="btn btn-secondary" type="button" id="classPreprocessCancel">Close</button>
        </div>
      </div>
    </div>
  `;
  restoreClassPreprocessScroll();
  const rightPane = document.getElementById('classPreprocessRightPane');
  if (rightPane) {{
    rightPane.onscroll = () => {{
      classPreprocessScrollTop = Number(rightPane.scrollTop || 0);
    }};
  }}
  const closeTop = document.getElementById('classPreprocessCloseTop');
  const cancel = document.getElementById('classPreprocessCancel');
  const deleteSampleBtn = document.getElementById('classPreprocessDeleteSample');
  const save = document.getElementById('classPreprocessSave');
  const modeSel = document.getElementById('classPreprocessMode');
  const bindRoi = (id, idx) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.onchange = () => {{
      rememberClassPreprocessScroll();
      const draft = normalizeClassPreprocessConfig(classPreprocessDraft);
      const roiNow = Array.isArray(draft.manual_roi) && draft.manual_roi.length === 4 ? draft.manual_roi.slice() : [0, 0, 1, 1];
      const n = Number(el.value);
      if (!isFinite(n)) return;
      roiNow[idx] = Math.max(0, Math.min(1, n));
      draft.manual_roi = roiNow;
      classPreprocessDraft = normalizeClassPreprocessConfig(draft);
      classPreprocessDirty = true;
      renderClassPreprocessModal();
      refreshClassProcessedPreview();
    }};
  }};
  bindRoi('classPreprocessX1', 0);
  bindRoi('classPreprocessY1', 1);
  bindRoi('classPreprocessX2', 2);
  bindRoi('classPreprocessY2', 3);
  host.onclick = (e) => {{
    if (e.target === host) closeClassPreprocessEditor();
  }};
  if (closeTop) closeTop.onclick = () => closeClassPreprocessEditor();
  if (cancel) cancel.onclick = () => closeClassPreprocessEditor();
  if (deleteSampleBtn) deleteSampleBtn.onclick = async () => {{
    if (!sampleFilename) return;
    await deleteSample(classPreprocessClass, sampleFilename);
  }};
  if (save) save.onclick = () => {{
    saveCurrentClassPreprocessSample(false);
  }};
  if (modeSel) modeSel.onchange = () => {{
      applyClassPreprocessMode(modeSel.value);
  }};
  // Dark Thr / Lum Thr.
  //
  // Two defects lived in this block.  It wrote the moved values into
  // STATE.class_preprocess and nothing else — but /preprocess/preview renders
  // from `sample_config`, which refreshClassProcessedPreview builds out of
  // classPreprocessDraft, so the backend never saw a move and re-binarised
  // with exact Otsu wherever the sliders sat.  And it re-armed at 250ms while
  // blanking the result pane on every tick, so even a correct preview would
  // flicker through "Rendering..." under the user's hand.
  //
  // Now: `input` (every drag frame, and every keystroke in the number boxes)
  // writes the DRAFT and re-renders the preview in place, debounced 40ms;
  // `change` (slider released, number committed) additionally persists the
  // class record — which is where the image branch has always kept these two
  // (image_preprocess.preprocess_for_label reads them off the class config).
  const lumSlider  = document.getElementById('classPreprocessLum');
  const lumNum     = document.getElementById('classPreprocessLumNum');
  const darkSlider = document.getElementById('classPreprocessDark');
  const darkNum    = document.getElementById('classPreprocessDarkNum');
  const clampThreshold = (value, lo, hi, fallback) => {{
    const n = Number(value);
    if (!isFinite(n)) return fallback;
    return Math.max(lo, Math.min(hi, Math.round(n)));
  }};
  const pushThresholds = () => {{
    const draft = normalizeClassPreprocessConfig(
      classPreprocessDraft || getClassPreprocessConfig(classPreprocessClass)
    );
    draft.bg_dark_thresh = clampThreshold(darkSlider ? darkSlider.value : 0, 0, 100, 0);
    draft.bg_lum_thresh = clampThreshold(lumSlider ? lumSlider.value : 100, 50, 255, 100);
    classPreprocessDraft = draft;
    // keepPrevious: the pane holds the last frame while this one renders, so
    // a drag reads as a filter being applied, not as a reload.
    refreshClassProcessedPreview({{ keepPrevious: true }});
  }};
  const persistThresholds = () => {{
    const existing = getClassPreprocessConfig(classPreprocessClass);
    existing.bg_lum_thresh = clampThreshold(lumSlider ? lumSlider.value : 100, 50, 255, 100);
    existing.bg_dark_thresh = clampThreshold(darkSlider ? darkSlider.value : 0, 0, 100, 0);
    setClassPreprocessConfig(classPreprocessClass, existing);
  }};
  const debouncedThresholds = () => {{
    if (classPreprocessThresholdTimer) clearTimeout(classPreprocessThresholdTimer);
    classPreprocessThresholdTimer = setTimeout(() => {{
      classPreprocessThresholdTimer = null;
      pushThresholds();
    }}, 40);
  }};
  const commitThresholds = () => {{
    if (classPreprocessThresholdTimer) {{
      clearTimeout(classPreprocessThresholdTimer);
      classPreprocessThresholdTimer = null;
    }}
    pushThresholds();
    persistThresholds();
  }};
  // Lum/B-G slider + number sync.  The slider is the authority for the pair:
  // its own min/max cannot leave the valid range, so it applies on every frame.
  if (lumSlider) lumSlider.oninput = () => {{ if (lumNum) lumNum.value = lumSlider.value; debouncedThresholds(); }};
  if (darkSlider) darkSlider.oninput = () => {{ if (darkNum) darkNum.value = darkSlider.value; debouncedThresholds(); }};
  if (lumSlider) lumSlider.onchange = commitThresholds;
  if (darkSlider) darkSlider.onchange = commitThresholds;
  // A number box mid-edit can hold anything ("", "-", "1" on the way to "150"),
  // so `input` only applies a value that already parsed inside the range —
  // otherwise the box would fight the user's typing.  `change` clamps for real.
  if (darkNum) darkNum.oninput = () => {{
    const v = Number(darkNum.value);
    if (!isFinite(v) || v < 0 || v > 100 || !darkSlider) return;
    darkSlider.value = v;
    debouncedThresholds();
  }};
  if (lumNum) lumNum.oninput = () => {{
    const v = Number(lumNum.value);
    if (!isFinite(v) || v < 50 || v > 255 || !lumSlider) return;
    lumSlider.value = v;
    debouncedThresholds();
  }};
  if (darkNum) darkNum.onchange = () => {{
    const v = clampThreshold(darkNum.value, 0, 100, 0);
    darkNum.value = v;
    if (darkSlider) darkSlider.value = v;
    commitThresholds();
  }};
  if (lumNum) lumNum.onchange = () => {{
    const v = clampThreshold(lumNum.value, 50, 255, 100);
    lumNum.value = v;
    if (lumSlider) lumSlider.value = v;
    commitThresholds();
  }};



  host.querySelectorAll('[data-preprocess-sample]').forEach((btn) => {{
    btn.onclick = () => {{
      if (!maybeDiscardClassPreprocessDraft()) return;
      rememberClassPreprocessScroll();
      classPreprocessSampleIndex = Number(btn.getAttribute('data-preprocess-sample') || 0);
      classPreprocessDraft = normalizeClassPreprocessConfig(
        getSampleEffectivePreprocessConfig(classPreprocessClass, selectedClassSampleFilename(classPreprocessClass))
      );
      classPreprocessDirty = false;
      renderClassPreprocessModal();
      refreshClassProcessedPreview();
    }};
  }});
  bindClassPreprocessStage();
    updateClassPreprocessOverlay(classPreprocessDraft && classPreprocessDraft.manual_roi);
}}
  document.addEventListener('keydown', (e) => {{
    if (!classPreprocessOpen) return;
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    const target = e.target;
    const tag = String(target && target.tagName || '').toUpperCase();
    const isTypingTarget = !!(target && (target.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'));
    if (isTypingTarget) return;
    const key = String(e.key || '').toUpperCase();
    const modeMap = {{
      F1: 'auto_by_label',
      F2: 'manual_roi',
    }};
    const nextMode = modeMap[key];
    if (nextMode) {{
      e.preventDefault();
      applyClassPreprocessMode(nextMode);
      return;
    }}
    if (key === 'ESCAPE') {{
      e.preventDefault();
      closeClassPreprocessEditor();
      return;
    }}
    if (key === 'S') {{
      e.preventDefault();
      if (classPreprocessClass && selectedClassSampleFilename(classPreprocessClass) && currentClassPreprocessDirty()) {{
        saveCurrentClassPreprocessSample(false);
      }}
      return;
    }}
    if (key === 'D') {{
      e.preventDefault();
      const sampleFilename = classPreprocessClass ? selectedClassSampleFilename(classPreprocessClass) : '';
      if (classPreprocessClass && sampleFilename) {{
        deleteSample(classPreprocessClass, sampleFilename);
      }}
      return;
    }}
    // ← → = navigate sequential   |   ↑ ↓ = navigate by row   |   Shift+arrows = nudge ROI
    if (key === 'ARROWLEFT' || key === 'ARROWRIGHT' || key === 'ARROWUP' || key === 'ARROWDOWN') {{
      e.preventDefault();
      // Shift held → nudge ROI (manual mode only)
      if (e.shiftKey && classPreprocessDraft && String(classPreprocessDraft.mode || '') === 'manual_roi') {{
        const step = 0.05;
        if (key === 'ARROWLEFT')  nudgeClassPreprocessRoi(-step, 0);
        if (key === 'ARROWRIGHT') nudgeClassPreprocessRoi( step, 0);
        if (key === 'ARROWUP')    nudgeClassPreprocessRoi(0, -step);
        if (key === 'ARROWDOWN')  nudgeClassPreprocessRoi(0,  step);
        return;
      }}
      // No Shift → navigate samples: ← → = prev/next, ↑ ↓ = previous/next row
      if (key === 'ARROWLEFT' || key === 'LEFT') {{
        moveClassPreprocessSample(-1);
        return;
      }}
      if (key === 'ARROWRIGHT' || key === 'RIGHT') {{
        moveClassPreprocessSample(1);
        return;
      }}
      if (key === 'ARROWUP' || key === 'UP') {{
        moveClassPreprocessSample(-getClassPreprocessGridColumnCount());
        return;
      }}
      if (key === 'ARROWDOWN' || key === 'DOWN') {{
        moveClassPreprocessSample(getClassPreprocessGridColumnCount());
        return;
      }}
    }}
  }});
restoreTrainCfgFromStorage();
if (window.__tmStageMark) window.__tmStageMark('train-cfg-restored');
function restorePreviewState() {{
  try {{
    const raw = window.localStorage.getItem(previewStorageKey);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (!data || typeof data !== 'object') return;
    previewInputOn = !!data.inputOn;
    if (data.source === 'webcam' || data.source === 'device' || data.source === 'upload') {{
      previewSource = String(data.source);
      STATE.preview_source = previewSource;
    }}
    if (['auto_by_label', 'sign', 'junction', 'manual_roi', 'none'].includes(String(data.preprocess || ''))) {{
      previewPreprocessMode = String(data.preprocess);
    }}
    previewShowRoi = !!data.showRoi;
    previewShowRaw = !!data.showRaw;
    previewShowScore = !!data.showScore;
    previewThreshUserModified = !!data.threshModified;
    previewOodSignPctMin = Number(data.oodSignPctMin ?? previewOodSignPctMin ?? 0.3);
    previewOodSignPctMax = Number(data.oodSignPctMax ?? previewOodSignPctMax ?? 70.0);
    const storageVersion = Number(data.version || 1);
    let cachedMaxProb = Number(
      data.oodMaxProbMin ?? data.signMatchMinProb
      ?? previewOodMaxProbMin ?? 0.40
    );
    let cachedEntropy = Number(
      data.oodEntropyRatioMax ?? previewOodEntropyRatioMax ?? 0.85
    );
    if (storageVersion < PREVIEW_STORAGE_VERSION) {{
      // A v1 cache wrote the old 0.60 / 0.70 defaults for every project, so
      // they overwrote the backend's new 0.40 / 0.85 values on load.
      // Migrate ONLY those exact defaults: a custom 0.25 / 0.30 survives.
      if (Math.abs(cachedMaxProb - 0.60) < 1e-9) cachedMaxProb = 0.40;
      if (Math.abs(cachedEntropy - 0.70) < 1e-9) cachedEntropy = 0.85;
    }}
    previewOodMaxProbMin = Math.max(0, Math.min(1, cachedMaxProb));
    previewOodEntropyRatioMax = Math.max(0, Math.min(1, cachedEntropy));
    if (previewThreshUserModified) {{
      previewDarkThresh = Number(data.darkThresh ?? STATE.preview_dark_thresh ?? 0);
      previewLumThresh = Number(data.lumThresh ?? STATE.preview_lum_thresh ?? 100);
    }}
    if (storageVersion < PREVIEW_STORAGE_VERSION) {{
      // One-time rewrite, so a later intentional 0.60 / 0.70 is not
      // converted again on the next load.
      persistPreviewState();
    }}
  }} catch (e) {{}}
}}
function persistPreviewState() {{
  STATE.preview_source = String(previewSource || 'webcam');
  try {{
    window.localStorage.setItem(previewStorageKey, JSON.stringify({{
      inputOn: !!previewInputOn,
      source: String(previewSource || 'webcam'),
      preprocess: String(previewPreprocessMode || 'auto_by_label'),
      showRoi: !!previewShowRoi,
      showRaw: !!previewShowRaw,
      showScore: !!previewShowScore,
      darkThresh: Number((previewDarkThresh ?? 0)),
      lumThresh: Number((previewLumThresh ?? 100)),
      oodSignPctMin: Number((previewOodSignPctMin ?? 0.3)),
      oodSignPctMax: Number((previewOodSignPctMax ?? 70.0)),
      oodMaxProbMin: Number((previewOodMaxProbMin ?? 0.40)),
      oodEntropyRatioMax: Number((previewOodEntropyRatioMax ?? 0.85)),
      // One threshold under both names; OOD MAX PROB is authoritative.
      signMatchMinProb: Number((previewOodMaxProbMin ?? 0.40)),
      threshModified: !!previewThreshUserModified,
      version: PREVIEW_STORAGE_VERSION
    }}));
  }} catch (e) {{}}
}}
async function pushPreviewOodToBackend() {{
  // OOD MAX PROB is the single per-slot probability threshold.  Pushing the
  // four Input Settings values immediately persists that unification in the
  // project train config (the export and /classes/sync_ocr both read it).
  try {{
    await fetch(`${{baseUrl}}/preview/ood`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        ood_sign_pct_min: Number(previewOodSignPctMin ?? 0.3),
        ood_sign_pct_max: Number(previewOodSignPctMax ?? 70.0),
        ood_max_prob_min: Number(previewOodMaxProbMin ?? 0.40),
        ood_entropy_ratio_max: Number(previewOodEntropyRatioMax ?? 0.85)
      }})
    }});
  }} catch (e) {{}}
}}
function previewStateForProject() {{
  // The block of Preview-pane state that travels WITH the project.  These are
  // not training hyperparameters, but they are not cosmetic either:
  // buildExportRuntimeConfig() stamps every one of them into model_settings.h
  // as the device's OOD gates, so a project that reopened them at the
  // fallbacks would re-export a device whose thresholds differ from the one
  // already flashed.  Saved and restored as a unit, under one key, so the
  // save path and the load path cannot disagree about which keys exist.
  return {{
    bg_dark_thresh: Number((previewDarkThresh ?? 0)),
    bg_lum_thresh: Number((previewLumThresh ?? 100)),
    thresh_modified: !!previewThreshUserModified,
    ood_sign_pct_min: Number((previewOodSignPctMin ?? 0.3)),
    ood_sign_pct_max: Number((previewOodSignPctMax ?? 70.0)),
    ood_max_prob_min: Number((previewOodMaxProbMin ?? 0.40)),
    ood_entropy_ratio_max: Number((previewOodEntropyRatioMax ?? 0.85)),
    // Compatibility alias; OOD MAX PROB is the single authoritative value.
    sign_match_min_prob: Number((previewOodMaxProbMin ?? 0.40)),
  }};
}}
function applyPreviewState(raw) {{
  // Absent (a v1 project, or /project/reset) -> keep the caller's fallbacks
  // rather than writing undefined over them.  Per key, so a project saved by
  // a build that knew about fewer gates still restores the ones it has.
  const p = raw && typeof raw === 'object' ? raw : null;
  if (!p) return;
  if (isFinite(Number(p.bg_dark_thresh))) previewDarkThresh = Number(p.bg_dark_thresh);
  if (isFinite(Number(p.bg_lum_thresh))) previewLumThresh = Number(p.bg_lum_thresh);
  if (typeof p.thresh_modified === 'boolean') previewThreshUserModified = p.thresh_modified;
  if (isFinite(Number(p.ood_sign_pct_min))) previewOodSignPctMin = Number(p.ood_sign_pct_min);
  if (isFinite(Number(p.ood_sign_pct_max))) previewOodSignPctMax = Number(p.ood_sign_pct_max);
  const projectMaxProb = isFinite(Number(p.ood_max_prob_min))
    ? Number(p.ood_max_prob_min)
    : (isFinite(Number(p.sign_match_min_prob)) ? Number(p.sign_match_min_prob) : null);
  if (projectMaxProb !== null) previewOodMaxProbMin = projectMaxProb;
  if (isFinite(Number(p.ood_entropy_ratio_max))) previewOodEntropyRatioMax = Number(p.ood_entropy_ratio_max);
}}
function clearPreviewUploadState() {{
  previewUploadImageSrc = '';
  previewUploadImageB64 = '';
  previewUploadFilename = '';
}}
  async function fileToB64(file) {{
    const buf = await file.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = '';
    for (let i = 0; i < bytes.length; i += 1) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }}
async function pickPreviewUploadFile() {{
    if (window.pywebview && window.pywebview.api && window.pywebview.api.pick_single_image_file) {{
      try {{
        const picked = await window.pywebview.api.pick_single_image_file();
        if (picked && picked.b64) {{
          previewUploadImageB64 = String(picked.b64 || '');
          previewUploadImageSrc = `data:${{picked.mime || 'image/png'}};base64,${{picked.b64}}`;
          previewUploadFilename = String(picked.name || 'upload');
          previewSource = 'upload';
          persistPreviewState();
          renderPreviewSettings();
          renderPreviewCard();
          if (previewInputOn) await runPreviewUploadPrediction();
          return true;
        }}
      }} catch (err) {{
        toast(String(err && err.message ? err.message : err));
      }}
      return false;
    }}
    return await new Promise((resolve) => {{
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/png,image/jpeg,image/jpg,image/bmp,image/gif';
      input.onchange = async () => {{
        const file = input.files && input.files[0];
        if (!file) {{
          resolve(false);
          return;
        }}
        try {{
          const b64 = await fileToB64(file);
          const mime = String(file.type || 'image/png');
          previewUploadImageB64 = String(b64 || '');
          previewUploadImageSrc = `data:${{mime}};base64,${{b64}}`;
          previewUploadFilename = String(file.name || 'upload');
          previewSource = 'upload';
          persistPreviewState();
          renderPreviewSettings();
          renderPreviewCard();
          if (previewInputOn) await runPreviewUploadPrediction();
          resolve(true);
        }} catch (err) {{
          toast(String(err && err.message ? err.message : err));
          resolve(false);
        }}
      }};
      input.click();
    }});
}}
restorePreviewState();
restorePreferences();
restoreAutoSeedEmnist();
if (window.__tmStageMark) window.__tmStageMark('preview-restored');
function restoreExportDir() {{
  try {{
    const raw = window.localStorage.getItem(exportDirStorageKey);
    if (raw) exportDir = String(raw || '');
  }} catch (e) {{}}
}}
function persistExportDir() {{
  try {{
    if (exportDir) window.localStorage.setItem(exportDirStorageKey, String(exportDir));
  }} catch (e) {{}}
}}
restoreExportDir();
function restoreExportSettings() {{
  try {{
    const rawName = window.localStorage.getItem(exportNameStorageKey);
    const rawArray = window.localStorage.getItem(exportArrayStorageKey);
    if (rawName) exportModelName = String(rawName || '').trim() || exportModelName;
    if (rawArray !== null && rawArray !== undefined) exportArrayName = String(rawArray || '').trim();
  }} catch (e) {{}}
}}
function persistExportSettings() {{
  try {{
    window.localStorage.setItem(exportNameStorageKey, String(exportModelName || 'tm'));
    window.localStorage.setItem(exportArrayStorageKey, String(exportArrayName || ''));
  }} catch (e) {{}}
}}
restoreExportSettings();
if (window.__tmStageMark) window.__tmStageMark('export-restored');
function parentUrl() {{
  try {{
    return new URL(window.parent.location.href);
  }} catch (e) {{
    return new URL(window.location.href);
  }}
}}
function navigateParent(url) {{
  try {{
    window.parent.location.href = url;
    return;
  }} catch (e) {{}}
  window.location.href = url;
}}
function cleanupWorkspaceFrameBeforeNavigate(reason) {{
  if (window.__tmNavigatingAway) return;
  try {{
    if (frameHeightRaf) {{
      try {{ cancelAnimationFrame(frameHeightRaf); }} catch (e) {{}}
      frameHeightRaf = 0;
    }}
  }} catch (e) {{}}
  try {{
    layoutResyncTimers.forEach((t) => {{
      try {{ clearTimeout(t); }} catch (e) {{}}
    }});
    layoutResyncTimers = [];
    mountReflowTimers.forEach((t) => {{
      try {{ clearTimeout(t); }} catch (e) {{}}
    }});
    mountReflowTimers = [];
    if (homeNavigateTimer) {{
      try {{ clearTimeout(homeNavigateTimer); }} catch (e) {{}}
      homeNavigateTimer = 0;
    }}
  }} catch (e) {{}}
  try {{
    sendStreamlitMessage('streamlit:setFrameHeight', {{height: 1}});
  }} catch (e) {{}}
  window.__tmNavigatingAway = true;
  try {{
    const targets = [];
    if (window.frameElement) targets.push(window.frameElement);
    if (window.frameElement && window.frameElement.parentElement) targets.push(window.frameElement.parentElement);
    if (window.frameElement && window.frameElement.parentElement && window.frameElement.parentElement.parentElement) {{
      targets.push(window.frameElement.parentElement.parentElement);
    }}
    for (const el of targets) {{
      if (!el || !el.style) continue;
      el.style.height = '';
      el.style.minHeight = '';
      el.style.maxHeight = '';
      el.style.width = '';
      el.style.minWidth = '';
      el.style.maxWidth = '';
      el.style.overflow = '';
      el.style.overflowX = '';
      el.style.overflowY = '';
      el.style.position = '';
      el.style.inset = '';
    }}
  }} catch (e) {{}}
  try {{
    for (const el of [document.documentElement, document.body]) {{
      if (!el || !el.style) continue;
      el.style.minHeight = '';
      el.style.maxHeight = '';
      el.style.overflow = '';
      el.style.overflowX = '';
      el.style.overflowY = '';
    }}
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
    window.scrollTo(0, 0);
  }} catch (e) {{}}
  try {{
    requestShellLayoutRefresh(String(reason || 'navigate-away'));
  }} catch (e) {{}}
  // #region debug-point B:return-home-cleanup
  dbgEvent('B', 'app.py:cleanupWorkspaceFrameBeforeNavigate', '[DEBUG] cleanup workspace frame before navigate', {{
    reason: String(reason || ''),
    hasFrameElement: !!window.frameElement,
    parentHref: parentUrl().toString(),
    innerWidth: window.innerWidth || 0,
    innerHeight: window.innerHeight || 0,
  }});
  // #endregion
}}
async function returnHome() {{
  // #region debug-point B:return-home-start
  dbgEvent('B', 'app.py:returnHome', '[DEBUG] returnHome start', {{
    session: String(STATE.session || ''),
    returnTarget: String(STATE.return_target || 'home'),
    openSourceClass,
    openSourceKind,
    href: parentUrl().toString(),
    innerWidth: window.innerWidth || 0,
    innerHeight: window.innerHeight || 0,
  }});
  // #endregion
  try {{
    await closeSourcePanel(false);
  }} catch (e) {{}}
  stopPreviewPredictLoop();
  previewInputOn = false;
  persistPreviewState();
  // Stop ANY in-progress hold recording (closeSourcePanel only fires /stop
  // while the source panel is open) and close ALL live sources of this
  // session (closeSourcePanel closes only the currently-open source; the
  // controller endpoint stops every source when no source param is given).
  // Fire-and-forget: the Python-side teardown (hidden-button click) is the
  // authoritative release; these are the fallback channel and must not
  // freeze the UI for the server-side joins (~4.5s worst case).
  try {{ fetch(`${{baseUrl}}/stop?session=${{encodeURIComponent(STATE.session)}}`, {{keepalive:true}}); }} catch (e) {{}}
  try {{ fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}`, {{keepalive:true}}); }} catch (e) {{}}
  const u = parentUrl();
  const target = String(STATE.return_target || 'home');
  const keys = [
    'open_class', 'open_kind', 'notice',
    'idx', 'name', 'serial_port', 'webcam_index'
  ];
  for (const key of keys) {{
    u.searchParams.delete(key);
  }}
  if (target === 'classified-import') {{
    u.searchParams.set('tm_project', 'image_classified_import');
    u.searchParams.set('tm_session', String(STATE.session || ''));
  }} else {{
    u.searchParams.delete('tm_project');
    u.searchParams.delete('tm_session');
  }}
  // Keep tm_action=home so a later Python rerun can self-heal (full teardown
  // + URL reset) even if the sandbox-safe channel below fails.
  u.searchParams.set('tm_action', 'home');
  // #region debug-point B:return-home-navigate
  dbgEvent('B', 'app.py:returnHome', '[DEBUG] returnHome navigating to home URL', {{
    session: String(STATE.session || ''),
    returnTarget: target,
    targetUrl: u.toString(),
  }});
  // #endregion
  // Sandbox-safe channel to Python: this iframe cannot navigate the parent
  // (components.html sandbox lacks allow-top-navigation) and Streamlit's
  // frontend ignores custom postMessages, so click the hidden "Return Home"
  // st.button rendered in the parent document (same-origin DOM access, the
  // same mechanism used by _tm_render_page_scroll_reset). Python then stops
  // live/record, drops the session and rewrites the parent URL via
  // st.query_params — the only channel that actually resets the address bar.
  try {{
    const wrappers = window.parent.document.querySelectorAll('div[data-testid="stButton"]');
    for (const wrapper of wrappers) {{
      const b = wrapper.querySelector('button');
      if (b && String(b.textContent || '').trim() === 'Return Home') {{
        b.click();
        break;
      }}
    }}
  }} catch (e) {{}}
  // Best-effort parent URL rewrite (allowed when the sandbox grants
  // same-origin history access); Python's st.query_params.clear() remains
  // the authoritative reset.
  try {{
    window.parent.history.replaceState({{}}, '', u.toString());
  }} catch (e) {{}}
  cleanupWorkspaceFrameBeforeNavigate('return-home');
  // NOTE: do NOT call navigateParent here — the components.html sandbox lacks
  // allow-top-navigation, so its fallback would navigate the IFRAME itself and
  // briefly mount a nested Streamlit app inside the old iframe.  The parent
  // URL reset is owned by window.parent.history.replaceState above + the
  // Python st.query_params.clear() on the hidden-button path.
}}
function reloadParent() {{
  try {{
    window.parent.location.reload();
    return;
  }} catch (e) {{}}
  window.location.reload();
}}
// -------- Class panel source live loop (sourcePreview-${{className}} <img>) --------
// ⚠️ Uses its own blob URL + abort controller to stay independent from the
// Preview-card visual/predict loops.  Previously they shared `previewBlobUrl`
// and `previewFrameController`, so Hold-to-Capture (which touches the
// Preview loop timers on purpose) would cancel or revoke the class panel's
// active fetch mid-flight and make the camera view appear frozen.
function stopSourceLiveLoop() {{
  if (classSourceAbortController) {{
    try {{ classSourceAbortController.abort(); }} catch (e) {{}}
    classSourceAbortController = null;
  }}
  classSourceRequestInFlight = false;
  classSourceInFlightAt = 0;
  if (sourceLiveTimer) {{
    clearInterval(sourceLiveTimer);
    sourceLiveTimer = null;
  }}
}}
function startSourceLiveLoop() {{
  stopSourceLiveLoop();
  refreshPreviewImage();
  sourceLiveTimer = window.setInterval(refreshPreviewImage, Math.max(40, Number(previewIntervalMs || 80)));
}}
// Legacy aliases — kept so call sites we haven't yet switched keep compiling;
// prefer the explicit names above.
const stopPreviewLoop = stopSourceLiveLoop;
const startPreviewLoop = startSourceLiveLoop;

// -------- Preview-card visual-only loop (updates <img> WITHOUT inference) --------
function stopPreviewVisualLoop() {{
  previewVisualToken += 1;
  previewVisualInFlight = false;
  if (previewVisualTimer) {{
    clearInterval(previewVisualTimer);
    previewVisualTimer = null;
  }}
}}
async function refreshPreviewVisualFrame(token) {{
  // Visual refresh drives the Preview pane only while Input is ON but the
  // predict timer is paused (hold-to-capture / single capture).  With
  // Input OFF the pane must NOT stream — previously the visual loop ran
  // regardless of Input, so the camera opened and the pane showed live
  // frames without the user toggling Input (the "preview already opened"
  // bug).
  if (previewSource === 'upload') return;
  if (previewSource !== 'webcam' && previewSource !== 'device') return;
  // The visual loop drives the pane ONLY while Input is ON but the predict
  // timer is paused (hold-to-capture / single capture).  With Input OFF the
  // pane must stay on the last prediction or the empty state — running the
  // visual loop then opened the camera and streamed live frames without the
  // user toggling Input (the "preview already opened" bug).  When the
  // predict loop runs it owns the pane (processed crop/roi image); two
  // writers to the same <img> flickered between raw and processed frames.
  if (!previewInputOn) return;
  if (previewPredictTimer) return;
  if (previewVisualInFlight) return;
  if (token !== previewVisualToken) return;
  const sess = String(STATE.session || '').trim();
  if (!sess) return;
  // The pane's real image element — a nonexistent 'previewPaneImg' id here
  // made this whole loop dead code, so the pane froze whenever inference
  // was paused (the "Preview freezes during Hold" half of the bug).
  const paneImg = document.getElementById('previewImage');
  if (!paneImg || !paneImg.closest('#previewPane')) return;
  previewVisualInFlight = true;
  try {{
    const res = await fetch(`${{baseUrl}}/live/frame?session=${{encodeURIComponent(sess)}}&source=${{encodeURIComponent(previewSource)}}&_ts=${{Date.now()}}`);
    if (!res.ok || token !== previewVisualToken) return;
    const blob = await res.blob();
    if (token !== previewVisualToken) return;
    if (previewPredictTimer) return;  // predict resumed mid-fetch
    if (previewBlobUrl) try {{ URL.revokeObjectURL(previewBlobUrl); }} catch (e) {{}}
    previewBlobUrl = URL.createObjectURL(blob);
    paneImg.src = previewBlobUrl;
  }} catch (e) {{
    // Visual refresh is best-effort — if the backend is busy saving samples
    // we don't want to spam errors.
  }} finally {{
    previewVisualInFlight = false;
  }}
}}
function startPreviewVisualLoop() {{
  stopPreviewVisualLoop();
  if (previewSource === 'upload') return;
  if (previewSource !== 'webcam' && previewSource !== 'device') return;
  // Never tick with Input OFF — the pane must not show live frames (or
  // open the camera) unless the user toggled Input ON.
  if (!previewInputOn) return;
  previewVisualToken += 1;
  const token = previewVisualToken;
  refreshPreviewVisualFrame(token);
  previewVisualTimer = window.setInterval(
    () => refreshPreviewVisualFrame(token),
    Math.max(40, Number(previewIntervalMs || 80))
  );
}}
function stopHoldSyncLoop() {{
  holdNextToken += 1;
  if (holdPollController) {{
    try {{ holdPollController.abort(); }} catch (e) {{}}
    holdPollController = null;
  }}
  if (holdSyncTimer) {{
    clearInterval(holdSyncTimer);
    holdSyncTimer = null;
  }}
}}
function clearPreviewBlob() {{
  if (!previewBlobUrl) return;
  // Never revoke a URL the pane <img> is still displaying — the visual
  // loop replaces it on the next tick, so a revoked-while-shown blob used
  // to blank the pane until then.
  try {{
    const paneImg = document.getElementById('previewImage');
    if (paneImg && paneImg.getAttribute('src') === previewBlobUrl) return;
  }} catch (e) {{}}
  try {{ URL.revokeObjectURL(previewBlobUrl); }} catch (e) {{}}
  previewBlobUrl = '';
}}
function syncSourceActionButtons(className) {{
  if (!className) return;
  const safe = cssSafe(className);
  const capBtn = document.getElementById(`sourceCapture-${{safe}}`);
  const holdBtn = document.getElementById(`sourceHold-${{safe}}`);
  const switching = sourceSwitchInFlight && sourceSwitchClass === className && (sourceSwitchKind === 'webcam' || sourceSwitchKind === 'device');
  const capturing = captureInFlight && openSourceClass === className;
  const holding = holdRecording && holdRecordClass === className;
  if (capBtn) {{
    capBtn.disabled = switching || capturing || holding;
    capBtn.textContent = switching ? 'Switching...' : (capturing ? 'Capturing...' : 'Capture');
  }}
  if (holdBtn) {{
    holdBtn.disabled = switching || capturing;
    holdBtn.textContent = switching ? 'Switching...' : (holding ? 'Recording...' : 'Hold to Capture');
  }}
}}
async function closeSourcePanel(shouldRender = true) {{
  stopPreviewLoop();
  clearPreviewBlob();
  sourceSettingsOpen = false;
  const prevClass = openSourceClass;
  const prevKind = openSourceKind;
  if (holdRecording) {{
    try {{
      await fetch(`${{baseUrl}}/stop?session=${{encodeURIComponent(STATE.session)}}`);
    }} catch (e) {{}}
    stopHoldSyncLoop();
    holdRecording = false;
    holdRecordClass = '';
    holdRecordSource = '';
  }}
  openSourceClass = '';
  openSourceKind = '';
  clearOpenSourceState();
  if (prevKind) {{
    try {{
      await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(prevKind)}}`);
    }} catch (e) {{}}
  }}
  if (shouldRender && prevClass) render();
}}
// The class-panel source note is SHARED: hold capture writes "Captured: N" to
// it once per saved sample, and this loop writes to it every tick.  At the
// default rate the preview tick (80 ms) lands ~1.6 times per captured sample
// (125 ms), so every blank here erases the counter inside the gap before the
// next one is written — the only progress feedback a hold has flickers and
// mostly reads as blank, and capture looks stalled while samples are in fact
// being saved at the full rate.  While a hold is running on this class the
// note belongs to the hold: this loop keeps replacing the IMAGE and leaves the
// text alone.  A genuine preview failure is still reported — the record thread
// has its own stall detector and surfaces it as a toast, so nothing is hidden.
function noteHeldByCapture(className) {{
  return !!(holdRecording && holdRecordClass && holdRecordClass === className);
}}
function refreshPreviewImage() {{
  if (!openSourceKind || !openSourceClass || openSourceKind === 'upload') return;
  if (classSourceRequestInFlight) {{
    if (classSourceAbortController && classSourceInFlightAt && (Date.now() - classSourceInFlightAt) > 1500) {{
      try {{ classSourceAbortController.abort(); }} catch (e) {{}}
      classSourceAbortController = null;
      classSourceRequestInFlight = false;
      classSourceInFlightAt = 0;
    }} else {{
      return;
    }}
  }}
  const sess = String(STATE.session || '').trim();
  if (!sess) return;
  const img = document.getElementById(`sourcePreview-${{cssSafe(openSourceClass)}}`);
  const note = document.getElementById(`sourceNote-${{cssSafe(openSourceClass)}}`);
  if (!img || !note) return;
  const url = `${{baseUrl}}/live/frame?session=${{encodeURIComponent(sess)}}&source=${{encodeURIComponent(openSourceKind)}}&_ts=${{Date.now()}}`;
  classSourceRequestInFlight = true;
  classSourceAbortController = new AbortController();
  classSourceInFlightAt = Date.now();
  const ourUrl = url;
  const ourController = classSourceAbortController;
  const abortTimer = window.setTimeout(() => {{
    if (ourController === classSourceAbortController) {{
      try {{ ourController.abort(); }} catch (e) {{}}
    }}
  }}, 1500);
  fetch(ourUrl, {{ signal: classSourceAbortController.signal }}).then(async (res) => {{
    if (ourController !== classSourceAbortController) return;
    if (!res.ok) {{
      let msg = '';
      let starting = false;
      try {{
        const data = await res.json();
        msg = String(data && data.error ? data.error : '');
        starting = String(data && data.starting ? data.starting : '0') === '1';
      }} catch (e) {{}}
      if (!msg) {{
        msg = openSourceKind === 'device'
          ? 'Unable to read from serial device. Check serial port and baudrate.'
          : 'Unable to open webcam. Check permission or whether the camera is in use.';
      }}
      if (starting) {{
        // The producer is (re)starting — KEEP the last good frame on screen
        // (never blank it) and retry on the next tick.  This is the H4
        // self-recovery path: a transient stall reads as a still image for
        // ≤1 tick, never as a permanent freeze.
        if (img.getAttribute('src')) {{
          if (!noteHeldByCapture(openSourceClass)) note.textContent = '';
          return;
        }}
        if (!noteHeldByCapture(openSourceClass)) note.textContent = msg || 'Preview starting...';
        return;
      }}
      throw new Error(msg);
    }}
    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    if (!contentType.includes('image/png')) {{
      let msg = 'Preview returned invalid data.';
      try {{
        const data = await res.json();
        msg = String(data && data.error ? data.error : msg);
      }} catch (e) {{}}
      throw new Error(msg);
    }}
    const blob = await res.blob();
    if (ourController !== classSourceAbortController) return;
    // Revoke ONLY the class-panel source blob URL — do not touch
    // `previewBlobUrl` (used by the Preview card visual/predict loops;
    // sharing one URL caused the source preview to lose its img src every
    // single frame during hold capture, making the camera view appear
    // completely frozen).
    if (classSourceBlobUrl) {{
      try {{ URL.revokeObjectURL(classSourceBlobUrl); }} catch (e) {{}}
      classSourceBlobUrl = '';
    }}
    classSourceBlobUrl = URL.createObjectURL(blob);
    img.src = classSourceBlobUrl;
    if (!noteHeldByCapture(openSourceClass)) note.textContent = '';
  }}).catch((err) => {{
    if (ourController !== classSourceAbortController) return;
    if (err && (err.name === 'AbortError' || String(err).includes('aborted'))) return;
    if (classSourceBlobUrl) {{
      try {{ URL.revokeObjectURL(classSourceBlobUrl); }} catch (e) {{}}
      classSourceBlobUrl = '';
    }}
    img.removeAttribute('src');
    if (!noteHeldByCapture(openSourceClass)) note.textContent = String(err && err.message ? err.message : err);
  }}).finally(() => {{
    try {{ clearTimeout(abortTimer); }} catch (e) {{}}
    if (ourController === classSourceAbortController) {{
      classSourceRequestInFlight = false;
      classSourceAbortController = null;
      classSourceInFlightAt = 0;
    }} else {{
      // Our request got superseded — still clear the in-flight flag if
      // we were the last owner (bounded to keep flags consistent).
      if (classSourceRequestInFlight && !classSourceAbortController) {{
        classSourceRequestInFlight = false;
        classSourceInFlightAt = 0;
      }}
    }}
  }});
}}
async function openSourcePanel(kind, className) {{
  // #region debug-point B:open-source-panel
  dbgEvent('B', 'app.py:openSourcePanel', '[DEBUG] openSourcePanel requested', {{kind, className, openSourceClass, openSourceKind}});
  // #endregion
  // Clicking the source tile whose drawer is already open folds it back up,
  // so the tile behaves as a toggle instead of a no-op.
  if (openSourceClass === className && openSourceKind === kind) {{
    await closeSourcePanel();
    return;
  }}
  await closeSourcePanel(false);
  openSourceClass = className;
  openSourceKind = kind;
  sourceSettingsOpen = false;
  persistOpenSourceState();
  render();
  if (kind === 'device') {{
    await refreshSerialPorts(false);
    render();
  }}
  await syncClassState(className);
  if (openSourceClass === className) updateOpenSamplesPanel(className);
  if (kind === 'webcam' || kind === 'device') {{
    try {{
      const res = await fetch(`${{baseUrl}}/live/open?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(kind)}}&frame_side=${{encodeURIComponent(String(currentSerialFrameSide || 96))}}&channels=${{encodeURIComponent(String(currentSerialChannels || 1))}}`);
      if (!res.ok) {{
        const data = await res.json().catch(() => ({{ok:'0'}}));
        toast(String(data && data.error ? data.error : 'Unable to open live preview.'));
      }}
    }} catch (e) {{}}
    startPreviewLoop();
  }}
}}
async function ensureOpenSourceLive(opts = {{}}) {{
  // #region debug-point B:ensure-open-source-live
  dbgEvent('B', 'app.py:ensureOpenSourceLive', '[DEBUG] ensureOpenSourceLive enter', {{openSourceClass, openSourceKind, session: String(STATE.session || '')}});
  // #endregion
  if (!openSourceKind || !openSourceClass) return;
  if (openSourceKind !== 'webcam' && openSourceKind !== 'device') return;
  const sess = String(STATE.session || '').trim();
  if (!sess) {{
    if (!opts.silent) toast('Workspace still initializing — try again in 1-2 seconds.', 3500);
    return;
  }}
  try {{
    const res = await fetch(`${{baseUrl}}/live/open?session=${{encodeURIComponent(sess)}}&source=${{encodeURIComponent(openSourceKind)}}&frame_side=${{encodeURIComponent(String(currentSerialFrameSide || 96))}}&channels=${{encodeURIComponent(String(currentSerialChannels || 1))}}`);
    if (!res.ok) {{
      const data = await res.json().catch(() => ({{ok:'0'}}));
      const msg = String(data && data.error ? data.error : 'Unable to open live preview.');
      if (!opts.silentFail) toast(msg);
    }}
  }} catch (e) {{}}
  startPreviewLoop();
}}
async function captureSource() {{
  if (!openSourceKind || !openSourceClass) return;
  if (captureInFlight) {{
    dbgEvent('A', 'app.py:captureSource', '[DEBUG] duplicate capture ignored', {{openSourceKind, openSourceClass}});
    return;
  }}
  const shouldResumePreviewPredict = !!previewInputOn;
  const previewSourceBeforeCapture = String(previewSource || 'webcam');
  captureInFlight = true;
  syncSourceActionButtons(openSourceClass);
  dbgEvent('A', 'app.py:captureSource', '[DEBUG] capture clicked', {{openSourceKind, openSourceClass, captureInFlight}});
  const url = `${{baseUrl}}/live/capture?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(openSourceKind)}}&class=${{encodeURIComponent(openSourceClass)}}`;
  try {{
    if (shouldResumePreviewPredict) {{
      stopPreviewPredictLoop();
      // NOTE: no /live/close here for ANY source — the backend single
      // capture now consumes the live worker's capture cache (device
      // included), and closing the shared worker used to freeze the
      // class-panel camera view until Preview Input was toggled off/on.
    }}
    const res = await fetch(url);
    const data = await res.json().catch(() => ({{ok:'0', error:'capture failed'}}));
    if (!res.ok || data.ok !== '1') {{
      toast(data.error || 'Capture failed.');
      return;
    }}
    if (data.image_b64) {{
      prependSamplePreview(openSourceClass, {{
        src: `data:image/png;base64,${{data.image_b64}}`,
        filename: String(data.filename || '')
      }});
    }}
    incrementSampleCount(openSourceClass, 1);
    recomputeTrainEnabled();
  refreshTrainRec();
    updateOpenSamplesPanel(openSourceClass);
    toast('1 sample captured.');
  }} finally {{
    captureInFlight = false;
    syncSourceActionButtons(openSourceClass);
    if (shouldResumePreviewPredict && previewInputOn) {{
      previewSource = previewSourceBeforeCapture === 'device' ? 'device' : 'webcam';
      startPreviewPredictLoop();
    }}
  }}
}}
function toast(msg, ms = 2400) {{
  const el = document.getElementById('toast');
  if (!msg) return;
  el.textContent = msg;
  el.style.display = 'block';
  el._toastClearAt = Date.now() + Math.max(600, Number(ms || 0));
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {{ el.style.display = 'none'; }}, Math.max(600, Number(ms || 0)));
}}
function bumpToastUntil(clearAt) {{
  const el = document.getElementById('toast');
  if (!el || !el.style.display || el.style.display === 'none') return;
  const target = Math.max(Number(el._toastClearAt || 0), Number(clearAt || 0));
  el._toastClearAt = target;
  clearTimeout(toast._t);
  const ms = Math.max(600, target - Date.now());
  toast._t = setTimeout(() => {{ el.style.display = 'none'; }}, ms);
}}
function themeVar(name, fallback = '') {{
  try {{
    const root = document.documentElement;
    const value = window.getComputedStyle(root).getPropertyValue(name);
    return String(value || '').trim() || String(fallback || '');
  }} catch (e) {{
    return String(fallback || '');
  }}
}}
function showConfirmDialog(titleText, bodyText, confirmText = 'Confirm') {{
  return new Promise((resolve) => {{
    let hostDocument = document;
    try {{
      if (window.parent && window.parent !== window && window.parent.document && window.parent.document.body) {{
        hostDocument = window.parent.document;
      }}
    }} catch (e) {{}}
    const overlay = hostDocument.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.inset = '0';
    overlay.style.background = themeVar('--overlay', 'rgba(32,33,36,0.56)');
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.zIndex = '2400';
    overlay.style.padding = '20px';

    const card = hostDocument.createElement('div');
    card.style.width = 'min(480px, 92vw)';
    card.style.background = themeVar('--surface-elevated', '#fff');
    card.style.borderRadius = '14px';
    card.style.boxShadow = '0 14px 40px rgba(0,0,0,0.28)';
    card.style.padding = '16px';
    card.style.color = themeVar('--text', '#202124');

    const title = hostDocument.createElement('div');
    title.style.fontSize = '16px';
    title.style.fontWeight = '700';
    title.style.marginBottom = '10px';
    title.textContent = String(titleText || 'Are you sure?');

    const body = hostDocument.createElement('div');
    body.style.fontSize = '13px';
    body.style.lineHeight = '1.5';
    body.style.whiteSpace = 'pre-wrap';
    body.textContent = String(bodyText || '');

    const actions = hostDocument.createElement('div');
    actions.style.display = 'flex';
    actions.style.justifyContent = 'flex-end';
    actions.style.gap = '10px';
    actions.style.marginTop = '14px';

    const cancelBtn = hostDocument.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.border = `1px solid ${{themeVar('--input-border', 'rgba(0,0,0,0.16)')}}`;
    cancelBtn.style.background = themeVar('--input-bg', '#fff');
    cancelBtn.style.color = themeVar('--text', '#202124');
    cancelBtn.style.padding = '9px 14px';
    cancelBtn.style.borderRadius = '10px';
    cancelBtn.style.cursor = 'pointer';

    const okBtn = hostDocument.createElement('button');
    okBtn.type = 'button';
    okBtn.textContent = String(confirmText || 'Confirm');
    okBtn.style.border = '0';
    okBtn.style.background = themeVar('--danger', '#d93025');
    okBtn.style.color = themeVar('--text-inverse', '#fff');
    okBtn.style.padding = '9px 14px';
    okBtn.style.borderRadius = '10px';
    okBtn.style.cursor = 'pointer';

    const cleanup = (result) => {{
      try {{ overlay.remove(); }} catch (e) {{}}
      resolve(!!result);
    }};
    cancelBtn.onclick = () => cleanup(false);
    okBtn.onclick = () => cleanup(true);
    overlay.onclick = (e) => {{
      if (e.target === overlay) cleanup(false);
    }};

    actions.appendChild(cancelBtn);
    actions.appendChild(okBtn);
    card.appendChild(title);
    card.appendChild(body);
    card.appendChild(actions);
    overlay.appendChild(card);
    hostDocument.body.appendChild(overlay);
  }});
}}
function showOverwriteConfirmDialog(conflicts) {{
  return new Promise((resolve) => {{
    let hostWindow = window;
    let hostDocument = document;
    try {{
      if (window.parent && window.parent !== window && window.parent.document && window.parent.document.body) {{
        hostWindow = window.parent;
        hostDocument = window.parent.document;
      }}
    }} catch (e) {{}}
    const overlay = hostDocument.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.inset = '0';
    overlay.style.background = themeVar('--overlay', 'rgba(32,33,36,0.56)');
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.zIndex = '2400';
    overlay.style.padding = '20px';

    const card = hostDocument.createElement('div');
    card.style.width = 'min(560px, 92vw)';
    card.style.background = themeVar('--surface-elevated', '#fff');
    card.style.borderRadius = '14px';
    card.style.boxShadow = '0 14px 40px rgba(0,0,0,0.28)';
    card.style.padding = '16px';
    card.style.color = themeVar('--text', '#202124');

    const title = hostDocument.createElement('div');
    title.style.fontSize = '16px';
    title.style.fontWeight = '700';
    title.style.marginBottom = '10px';
    title.textContent = 'Overwrite existing export files?';

    const body = hostDocument.createElement('div');
    body.style.fontSize = '13px';
    body.style.lineHeight = '1.5';
    body.style.whiteSpace = 'pre-wrap';
    let msg = 'Files already exist in this export folder and will be overwritten.';
    const nl = String.fromCharCode(10);
    if (Array.isArray(conflicts) && conflicts.length) {{
      msg += nl + nl + 'Existing files:' + nl + '- ' + conflicts.join(nl + '- ');
    }}
    body.textContent = msg;

    const actions = hostDocument.createElement('div');
    actions.style.display = 'flex';
    actions.style.justifyContent = 'flex-end';
    actions.style.gap = '10px';
    actions.style.marginTop = '14px';

    const cancelBtn = hostDocument.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.border = '0';
    cancelBtn.style.borderRadius = '8px';
    cancelBtn.style.padding = '10px 12px';
    cancelBtn.style.fontWeight = '700';
    cancelBtn.style.cursor = 'pointer';
    cancelBtn.style.background = themeVar('--surface-soft', '#eef1f5');
    cancelBtn.style.color = themeVar('--text', '#202124');

    const okBtn = hostDocument.createElement('button');
    okBtn.type = 'button';
    okBtn.textContent = 'Overwrite';
    okBtn.style.border = '0';
    okBtn.style.borderRadius = '8px';
    okBtn.style.padding = '10px 12px';
    okBtn.style.fontWeight = '700';
    okBtn.style.cursor = 'pointer';
    okBtn.style.background = themeVar('--blue', '#1a73e8');
    okBtn.style.color = themeVar('--text-inverse', '#fff');

    let done = false;
    function finish(value) {{
      if (done) return;
      done = true;
      try {{ overlay.remove(); }} catch (e) {{}}
      resolve(!!value);
    }}

    cancelBtn.onclick = () => finish(false);
    okBtn.onclick = () => finish(true);
    overlay.onclick = (e) => {{
      if (e.target === overlay) finish(false);
    }};

    actions.appendChild(cancelBtn);
    actions.appendChild(okBtn);
    card.appendChild(title);
    card.appendChild(body);
    card.appendChild(actions);
    overlay.appendChild(card);
    try {{
      if (hostWindow && typeof hostWindow.scrollTo === 'function') hostWindow.scrollTo(0, 0);
    }} catch (e) {{}}
    hostDocument.body.appendChild(overlay);
  }});
}}
function renderNavMenu() {{
  const menu = document.getElementById('navMenu');
  if (!menu) return;
  if (navMenuOpen) menu.classList.add('open');
  else menu.classList.remove('open');
  try {{
    window.requestAnimationFrame(() => updateFlow());
  }} catch (e) {{}}
}}
function closeToolbarDropdowns() {{
  try {{
    const d1 = document.getElementById('menuFileDropdown');
    const d2 = document.getElementById('menuEditDropdown');
    if (d1) d1.classList.remove('open');
    if (d2) d2.classList.remove('open');
  }} catch (e) {{}}
}}
function closeNavMenu() {{
  navMenuOpen = false;
  renderNavMenu();
  closeToolbarDropdowns();
}}
function toggleNavMenu() {{
  navMenuOpen = !navMenuOpen;
  renderNavMenu();
}}
function openToolbarDropdown(id) {{
  closeToolbarDropdowns();
  const el = document.getElementById(id);
  if (el) el.classList.add('open');
}}
function openGlobalSettings() {{
  const modal = document.getElementById('globalSettingsModal');
  if (!modal) return;
  try {{
    const wSel = document.getElementById('settingsWebcam');
    if (wSel && Array.isArray(STATE.webcam_options)) {{
      wSel.innerHTML = '';
      STATE.webcam_options.forEach((opt, i) => {{
        const o = document.createElement('option');
        o.value = String(opt.index ?? i);
        o.textContent = String(opt.label || opt.name || `Camera ${{i + 1}}`);
        if (String(opt.index ?? i) === String(STATE.current_webcam_index ?? 0)) o.selected = true;
        wSel.appendChild(o);
      }});
      if (!wSel.options.length) {{
        const o = document.createElement('option');
        o.value = '0'; o.textContent = 'No cameras detected';
        wSel.appendChild(o);
      }}
    }}
    const pSel = document.getElementById('settingsSerialPort');
    if (pSel && Array.isArray(STATE.serial_ports)) {{
      const cur = String(STATE.current_serial_port || '');
      // Always start from a clean slate — appending without clearing causes
      // duplicate port entries every time the modal is reopened.
      pSel.innerHTML = '';
      // Deduplicate by device path. Some serial adapters enumerate the same
      // port under multiple interface names; the first occurrence wins.
      const seen = new Set();
      const uniquePorts = [];
      for (const opt of STATE.serial_ports) {{
        const key = String(opt.path || opt.device || '').trim();
        if (!key || seen.has(key)) continue;
        seen.add(key);
        uniquePorts.push(opt);
      }}
      for (const opt of uniquePorts) {{
        const o = document.createElement('option');
        o.value = String(opt.path || opt.device || '');
        const lbl = String(opt.path || opt.device || opt.label || '');
        const desc = String(opt.description || opt.label || '').trim();
        o.textContent = desc ? `${{lbl}} (${{desc}})` : lbl || '(unknown)';
        if (o.value === cur) o.selected = true;
        pSel.appendChild(o);
      }}
      // Always offer a "not connected" sentinel. Insert it at the top so it
      // is the default when nothing is selected.
      const notConnected = document.createElement('option');
      notConnected.value = '';
      notConnected.textContent = '— Not connected —';
      pSel.insertBefore(notConnected, pSel.firstChild);
      if (!cur) notConnected.selected = true;
      // Restore the user's previous selection in case it was lost during the
      // clear + rebuild above (e.g. if the selected port had been appended
      // before the sentinel was inserted).
      if (cur) pSel.value = cur;
    }}
    const fps = document.getElementById('settingsFps');
    if (fps) fps.value = Number(STATE.fps ?? 8.0);
    // ...then ask the SERVER what it is actually pacing at.  fps is per
    // session and outlives the page: set it to 20, reload, and the seeded
    // value above would claim 8.0 while the record thread kept running at 20.
    // The field is the only readout of the capture rate, so it reads it back
    // from the authority rather than from a client constant.
    syncFpsFromServer();
    const bSel = document.getElementById('settingsSerialBaud');
    if (bSel) {{
      const v = String(STATE.current_serial_baud ?? 115200);
      for (const o of Array.from(bSel.options)) {{ if (o.value === v) o.selected = true; }}
    }}
    const fs = document.getElementById('settingsFrameSide');
    if (fs) fs.value = Number(STATE.current_serial_frame_side ?? 96);
    const ch = document.getElementById('settingsChannels');
    if (ch) {{
      const v = String(STATE.current_serial_channels ?? 1);
      for (const o of Array.from(ch.options)) {{ if (o.value === v) o.selected = true; }}
    }}
    const sw = document.getElementById('settingsSerialSync');
    if (sw) sw.value = String(STATE.current_serial_sync ?? 'AA 55 AA');
    // ── Advanced: EMNIST template samples (developer path) ──
    const autoSeedBox = document.getElementById('settingsAutoSeedEmnist');
    if (autoSeedBox) autoSeedBox.checked = !!autoSeedEmnistSamples;
    const seedHint = document.getElementById('settingsSeedEmnistHint');
    if (seedHint) {{
      const chars = (Array.isArray(STATE.classes) ? STATE.classes : []).filter(isTemplateCharClass);
      seedHint.textContent = chars.length
        ? `Fills every EMPTY class among ${{chars.join(', ')}} with 24 dark-ink-on-light template captures. Classes with real camera samples are kept.`
        : 'Type a character (A-Z, 0-9) in the target bar first — there are no character classes to fill yet.';
    }}
    // ── Preview group: input source + OOD thresholds (reuse existing state vars) ──
    const srcSel = document.getElementById('settingsPreviewSource');
    if (srcSel) {{
      const v = String(previewSource || 'webcam');
      for (const o of Array.from(srcSel.options)) {{ if (o.value === v) o.selected = true; }}
    }}
    const oodMin = document.getElementById('settingsOodSignPctMin');
    if (oodMin) oodMin.value = Number(previewOodSignPctMin ?? 0.3);
    const oodMax = document.getElementById('settingsOodSignPctMax');
    if (oodMax) oodMax.value = Number(previewOodSignPctMax ?? 70.0);
    const oodProb = document.getElementById('settingsOodMaxProbMin');
    if (oodProb) oodProb.value = Number(previewOodMaxProbMin ?? 0.40);
    const oodEnt = document.getElementById('settingsOodEntropyRatioMax');
    if (oodEnt) oodEnt.value = Number(previewOodEntropyRatioMax ?? 0.85);
    // ── Training Hyperparameters group (live TrainConfig state) ──
    try {{
      if (typeof syncAdvancedInlineInputs === 'function') syncAdvancedInlineInputs();
      if (typeof refreshRecHints === 'function') refreshRecHints();
    }} catch (e) {{ console.error('[settings] train cfg sync failed', e); }}
  }} catch (err) {{ console.error('[settings] populate failed', err); }}
  modal.classList.add('open');
}}
function closeGlobalSettings() {{
  const modal = document.getElementById('globalSettingsModal');
  if (modal) modal.classList.remove('open');
}}
function restorePreferences() {{
  try {{
    const raw = window.localStorage.getItem(preferencesStorageKey);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (!data || typeof data !== 'object') return;
    if (Number.isFinite(Number(data.maxSamplesDisplay)) && Number(data.maxSamplesDisplay) > 0) {{
      PREF_MAX_SAMPLES_DISPLAY = Math.max(1, Math.min(100, Math.round(Number(data.maxSamplesDisplay))));
    }}
  }} catch (e) {{}}
}}
function persistPreferences() {{
  try {{
    window.localStorage.setItem(preferencesStorageKey, JSON.stringify({{
      maxSamplesDisplay: Number(PREF_MAX_SAMPLES_DISPLAY),
    }}));
  }} catch (e) {{}}
}}
function openPreferences() {{
  const modal = document.getElementById('preferencesModal');
  if (!modal) return;
  const input = document.getElementById('prefMaxSamples');
  if (input) input.value = Number(PREF_MAX_SAMPLES_DISPLAY);
  modal.classList.add('open');
}}
function closePreferences() {{
  const modal = document.getElementById('preferencesModal');
  if (modal) modal.classList.remove('open');
}}
// The capture rate lives in SessionConfig, not in the page, so a reload has to
// read it back before it can show it.  Fire-and-forget: the seeded value stays
// on screen if the controller cannot be reached, and a failed read must never
// throw into the load handler that calls this.
async function syncFpsFromServer() {{
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}`);
    const data = await res.json().catch(() => ({{}}));
    const v = Number(data && data.fps);
    if (!Number.isFinite(v) || v <= 0) return;
    STATE.fps = v;
    const el = document.getElementById('settingsFps');
    if (el) el.value = v;
  }} catch (e) {{}}
}}
function savePreferences() {{
  try {{
    const input = document.getElementById('prefMaxSamples');
    if (input) {{
      const v = Number(input.value);
      if (Number.isFinite(v) && v > 0) {{
        PREF_MAX_SAMPLES_DISPLAY = Math.max(1, Math.min(100, Math.round(v)));
      }}
    }}
    persistPreferences();
    closePreferences();
    render();
    toast('Preferences saved.');
  }} catch (e) {{
    toast('Failed to save preferences.');
  }}
}}
async function saveGlobalSettings() {{
  try {{
    const payload = {{}};
    const wSel = document.getElementById('settingsWebcam');
    if (wSel) payload.webcam_index = Number(wSel.value);
    const pSel = document.getElementById('settingsSerialPort');
    if (pSel) payload.serial_port = String(pSel.value || '');
    const fps = document.getElementById('settingsFps');
    // Only carry a fps we can actually read.  `Number('' || 0)` is 0, and the
    // clamp below turns that into 1 — harmless while this only fed client
    // state, but fps now paces the record thread, so an emptied field would
    // silently pin capture to one sample per second.  Absent means "leave the
    // session's rate alone".
    if (fps) {{
      const fpsValue = Number(fps.value);
      if (isFinite(fpsValue) && fpsValue > 0) {{
        payload.fps = Math.max(1, Math.min(60, fpsValue));
      }}
    }}
    const bSel = document.getElementById('settingsSerialBaud');
    if (bSel) payload.serial_baud = Number(bSel.value);
    const fs = document.getElementById('settingsFrameSide');
    if (fs) payload.serial_frame_side = Math.max(32, Math.min(512, Number(fs.value || 96)));
    const ch = document.getElementById('settingsChannels');
    if (ch) payload.serial_channels = Number(ch.value);
    const sw = document.getElementById('settingsSerialSync');
    if (sw) payload.serial_sync = String(sw.value || 'AA 55 AA').trim().toUpperCase() || 'AA 55 AA';

    // ── Preview group: input source + OOD thresholds (reuse existing state vars + persistPreviewState) ──
    const srcSel = document.getElementById('settingsPreviewSource');
    if (srcSel) {{
      const next = String(srcSel.value || 'webcam');
      previewSource = (next === 'device' || next === 'upload') ? next : 'webcam';
    }}
    const oodMinEl = document.getElementById('settingsOodSignPctMin');
    if (oodMinEl) previewOodSignPctMin = Math.max(0, Math.min(100, Number(oodMinEl.value) || 0.3));
    const oodMaxEl = document.getElementById('settingsOodSignPctMax');
    if (oodMaxEl) previewOodSignPctMax = Math.max(0, Math.min(100, Number(oodMaxEl.value) || 70.0));
    const oodProbEl = document.getElementById('settingsOodMaxProbMin');
    if (oodProbEl) previewOodMaxProbMin = Math.max(0, Math.min(1, Number(oodProbEl.value) || 0.40));
    const oodEntEl = document.getElementById('settingsOodEntropyRatioMax');
    if (oodEntEl) previewOodEntropyRatioMax = Math.max(0, Math.min(1, Number(oodEntEl.value) || 0.85));
    persistPreviewState();
    await pushPreviewOodToBackend();

    // Resolve webcam uid from the selected option
    let webcamUid = '';
    if (wSel && Array.isArray(STATE.webcam_options)) {{
      const selIdx = Number(payload.webcam_index || 0);
      const found = STATE.webcam_options.find((o) => o && Number(o.index ?? -1) === selIdx);
      if (found) webcamUid = String(found.unique_id || found.uid || found.deviceId || '');
    }}

    // Apply locally first so the UI reflects the new values immediately
    currentWebcamIndex = Number(payload.webcam_index || 0);
    currentWebcamUid = webcamUid;
    currentSerialPort = String(payload.serial_port || '');
    currentSerialBaud = Number(payload.serial_baud || 115200);
    currentSerialFrameSide = Number(payload.serial_frame_side || 96);
    currentSerialChannels = Number(payload.serial_channels || 1);
    currentSerialSync = String(payload.serial_sync || 'AA 55 AA');
    STATE.current_webcam_index = currentWebcamIndex;
    STATE.current_webcam_id = currentWebcamUid;
    STATE.current_serial_port = currentSerialPort;
    STATE.current_serial_baud = currentSerialBaud;
    STATE.current_serial_frame_side = currentSerialFrameSide;
    STATE.current_serial_channels = currentSerialChannels;
    STATE.current_serial_sync = currentSerialSync;
    if (payload.fps != null) STATE.fps = payload.fps;

    // Persist to the controller via /live/config — no page reload
    const params = new URLSearchParams();
    params.set('session', String(STATE.session || ''));
    params.set('webcam_index', String(currentWebcamIndex));
    if (webcamUid) params.set('webcam_id', webcamUid);
    params.set('serial_port', currentSerialPort);
    params.set('serial_baud', String(currentSerialBaud));
    params.set('serial_frame_side', String(currentSerialFrameSide));
    params.set('serial_channels', String(currentSerialChannels));
    params.set('serial_sync', currentSerialSync);
    // FPS is the hold-capture PACING — the record thread sleeps 1/fps between
    // samples — so it only means anything once it reaches the controller's
    // SessionConfig.  Updating STATE.fps alone left this control inert: the
    // session kept the default and capture could never be made faster than it.
    if (payload.fps != null) params.set('fps', String(payload.fps));
    const res = await fetch(`${{baseUrl}}/live/config?${{params.toString()}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to save settings.');

    // Re-render so the open source panel reflects the new webcam/port labels
    // (currentCamLabel / currentPortLabel are computed inside render()).
    // render() also re-opens the live source via ensureOpenSourceLive().
    try {{
      if (typeof render === 'function') render();
    }} catch (e) {{
      console.warn('[settings] render after save failed', e);
    }}

    closeGlobalSettings();
    toast('Settings saved.');
  }} catch (err) {{
    console.error('[settings] save failed', err);
    toast(`Settings save failed: ${{String(err && err.message ? err.message : err)}}`);
  }}
}}
// ── Settings modal keyboard shortcuts: Escape = Cancel, Enter = Save ──
// The modal is rebuilt by every render(), so the listener lives on `window`
// rather than on any node inside it: a rebuild can neither detach it nor
// stack a second copy.  The flag makes installation idempotent even if this
// script is evaluated again in the same window.
function settingsModalIsOpen() {{
  const modal = document.getElementById('globalSettingsModal');
  if (!modal || modal.hidden || !modal.classList.contains('open')) return false;
  const style = window.getComputedStyle(modal);
  return style.display !== 'none' && style.visibility !== 'hidden';
}}
function installSettingsKeyboardShortcuts() {{
  if (window.__tmSettingsShortcutBound) return;
  window.__tmSettingsShortcutBound = true;
  window.addEventListener('keydown', (e) => {{
    const key = String(e.key || '');
    const isEscape = key === 'Escape' || key === 'Esc';
    const isEnter = key === 'Enter';
    if (!isEscape && !isEnter) return;
    if (!settingsModalIsOpen()) return;
    if (e.defaultPrevented) return;

    if (isEscape) {{
      // Cancel, never Save.  Owning the key outright also keeps the
      // workspace Escape stack and window.tmTour from acting on a press
      // that has already dismissed the modal — the tour hands Escape back
      // while a layer is open, but this listener is registered first.
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      closeGlobalSettings();
      return;
    }}

    // Enter saves, except where Enter means something else.
    if (e.isComposing || e.keyCode === 229) return;   // IME candidate window
    if (e.shiftKey || e.ctrlKey || e.altKey || e.metaKey) return;
    const active = document.activeElement;
    const tag = String((active && active.tagName) || '').toUpperCase();
    if (tag === 'TEXTAREA') return;                    // newline, not Save
    if (active && active.isContentEditable) return;
    if (tag === 'BUTTON') {{
      const id = String((active && active.id) || '');
      // Enter on Cancel / Close / Reset must never commit the form.
      const dismiss = id === 'settingsCancel' || id === 'settingsClose' ||
        id === 'settingsReset' ||
        /(?:^|[-_])(cancel|close|reset)(?:$|[-_])/i.test(id);
      if (dismiss) return;
    }}

    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
    const saveBtn = document.getElementById('settingsSave');
    if (saveBtn) saveBtn.click();
    else saveGlobalSettings();
  }}, true);
}}
installSettingsKeyboardShortcuts();
function applyProjectState(state) {{
  const s = state && typeof state === 'object' ? state : {{}};
  // #region debug-point C:apply-project-state
  dbgEvent('C', 'app.py:applyProjectState', '[DEBUG] applyProjectState received state payload', {{
    classes: Array.isArray(s.classes) ? s.classes.slice() : [],
    countKeys: s.counts && typeof s.counts === 'object' ? Object.keys(s.counts) : [],
    previewKeys: s.sample_previews && typeof s.sample_previews === 'object' ? Object.keys(s.sample_previews) : [],
    exportEnabled: String(s.export_enabled || '0'),
  }});
  // #endregion
  // A new/opened project gets a fresh auto-seed ledger: a character class
  // that was already offered in the previous project must be eligible here.
  autoSeedEmnistAttempted = new Set();
  stopHoldSyncLoop();
  stopPreviewPredictLoop();
  stopPreviewLoop();
  clearPreviewBlob();
  previewSettingsOpen = false;
  sourceSettingsOpen = false;
  navMenuOpen = false;
  renderNavMenu();
  holdRecording = false;
  holdRecordClass = '';
  holdRecordSource = '';
  openSourceClass = '';
  openSourceKind = '';
  stopClassPreprocessPreviewWork();
  classPreprocessOpen = false;
  classPreprocessClass = '';
  classPreprocessDraft = null;
  classPreprocessSampleIndex = 0;
  classPreprocessProcessedSrc = '';
  classPreprocessBusy = false;
  classPreprocessAutoCrop = null;
  clearOpenSourceState();
  previewInputOn = false;
  persistPreviewState();
  trainInFlight = false;
  // Kill any in-flight train poll from the previous project and reset the
  // train UI + preview pane so nothing stale survives the switch.
  trainPollToken += 1;
  showTrainProgress(false, 0, '');
  const projPane = document.getElementById('previewPane');
  if (projPane) {{ projPane.removeAttribute('data-ready'); }}
  // Preview transforms/gates back to defaults; the model meta re-applies
  // per-project settings on the next predict.
  previewPreprocessMode = 'auto_by_label';
  previewThreshUserModified = false;
  previewOodSignPctMin = 0.3;
  previewOodSignPctMax = 70.0;
  previewOodMaxProbMin = 0.40;
  previewOodEntropyRatioMax = 0.85;
  const cls = Array.isArray(s.classes) ? s.classes.slice() : [];
  STATE.classes = cls;
  const cnt = s.counts && typeof s.counts === 'object' ? s.counts : {{}};
  const nextCounts = {{}};
  for (const [k, v] of Object.entries(cnt)) nextCounts[String(k)] = Number(v || 0);
  STATE.counts = nextCounts;
  const prev = s.sample_previews && typeof s.sample_previews === 'object' ? s.sample_previews : {{}};
  const nextPrev = {{}};
  for (const [k, v] of Object.entries(prev)) nextPrev[String(k)] = normalizePreviewList(v);
  STATE.sample_previews = nextPrev;
  const processedPrev = s.processed_previews && typeof s.processed_previews === 'object' ? s.processed_previews : {{}};
  const nextProcessedPrev = {{}};
  for (const [k, v] of Object.entries(processedPrev)) nextProcessedPrev[String(k)] = normalizePreviewList(v);
  STATE.processed_previews = nextProcessedPrev;
  STATE.class_preprocess = s.class_preprocess && typeof s.class_preprocess === 'object' ? Object.assign({{}}, s.class_preprocess) : {{}};
  STATE.sample_preprocess = normalizeSamplePreprocessMap(s.sample_preprocess);
  STATE.export_enabled = String(s.export_enabled || '0') === '1';
  // The opened file states what IT is trained with, so its train_cfg
  // REPLACES this one rather than merging into it.  A merge kept every knob
  // the previous project had set and the new file did not mention — an OCR
  // session's EMNIST mixing, ink polarity and sign_sequence survived into an
  // Image project, which has no use for any of them and would have trained
  // with them anyway.  This build's defaults fill in BEHIND the file's own
  // values, so a legacy archive with a partial config still leaves a
  // complete one rather than a train_cfg with holes in it.
  if (s.train_cfg && typeof s.train_cfg === 'object') {{
    STATE.train_cfg = Object.assign({{}}, STATE.train_cfg_defaults || {{}}, s.train_cfg);
    persistTrainCfgStorage();
  }}
  // Which workspace this is.  Every OCR-only affordance (the Edit menu's
  // character entry, the glyph settings item, the OCR blocks in the Advanced
  // panel) is gated on STATE.project_type, and the project being opened is
  // the authority on it — the SPA is mounted once and does NOT re-run the
  // server page on an in-workspace open, so without this an OCR project
  // opened from an image session would keep the image session's menu.
  // Absent (a legacy file with no recorded type, /project/reset) -> keep the
  // current one.
  if (typeof s.project_type === 'string' && s.project_type) {{
    STATE.project_type = s.project_type;
  }}
  // ...and re-sync unconditionally, because the panel on screen must agree
  // with whichever of the two above ended up in effect.
  syncOcrPanelVisibility();
  // The payload's own flat keys first (the v1 contract), then the grouped
  // block that supersedes them — it carries the four OOD gates the flat keys
  // never did.
  if (typeof s.bg_dark_thresh === 'number') previewDarkThresh = s.bg_dark_thresh;
  if (typeof s.bg_lum_thresh === 'number') previewLumThresh = s.bg_lum_thresh;
  applyPreviewState(s.preview);
  persistPreviewState();
  recomputeTrainEnabled();
  refreshTrainRec();
  STATE.trainClassAccuracies = null;
  STATE._trained_fingerprint = null;
  if ('current_project_path' in s) {{
    if (typeof s.current_project_path === 'string' && s.current_project_path) {{
      STATE.currentProjectPath = s.current_project_path;
    }} else {{
      STATE.currentProjectPath = null;
    }}
  }}
  try {{
    document.documentElement.scrollTop = 0;
    document.documentElement.scrollLeft = 0;
    document.body.scrollTop = 0;
    document.body.scrollLeft = 0;
    window.scrollTo(0, 0);
  }} catch (e) {{}}
  scheduleLayoutResync();
}}
async function exportDataset() {{
  try {{
    const pickRes = await fetch(`${{baseUrl}}/export/pick_dir?session=${{encodeURIComponent(STATE.session)}}`);
    const pick = await pickRes.json().catch(() => ({{ok:'0'}}));
    if (!pickRes.ok || pick.ok !== '1') {{
      if (pick && pick.canceled === '1') return;
      throw new Error(pick.error || 'Unable to choose folder.');
    }}
    const exportDir = String(pick.export_dir || '').trim();
    if (!exportDir) return;
    const res = await fetch(`${{baseUrl}}/dataset/export`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session, export_dir: exportDir}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Dataset export failed.');
    toast(`Dataset exported to: ${{String(data.export_dir || exportDir)}}`);
  }} catch (e) {{
    toast(String(e && e.message ? e.message : e));
  }}
}}
function collectProjectState() {{
  // Everything a save sends, in one place.  saveProjectAs and saveProjectQuick
  // previously built this object twice, character for character; a field the
  // project format gained (the Preview gates, the workspace type) could only
  // ever be added to one of them.
  return {{
    // STATE.train_cfg IS the training configuration — every TrainConfig field
    // the SPA knows about, because the server seeded it from the dataclass.
    // class_preprocess rides along inside it for the run's own record, and
    // again at the root where the backend reads it as authority.
    train_cfg: Object.assign({{}}, STATE.train_cfg || {{}}, {{
      class_preprocess: Object.assign({{}}, STATE.class_preprocess || {{}})
    }}),
    class_preprocess: Object.assign({{}}, STATE.class_preprocess || {{}}),
    sample_preprocess: normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}}),
    // Which workspace saved this, so the file routes back to the same one even
    // if its manifest is ever lost.  The backend reconciles it against the
    // session config and the manifest before writing.
    project_type: String(STATE.project_type || ''),
    preview: previewStateForProject(),
    // Kept at the root as well: they were the v1 contract, and a project saved
    // by this build should still be readable by one that only knows them.
    bg_dark_thresh: Number((previewDarkThresh ?? 0)),
    bg_lum_thresh: Number((previewLumThresh ?? 100))
  }};
}}
async function saveProjectAs() {{
  try {{
    const pickRes = await fetch(`${{baseUrl}}/project/pick_save?session=${{encodeURIComponent(STATE.session)}}`);
    const pick = await pickRes.json().catch(() => ({{ok:'0'}}));
    if (!pickRes.ok || pick.ok !== '1') {{
      if (pick && pick.canceled === '1') return;
      throw new Error(pick.error || 'Unable to choose file.');
    }}
    const savePath = String(pick.save_path || '').trim();
    if (!savePath) return;
    const res = await fetch(`${{baseUrl}}/project/save`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        save_path: savePath,
        project_state: collectProjectState()
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Save failed.');
    const finalPath = String(data.save_path || savePath);
    STATE.currentProjectPath = finalPath;
    toast(`Saved: ${{finalPath}}`);
  }} catch (e) {{
    toast(String(e && e.message ? e.message : e));
  }}
}}
async function saveProjectQuick() {{
  const existing = String(STATE.currentProjectPath || '').trim();
  if (existing) {{
    try {{
      const res = await fetch(`${{baseUrl}}/project/save`, {{
        method: 'POST',
        headers: {{'Content-Type':'application/json'}},
        body: JSON.stringify({{
          session: STATE.session,
          save_path: existing,
          project_state: collectProjectState()
        }})
      }});
      const data = await res.json().catch(() => ({{ok:'0'}}));
      if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Save failed.');
      const finalPath = String(data.save_path || existing);
      STATE.currentProjectPath = finalPath;
      toast(`Saved: ${{finalPath}}`);
      return;
    }} catch (e) {{
      toast(String(e && e.message ? e.message : e));
      return;
    }}
  }}
  await saveProjectAs();
}}
async function newProject() {{
  const ok = window.confirm('Start a new project? Unsaved changes will be lost.');
  if (!ok) return;
  try {{
    const res = await fetch(`${{baseUrl}}/project/reset`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session, confirm: '1'}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'New project failed.');
    const resetMerged = Object.assign({{}}, data.state || {{}}, {{current_project_path: null}});
    applyProjectState(resetMerged);
    render();
    scheduleLayoutResync();
    requestShellLayoutRefresh('new-project');
    toast('New project.');
  }} catch (e) {{
    toast(String(e && e.message ? e.message : e));
  }}
}}
function toastProjectFormat(state) {{
  // What the archive said about itself on open.  An in-workspace open does
  // not re-run the server page, so the Start Page's notice never reaches
  // here — this is the same message raised from the payload instead.  The
  // point is not ceremony: a v1 file cannot state how it mixed EMNIST, so a
  // retrain would silently use this build's defaults, and the user should
  // hear that from the app rather than from a model that behaves differently.
  const fmt = state && state.project_format;
  if (!fmt || typeof fmt !== 'object') return;
  const missing = Array.isArray(fmt.missing_train_cfg_fields) ? fmt.missing_train_cfg_fields : [];
  if (!missing.length) {{
    if (fmt.legacy) toast('Opened a project saved before this version.');
    return;
  }}
  toast(`Opened a v${{String(fmt.version || '1.0')}} project: ${{missing.length}} training settings it predates came back at their defaults. Check Advanced settings before retraining.`);
}}
async function openProject() {{
  try {{
    // #region debug-point A:open-project-start
    dbgEvent('A', 'app.py:openProject', '[DEBUG] openProject start', {{
      session: String(STATE.session || ''),
      href: parentUrl().toString(),
      innerWidth: window.innerWidth || 0,
      innerHeight: window.innerHeight || 0,
      classes: Array.isArray(STATE.classes) ? STATE.classes.slice() : [],
    }});
    // #endregion
    const pickRes = await fetch(`${{baseUrl}}/project/pick_open?session=${{encodeURIComponent(STATE.session)}}`);
    const pick = await pickRes.json().catch(() => ({{ok:'0'}}));
    if (!pickRes.ok || pick.ok !== '1') {{
      if (pick && pick.canceled === '1') return;
      throw new Error(pick.error || 'Unable to choose file.');
    }}
    const openPath = String(pick.open_path || '').trim();
    if (!openPath) return;
    const res = await fetch(`${{baseUrl}}/project/open`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session, open_path: openPath}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Open failed.');
    const statePayload = data && data.state && typeof data.state === 'object' ? data.state : {{}};
    // #region debug-point D:open-project-response
    dbgEvent('D', 'app.py:openProject', '[DEBUG] openProject response received', {{
      openPath,
      stateClasses: Array.isArray(statePayload.classes) ? statePayload.classes.slice() : [],
      statePreviewKeys: statePayload.sample_previews && typeof statePayload.sample_previews === 'object' ? Object.keys(statePayload.sample_previews) : [],
      stateCounts: statePayload.counts || {{}},
    }});
    // #endregion
    const mergedState = Object.assign({{}}, statePayload, {{current_project_path: openPath}});
    applyProjectState(mergedState);
    toastProjectFormat(statePayload);
    render();
    scheduleLayoutResync();
    requestShellLayoutRefresh('open-project');
    window.setTimeout(() => {{
      // #region debug-point A:open-project-post-render-1
      dbgEvent('A', 'app.py:openProject', '[DEBUG] openProject post-render metrics (120ms)', {{
        innerWidth: window.innerWidth || 0,
        innerHeight: window.innerHeight || 0,
        docClientHeight: document.documentElement ? document.documentElement.clientHeight : 0,
        docScrollHeight: document.documentElement ? document.documentElement.scrollHeight : 0,
        bodyScrollHeight: document.body ? document.body.scrollHeight : 0,
      }});
      // #endregion
    }}, 120);
    window.setTimeout(() => {{
      // #region debug-point A:open-project-post-render-2
      dbgEvent('A', 'app.py:openProject', '[DEBUG] openProject post-render metrics (500ms)', {{
        innerWidth: window.innerWidth || 0,
        innerHeight: window.innerHeight || 0,
        docClientHeight: document.documentElement ? document.documentElement.clientHeight : 0,
        docScrollHeight: document.documentElement ? document.documentElement.scrollHeight : 0,
        bodyScrollHeight: document.body ? document.body.scrollHeight : 0,
      }});
      // #endregion
    }}, 500);
  }} catch (e) {{
    toast(String(e && e.message ? e.message : e));
  }}
}}
async function resetProject() {{
  const ok = window.confirm('Reset project? This will delete samples, classes, and trained model.');
  if (!ok) return;
  try {{
    const res = await fetch(`${{baseUrl}}/project/reset`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session, confirm: '1'}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Reset failed.');
    const resetMerged = Object.assign({{}}, data.state || {{}}, {{current_project_path: null}});
    applyProjectState(resetMerged);
    render();
    scheduleLayoutResync();
    requestShellLayoutRefresh('reset-project');
  }} catch (e) {{
    toast(String(e && e.message ? e.message : e));
  }}
}}
function defaultExportDir() {{
  if (exportDir) return exportDir;
  return `~/Documents/TFLiteTraining/exports`;
}}
function buildExportRuntimeConfig() {{
  // Collects the Preview-time OOD + preprocessing thresholds so the device
  // export can stamp them into model_settings.h and eliminate hardcoded
  // magic numbers on the ESP32.  Reuses the exact same JS state the predict
  // loop sends to /preview/predict.
  return {{
    ood_sign_min: Number(previewOodSignPctMin ?? 0.3),
    ood_sign_max: Number(previewOodSignPctMax ?? 70.0),
    ood_max_prob: Number(previewOodMaxProbMin ?? 0.40),
    ood_entropy_max: Number(previewOodEntropyRatioMax ?? 0.85),
    bg_dark_thresh: previewThreshUserModified ? Number(previewDarkThresh ?? 0) : 0,
    bg_lum_thresh: previewThreshUserModified ? Number(previewLumThresh ?? 100) : 100,
    // sign_ocr only.  The server ignores these for every other project
    // type; sending them unconditionally keeps this one function the copy
    // point for "what the device is told", instead of a second branch that
    // only sign projects take.  The target sequence is what the model was
    // trained to read, so it must come from the same TrainConfig state the
    // trainer and the preview use.
    sign_sequence: signSequence().join(''),
    sign_sequences: signSequences(),
    capture_frame_side: Number(currentSerialFrameSide || 96),
    reading_order: String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr')),
    // ONE probability threshold: OOD MAX PROB is also the per-slot match
    // floor, and the exporter writes it to both kOodMaxProbMin and
    // kSignMatchMinProb.  No hidden 0.40 floor.
    match_min_prob: Number(previewOodMaxProbMin ?? 0.40),
  }};
}}
async function exportRunWithOverwriteConfirm(exportDirValue, modelNameValue, arrayNameValue) {{
  const runtimeConfig = buildExportRuntimeConfig();
  const payload = {{
    session: STATE.session,
    export_dir: String(exportDirValue || '').trim(),
    model_name: String(modelNameValue || 'tm'),
    array_name: String(arrayNameValue || ''),
    ...runtimeConfig
  }};
  let res = await fetch(`${{baseUrl}}/export/run`, {{
    method: 'POST',
    headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify(payload)
  }});
  let data = await res.json().catch(() => ({{ok:'0'}}));
  if (res.ok && data.ok === '1') {{
    if (Array.isArray(data.legacy_removed) && data.legacy_removed.length) {{
      toast(`Cleaned up legacy duplicates: ${{data.legacy_removed.join(', ')}}`);
    }}
    return data;
  }}
  if (data && data.needs_confirm === '1') {{
    const files = Array.isArray(data.conflicts) ? data.conflicts.slice(0, 8) : [];
    const confirmed = await showOverwriteConfirmDialog(files);
    if (!confirmed) {{
      toast('Export canceled.');
      return {{ok:'0', canceled:'1'}};
    }}
    const overwritePayload = {{
      session: payload.session,
      export_dir: payload.export_dir,
      model_name: payload.model_name,
      array_name: payload.array_name,
      overwrite: true,
      ...runtimeConfig
    }};
    res = await fetch(`${{baseUrl}}/export/run`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify(overwritePayload)
    }});
    data = await res.json().catch(() => ({{ok:'0'}}));
    if (res.ok && data.ok === '1') {{
      toast('Existing export files were overwritten.');
      if (Array.isArray(data.legacy_removed) && data.legacy_removed.length) {{
        toast(`Cleaned up legacy duplicates: ${{data.legacy_removed.join(', ')}}`);
      }}
      return data;
    }}
  }}
  throw new Error((data && data.error) || ('Export failed (status ' + String(res && res.status ? res.status : 0) + ').'));
}}
async function exportModel() {{
  if (!STATE.export_enabled) return;
  const exportBtn = document.getElementById('exportBtn');
  if (exportBtn) exportBtn.disabled = true;
  try {{
    const pickRes = await fetch(`${{baseUrl}}/export/pick_dir?session=${{encodeURIComponent(STATE.session)}}`);
    const pick = await pickRes.json().catch(() => ({{ok:'0'}}));
    if (!pickRes.ok || pick.ok !== '1') {{
      if (pick && pick.canceled === '1') return;
      throw new Error(pick.error || 'Unable to choose folder.');
    }}
    const chosenDir = String(pick.export_dir || '').trim();
    if (!chosenDir) return;
    exportDir = chosenDir;
    persistExportDir();
    const runData = await exportRunWithOverwriteConfirm(
      exportDir,
      String(exportModelName || 'model'),
      String(exportArrayName || 'g_model')
    );
    if (!runData || runData.canceled === '1') return;
    toast(`Exported to: ${{String(runData.export_dir || exportDir)}}`);
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    if (exportBtn) exportBtn.disabled = !STATE.export_enabled;
  }}
}}
function showTrainProgress(show, pct, text) {{
  const wrap = document.getElementById('trainProgress');
  const fill = document.getElementById('trainProgressFill');
  const label = document.getElementById('trainProgressText');
  if (!wrap || !fill || !label) return;
  wrap.style.display = show ? 'block' : 'none';
  const clamped = Math.max(0, Math.min(1, Number(pct || 0)));
  fill.style.width = `${{Math.round(clamped * 1000) / 10}}%`;
  label.textContent = String(text || '');
}}
function syncTrainUi() {{
  const trainBtn = document.getElementById('trainBtn');
  if (!trainBtn) return;
  trainBtn.classList.toggle('enabled', !!STATE.train_enabled);
  trainBtn.disabled = !STATE.train_enabled || !!trainInFlight;
}}
function datasetFingerprint() {{
  try {{
    const cls = Array.isArray(STATE.classes) ? STATE.classes.slice().sort() : [];
    const counts = STATE.counts && typeof STATE.counts === 'object' ? STATE.counts : {{}};
    const entries = cls.map((c) => `${{c}}:${{Number(counts[c] || 0)}}`);
    let total = 0;
    for (const c of cls) total += Number(counts[c] || 0);
    return `${{cls.length}}|${{total}}|${{entries.join(';')}}`;
  }} catch (e) {{ return ''; }}
}}
async function startTrain() {{
  if (trainInFlight) return;
  if (!STATE.train_enabled) return;
  trainInFlight = true;
  syncTrainUi();
  trainPollToken += 1;
  const token = trainPollToken;
  showTrainProgress(true, 0.01, 'Starting...');
  STATE.trainClassAccuracies = null;
  STATE.export_enabled = false;
  renderTrainStatus();
  try {{
    // Global ink-polarity master override: the single Ink Polarity dropdown in
    // the inline Advanced panel wins for every class. Per-class ink_polarity
    // values (if any were stored by older builds) are ignored at train time so
    // the user only ever has to set it once. The override is applied here in
    // the payload, not persisted back to class_preprocess, so reopening a
    // project still shows the original per-class records.
    const globalInkPolarity = normalizeInkPolarity((STATE.train_cfg || {{}}).ink_polarity);
    const classesForTrain = Array.isArray(STATE.classes) ? STATE.classes : [];
    const classPreprocessForTrain = Object.assign({{}}, STATE.class_preprocess || {{}});
    for (const name of classesForTrain) {{
      const existing = classPreprocessForTrain[name] && typeof classPreprocessForTrain[name] === 'object'
        ? classPreprocessForTrain[name] : {{}};
      classPreprocessForTrain[name] = Object.assign({{}}, existing, {{
        ink_polarity: globalInkPolarity,
      }});
    }}
    const res = await fetch(`${{baseUrl}}/train/start`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        cfg: Object.assign({{}}, STATE.train_cfg || {{}}, {{
          ink_polarity: globalInkPolarity,
          class_preprocess: classPreprocessForTrain,
          sample_preprocess: normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}})
        }})
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') {{
      throw new Error(data.error || 'Unable to start training.');
    }}
    while (token === trainPollToken) {{
      const stRes = await fetch(`${{baseUrl}}/train/status?session=${{encodeURIComponent(STATE.session)}}`);
      const stData = await stRes.json().catch(() => ({{ok:'0'}}));
      if (!stRes.ok || stData.ok !== '1') {{
        throw new Error(stData.error || 'Unable to read training status.');
      }}
      const p = Number(stData.progress || 0);
      const msg = String(stData.message || '');
      if (stData.class_accuracies) {{
        if (stData.class_accuracies && Object.keys(stData.class_accuracies).length > 0 &&
            (!STATE.trainClassAccuracies ||
             JSON.stringify(stData.class_accuracies) !== JSON.stringify(STATE.trainClassAccuracies))) {{
          STATE.trainClassAccuracies = stData.class_accuracies;
          showClassAccuracyModal();  // 提示出现: auto-pop, click Close to dismiss
        }} else if (stData.class_accuracies) {{
          STATE.trainClassAccuracies = stData.class_accuracies;
        }}
      }}
      showTrainProgress(true, p, msg);
      if (String(stData.done || '0') === '1') {{
        const err = String(stData.error || '');
        if (err) {{
          toast(err);
          showTrainProgress(true, 1.0, 'Failed.');
        }} else {{
          STATE.export_enabled = true;
          STATE._trained_fingerprint = datasetFingerprint();
          renderTrainStatus();
          // A sign project's synthetic test plate was drawn for the model
          // (and possibly the target) from BEFORE this training run.  Drop
          // it so the card asks for a fresh MATCH plate against the new
          // model — a real upload is left exactly where the user put it.
          if (isSignOcrProject() && previewSource === 'upload' && /^synth_/.test(String(previewUploadFilename || ''))) {{
            previewUploadImageB64 = '';
            previewUploadImageSrc = '';
            previewUploadFilename = '';
            previewLastPredictData = null;
          }}
          // Force the preview pane to re-render (fresh static image / fresh
          // predict loop) so the NEW model is visible immediately.
          const trainedPane = document.getElementById('previewPane');
          if (trainedPane) {{ trainedPane.removeAttribute('data-ready'); }}
          renderPreviewCard();
          // If the upload source was already on, the loop now draws the
          // fresh plate automatically (runPreviewUploadPrediction's
          // no-image branch).
          maybeAutoSynthesizeSignPreview();
          showTrainProgress(true, 1.0, msg || 'Done.');
          if (msg && msg.indexOf('WARNINGS') >= 0) {{
            showTrainWarningModal(msg);
          }} else {{
            toast('Training complete.');
          }}
          const exportBtn = document.getElementById('exportBtn');
          if (exportBtn) exportBtn.disabled = false;
        }}
        break;
      }}
      await new Promise((r) => setTimeout(r, 250));
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
    showTrainProgress(false, 0, '');
  }} finally {{
    if (token === trainPollToken) {{
      trainInFlight = false;
      syncTrainUi();
    }}
  }}
}}
function setAction(action, params) {{
  dbgEvent(action === 'open_source' ? 'B' : (action === 'set_webcam_index' ? 'D' : 'A'), 'app.py:setAction', '[DEBUG] setAction navigation', {{action, params, href: parentUrl().toString()}});
  const u = parentUrl();
  u.searchParams.set('tm_action', action);
  if (action !== 'home') {{
    u.searchParams.set('tm_project', 'image');
    u.searchParams.set('tm_session', String(STATE.session || ''));
  }} else {{
    u.searchParams.delete('tm_project');
    u.searchParams.delete('tm_session');
  }}
  for (const [k,v] of Object.entries(params || {{}})) {{
    u.searchParams.set(k, String(v));
  }}
  navigateParent(u.toString());
}}
async function fileToBase64(f) {{
  const buf = await f.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let bin = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {{
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }}
  return btoa(bin);
}}
async function uploadOneFile(className, f) {{
  const maxBytes = Number(UPLOAD_MAX_MB || 25) * 1024 * 1024;
  if (Number(f.size || 0) > maxBytes) {{
    return {{ ok: false, name: String(f.name || 'file'), error: `File too large (>${{UPLOAD_MAX_MB}}MB).` }};
  }}
  try {{
    const b64 = await fileToBase64(f);
    const res = await fetch(`${{baseUrl}}/upload`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session, class: className, image_b64: b64, filename: String(f.name || '')}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (res.ok && data.ok === '1') {{
      const filename = String(data.filename || f.name || '');
      const thumb = typeof data.thumb_b64 === 'string' && data.thumb_b64 ? String(data.thumb_b64) : '';
      return {{
        ok: true,
        name: String(f.name || filename),
        filename,
        thumb_b64: thumb,
      }};
    }}
    const why = String(data && data.error ? data.error : '');
    return {{ ok: false, name: String(f.name || 'file'), error: why || 'Upload rejected by server.' }};
  }} catch (e) {{
    return {{ ok: false, name: String(f.name || 'file'), error: String(e && e.message ? e.message : e) }};
  }}
}}
async function uploadFiles(className, files) {{
  if (!files || files.length === 0) return;
  const all = Array.from(files);
  const total = all.length;
  let uploaded = 0;
  const failed = [];
  let next = 0;
  const progressUntil = Date.now() + 120000;
  const report = (done) => {{
    const msg = `Uploading ${{className}}: ${{done}}/${{total}}${{failed.length ? ` (failed ${{failed.length}})` : ''}}…`;
    toast(msg, 20000);
    bumpToastUntil(progressUntil);
  }};
  const worker = async () => {{
    while (next < total) {{
      const i = next++;
      const f = all[i];
      try {{
        const r = await uploadOneFile(className, f);
        if (r && r.ok) {{
          uploaded += 1;
          if (r.thumb_b64) {{
            prependSamplePreview(className, {{src: `data:image/png;base64,${{r.thumb_b64}}`, filename: String(r.filename || r.name || '')}});
          }}
          incrementSampleCount(className, 1);
          recomputeTrainEnabled();
        }} else {{
          const why = String(r && r.error ? r.error : '');
          const fname = String(r && r.name ? r.name : 'file');
          failed.push(why ? `${{fname}} (${{why}})` : fname);
          toast(`Upload failed: ${{fname}}${{why ? ' — ' + why : ''}}`, 5000);
        }}
      }} catch (e) {{
        const fname = String(f && f.name ? f.name : 'file');
        failed.push(fname);
        toast(`Upload failed: ${{fname}}`, 4000);
      }} finally {{
        report(next);
      }}
    }}
  }};
  report(0);
  const workers = Math.max(1, Math.min(Number(UPLOAD_CONCURRENCY || 3), total));
  const tasks = [];
  for (let i = 0; i < workers; i += 1) tasks.push(worker());
  await Promise.all(tasks);
  syncTrainUi();
  refreshTrainRec();
  if (openSourceClass === className) updateOpenSamplesPanel(className);
  const lines = [`Uploaded ${{uploaded}}/${{total}} image(s) to ${{className}}.`];
  if (failed.length) lines.push(`Failed ${{failed.length}}: ${{failed.slice(0, 8).join('; ')}}${{failed.length > 8 ? '; ...' : ''}}`);
  toast(lines.join(' '), Math.max(4000, 2000 + Math.min(6000, failed.length * 600)));
}}
function cssSafe(name) {{
  return String(name).replace(/[^a-zA-Z0-9_-]/g, '_');
}}
function normalizePreviewItem(item) {{
  if (!item) return null;
  if (typeof item === 'string') {{
    return {{src: String(item), filename: ''}};
  }}
  if (typeof item === 'object') {{
    const src = String(item.src || item.image || '');
    if (!src) return null;
    return {{
      src,
      filename: String(item.filename || '')
    }};
  }}
  return null;
}}
function normalizePreviewList(items) {{
  if (!Array.isArray(items)) return [];
  const out = [];
  for (const item of items) {{
    const normalized = normalizePreviewItem(item);
    if (normalized) out.push(normalized);
  }}
  return out;
}}
function previewSrc(item) {{
  const normalized = normalizePreviewItem(item);
  return normalized ? String(normalized.src || '') : '';
}}
function previewFilename(item) {{
  const normalized = normalizePreviewItem(item);
  return normalized ? String(normalized.filename || '') : '';
}}
function escapeHtml(value) {{
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}
function nullish(value, fallback) {{
  return value === null || value === undefined ? fallback : value;
}}
function prependSamplePreview(className, item) {{
  const normalized = normalizePreviewItem(item);
  if (!className || !normalized) return;
  const items = Array.isArray(STATE.sample_previews[className]) ? STATE.sample_previews[className].slice() : [];
  items.unshift(normalized);
  STATE.sample_previews[className] = normalizePreviewList(items).slice(0, PREF_MAX_SAMPLES_DISPLAY);
}}
function removePreviewItemLocal(className, filename) {{
  if (!className || !filename) return;
  const items = normalizePreviewList(STATE.sample_previews[className]);
  STATE.sample_previews[className] = items.filter((item) => previewFilename(item) !== filename);
}}
function incrementSampleCount(className, delta) {{
  const next = Math.max(0, Number(STATE.counts[className] || 0) + Number(delta || 0));
  STATE.counts[className] = next;
}}
function recomputeTrainEnabled() {{
  const classes = Array.isArray(STATE.classes) ? STATE.classes : [];
  const counts = STATE.counts || {{}};
  const nonEmpty = classes.filter((name) => Number(counts[name] || 0) > 0).length;
  const total = classes.reduce((sum, name) => sum + Number(counts[name] || 0), 0);
  STATE.train_enabled = classes.length >= 2 && nonEmpty >= 2 && nonEmpty === classes.length && total > 0;
}}
async function syncClassState(className) {{
  if (!className) return;
  try {{
    const res = await fetch(`${{baseUrl}}/class_state?session=${{encodeURIComponent(STATE.session)}}&class=${{encodeURIComponent(className)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to refresh class state.');
    STATE.counts[className] = Number(data.count || 0);
    STATE.sample_previews[className] = normalizePreviewList(data.previews).slice(0, PREF_MAX_SAMPLES_DISPLAY);
    recomputeTrainEnabled();
  refreshTrainRec();
    syncTrainUi();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
// ── EMNIST template samples (developer / demo path) ──────────────────────
// The main workspace belongs to real camera captures; demo data must never
// clutter it.  The seeder therefore lives in Settings -> Advanced and is
// reachable two ways:
//   * manual: the button fills every character class in the project now
//   * auto:   a character class that arrives empty is filled on creation
// Both POST /samples/seed_emnist, which renders DARK-INK-ON-LIGHT 96x96
// frames — the polarity the capture pipeline expects (raw EMNIST is
// bright-on-dark and would binarize inverted).  Seeded files carry an
// `emnist_seed_` prefix, so re-seeding replaces only its own output.
// (The state variables live up with PREF_MAX_SAMPLES_DISPLAY so the boot
// restore can run before this block is evaluated.)

function restoreAutoSeedEmnist() {{
  try {{
    autoSeedEmnistSamples = window.localStorage.getItem(AUTO_SEED_EMNIST_STORAGE_KEY) === '1';
  }} catch (e) {{
    autoSeedEmnistSamples = false;
  }}
}}
function setAutoSeedEmnistEnabled(on) {{
  autoSeedEmnistSamples = !!on;
  try {{
    window.localStorage.setItem(AUTO_SEED_EMNIST_STORAGE_KEY, autoSeedEmnistSamples ? '1' : '0');
  }} catch (e) {{}}
}}
function isTemplateCharClass(name) {{
  const ch = String(name == null ? '' : name).trim().toUpperCase();
  return ch.length === 1 && 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.indexOf(ch) >= 0;
}}
// The one place the SPA asks the backend to draw template samples.  Returns
// the class names the server actually seeded, or null when nothing ran.
async function seedEmnistSamples(names, opts) {{
  const options = opts || {{}};
  const target = (Array.isArray(names) ? names : [])
    .map((n) => String(n == null ? '' : n))
    .filter((n, i, all) => n && all.indexOf(n) === i);
  if (!target.length || !isOcrProject() || autoSeedEmnistInFlight) return null;
  autoSeedEmnistInFlight = true;
  try {{
    const res = await fetch(`${{baseUrl}}/samples/seed_emnist`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(Object.assign(
        {{session: STATE.session, classes: target, count: 24}},
        isSignOcrProject() ? {{
          sign_sequence: signSequence().join(''),
          sign_sequences: signSequences(),
          reading_order: String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr')),
        }} : {{}}
      ))
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to load template samples.');
    const states = data.states && typeof data.states === 'object' ? data.states : {{}};
    // `classes` is the project's COMPLETE class list and `seeded` names what
    // this call drew; fall back to the old shape for a stale backend.
    const seeded = (Array.isArray(data.seeded) ? data.seeded : [])
      .map((n) => String(n))
      .concat(
        (Array.isArray(data.classes) ? data.classes : target)
          .map((n) => String(n))
          .filter((n) => !!states[n])
      )
      .filter((n, i, all) => all.indexOf(n) === i);
    // The backend now returns the whole runtime state (classes, counts,
    // sample_previews, train_enabled, train_rec).  Applying it is what makes
    // a class that was ALREADY on screen update in place — no reload, no
    // re-creating the class — so the card shows "24 Image Samples" and the
    // Train button enables the moment the response lands.  A response that
    // carries only the per-class `states` (older backend) is folded into the
    // same shape first, so the cards still update.
    if (!data.counts && Object.keys(states).length) {{
      const patchedCounts = Object.assign({{}}, STATE.counts || {{}});
      const patchedPreviews = Object.assign({{}}, STATE.sample_previews || {{}});
      Object.keys(states).forEach((name) => {{
        const state = states[name] || {{}};
        if (state.count !== undefined) patchedCounts[name] = Number(state.count || 0);
        if (Array.isArray(state.previews)) {{
          patchedPreviews[name] = normalizePreviewList(state.previews).slice(0, PREF_MAX_SAMPLES_DISPLAY);
        }}
      }});
      data.counts = patchedCounts;
      data.sample_previews = patchedPreviews;
    }}
    applyRuntimeStateAndRender(data);
    maybeAutoSynthesizeSignPreview();
    if (!options.quiet) {{
      const skipped = Array.isArray(data.skipped) ? data.skipped.length : 0;
      toast(seeded.length
        ? (skipped
            ? `Loaded 24 template samples per class (skipped ${{skipped}} non-character class${{skipped === 1 ? '' : 'es'}}).`
            : `Loaded 24 template samples per class.`)
        : 'Every character class already has samples — nothing to fill.');
    }}
    return seeded;
  }} catch (err) {{
    if (!options.quiet) toast(String(err && err.message ? err.message : err));
    return null;
  }} finally {{
    autoSeedEmnistInFlight = false;
  }}
}}
// Auto path: called right after classes are created/synced.  Only classes
// that are empty, are single sign characters, and have not already been
// offered this project are seeded — a class with real captures is never
// touched, and "Clear Samples" stays cleared.
function autoSeedEmnistForNewClasses(names) {{
  if (!isOcrProject()) return;
  // Same OR the backend applies: the browser-local checkbox and the
  // project's persisted TrainConfig flag are independent, and either being
  // on means auto-fill is on for this project.
  const autoSeedOn = !!(autoSeedEmnistSamples
    || (STATE.train_cfg && STATE.train_cfg.auto_seed_emnist_samples));
  if (!autoSeedOn) return;
  const fresh = (Array.isArray(names) ? names : []).filter((name) => {{
    const key = String(name == null ? '' : name);
    if (!key || autoSeedEmnistAttempted.has(key)) return false;
    if (!isTemplateCharClass(key)) return false;
    return Number((STATE.counts && STATE.counts[key]) || 0) === 0;
  }});
  if (!fresh.length) return;
  if (autoSeedEmnistInFlight) {{
    // seedEmnistSamples would return without starting, so marking these
    // names now would make the skip permanent.  Wait for the running
    // request to settle, then re-ask with the same names.
    window.setTimeout(() => autoSeedEmnistForNewClasses(names), 250);
    return;
  }}
  // Marked before the request so a reload/resync cannot double-fire it; the
  // server response then replaces the counts for the classes it seeded.
  fresh.forEach((name) => autoSeedEmnistAttempted.add(String(name)));
  seedEmnistSamples(fresh, {{quiet: true}}).then((seeded) => {{
    // null = the request did not run/failed.  Un-mark so a later class
    // write retries instead of leaving the class at 0 for the session.
    if (!seeded) fresh.forEach((name) => autoSeedEmnistAttempted.delete(String(name)));
  }}).catch(() => {{
    fresh.forEach((name) => autoSeedEmnistAttempted.delete(String(name)));
  }});
}}
// Apply the full runtime state every class-write response carries.  This is
// deliberately a REPLACE and not a merge: the server's class list, counts and
// previews are authoritative, and merging could keep a record for a class the
// response just removed.  Missing keys are left alone so an older backend
// that only sends `classes` still works.
function applyRuntimeState(data) {{
  const d = data && typeof data === 'object' ? data : {{}};
  if (Array.isArray(d.classes)) {{
    STATE.classes = d.classes.map((n) => String(n));
  }}
  if (d.counts && typeof d.counts === 'object') {{
    const counts = {{}};
    Object.keys(d.counts).forEach((name) => {{
      counts[name] = Number(d.counts[name] || 0);
    }});
    STATE.counts = counts;
  }}
  if (d.sample_previews && typeof d.sample_previews === 'object') {{
    const previews = {{}};
    Object.keys(d.sample_previews).forEach((name) => {{
      previews[name] = normalizePreviewList(d.sample_previews[name]).slice(0, PREF_MAX_SAMPLES_DISPLAY);
    }});
    STATE.sample_previews = previews;
  }}
  if (d.processed_previews && typeof d.processed_previews === 'object') {{
    const processed = {{}};
    Object.keys(d.processed_previews).forEach((name) => {{
      processed[name] = normalizePreviewList(d.processed_previews[name]);
    }});
    STATE.processed_previews = processed;
  }}
  if (typeof d.train_enabled === 'boolean') STATE.train_enabled = d.train_enabled;
  else recomputeTrainEnabled();
  if (d.train_rec && typeof d.train_rec === 'object') STATE.train_rec = d.train_rec;
  return d;
}}
// The class column is built by render(); this is the named entry point a
// state response calls so "update the cards" reads as one action.  It is
// intentionally the same path as a normal render: counts, previews, the
// Train button and the open sample panel must all agree with STATE, and a
// second, partial renderer would be a second place for them to disagree.
function renderClassCards() {{
  render();
}}
// Refresh the Training card's derived UI (recommendation, enablement,
// status) without rebuilding the class column.
function renderTrainingCard() {{
  recomputeTrainEnabled();
  refreshTrainRec();
  syncTrainUi();
  renderTrainStatus();
}}
function applyRuntimeStateAndRender(data) {{
  applyRuntimeState(data);
  renderClassCards();
  renderTrainingCard();
}}
// The Settings checkbox is the master switch for the auto-fill workflow.
// Flipping it persists the flag on the project AND asks the controller to
// backfill every class that is ALREADY on disk with zero samples — the
// classes do not have to be re-created for the fill to happen.
async function applyAutoSeedEmnistSetting(enabled) {{
  const on = !!enabled;
  setAutoSeedEmnistEnabled(on);
  STATE.train_cfg = Object.assign({{}}, STATE.train_cfg || {{}}, {{
    auto_seed_emnist_samples: on,
  }});
  persistTrainCfgStorage();
  try {{
    const params = new URLSearchParams();
    params.set('session', String(STATE.session || ''));
    params.set('auto_seed_emnist', on ? '1' : '0');
    params.set('include_state', '1');
    if (isSignOcrProject()) {{
      // The target travels with the toggle so the backend can persist it
      // and draw a MATCH preview plate during the same backfill pass.
      params.set('sign_sequence', signSequence().join(''));
      params.set('reading_order', String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr')));
    }}
    const res = await fetch(`${{baseUrl}}/live/config?${{params.toString()}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update the auto-fill setting.');
    applyRuntimeStateAndRender(data);
    if (on) maybeAutoSynthesizeSignPreview();
    toast(on
      ? 'Auto-fill on — empty character classes now hold 24 template samples.'
      : 'Template auto-fill off.');
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
// One-shot "fill the empty ones NOW", independent of the checkbox: the
// button is an explicit request, so it runs even while auto-fill is off.
async function fillEmptyClassesWithEmnist() {{
  try {{
    const res = await fetch(`${{baseUrl}}/classes/seed_emnist`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(Object.assign(
        {{ session: STATE.session, only_empty: true, count: 24 }},
        isSignOcrProject() ? {{
          sign_sequence: signSequence().join(''),
          sign_sequences: signSequences(),
          reading_order: String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr')),
        }} : {{}}
      ))
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to fill empty classes.');
    applyRuntimeStateAndRender(data);
    maybeAutoSynthesizeSignPreview();
    return data;
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
    return null;
  }}
}}
async function startHoldCapture() {{
  if (holdRecording || !openSourceClass || !openSourceKind || openSourceKind === 'upload') return;
  try {{
    holdResumePreviewPredict = !!previewInputOn;
    holdPreviewSourceBeforeCapture = String(previewSource || 'webcam');
    // Pause prediction (TFLite + preprocess CPU) during hold capture so
    // PNG saves + ROI focus don't contend.  VISUAL UPDATE LOOP and class
    // panel source loop are LEFT RUNNING — this is the exact fix for
    // "preview pane freezes the moment hold starts".
    if (holdResumePreviewPredict) {{
      stopPreviewPredictLoop();
    }}
    // Make sure the Preview visual loop stays alive for the webcam path even
    // if the user turned Input on a split-second before pressing Hold (in
    // which case startPreviewPredictLoop() may not have fired yet).
    holdResumePreviewVisual = false;
    if (openSourceKind === 'webcam' && previewInputOn && previewSource === 'webcam') {{
      if (!previewVisualTimer) {{
        startPreviewVisualLoop();
        holdResumePreviewVisual = true;
      }}
    }}
    // BOTH sources keep EVERYTHING running during hold.  NO close, NO stop,
    // NO restart.  The backend record thread now consumes the live worker's
    // capture cache (device included — the serial reader stays single), so
    // the class-panel /live/frame loop, the Preview visual loop and the
    // backend live thread all keep going.  Previously the device path
    // stopped the class loop + /live/closed the worker for the whole hold —
    // the class-panel view froze on press every single time.
    if (!sourceLiveTimer) {{
      startSourceLiveLoop();
    }}
    const res = await fetch(`${{baseUrl}}/start?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(openSourceKind)}}&class=${{encodeURIComponent(openSourceClass)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to start hold capture.');
    holdRecording = true;
    holdRecordClass = openSourceClass;
    holdRecordSource = openSourceKind;
    syncSourceActionButtons(openSourceClass);
    stopHoldSyncLoop();
    holdSeq = 0;
    const token = holdNextToken + 1;
    holdNextToken = token;
    holdPollController = new AbortController();
    (async () => {{
      while (holdRecording && holdRecordClass && holdNextToken === token) {{
        try {{
          const nextRes = await fetch(
            `${{baseUrl}}/record/next?session=${{encodeURIComponent(STATE.session)}}&since=${{encodeURIComponent(String(holdSeq))}}`,
            {{ signal: holdPollController ? holdPollController.signal : undefined }}
          );
          const nextData = await nextRes.json().catch(() => ({{ok:'0'}}));
          if (!holdRecording || holdNextToken !== token) break;
          if (!nextRes.ok || nextData.ok !== '1') throw new Error(nextData.error || 'Unable to read hold capture updates.');
          const serverError = String(nextData.error || '').trim();
          if (serverError) {{
            toast(serverError);
            break;
          }}
          const seq = Number(nextData.seq || holdSeq || 0);
          if (seq > holdSeq && nextData.image_b64) {{
            holdSeq = seq;
            try {{
              // Progress goes to the note only — NEVER overwrite the live
              // camera view with the 96×96 processed sample (that made the
              // view window show the tiny resized image during hold).
              const note = document.getElementById(`sourceNote-${{cssSafe(holdRecordClass)}}`);
              if (note) note.textContent = `Captured: ${{String(nextData.count || '')}}`;
            }} catch (e) {{}}
            prependSamplePreview(holdRecordClass, {{
              src: `data:image/png;base64,${{nextData.image_b64}}`,
              filename: String(nextData.filename || '')
            }});
            STATE.counts[holdRecordClass] = Number(nextData.count || STATE.counts[holdRecordClass] || 0);
            recomputeTrainEnabled();
            refreshTrainRec();
            // Keep hold capture responsive: avoid rebuilding the whole sample
            // panel on every frame. We sync the full panel once on release.
            setClassSampleCountLabel(holdRecordClass);
          }} else {{
            holdSeq = seq;
          }}
          if (String(nextData.recording || '1') !== '1') break;
        }} catch (err) {{
          if (err && (err.name === 'AbortError' || String(err).includes('aborted'))) break;
          toast(String(err && err.message ? err.message : err));
          await new Promise((r) => setTimeout(r, 250));
        }}
      }}
      if (holdNextToken === token) holdPollController = null;
    }})().catch(() => {{}});
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function stopHoldCapture() {{
  if (!holdRecording) return;
  const className = holdRecordClass;
  const sourceKind = holdRecordSource;
  const shouldResumePreviewPredict = holdResumePreviewPredict;
  const previewSourceBeforeCapture = holdPreviewSourceBeforeCapture === 'device' ? 'device' : 'webcam';
  stopHoldSyncLoop();
  holdRecording = false;
  holdRecordClass = '';
  holdRecordSource = '';
  holdResumePreviewPredict = false;
  holdPreviewSourceBeforeCapture = 'webcam';
  syncSourceActionButtons(className);
  try {{
    await fetch(`${{baseUrl}}/stop?session=${{encodeURIComponent(STATE.session)}}`);
  }} catch (e) {{}}
  await syncClassState(className);
  if (openSourceClass === className) updateOpenSamplesPanel(className);
  // Restore the class-panel live loop FIRST and UNCONDITIONALLY — the old
  // `else if` skipped it whenever Preview Input was ON (only the predict
  // loop resumed), so the class view stayed frozen after a hold.
  if (openSourceClass === className && openSourceKind === sourceKind && (sourceKind === 'webcam' || sourceKind === 'device')) {{
    await ensureOpenSourceLive();
  }}
  if (shouldResumePreviewPredict && previewInputOn) {{
    previewSource = previewSourceBeforeCapture;
    startPreviewPredictLoop();
  }}
  if (holdResumePreviewVisual && !previewVisualTimer) {{
    startPreviewVisualLoop();
  }}
  holdResumePreviewVisual = false;
  toast('Hold capture stopped.');
}}
function buildDeviceOptions(selected) {{
  const ports = Array.isArray(STATE.serial_ports) ? STATE.serial_ports : [];
  const opts = ['<option value="">Select device port</option>'];
  for (const p of ports) {{
    const device = String(p.device || '').trim();
    if (!device) continue;  // skip entries with empty device name
    const label = String(p.label || device);
    const sel = device === selected ? ' selected' : '';
    opts.push(`<option value="${{device.replace(/"/g, '&quot;')}}"${{sel}}>${{label}}</option>`);
  }}
  return opts.join('');
}}
function refillDeviceSelect(selectEl, selectedValue) {{
  if (!selectEl) return;
  const current = String(selectedValue == null ? (selectEl.value || '') : selectedValue);
  selectEl.innerHTML = buildDeviceOptions(current);
  selectEl.value = current;
}}
async function refreshSerialPorts(shouldRerender = true, targetSelectId = '') {{
  try {{
    const res = await fetch(`${{baseUrl}}/serial/ports?session=${{encodeURIComponent(STATE.session)}}&_ts=${{Date.now()}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to refresh serial ports.');
    const ports = Array.isArray(data.ports) ? data.ports : [];
    STATE.serial_ports = ports.map((p) => ({{
      device: String(p && p.device ? p.device : ''),
      label: String(p && p.label ? p.label : (p && p.device ? p.device : '')),
    }})).filter((p) => !!p.device);
    const stillExists = STATE.serial_ports.some((p) => String(p.device || '') === String(currentSerialPort || ''));
    if (!stillExists && currentSerialPort) {{
      STATE.serial_ports = [{{device: String(currentSerialPort), label: String(currentSerialPort)}}].concat(STATE.serial_ports);
    }}
    if (shouldRerender) {{
      render();
    }} else if (targetSelectId) {{
      const selectEl = document.getElementById(targetSelectId);
      refillDeviceSelect(selectEl, currentSerialPort);
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
function buildWebcamOptions(selected, selectedUid) {{
  const cams = Array.isArray(STATE.webcam_options) ? STATE.webcam_options : [];
  dbgEvent('E', 'app.py:buildWebcamOptions', '[DEBUG] frontend webcam options', {{selected, cams}});
  const opts = ['<option value="">Select camera</option>'];
  for (const c of cams) {{
    const idx = Number(c.index);
    const label = String(c.label || `Camera ${{idx}}`);
    const uid = String(c.unique_id || '');
    const sel = (uid && uid === selectedUid) || (!selectedUid && idx === Number(selected)) ? ' selected' : '';
    opts.push(`<option value="${{idx}}" data-uid="${{uid}}"${{sel}}>${{label}}</option>`);
  }}
  return opts.join('');
}}
function buildSerialBaudOptions(selected) {{
  const values = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600];
  return values.map((baud) => `<option value="${{baud}}"${{Number(selected) === baud ? ' selected' : ''}}>${{baud}}</option>`).join('');
}}
function buildSerialFrameSideOptions(selected) {{
  const values = [96, 160, 384];
  return values.map((side) => `<option value="${{side}}"${{Number(selected) === side ? ' selected' : ''}}>${{side}}×${{side}}</option>`).join('');
}}
function buildSerialColorModeOptions(selected) {{
  const sel = Number(selected) === 3 ? 3 : 1;
  return `<option value="1"${{sel === 1 ? ' selected' : ''}}>Grayscale</option><option value="3"${{sel === 3 ? ' selected' : ''}}>RGB</option>`;
}}
function deviceHelpText() {{
  const ports = Array.isArray(STATE.serial_ports) ? STATE.serial_ports : [];
  if (!ports.length) return 'No compatible serial UART device is currently detected. Connect the board or install the required USB serial driver first.';
  return 'Choose the serial port connected to your device before capturing. Image Size must match the firmware output exactly (e.g. 96×96 or 160×160). Sync is the hex frame header used to find the start of each serial image packet, for example AA 55 AA.';
}}
function webcamHelpText() {{
  return 'Use the settings button (⚙) to choose the camera source for live capture.';
}}
function buildSourceSettingsMarkup(className) {{
  if (openSourceClass !== className || !sourceSettingsOpen || openSourceKind === 'upload') return '';
  if (openSourceKind === 'device') {{
    return `
      <div class="source-settings-panel" id="sourceSettingsPanel-${{cssSafe(className)}}">
        <div class="source-settings-grid">
          <label>
            Image Size
            <select id="sourceFrameSide-${{cssSafe(className)}}">${{buildSerialFrameSideOptions(currentSerialFrameSide)}}</select>
          </label>
          <label>
            Color Mode
            <select id="sourceColorMode-${{cssSafe(className)}}">${{buildSerialColorModeOptions(currentSerialChannels)}}</select>
          </label>
          <label>
            Baud Rate
            <select id="sourceBaud-${{cssSafe(className)}}">${{buildSerialBaudOptions(currentSerialBaud)}}</select>
          </label>
          <label>
            Sync Header
            <input id="sourceSync-${{cssSafe(className)}}" value="${{escapeHtml(String(currentSerialSync || 'AA 55 AA'))}}" placeholder="AA 55 AA"/>
          </label>
        </div>
        <div class="source-settings-actions">
          <button class="source-settings-cancel" type="button" id="sourceSettingsCancel-${{cssSafe(className)}}">Close</button>
          <button class="source-settings-save" type="button" id="sourceSettingsSave-${{cssSafe(className)}}">Apply</button>
        </div>
      </div>
    `;
  }}
  return `
    <div class="source-settings-panel" id="sourceSettingsPanel-${{cssSafe(className)}}">
      <div class="source-settings-grid">
        <label>
          Camera
          <select id="sourceCamera-${{cssSafe(className)}}">${{buildWebcamOptions(currentWebcamIndex, currentWebcamUid)}}</select>
        </label>
        <label>
          Preview Rate
          <select id="sourcePreviewRate-${{cssSafe(className)}}">
            <option value="60"${{Number(previewIntervalMs) === 60 ? ' selected' : ''}}>Fast</option>
            <option value="80"${{Number(previewIntervalMs) === 80 ? ' selected' : ''}}>Balanced</option>
            <option value="120"${{Number(previewIntervalMs) === 120 ? ' selected' : ''}}>Stable</option>
            <option value="180"${{Number(previewIntervalMs) === 180 ? ' selected' : ''}}>Low CPU</option>
          </select>
        </label>
      </div>
      <div class="source-settings-actions">
        <button class="source-settings-cancel" type="button" id="sourceSettingsCancel-${{cssSafe(className)}}">Close</button>
        <button class="source-settings-save" type="button" id="sourceSettingsSave-${{cssSafe(className)}}">Apply</button>
      </div>
    </div>
  `;
}}
async function changeSerialBaud(className, value) {{
  const nextBaud = Number(value || currentSerialBaud || 115200);
  currentSerialBaud = nextBaud;
  STATE.current_serial_baud = currentSerialBaud;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'device';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_baud=${{encodeURIComponent(String(currentSerialBaud))}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update baud rate.');
    if (openSourceClass === className && openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
async function changeSerialSync(className, value) {{
  const nextSync = String(value || currentSerialSync || 'AA 55 AA').trim() || 'AA 55 AA';
  currentSerialSync = nextSync;
  STATE.current_serial_sync = currentSerialSync;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'device';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_sync=${{encodeURIComponent(currentSerialSync)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update sync header.');
    currentSerialSync = String(data.serial_sync || currentSerialSync || 'AA 55 AA');
    STATE.current_serial_sync = currentSerialSync;
    if (openSourceClass === className && openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
async function changeSerialFrameSide(className, value) {{
  const nextSide = Number(value || currentSerialFrameSide || 96);
  currentSerialFrameSide = nextSide;
  STATE.current_serial_frame_side = currentSerialFrameSide;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'device';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_frame_side=${{encodeURIComponent(String(currentSerialFrameSide))}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update device image size.');
    currentSerialFrameSide = Number(data.serial_frame_side || currentSerialFrameSide || 96);
    STATE.current_serial_frame_side = currentSerialFrameSide;
    if (openSourceClass === className && openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
async function changeDirectTensorStream(value) {{
  directTensorStream = !!value;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&direct_tensor_stream=${{directTensorStream ? '1' : '0'}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update device stream mode.');
    directTensorStream = String(data.direct_tensor_stream || '0') === '1';
    if (openSourceClass && openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function changeSerialChannels(className, value) {{
  const nextCh = Number(value || currentSerialChannels || 1);
  currentSerialChannels = nextCh;
  STATE.current_serial_channels = currentSerialChannels;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'device';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_channels=${{encodeURIComponent(String(currentSerialChannels))}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update device color mode.');
    currentSerialChannels = Number(data.serial_channels || currentSerialChannels || 1);
    STATE.current_serial_channels = currentSerialChannels;
    if (openSourceClass === className && openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
function toggleSourceSettings(className) {{
  if (openSourceClass !== className || !openSourceKind || openSourceKind === 'upload') return;
  sourceSettingsOpen = !sourceSettingsOpen;
  render();
}}
async function applySourceSettings(className) {{
  if (openSourceClass !== className) return;
  try {{
    if (openSourceKind === 'device') {{
      const sideEl = document.getElementById(`sourceFrameSide-${{cssSafe(className)}}`);
      const modeEl = document.getElementById(`sourceColorMode-${{cssSafe(className)}}`);
      const baudEl = document.getElementById(`sourceBaud-${{cssSafe(className)}}`);
      const syncEl = document.getElementById(`sourceSync-${{cssSafe(className)}}`);
      await changeSerialFrameSide(className, sideEl ? sideEl.value : String(currentSerialFrameSide));
      await changeSerialChannels(className, modeEl ? modeEl.value : String(currentSerialChannels));
      await changeSerialBaud(className, baudEl ? baudEl.value : String(currentSerialBaud));
      await changeSerialSync(className, syncEl ? syncEl.value : String(currentSerialSync));
    }} else if (openSourceKind === 'webcam') {{
      const camEl = document.getElementById(`sourceCamera-${{cssSafe(className)}}`);
      const rateEl = document.getElementById(`sourcePreviewRate-${{cssSafe(className)}}`);
      const nextCam = camEl ? camEl.value : String(currentWebcamIndex);
      const nextUid = camEl && camEl.selectedOptions && camEl.selectedOptions[0] ? String(camEl.selectedOptions[0].dataset.uid || '') : '';
      const indexChanged = String(nextCam) !== String(currentWebcamIndex);
      const uidChanged = String(nextUid) !== String(currentWebcamUid || '');
      if (indexChanged || uidChanged) {{
        await changeWebcamIndex(className, nextCam, nextUid);
      }}
      previewIntervalMs = Number(rateEl ? rateEl.value : previewIntervalMs || 80);
      if (openSourceClass === className && openSourceKind === 'webcam') startPreviewLoop();
    }}
    sourceSettingsOpen = false;
    render();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function changeDevicePort(className, value) {{
  if (captureInFlight && openSourceClass === className) {{
    toast('Wait for the current capture to finish before switching device.');
    syncSourceActionButtons(className);
    return;
  }}
  if (holdRecording && holdRecordClass === className && holdRecordSource === 'device') {{
    await stopHoldCapture();
  }}
  currentSerialPort = String(value || '');
  STATE.current_serial_port = currentSerialPort;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'device';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_port=${{encodeURIComponent(currentSerialPort)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update serial port.');
    if (openSourceClass === className && openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
async function refreshWebcamOptions() {{
  try {{
    const res = await fetch(`${{baseUrl}}/webcam/options`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1' || !Array.isArray(data.options)) return;
    STATE.webcam_options = data.options;
    const camSel = document.getElementById('previewWebcamSelect') || document.getElementById('previewCamera');
    if (camSel) camSel.innerHTML = buildWebcamOptions(currentWebcamIndex, currentWebcamUid);
    for (const cls of (Array.isArray(STATE.classes) ? STATE.classes : [])) {{
      const sel = document.getElementById(`sourceCamera-${{cssSafe(cls)}}`);
      if (sel) sel.innerHTML = buildWebcamOptions(currentWebcamIndex, currentWebcamUid);
    }}
  }} catch (e) {{}}
}}
async function changeWebcamIndex(className, value, uid) {{
  if (captureInFlight && openSourceClass === className) {{
    toast('Wait for the current capture to finish before switching camera.');
    syncSourceActionButtons(className);
    return;
  }}
  if (holdRecording && holdRecordClass === className && holdRecordSource === 'webcam') {{
    await stopHoldCapture();
  }}
  currentWebcamIndex = Number(value || 0);
  currentWebcamUid = String(uid || '');
  STATE.current_webcam_index = currentWebcamIndex;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'webcam';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&webcam_index=${{encodeURIComponent(currentWebcamIndex)}}&webcam_id=${{encodeURIComponent(currentWebcamUid)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update webcam.');
    if (openSourceClass === className && openSourceKind === 'webcam') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=webcam`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
function sourceLabel(kind) {{
  if (kind === 'device') return 'Device';
  if (kind === 'upload') return 'Upload';
  return 'Webcam';
}}
function sourceSamplesTitle(kind) {{
  if (kind === 'upload') return 'Uploaded Samples';
  if (kind === 'device') return 'Device Samples';
  return 'Camera Samples';
}}
function buildSampleTileMarkup(item, idx) {{
  const src = escapeHtml(previewSrc(item));
  const filename = escapeHtml(previewFilename(item));
  return `
    <div class="sample-item">
      <img class="sample-thumb" src="${{src}}" loading="lazy" decoding="async" alt="Sample ${{idx+1}}"/>
      <button class="sample-delete" type="button" data-filename="${{filename}}" aria-label="Delete sample">✕</button>
    </div>
  `;
}}
function buildSamplesMarkup(className) {{
  const items = normalizePreviewList((STATE.sample_previews && STATE.sample_previews[className]) || []).slice(0, PREF_MAX_SAMPLES_DISPLAY);
  if (!items.length) return '<div class="samples-empty">No samples yet.</div>';
  const totalSamples = Number(STATE.counts && STATE.counts[className] ? STATE.counts[className] : items.length);
  const hiddenCount = Math.max(0, totalSamples - items.length);
  const more = hiddenCount > 0 ? `<div class="samples-grid-more">+${{hiddenCount}}</div>` : '';
  return `<div class="samples-grid">${{items.map((item, idx) => buildSampleTileMarkup(item, idx)).join('')}}${{more}}</div>`;
}}
function buildSamplesStripMarkup(className) {{
  const allItems = normalizePreviewList((STATE.sample_previews && STATE.sample_previews[className]) || []);
  const items = allItems.slice(0, PREF_MAX_SAMPLES_DISPLAY);
  if (!items.length) return '<div class="samples-strip-empty">No samples yet.</div>';
  const totalSamples = Number(STATE.counts && STATE.counts[className] ? STATE.counts[className] : allItems.length);
  const hiddenCount = Math.max(0, totalSamples - items.length);
  const more = hiddenCount > 0 ? `<div class="samples-strip-more">+${{hiddenCount}}</div>` : '';
  return `<div class="samples-strip">${{items.map((item, idx) => buildSampleTileMarkup(item, idx)).join('')}}${{more}}</div>`;
}}
function createIcon(kind) {{
  if (kind === 'webcam') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="14" height="12" rx="2"></rect><path d="M22 8l-6 4 6 4V8z"></path></svg>`;
  if (kind === 'upload') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>`;
  if (kind === 'device') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"></rect><line x1="9" y1="9" x2="15" y2="9"></line><line x1="9" y1="15" x2="15" y2="15"></line></svg>`;
  if (kind === 'rename') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>`;
  if (kind === 'clear') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20H7L3 16a2 2 0 0 1 0-2.83l9.17-9.17a2 2 0 0 1 2.83 0l5 5a2 2 0 0 1 0 2.83L11 20"></path><line x1="18" y1="13" x2="9" y2="4"></line><line x1="5" y1="16" x2="13" y2="8"></line></svg>`;
  if (kind === 'delete') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;
  // A cross means "close" everywhere else in this UI, so the destructive
  // "delete the whole class" action uses an unmistakable trash can instead.
  if (kind === 'trash') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>`;
  if (kind === 'cpu') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>`;
  if (kind === 'settings') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.19.42.58.74 1.06.91.48.17.99.22 1.51.09"></path></svg>`;
  if (kind === 'file') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>`;
  if (kind === 'folder') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>`;
  if (kind === 'save') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path><polyline points="17 21 17 13 7 13 7 21"></polyline><polyline points="7 3 7 8 15 8"></polyline></svg>`;
  if (kind === 'download') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>`;
  if (kind === 'box') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>`;
  if (kind === 'refresh') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>`;
  if (kind === 'eye') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>`;
  if (kind === 'plus') return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>`;
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"></rect><line x1="9" y1="9" x2="15" y2="9"></line><line x1="9" y1="15" x2="15" y2="15"></line></svg>`;
}}
function setFlowPath(id, d) {{
  const p = document.getElementById(id);
  if (p) p.setAttribute('d', d);
}}
function setClassSampleCountLabel(className) {{
  const count = Number(STATE.counts[className] || 0);
  const summary = document.getElementById(`summaryTitle-${{cssSafe(className)}}`);
  if (summary) summary.textContent = `${{count}} ${{count === 1 ? 'Image Sample' : 'Image Samples'}}`;
  const sourceCount = document.getElementById(`sourceCount-${{cssSafe(className)}}`);
  if (sourceCount) sourceCount.innerHTML = `${{count}}<small>${{count === 1 ? 'Image Sample' : 'Image Samples'}}</small>`;
}}
function latestPreviewImage() {{
  if (openSourceClass) {{
    const items = normalizePreviewList(STATE.sample_previews && STATE.sample_previews[openSourceClass]);
    if (items.length) return previewSrc(items[0]);
  }}
  for (const name of (Array.isArray(STATE.classes) ? STATE.classes : [])) {{
    const items = normalizePreviewList(STATE.sample_previews && STATE.sample_previews[name]);
    if (items.length) return previewSrc(items[0]);
  }}
  return '';
}}
function stopPreviewPredictLoop() {{
  previewPredictToken += 1;
  if (previewPredictController) {{
    try {{ previewPredictController.abort(); }} catch (e) {{}}
    previewPredictController = null;
  }}
  previewPredictInFlight = false;
  if (previewPredictTimer) {{
    clearInterval(previewPredictTimer);
    previewPredictTimer = null;
  }}
}}
// What the preview note's headline should say.  A ranking preview reports a
// "top" class; a sign preview reports a VERDICT, and calling that "Top:" would
// describe a decision as a leaderboard entry.  One helper so the live and
// upload paths cannot word the same answer differently.
function previewHeadline(data) {{
  const d = data || {{}};
  const text = d.top_label ? String(d.top_label) : '';
  // The synth endpoint always carries match_text; fall back to it so a
  // verdict is never described as "no verdict" just because top_label was
  // left off a payload.
  if (d.sign_ocr) return text || (d.match_text ? String(d.match_text) : '');
  if (!text) return '';
  const p = Math.round(Number(d.top_prob || 0) * 1000) / 10;
  return `Top: ${{text}} (${{p}}%)`;
}}

// The one output renderer, so a sign project and a ranking project cannot end
// up drawing into the host in two different shapes depending on which fetch
// happened to fire last.
function renderPreviewOutput(data) {{
  const d = data || {{}};
  if (d.sign_ocr) {{
    renderSignMatch(d);
    return;
  }}
  renderOutputBars(d.labels || [], d.probs || []);
}}

// The MATCH banner, which REPLACES the score bars rather than sitting above
// them.  The bars rank classes, and "is this the sign I trained?" is not a
// ranking question — showing both would invite reading whichever answer is
// more agreeable.  Everything shown here comes from the payload's own slots,
// so what the user sees is what the decision was actually made on.
function renderSignMatch(data) {{
  const host = document.getElementById('previewOutput');
  if (!host) return;
  host.innerHTML = '';
  host._lastLabels = null;
  host._lastProbs = null;

  const want = Array.isArray(data.sign_sequence) ? data.sign_sequence : [];
  const slots = Array.isArray(data.slots) ? data.slots : [];
  const shots = Array.isArray(data.slot_images_b64) ? data.slot_images_b64 : [];
  const hasTarget = want.some((c) => c && c !== SIGN_SEQUENCE_PAD);
  const anyLabel = slots.some((s) => s && s.label);
  // Both "no question asked" and "no model to answer with" are IDLE, not a
  // verdict: painting NO MODEL YET red would read as "this sign is wrong"
  // when nothing was compared at all.
  const state = (!hasTarget || !anyLabel) ? 'idle' : (data.match ? 'yes' : 'no');

  const box = document.createElement('div');
  box.className = 'signmatch ' + state;

  const main = document.createElement('div');
  main.className = 'signmatch-main';
  // Three distinct reasons the answer is not a verdict, and they need three
  // different next steps — so the idle case says which one it is instead of
  // leaving a stale red NO on screen for a question never asked.
  if (!hasTarget) {{
    main.textContent = '◻ SET THE TARGET SIGN';
  }} else if (!anyLabel) {{
    main.textContent = '◻ NO MODEL YET';
  }} else {{
    main.textContent = String(data.match_text || (data.match ? SIGN_MATCH_TEXT : SIGN_NOMATCH_TEXT));
  }}
  box.appendChild(main);

  const row = document.createElement('div');
  row.className = 'signmatch-slots';
  const minP = Number(data.match_min_prob || 0);
  // The payload's slots are already in READING order — the split ran
  // left-to-right and was reversed only if the project reads right-to-left —
  // so they are drawn in array order and never re-sorted here.  Sorting by
  // position in the frame would silently undo an RTL project.
  slots.forEach((slot, idx) => {{
    const s = slot || {{}};
    const wantCh = String(s.char || want[idx] || SIGN_SEQUENCE_PAD);
    const cell = document.createElement('div');
    const ok = !!s.match;
    cell.className = 'signmatch-slot ' + (ok ? 'ok' : 'bad');
    const wantEl = document.createElement('div');
    wantEl.className = 'want';
    wantEl.textContent = `WANT ${{wantCh}}`;
    cell.appendChild(wantEl);
    const gotEl = document.createElement('div');
    gotEl.className = 'got';
    gotEl.textContent = s.label ? String(s.label) : '—';
    gotEl.title = s.background
      ? 'Rejected as background — the crop held no glyph the model recognises.'
      : (s.empty ? 'The split found nothing in this slot.' : '');
    cell.appendChild(gotEl);
    if (shots[idx]) {{
      const img = document.createElement('img');
      img.alt = `Slot ${{idx + 1}} glyph`;
      img.src = `data:image/png;base64,${{String(shots[idx])}}`;
      cell.appendChild(img);
    }}
    const probEl = document.createElement('div');
    probEl.className = 'prob';
    probEl.textContent = `${{Math.round(Number(s.prob || 0) * 1000) / 10}}%`;
    cell.appendChild(probEl);
    row.appendChild(cell);
  }});
  box.appendChild(row);

  if (!hasTarget) {{
    const hint = document.createElement('div');
    hint.className = 'signmatch-hint';
    hint.textContent = 'Open Edit ▸ Edit sign characters… to enter the three characters this sign reads. The preview splits one frame into three slots and answers only once there is a target to answer against.';
    box.appendChild(hint);
  }} else if (!anyLabel) {{
    const hint = document.createElement('div');
    hint.className = 'signmatch-hint';
    hint.textContent = 'The split is live — these are the three crops the model will be given — but until a model is trained there is no reading to compare. Train, then come back.';
    box.appendChild(hint);
  }} else if (!data.match) {{
    const fails = slots.filter((s) => s && !s.match);
    const reasons = fails.map((s, i) => {{
      const idx = Number(s.index != null ? s.index : i) + 1;
      const got = s.label
        ? `read "${{String(s.label)}}" at ${{Math.round(Number(s.prob || 0) * 1000) / 10}}%`
        : 'produced no glyph';
      // If an OOD gate fired, name THAT gate: a correct character below the
      // entropy floor must not be explained as a confidence failure.
      if (s.ood && s.ood_reason) {{
        return `slot ${{idx}} ${{got}} — rejected by OOD (${{String(s.ood_reason)}})`;
      }}
      if (s.background) return `slot ${{idx}} came back as background`;
      if (s.empty) return `slot ${{idx}} contained no ink`;
      return `slot ${{idx}} ${{got}} (wanted "${{String(s.char || SIGN_SEQUENCE_PAD)}}")`;
    }});
    const hint = document.createElement('div');
    hint.className = 'signmatch-hint';
    hint.textContent = `All three must match (each at ${{Math.round(minP * 1000) / 10}}% or better) — ${{reasons.join('; ')}}.`;
    box.appendChild(hint);
  }}

  host.appendChild(box);
}}

function renderOutputBars(labels, probs) {{
  const host = document.getElementById('previewOutput');
  if (!host) return;
  const ls = Array.isArray(labels) ? labels : [];
  const ps = Array.isArray(probs) ? probs : [];
  if (!ls.length || !ps.length) {{
    host.innerHTML = '';
    return;
  }}
  // Build persistent structure ONCE — the toggle button must never be
  // destroyed by the prediction loop, or clicks get lost mid-press.
  let rows = host.querySelector('.out-rows');
  let btn = host.querySelector('.out-toggle');
  if (!rows || !btn) {{
    host.innerHTML = '<div class="out-rows"></div>';
    rows = host.querySelector('.out-rows');
    btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'out-toggle';
    btn.addEventListener('click', () => {{
      previewShowScore = !previewShowScore;
      try {{ persistPreviewState(); }} catch (e) {{}}
      renderOutputBars(host._lastLabels || [], host._lastProbs || []);
    }});
    host.appendChild(btn);
  }}
  host._lastLabels = ls;
  host._lastProbs = ps;
  rows.innerHTML = ls.map((label, idx) => {{
    const p = Math.max(0, Math.min(1, Number(ps[idx] || 0)));
    const pct = Math.round(p * 1000) / 10;
    const disp = previewShowScore ? p.toFixed(3) : `${{pct}}%`;
    return `
      <div class="out-row">
        <div>${{String(label)}}</div>
        <div class="out-bar">
          <div class="out-fill" style="width:${{pct}}%"></div>
          <div class="out-pct">${{disp}}</div>
        </div>
      </div>
    `;
  }}).join('');
  btn.textContent = previewShowScore ? 'Show %' : 'Show Score';
}}

async function refreshPreviewPrediction(token) {{
  if (!previewInputOn || !STATE.export_enabled) return;
  if (previewPredictInFlight) return;
  if (token !== previewPredictToken) return;
  previewPredictInFlight = true;
  previewPredictController = new AbortController();
  const note = document.getElementById('previewNote');
  try {{
    const res = await fetch(`${{baseUrl}}/preview/predict?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(previewSource)}}&preprocess=${{encodeURIComponent(previewPreprocessMode)}}&bg_dark=${{encodeURIComponent(previewThreshUserModified ? String(previewDarkThresh ?? '') : '')}}&bg_lum=${{encodeURIComponent(previewThreshUserModified ? String(previewLumThresh ?? '') : '')}}&ood_sign_min=${{encodeURIComponent(String(previewOodSignPctMin ?? 0.3))}}&ood_sign_max=${{encodeURIComponent(String(previewOodSignPctMax ?? 70.0))}}&ood_max_prob=${{encodeURIComponent(String(previewOodMaxProbMin ?? 0.40))}}&ood_entropy_max=${{encodeURIComponent(String(previewOodEntropyRatioMax ?? 0.85))}}&sign_seq=${{encodeURIComponent(signSequence().join(''))}}&sign_seqs=${{encodeURIComponent(signSequences().join(','))}}&reading=${{encodeURIComponent(String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr')))}}&_ts=${{Date.now()}}`, {{ signal: previewPredictController.signal }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Preview failed.');
    previewLastPredictData = data;
    applyPreviewSrcToPane(data);
    renderPreviewOutput(data);
    if (note) {{
      const top = previewHeadline(data);
      const roiMsg = data.processed_variant ? (previewPreprocessMode === 'manual_roi' ? ' · ROI/manual crop input' : ' · sign search + crop input') : '';
      const debugMsg = data.ood
        ? ` · OOD: ${{String(data.ood_reason || 'suppressed')}} · sign=${{Number(data.sign_pct || 0).toFixed(1)}}% · max=${{Number(data.max_prob || 0).toFixed(3)}} · ent=${{Number(data.entropy_ratio || 0).toFixed(3)}}`
        : ` · sign=${{Number(data.sign_pct || 0).toFixed(1)}}% · max=${{Number(data.max_prob || 0).toFixed(3)}} · ent=${{Number(data.entropy_ratio || 0).toFixed(3)}}`;
      note.textContent = top ? `${{top}}${{roiMsg}}${{debugMsg}}` : `${{roiMsg.trim()}}${{debugMsg}}`.trim();
    }}
  }} catch (err) {{
    if (err && (err.name === 'AbortError' || String(err).includes('aborted'))) return;
    if (note) note.textContent = String(err && err.message ? err.message : err);
    const outHost = document.getElementById('previewOutput');
    if (outHost) outHost.innerHTML = '';
  }} finally {{
    previewPredictInFlight = false;
    previewPredictController = null;
  }}
}}
// ── Preview ROI overlay: auto search box (same UX as the class-edit page) ──
// Green border = auto-detected shadow-search box, blue = manual ROI box.
function getPreviewImageContentRect() {{
  const stage = document.querySelector('#previewPane .preview-stage');
  const img = document.getElementById('previewImage');
  if (!stage || !img) return null;
  const stageRect = stage.getBoundingClientRect();
  if (!stageRect.width || !stageRect.height) return null;
  const naturalW = Number(img.naturalWidth || 0);
  const naturalH = Number(img.naturalHeight || 0);
  if (naturalW <= 0 || naturalH <= 0) {{
    return {{ offsetX: 0, offsetY: 0, width: stageRect.width, height: stageRect.height }};
  }}
  const scale = Math.min(stageRect.width / naturalW, stageRect.height / naturalH);
  const width = naturalW * scale;
  const height = naturalH * scale;
  const offsetX = (stageRect.width - width) * 0.5;
  const offsetY = (stageRect.height - height) * 0.5;
  return {{ offsetX, offsetY, width, height }};
}}
function updatePreviewRoiOverlay(crop, isManual, frameMatches = true) {{
  const overlay = document.getElementById('previewRoiOverlay');
  if (!overlay) return;
  const valid = Array.isArray(crop) && crop.length === 4;
  // ROI is a crop toggle: ON shows the cropped image itself, so the box is
  // hidden; OFF shows the full frame and draws the box to locate the ROI.
  // `crop` is normalized to the ORIGINAL frame, so it only maps onto a view
  // that keeps that frame (the raw or threshold full frame); cropped views
  // pass frameMatches = false as a second guard.
  const show = valid && !previewShowRoi && frameMatches;
  if (!show) {{
    overlay.style.display = 'none';
    return;
  }}
  const rect = getPreviewImageContentRect();
  if (!rect || !rect.width || !rect.height) {{
    overlay.style.display = 'none';
    return;
  }}
  const x1 = Math.max(0, Math.min(1, Number(crop[0] || 0)));
  const y1 = Math.max(0, Math.min(1, Number(crop[1] || 0)));
  const x2 = Math.max(x1, Math.min(1, Number(crop[2] || x1)));
  const y2 = Math.max(y1, Math.min(1, Number(crop[3] || y1)));
  overlay.style.display = 'block';
  overlay.style.left = `${{rect.offsetX + x1 * rect.width}}px`;
  overlay.style.top = `${{rect.offsetY + y1 * rect.height}}px`;
  overlay.style.width = `${{Math.max(0, (x2 - x1) * rect.width)}}px`;
  overlay.style.height = `${{Math.max(0, (y2 - y1) * rect.height)}}px`;
  // Green border = auto-detected, blue border = manual ROI.  Same missing
  // --accent as the class-preprocess overlay; see the note there.
  overlay.style.borderColor = isManual ? 'var(--accent, #1a73e8)' : '#4caf50';
}}
async function runPreviewUploadPrediction() {{
  if (!previewInputOn || !STATE.export_enabled) return;
  if (!previewUploadImageB64) {{
    // A sign project has a target but often no road-sign photo on disk —
    // draw a MATCH plate instead of showing an empty preview.  The guard
    // inside the synth handler prevents a request loop: it stores the plate
    // before any later predict call sees an empty image again.
    if (isSignOcrProject() && !previewSynthInFlight) {{
      const made = await handleSynthesizeSignPreview('match', {{auto: true}});
      if (made) return;
    }}
    const note = document.getElementById('previewNote');
    if (note) note.textContent = 'Click 📁 Upload Image... above (or drop a file on this frame) to test an upload.';
    return;
  }}
  if (previewPredictInFlight) return;
  previewPredictInFlight = true;
  const pane = document.getElementById('previewPane');
  const note = document.getElementById('previewNote');
  try {{
    const res = await fetch(`${{baseUrl}}/preview/upload`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        image_b64: previewUploadImageB64,
        filename: String(previewUploadFilename || ''),
        preprocess: previewPreprocessMode,
        bg_dark: previewThreshUserModified ? Number(previewDarkThresh ?? 0) : '',
        bg_lum: previewThreshUserModified ? Number(previewLumThresh ?? 0) : '', 
        ood_sign_min: Number(previewOodSignPctMin ?? 0.3),
        ood_sign_max: Number(previewOodSignPctMax ?? 70.0),
        ood_max_prob: Number(previewOodMaxProbMin ?? 0.40),
        ood_entropy_max: Number(previewOodEntropyRatioMax ?? 0.85),
        // The target sign travels with the request rather than being read
        // from the session: it is edited in the SPA and lives in STATE.train_cfg
        // until the project is saved, so the server has no other way to know
        // which sign this test image is being compared against.
        sign_sequence: signSequence().join(''),
        reading_order: String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr')),
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Preview failed.');
    previewLastPredictData = data;
    if (data.filename) previewUploadFilename = String(data.filename);
    const uploadName = document.getElementById('previewUploadName');
    if (uploadName) uploadName.textContent = previewUploadFilename || 'No image selected';
    applyPreviewSrcToPane(data);
    if (pane && !pane.dataset.ready) {{
      pane.innerHTML = '<div class="preview-empty">Click 📁 Upload Image... above, or drop an image here.</div>';
    }}
    renderPreviewOutput(data);
    if (note) {{
      const top = previewHeadline(data);
      const roiMsg = data.processed_variant ? (previewPreprocessMode === 'manual_roi' ? ' · ROI/manual crop input' : ' · sign search + crop input') : '';
      const debugMsg = data.ood
        ? ` · OOD: ${{String(data.ood_reason || 'suppressed')}} · sign=${{Number(data.sign_pct || 0).toFixed(1)}}% · max=${{Number(data.max_prob || 0).toFixed(3)}} · ent=${{Number(data.entropy_ratio || 0).toFixed(3)}}`
        : ` · sign=${{Number(data.sign_pct || 0).toFixed(1)}}% · max=${{Number(data.max_prob || 0).toFixed(3)}} · ent=${{Number(data.entropy_ratio || 0).toFixed(3)}}`;
      note.textContent = top
        ? `Upload: ${{String(previewUploadFilename || 'image')}} · ${{top}}${{roiMsg}}${{debugMsg}}`
        : `Upload: ${{String(previewUploadFilename || 'image')}}${{roiMsg}}${{debugMsg}}`;
    }}
  }} catch (err) {{
    if (note) note.textContent = String(err && err.message ? err.message : err);
  }} finally {{
    previewPredictInFlight = false;
  }}
}}
function startPreviewPredictLoop() {{
  stopPreviewPredictLoop();
  previewPredictToken += 1;
  const token = previewPredictToken;
  // Never kill the class-panel source live loop from here — these are two
  // independent views now.  The only shared thing is the backend live thread.
  if (previewSource === 'upload') {{
    runPreviewUploadPrediction();
    return;
  }}
  if (previewSource === 'webcam' || previewSource === 'device') {{
    // Keep the visual timer armed so the pane stays live whenever the
    // predict timer is paused (hold-to-capture / single capture).  While
    // the predict timer runs, the visual loop yields to it (gate inside
    // refreshPreviewVisualFrame).
    if (!previewVisualTimer) startPreviewVisualLoop();
  }}
  refreshPreviewPrediction(token);
  previewPredictTimer = window.setInterval(
    () => refreshPreviewPrediction(token),
    Math.max(50, Number(previewIntervalMs || 80))
  );
}}
async function setWebcamIndexGlobal(value, uid) {{
  const next = Number(value || 0);
  currentWebcamIndex = next;
  currentWebcamUid = String(uid || '');
  STATE.current_webcam_index = currentWebcamIndex;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&webcam_index=${{encodeURIComponent(currentWebcamIndex)}}&webcam_id=${{encodeURIComponent(currentWebcamUid)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update webcam.');
    if (openSourceKind === 'webcam') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=webcam`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function setDevicePortGlobal(value) {{
  currentSerialPort = String(value || '');
  STATE.current_serial_port = currentSerialPort;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_port=${{encodeURIComponent(currentSerialPort)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update serial port.');
    if (openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function setSerialBaudGlobal(value) {{
  currentSerialBaud = Number(value || currentSerialBaud || 115200);
  STATE.current_serial_baud = currentSerialBaud;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_baud=${{encodeURIComponent(String(currentSerialBaud))}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update baud rate.');
    if (openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function setSerialSyncGlobal(value) {{
  currentSerialSync = String(value || currentSerialSync || 'AA 55 AA').trim() || 'AA 55 AA';
  STATE.current_serial_sync = currentSerialSync;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_sync=${{encodeURIComponent(currentSerialSync)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update sync header.');
    currentSerialSync = String(data.serial_sync || currentSerialSync || 'AA 55 AA');
    STATE.current_serial_sync = currentSerialSync;
    if (openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function setSerialFrameSideGlobal(value) {{
  currentSerialFrameSide = Number(value || currentSerialFrameSide || 96);
  STATE.current_serial_frame_side = currentSerialFrameSide;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_frame_side=${{encodeURIComponent(String(currentSerialFrameSide))}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update device image size.');
    currentSerialFrameSide = Number(data.serial_frame_side || currentSerialFrameSide || 96);
    STATE.current_serial_frame_side = currentSerialFrameSide;
    if (openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function setSerialChannelsGlobal(value) {{
  currentSerialChannels = Number(value || currentSerialChannels || 1);
  STATE.current_serial_channels = currentSerialChannels;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_channels=${{encodeURIComponent(String(currentSerialChannels))}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update device color mode.');
    currentSerialChannels = Number(data.serial_channels || currentSerialChannels || 1);
    STATE.current_serial_channels = currentSerialChannels;
    if (openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
function buildPreviewSettingsMarkup() {{
  if (!previewSettingsOpen || !STATE.export_enabled) return '';
  // The input SOURCE is chosen in Settings, once, and the card's context bar
  // shows only the control that source needs.  It is deliberately absent
  // here: two switchers for one choice was the confusing part.
  const exportFields = `
    <label>
      Export Name
      <input id="previewExportName" value="${{String(exportModelName || 'tm')}}" placeholder="person_detect"/>
    </label>
    <label>
      Array Name (optional)
      <input id="previewExportArray" value="${{String(exportArrayName || '')}}" placeholder="g_person_detect_model_data"/>
    </label>
  `;
  const oodFields = `
    <label title="Minimum sign_pct before Preview treats the frame as empty / invalid.">
      OOD sign min %
      <input id="previewOodSignPctMin" type="number" min="0" max="100" step="0.1" value="${{Number(previewOodSignPctMin ?? 0.3)}}"/>
    </label>
    <label title="Maximum sign_pct before Preview treats the frame as too full / shadowed.">
      OOD sign max %
      <input id="previewOodSignPctMax" type="number" min="0" max="100" step="0.1" value="${{Number(previewOodSignPctMax ?? 70.0)}}"/>
    </label>
    <label title="Minimum softmax confidence required before Preview accepts a class result.">
      OOD max prob
      <input id="previewOodMaxProbMin" type="number" min="0" max="1" step="0.01" value="${{Number(previewOodMaxProbMin ?? 0.40)}}"/>
    </label>
    <label title="Maximum entropy ratio allowed before Preview suppresses to No Sign.">
      OOD entropy max
      <input id="previewOodEntropyRatioMax" type="number" min="0" max="1" step="0.01" value="${{Number(previewOodEntropyRatioMax ?? 0.85)}}"/>
    </label>
  `;
  // One panel regardless of source: the source-specific CONTROLS live in
  // #previewContextBar (camera picker / serial port / upload), so the gear
  // panel only owns things that are not a source choice — export names,
  // OOD gates and the webcam's preview rate.
  const rateField = previewSource === 'webcam'
    ? `
      <label>
        Preview Rate
        <select id="previewPreviewRate">
          <option value="60"${{Number(previewIntervalMs) === 60 ? ' selected' : ''}}>Fast</option>
          <option value="80"${{Number(previewIntervalMs) === 80 ? ' selected' : ''}}>Balanced</option>
          <option value="120"${{Number(previewIntervalMs) === 120 ? ' selected' : ''}}>Stable</option>
          <option value="180"${{Number(previewIntervalMs) === 180 ? ' selected' : ''}}>Low CPU</option>
        </select>
      </label>`
    : '';
  return `
    <div class="source-settings-panel" id="previewSettingsPanel">
      <div class="source-settings-grid">
        ${{exportFields}}
        ${{oodFields}}
        ${{rateField}}
      </div>
      <div class="source-settings-actions">
        <button class="source-settings-cancel" type="button" id="previewSettingsCancel">Close</button>
        <button class="source-settings-save" type="button" id="previewSettingsSave">Apply</button>
      </div>
    </div>
  `;
}}
function renderPreviewSettings() {{
  const host = document.getElementById('previewSettingsHost');
  if (!host) return;
  host.innerHTML = buildPreviewSettingsMarkup();
  updateFlow();
  const cancel = document.getElementById('previewSettingsCancel');
  const save = document.getElementById('previewSettingsSave');
  if (cancel) cancel.onclick = () => {{
    previewSettingsOpen = false;
    renderPreviewSettings();
  }};
  if (save) save.onclick = async () => {{
    try {{
      const nameEl = document.getElementById('previewExportName');
      const arrayEl = document.getElementById('previewExportArray');
      exportModelName = String(nameEl ? nameEl.value : exportModelName || 'tm').trim() || 'tm';
      exportArrayName = String(arrayEl ? arrayEl.value : exportArrayName || '').trim();
      persistExportSettings();
      const rateEl = document.getElementById('previewPreviewRate');
      if (rateEl) previewIntervalMs = Number(rateEl.value || previewIntervalMs || 80);
      const oodSignMinEl = document.getElementById('previewOodSignPctMin');
      const oodSignMaxEl = document.getElementById('previewOodSignPctMax');
      const oodMaxProbEl = document.getElementById('previewOodMaxProbMin');
      const oodEntropyEl = document.getElementById('previewOodEntropyRatioMax');
      previewOodSignPctMin = Math.max(0, Math.min(100, Number(oodSignMinEl ? oodSignMinEl.value : (previewOodSignPctMin ?? 0.3))));
      previewOodSignPctMax = Math.max(0, Math.min(100, Number(oodSignMaxEl ? oodSignMaxEl.value : (previewOodSignPctMax ?? 70.0))));
      previewOodMaxProbMin = Math.max(0, Math.min(1, Number(oodMaxProbEl ? oodMaxProbEl.value : (previewOodMaxProbMin ?? 0.40))));
      previewOodEntropyRatioMax = Math.max(0, Math.min(1, Number(oodEntropyEl ? oodEntropyEl.value : (previewOodEntropyRatioMax ?? 0.85))));
      persistPreviewState();
      await pushPreviewOodToBackend();
      previewSettingsOpen = false;
      renderPreviewSettings();
      renderPreviewCard();
      // Upload mode re-evaluates the stored image with the just-saved
      // thresholds; the other sources restart their normal loop.
      if (previewSource === 'upload') {{
        if (previewInputOn) runPreviewUploadPrediction();
      }} else if (previewInputOn) {{
        if (previewPredictTimer) stopPreviewPredictLoop();
        startPreviewPredictLoop();
      }} else {{
        if (previewVisualTimer) stopPreviewVisualLoop();
        startPreviewLoop();
      }}
    }} catch (err) {{
      toast(String(err && err.message ? err.message : err));
    }}
  }};
  scheduleLayoutResync();
}}
// ── Preview context bar ─────────────────────────────────────────────────
// Settings owns WHICH input the Preview card reads from.  The card shows
// only the one control that source needs, so the page never asks the user
// to pick the same thing twice:
//   webcam → compact camera select
//   device → compact serial-port select + Refresh
//   upload → an Upload button, a persistent file input, the file name
//            (plus drag-and-drop onto the preview frame)
function previewContextSource() {{
  // STATE.preview_source is the Settings-facing name for the same choice
  // (set by the master selector and persisted with the SPA).  Prefer it
  // when valid so a caller that updates STATE sees the bar follow even
  // before the next persistPreviewState().
  const fromState = String((STATE && STATE.preview_source) || '');
  const src = (fromState === 'device' || fromState === 'upload' || fromState === 'webcam')
    ? fromState
    : String(previewSource || 'webcam');
  return src === 'device' ? 'device' : (src === 'upload' ? 'upload' : 'webcam');
}}
function previewContextBarSignature() {{
  // The sign flag and target are part of the signature: switching project
  // type or typing a target character must add/remove/enable the synth
  // pills, and the bar otherwise only rebuilds when the source or option
  // lists change.
  const sign = isSignOcrProject()
    ? `1:${{signSequence().join('')}}:${{signSequences().join(',')}}:${{String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr'))}}:${{STATE.export_enabled ? '1' : '0'}}`
    : '0';
  return `${{previewContextSource()}}|${{(STATE.webcam_options || []).length}}|${{(STATE.serial_ports || []).length}}|${{sign}}`;
}}
function renderPreviewContextBar() {{
  const host = document.getElementById('previewContextBar');
  if (!host) return;
  const src = previewContextSource();
  // Rebuild only when the source (or the option lists it draws from)
  // changes.  Rebuilding a <select> on every render would drop focus and
  // reset a value the user is in the middle of choosing.
  if (host.dataset.signature !== previewContextBarSignature()) {{
    host.dataset.signature = previewContextBarSignature();
    host.dataset.source = src;
    host.innerHTML = '';
    const row = document.createElement('div');
    row.className = 'preview-context-row';
    if (src === 'webcam') {{
      const sel = document.createElement('select');
      sel.id = 'previewWebcamSelect';
      sel.title = 'Camera used by the Preview input';
      sel.innerHTML = buildWebcamOptions(currentWebcamIndex, currentWebcamUid);
      row.appendChild(sel);
    }} else if (src === 'device') {{
      const sel = document.createElement('select');
      sel.id = 'previewSerialPortSelect';
      sel.title = 'Serial port of the MCU device';
      sel.innerHTML = buildDeviceOptions(currentSerialPort);
      const refresh = document.createElement('button');
      refresh.type = 'button';
      refresh.id = 'previewSerialRefreshBtn';
      refresh.className = 'tm-btn';
      refresh.textContent = 'Refresh';
      refresh.title = 'Re-scan serial ports';
      row.appendChild(sel);
      row.appendChild(refresh);
    }} else {{
      const upload = document.createElement('button');
      upload.type = 'button';
      upload.id = 'previewUploadBtn';
      upload.className = 'upload-pick';
      upload.textContent = '📁 Upload Image...';
      const fileInput = document.createElement('input');
      fileInput.type = 'file';
      fileInput.id = 'previewFileInput';
      fileInput.accept = 'image/*';
      fileInput.style.display = 'none';
      const fname = document.createElement('span');
      fname.id = 'previewUploadName';
      fname.className = 'preview-upload-name';
      fname.textContent = previewUploadFilename || 'No image selected';
      row.appendChild(upload);
      row.appendChild(fileInput);
      row.appendChild(fname);
      // A sign project has a target and no road-sign photo.  These pills
      // draw one from the project's own samples, so the three-slot split
      // and the YES/NO verdict are testable with zero files on disk.  They
      // live in the upload source because that is the input they replace.
      if (isSignOcrProject()) {{
        const group = document.createElement('span');
        group.className = 'preview-synth-group';
        const plateSeqs = signSequences();
        if (plateSeqs.length > 1) {{
          const plateSel = document.createElement('select');
          plateSel.id = 'previewSynthPlateSelect';
          plateSel.title = 'Which allowed plate Synth should draw';
          plateSeqs.forEach((text, idx) => {{
            const opt = document.createElement('option');
            opt.value = String(idx);
            opt.textContent = `Plate #${{idx + 1}} (${{text}})`;
            plateSel.appendChild(opt);
          }});
          plateSel.value = String(Math.max(0, Math.min(plateSeqs.length - 1, Number(previewSynthPlateIndex) || 0)));
          plateSel.onchange = () => {{
            previewSynthPlateIndex = Math.max(0, Math.min(plateSeqs.length - 1, Number(plateSel.value) || 0));
          }};
          group.appendChild(plateSel);
        }}
        const defs = [
          ['match', '🧪 Synth Match', 'Combine current 3 target characters into a synthetic sign image', 'previewSynthMatchBtn'],
          ['reversed', '🔀 Synth Reversed', 'Combine in reversed order to test LTR/RTL direction', 'previewSynthRevBtn'],
          ['mismatch', '❌ Synth Mismatch', 'Combine with a wrong character to test NO verdict', 'previewSynthMismatchBtn'],
        ];
        defs.forEach(([variant, label, tip, id]) => {{
          const pill = document.createElement('button');
          pill.type = 'button';
          pill.id = id;
          pill.className = 'preview-synth-pill';
          pill.textContent = label;
          pill.title = tip;
          pill.dataset.synthVariant = variant;
          // No trained model = no verdict to test; no target = nothing to
          // combine.  Both are disabled rather than failing on click.
          pill.disabled = !signSequenceIsSet() || !STATE.export_enabled;
          group.appendChild(pill);
        }});
        row.appendChild(group);
      }}
    }}
    host.appendChild(row);
    bindPreviewContextBar();
    if (src === 'device') refreshSerialPorts(false, 'previewSerialPortSelect').catch(() => {{}});
  }} else if (src === 'upload') {{
    const fname = document.getElementById('previewUploadName');
    if (fname) fname.textContent = previewUploadFilename || 'No image selected';
  }}
}}
function bindPreviewContextBar() {{
  const camSel = document.getElementById('previewWebcamSelect');
  if (camSel) camSel.onchange = async () => {{
    const uid = camSel.selectedOptions && camSel.selectedOptions[0]
      ? String(camSel.selectedOptions[0].dataset.uid || '') : '';
    await setWebcamIndexGlobal(camSel.value, uid);
    persistPreviewState();
    if (previewInputOn) {{ stopPreviewPredictLoop(); startPreviewPredictLoop(); }}
  }};
  const portSel = document.getElementById('previewSerialPortSelect');
  if (portSel) portSel.onchange = async () => {{
    await setDevicePortGlobal(portSel.value);
    persistPreviewState();
    if (previewInputOn) {{ stopPreviewPredictLoop(); startPreviewPredictLoop(); }}
  }};
  const refreshBtn = document.getElementById('previewSerialRefreshBtn');
  if (refreshBtn) refreshBtn.onclick = () => {{
    refreshSerialPorts(false, 'previewSerialPortSelect').catch(() => {{}});
  }};
  const fileInput = document.getElementById('previewFileInput');
  if (fileInput) fileInput.onchange = () => {{
    handlePreviewUploadFile(fileInput.files && fileInput.files[0]);
  }};
  const uploadBtn = document.getElementById('previewUploadBtn');
  if (uploadBtn) uploadBtn.onclick = () => {{
    const el = document.getElementById('previewFileInput');
    if (el) el.click();
  }};
  document.querySelectorAll('#previewContextBar [data-synth-variant]').forEach((pill) => {{
    pill.onclick = () => {{ handleSynthesizeSignPreview(pill.dataset.synthVariant); }};
  }});
}}
// Drag-and-drop is part of the upload source, not a global gesture: a drop
// while Preview is reading the camera would silently switch what the model
// is looking at.
function bindPreviewDropTarget() {{
  const pane = document.getElementById('previewPane');
  if (!pane || pane.dataset.dropBound === '1') return;
  pane.dataset.dropBound = '1';
  const clear = () => pane.classList.remove('preview-drop-target');
  pane.addEventListener('dragover', (e) => {{
    if (previewContextSource() !== 'upload') return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
    pane.classList.add('preview-drop-target');
  }});
  pane.addEventListener('dragleave', clear);
  pane.addEventListener('drop', async (e) => {{
    if (previewContextSource() !== 'upload') return;
    e.preventDefault();
    clear();
    const file = e.dataTransfer && e.dataTransfer.files ? e.dataTransfer.files[0] : null;
    await handlePreviewUploadFile(file);
  }});
}}
// One-click "make me a sign to test": draws a composite plate from the
// project's own glyph samples (EMNIST fallback) on the backend, then shows
// it exactly like an upload — original plate, processed ROI, three 28x28
// slot chips and the YES/NO banner.  The backend also stores the bytes as
// the session's active preview frame, so a later Dark/Lum slider move or
// ROI/Orig tab switch re-predicts the SAME plate through `reuse=1`.
async function handleSynthesizeSignPreview(variant, opts) {{
  const options = opts || {{}};
  if (!isSignOcrProject()) {{
    if (!options.auto) toast('Synthetic signs are only available in a sign project.');
    return null;
  }}
  if (!STATE.export_enabled) {{
    if (!options.auto) toast('Train a model first — the synth plate needs a model to read it.');
    return null;
  }}
  if (!signSequenceIsSet()) {{
    if (!options.auto) toast('Set the 3-character target sign first.');
    return null;
  }}
  if (previewSynthInFlight) return null;
  previewSynthInFlight = true;
  const note = document.getElementById('previewNote');
  if (note && !options.auto) note.textContent = 'Drawing a synthetic sign from the project samples…';
  try {{
    const res = await fetch(`${{baseUrl}}/preview/synthesize_sign`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        variant: String(variant || 'match'),
        // The target is the SPA's editor state; the server persists it but
        // this request must test what is on screen right now.
        sign_sequence: signSequence().join(''),
        sign_sequences: signSequences(),
        plate_index: Number(previewSynthPlateIndex ?? 0),
        reading_order: String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr')),
        bg_dark: previewThreshUserModified ? Number(previewDarkThresh ?? 0) : '',
        bg_lum: previewThreshUserModified ? Number(previewLumThresh ?? 100) : '',
        // OOD MAX PROB is the single per-slot probability threshold.
        ood_max_prob: Number(previewOodMaxProbMin ?? 0.40),
        ood_entropy_max: Number(previewOodEntropyRatioMax ?? 0.85),
        ood_sign_min: Number(previewOodSignPctMin ?? 0.3),
        ood_sign_max: Number(previewOodSignPctMax ?? 70.0),
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to synthesize a sign image.');

    // A synth click means "read the model with this plate", so the Input
    // light must be ON and the source must be upload — the same state a
    // real file leaves behind.  persistPreviewState keeps that choice
    // across the SPA's re-renders.
    previewSource = 'upload';
    STATE.preview_source = 'upload';
    previewInputOn = true;
    const toggle = document.getElementById('previewInputToggle');
    if (toggle) toggle.checked = true;
    previewUploadImageB64 = String(data.image_b64 || '');
    previewUploadImageSrc = data.image_b64 ? `data:image/png;base64,${{data.image_b64}}` : '';
    previewUploadFilename = String(data.filename || 'synth_sign.png');
    previewLastPredictData = data;
    persistPreviewState();
    renderPreviewContextBar();
    applyPreviewSrcToPane(data);
    renderPreviewOutput(data);

    const uploadName = document.getElementById('previewUploadName');
    if (uploadName) uploadName.textContent = previewUploadFilename;
    if (note) {{
      const verdict = previewHeadline(data) || 'no verdict';
      const slots = (Array.isArray(data.slots) ? data.slots : [])
        .map((s) => `${{String((s && s.char) || '?')}}→${{String((s && s.label) || '—')}}`)
        .join(' ');
      const kind = String(data.synth_variant || variant || 'match');
      note.textContent = `Synth (${{kind}}): ${{previewUploadFilename}} · ${{verdict}}${{slots ? ' · ' + slots : ''}}`;
    }}
    if (!options.auto) toast(`Synthetic sign ready — ${{previewHeadline(data) || previewUploadFilename}}`);
    return data;
  }} catch (err) {{
    if (note) note.textContent = String(err && err.message ? err.message : err);
    if (!options.auto) toast(String(err && err.message ? err.message : err));
    return null;
  }} finally {{
    previewSynthInFlight = false;
  }}
}}
// The upload source with no file of its own asks for a match plate instead
// of showing an empty pane.  Called after auto-fill / training too, so the
// "ready-to-preview synthetic sign" the backend stored is what the user
// actually sees.
function maybeAutoSynthesizeSignPreview() {{
  if (!isSignOcrProject() || !STATE.export_enabled) return null;
  if (previewContextSource() !== 'upload' || !previewInputOn) return null;
  if (previewUploadImageB64) return null;
  return handleSynthesizeSignPreview('match', {{auto: true}});
}}
async function handlePreviewUploadFile(file) {{
  if (!file) return false;
  try {{
    const b64 = await fileToB64(file);
    const mime = String(file.type || 'image/png');
    previewUploadImageB64 = String(b64 || '');
    previewUploadImageSrc = `data:${{mime}};base64,${{b64}}`;
    previewUploadFilename = String(file.name || 'upload');
    previewSource = 'upload';
    previewInputOn = true;
    const inputToggle = document.getElementById('previewInputToggle');
    if (inputToggle) inputToggle.checked = true;
    persistPreviewState();
    renderPreviewCard();
    renderPreviewContextBar();
    await runPreviewUploadPrediction();
    return true;
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
    return false;
  }}
}}
function showTrainWarningModal(text) {{
  let m = document.getElementById('trainWarningModal');
  if (!m) {{
    m = document.createElement('div');
    m.id = 'trainWarningModal';
    m.style.cssText = 'position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);background:' + themeVar('--surface-elevated', '#fff') + ';border:2px solid #c62828;border-radius:12px;box-shadow:0 8px 40px rgba(0,0,0,0.35);padding:20px 24px;max-width:560px;width:90%;z-index:10000;font-size:14px;line-height:1.7;color:' + themeVar('--text', '#333') + ';';
    document.body.appendChild(m);
  }}
  const lines = String(text || '').split('|').map((l) => l.trim()).filter(Boolean);
  m.innerHTML =
    '<div style="font-weight:700;color:#c62828;margin-bottom:10px;font-size:15px;">Training Warnings</div>' +
    lines.map((l) => '<div style="margin-bottom:6px;">' + String(l).replace(/</g, '&lt;') + '</div>').join('') +
    '<button style="margin-top:12px;padding:8px 18px;border:0;border-radius:8px;background:#c62828;color:#fff;cursor:pointer;font-weight:600;" onclick="this.parentNode.remove()">Close</button>';
}}
function classAccuracyLines() {{
  const accs = STATE.trainClassAccuracies || null;
  if (!accs || Object.keys(accs).length === 0) return null;
  const lines = [];
  for (const k of Object.keys(accs)) {{
    const v = Number(accs[k]);
    lines.push(`${{k}}: ${{Math.round(v * 100)}}%`);
  }}
  return lines;
}}
function showClassAccuracyModal() {{
  const lines = classAccuracyLines();
  if (!lines) return;
  showTrainWarningModal('Model Trained — per-class accuracy|' + lines.join('|'));
}}
function renderTrainStatus() {{
  const el = document.getElementById('trainStatus');
  if (!el) return;
  const lines = classAccuracyLines();
  const staleWarn = (STATE.export_enabled && STATE._trained_fingerprint &&
                    datasetFingerprint() !== STATE._trained_fingerprint) ? ' ⚠ Stale' : '';
  if (STATE.export_enabled && lines) {{
    el.textContent = 'Model Trained · tap for accuracy' + staleWarn;
    el.title = lines.join(' · ') + (staleWarn ? ' | Dataset changed since last training' : '');
    el.style.cursor = 'pointer';
    el.style.color = staleWarn ? '#ef6c00' : (lines.some((t) => Number(t.split(': ')[1].replace('%','')) < 80) ? '#c62828' : '#2e7d32');
    el.onclick = () => showClassAccuracyModal();
  }} else {{
    el.textContent = (STATE.export_enabled ? 'Model Trained' : 'Not trained') + staleWarn;
    el.title = staleWarn ? 'Dataset changed since last training' : '';
    el.style.cursor = '';
    el.style.color = staleWarn ? '#ef6c00' : '';
    el.onclick = null;
  }}
}}
function bindPreviewThreshBar() {{
  const darkSlider = document.getElementById('previewDarkSlider');
  const darkNum = document.getElementById('previewDarkNum');
  const lumSlider = document.getElementById('previewLumSlider');
  const lumNum = document.getElementById('previewLumNum');
  if (!darkSlider || !lumSlider) return;
  if (darkSlider.dataset.bound === '1') {{
    darkSlider.value = String((previewDarkThresh ?? 0));
    darkNum.value = String((previewDarkThresh ?? 0));
    lumSlider.value = String((previewLumThresh ?? 100));
    lumNum.value = String((previewLumThresh ?? 100));
    return;
  }}
  darkSlider.dataset.bound = '1';
  const syncDark = (src) => {{
    const v = Math.max(0, Math.min(100, Number(src.value ?? 0)));
    if (previewDarkThresh !== v) {{
      previewDarkThresh = v;
      previewThreshUserModified = true;
      persistPreviewState();
      if (previewInputOn) {{ if (previewPredictTimer) stopPreviewPredictLoop(); startPreviewPredictLoop(); }}
    }}
    darkSlider.value = String(v);
    darkNum.value = String(v);
  }};
  const syncLum = (src) => {{
    const v = Math.max(50, Math.min(255, Number(src.value ?? 100)));
    if (previewLumThresh !== v) {{
      previewLumThresh = v;
      previewThreshUserModified = true;
      persistPreviewState();
      if (previewInputOn) {{ if (previewPredictTimer) stopPreviewPredictLoop(); startPreviewPredictLoop(); }}
    }}
    lumSlider.value = String(v);
    lumNum.value = String(v);
  }};
  darkSlider.oninput = () => syncDark(darkSlider);
  darkNum.onchange = () => syncDark(darkNum);
  lumSlider.oninput = () => syncLum(lumSlider);
  lumNum.onchange = () => syncLum(lumNum);
  darkSlider.value = String((previewDarkThresh ?? 0));
  darkNum.value = String((previewDarkThresh ?? 0));
  lumSlider.value = String((previewLumThresh ?? 100));
  lumNum.value = String((previewLumThresh ?? 100));
}}
// Set a checkbox's .checked WITHOUT firing any registered onchange /
// addEventListener('change') handler.  This is required for every "sync
// widget to memory-state" assignment during renderPreviewCard /
// bindPreviewControls: if we naively write `el.checked = X` while an
// onchange is already attached, the browser fires the change event, which
// (in our handlers) calls persistPreviewState() and triggers a Streamlit
// rerun MID-RENDER.  That Streamlit rerun aborts the in-flight render()
// before class panels get drawn — the exact symptom in the screenshot.
function setCheckedSilent(el, value) {{
  if (!el) return;
  value = !!value;
  if (el.checked === value) return;
  const cachedChange = el.onchange;
  const cachedHandlers = Object.prototype.hasOwnProperty.call(el, '_stHandlers') ? el._stHandlers : null;
  el.onchange = null;
  // removeEventListener cache not stored on el in this codebase — onchange
  // only via .onchange property, so the above is sufficient.
  try {{ el.checked = value; }} finally {{
    el.onchange = cachedChange;
    if (cachedHandlers !== null) el._stHandlers = cachedHandlers;
  }}
}}
function renderPreviewCard() {{
  // Keep the module-level source and STATE.preview_source (the name the
  // Settings selector persists) in lockstep — the bar, the loops and the
  // upload path all branch on the same value.
  const stateSource = String((STATE && STATE.preview_source) || '');
  if (stateSource === 'webcam' || stateSource === 'device' || stateSource === 'upload') {{
    previewSource = stateSource;
  }}
  const pane = document.getElementById('previewPane');
  const note = document.getElementById('previewNote');
  const output = document.getElementById('previewOutput');
  const toggle = document.getElementById('previewInputToggle');
  const rawToggle = document.getElementById('previewRawToggle');
  if (rawToggle) {{
    setCheckedSilent(rawToggle, !!previewShowRaw);
    // IMPORTANT: do NOT reassign rawToggle.onchange here.  The change
    // handler (with proper defer to avoid mid-render Streamlit reruns) is
    // registered ONCE in bindPreviewControls() via dataset.bound='1'.
    // Overwriting it here would turn the deferral off and cause class
    // panels to render blank (the exact screenshot bug).
  }}
  const roiToggle = document.getElementById('previewRoiToggle');
  const settingsBtn = document.getElementById('previewSettingsToggle');
  const modeTabs = Array.from(document.querySelectorAll('[data-preview-mode]'));
  if (!pane || !note || !output || !toggle || !roiToggle) return;
  if (!STATE.export_enabled) {{
    stopPreviewPredictLoop();
    previewSettingsOpen = false;
    renderPreviewSettings();
    const ctx = document.getElementById('previewContextBar');
    if (ctx) {{
      ctx.innerHTML = '';
      ctx.dataset.signature = '';
      ctx.dataset.source = '';
    }}
    pane.removeAttribute('data-ready');
    pane.innerHTML = '<div class="preview-empty">Train a model on the left to enable preview.</div>';
    output.innerHTML = '';
    note.textContent = 'You must train a model on the left before you can preview it here.';
    setCheckedSilent(toggle, false);
    toggle.disabled = true;
    setCheckedSilent(roiToggle, false);
    roiToggle.disabled = true;
    if (rawToggle) {{
      setCheckedSilent(rawToggle, false);
      rawToggle.disabled = true;
    }}
    if (settingsBtn) settingsBtn.disabled = true;
    const bar = document.getElementById('previewThreshBar');
    if (bar) bar.style.display = 'none';
    modeTabs.forEach((btn) => {{
      btn.disabled = true;
      btn.classList.remove('active');
    }});
    return;
  }}
  toggle.disabled = false;
  roiToggle.disabled = false;
  if (rawToggle) rawToggle.disabled = false;
  if (settingsBtn) settingsBtn.disabled = false;
  const bar = document.getElementById('previewThreshBar');
  if (bar) bar.style.display = '';
  modeTabs.forEach((btn) => {{
    const active = String(btn.getAttribute('data-preview-mode') || '') === String(previewPreprocessMode || 'auto_by_label');
    btn.disabled = false;
    btn.classList.toggle('active', active);
  }});
  setCheckedSilent(toggle, !!previewInputOn);
  setCheckedSilent(roiToggle, !!previewShowRoi);
  bindPreviewThreshBar();
  renderPreviewSettings();
  renderPreviewContextBar();
  bindPreviewDropTarget();
  const staticSrc = previewSource === 'upload' ? previewUploadImageSrc : latestPreviewImage();
  if (!pane.dataset.ready) {{
    pane.innerHTML = staticSrc
      ? `<div class="preview-stage"><img id="previewImage" src="${{staticSrc}}" alt="Preview"/><div class="roi-overlay" id="previewRoiOverlay" style="display:none"></div></div>`
      : '<div class="preview-empty">Turn on Input to preview live predictions.</div>';
    pane.dataset.ready = '1';
  }} else if (!previewInputOn) {{
    // Input OFF: hold the last live frame on screen (closing Input should
    // freeze the most recent prediction, not swap in an old sample image).
    // The project's static sample image is only used when the pane was
    // just (re)initialized — fresh load or project reopen — and has no
    // image yet.
    const img = document.getElementById('previewImage');
    if (img && !img.getAttribute('src')) {{
      if (staticSrc) img.src = staticSrc;
    }}
    // No new prediction is coming, so re-apply the cached one: the ROI
    // toggle must still show/hide its box on the held frame.
    if (previewLastPredictData) applyPreviewSrcToPane(previewLastPredictData);
    else updatePreviewRoiOverlay(null, false);
  }}
  bindLayoutImageObservers(pane);
  scheduleLayoutResync();
  if (previewInputOn) {{
    if (!previewPredictTimer) startPreviewPredictLoop();
  }} else {{
    if (previewPredictTimer) stopPreviewPredictLoop();
    output.innerHTML = '';
    note.textContent = previewSource === 'upload'
      ? (previewUploadFilename ? `Upload ready: ${{previewUploadFilename}}` : 'Choose an image in settings to test upload preview.')
      : 'Model ready. Turn on Input to preview it here.';
    // Webcam/Device path on Input OFF: the Preview pane stays on the last
    // prediction / empty state — the visual loop must NOT run here (it
    // opened the camera and streamed live frames without the user toggling
    // Input).  The class-panel source loop (startPreviewLoop alias) is
    // independent and only ticks when a class source is actually open
    // (guarded inside refreshPreviewImage).
    if (previewVisualTimer) stopPreviewVisualLoop();
    if (openSourceClass && (openSourceKind === 'webcam' || openSourceKind === 'device')) startPreviewLoop();
  }}
}}
function refreshStaticPreviewFromDom() {{
  const img = document.getElementById('previewImage');
  if (img && img.getAttribute('src')) previewLastStaticImageSrc = img.getAttribute('src');
}}
function applyPreviewSrcToPane(data) {{
  const pane = document.getElementById('previewPane');
  if (!pane) return;
  const hasData = !!data;
  const rawFromData = hasData && data.image_b64 ? `data:image/png;base64,${{data.image_b64}}` : '';
  const roiFromData = hasData && data.processed_image_b64 ? `data:image/png;base64,${{data.processed_image_b64}}` : '';
  const cropFromData = hasData && data.crop_image_b64 ? `data:image/png;base64,${{data.crop_image_b64}}` : '';
  const fullFromData = hasData && data.full_image_b64 ? `data:image/png;base64,${{data.full_image_b64}}` : '';
  const uploadedSrc = previewSource === 'upload' ? previewUploadImageSrc : '';
  const rawSrc = rawFromData || uploadedSrc || previewLastStaticImageSrc || '';
  const fullSrc = fullFromData || rawSrc;
  const rawCropSrc = cropFromData || rawSrc;
  // ROI is a CROP toggle:
  //   OFF -> full frame (threshold by default, raw when Orig is on) + box
  //   ON  -> cropped content inside the ROI (threshold/raw), no box
  // Orig is the raw/threshold toggle:
  //   OFF -> thresholded black-on-white view (default, so sliders are visible)
  //   ON  -> original unprocessed grayscale
  let src;
  let fullFrame;
  if (previewShowRoi) {{
    src = previewShowRaw ? rawCropSrc : (roiFromData || rawSrc);
    fullFrame = false;
  }} else {{
    src = previewShowRaw ? rawSrc : fullSrc;
    fullFrame = true;
  }}
  // `ready` is set even when the pane shows the Input-off empty state, so
  // the FIRST prediction (including a synth plate clicked before any image
  // was selected) must be able to replace that placeholder with a stage.
  if (!pane.dataset.ready || (src && !document.getElementById('previewImage'))) {{
    pane.innerHTML = `<div class="preview-stage"><img id="previewImage" alt="Preview"/><div class="roi-overlay" id="previewRoiOverlay" style="display:none"></div></div>`;
    pane.dataset.ready = '1';
  }}
  const img = document.getElementById('previewImage');
  if (img && src) {{
    img.src = src;
    previewLastStaticImageSrc = src;
  }}
  const cropRect = hasData && Array.isArray(data.crop) && data.crop.length === 4 ? data.crop : null;
  updatePreviewRoiOverlay(cropRect, previewPreprocessMode === 'manual_roi', fullFrame);
  return src;
}}
function rerenderPreviewPaneFromCache() {{
  refreshStaticPreviewFromDom();
  applyPreviewSrcToPane(previewLastPredictData || null);
}}
function bindPreviewControls() {{
  const toggle = document.getElementById('previewInputToggle');
  const rawToggle = document.getElementById('previewRawToggle');
  const roiToggle = document.getElementById('previewRoiToggle');
  const settings = document.getElementById('previewSettingsToggle');
  const modeTabs = Array.from(document.querySelectorAll('[data-preview-mode]'));
  if (!toggle || !roiToggle) return;
  if (STATE.export_enabled) {{
    setCheckedSilent(rawToggle, !!previewShowRaw);
    setCheckedSilent(roiToggle, !!previewShowRoi);
    setCheckedSilent(toggle, !!previewInputOn);
  }}
  // Bind each control independently so renderPreviewCard() can be re-run
  // after a view toggle without dropping change handlers.
  const onViewToggleChanged = () => {{
    // First sync variables from the real DOM state — if the user just clicked
    // this toggle, .checked is up-to-date; if the call came from persist, the
    // values equal anyway (no-op).
    if (rawToggle) previewShowRaw = !!rawToggle.checked;
    previewShowRoi = !!roiToggle.checked;
    rerenderPreviewPaneFromCache();
    // Defer anything that triggers Streamlit rerun / re-render / async HTTP
    // out of this synchronous change handler — otherwise persistPreviewState
    // fires a widget sync mid-render, the parent iframe tears down the DOM
    // we're half-way building, and class panels render blank.
    setTimeout(() => {{
      persistPreviewState();
      renderPreviewCard();
      if (previewInputOn && previewSource === 'upload') {{
        runPreviewUploadPrediction();
      }} else if (previewInputOn) {{
        if (previewPredictTimer) stopPreviewPredictLoop();
        startPreviewPredictLoop();
      }}
    }}, 0);
  }};
  if (rawToggle && rawToggle.dataset.bound !== '1') {{
    rawToggle.dataset.bound = '1';
    rawToggle.onchange = onViewToggleChanged;
  }}
  if (roiToggle.dataset.bound !== '1') {{
    roiToggle.dataset.bound = '1';
    roiToggle.onchange = onViewToggleChanged;
  }}
  if (toggle.dataset.bound !== '1') {{
    toggle.dataset.bound = '1';
    toggle.onchange = () => {{
      // Preview Input toggle change.  IMPORTANT: defer Streamlit state sync
      // and HTTP calls using setTimeout(0) so that a programmatic .checked=
      // assignment (from the earlier render/bind) doesn't abort the current
      // render via Streamlit rerun.
      previewInputOn = !!toggle.checked;
      if (!previewInputOn) {{
        stopPreviewPredictLoop();
        // Input OFF: the Preview pane freezes on the last prediction — the
        // visual loop must not keep streaming (it opened the camera and
        // showed live frames without the user toggling Input).  The backend
        // live worker is released in the deferred block below — if a
        // class-panel view still uses this source, its /live/frame poll
        // revives the shared worker on the next tick (server-side
        // self-healing), so the class view never freezes.
        stopPreviewVisualLoop();
        if (previewSource === 'device') {{
          clearPreviewBlob();
        }}
      }}
      setTimeout(async () => {{
        persistPreviewState();
        if (!previewInputOn) {{
          // Release the camera / serial session when inference is turned
          // off — the camera must not stay on once nothing is displaying
          // it.  Any class-panel view still using the source self-heals
          // via /live/frame revival.
          try {{
            await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(previewSource)}}`);
          }} catch (e) {{}}
        }} else if (previewInputOn) {{
          if (previewSource === 'upload') {{
            startPreviewPredictLoop();
          }} else if (previewSource === 'webcam' || previewSource === 'device') {{
            try {{ await ensureOpenSourceLive({{silent:true, silentFail:true}}); }} catch (e) {{}}
            startPreviewPredictLoop();
          }}
        }}
        renderPreviewCard();
      }}, 0);
    }};
  }}
  if (settings && settings.dataset.bound !== '1') {{
    settings.dataset.bound = '1';
    settings.onclick = () => {{
      previewSettingsOpen = !previewSettingsOpen;
      renderPreviewSettings();
    }};
  }}
  modeTabs.forEach((btn) => {{
    if (btn.dataset.bound === '1') return;
    btn.dataset.bound = '1';
    btn.onclick = () => {{
      previewPreprocessMode = String(btn.getAttribute('data-preview-mode') || 'auto_by_label');
      setTimeout(() => {{
        persistPreviewState();
        renderPreviewCard();
        if (previewInputOn) {{
          if (previewPredictTimer) stopPreviewPredictLoop();
          startPreviewPredictLoop();
        }}
      }}, 0);
    }};
  }});
}}
function updateOpenSamplesPanel(className) {{
  const host = document.getElementById(`samplesHost-${{cssSafe(className)}}`);
  if (!host) return;
  host.innerHTML = buildSamplesMarkup(className);
  bindSampleDeleteButtons(host, className);
  bindLayoutImageObservers(host);
  setClassSampleCountLabel(className);
  renderPreviewCard();
  syncTrainUi();
  scheduleLayoutResync();
}}
function applyClassesState(nextClasses, oldClasses) {{
  const prevClasses = Array.isArray(oldClasses) ? oldClasses.slice() : STATE.classes.slice();
  const oldCounts = Object.assign({{}}, STATE.counts || {{}});
  const oldPreviews = Object.assign({{}}, STATE.sample_previews || {{}});
  const oldProcessed = Object.assign({{}}, STATE.processed_previews || {{}});
  const oldClassPreprocess = Object.assign({{}}, STATE.class_preprocess || {{}});
  const oldSamplePreprocess = normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}});
  const nextCounts = {{}};
  const nextPreviews = {{}};
  const nextProcessed = {{}};
  const nextClassPreprocess = {{}};
  const nextSamplePreprocess = {{}};
  (Array.isArray(nextClasses) ? nextClasses : []).forEach((name, idx) => {{
    const prevName = prevClasses[idx];
    nextCounts[name] = Number(nullish(oldCounts[name], nullish(oldCounts[prevName], 0)));
    nextPreviews[name] = normalizePreviewList(
      Array.isArray(oldPreviews[name]) ? oldPreviews[name] : oldPreviews[prevName]
    );
    nextProcessed[name] = normalizePreviewList(
      Array.isArray(oldProcessed[name]) ? oldProcessed[name] : oldProcessed[prevName]
    );
    nextClassPreprocess[name] = normalizeClassPreprocessConfig(oldClassPreprocess[name] || oldClassPreprocess[prevName]);
    const sampleMap = oldSamplePreprocess[name] || oldSamplePreprocess[prevName];
    if (sampleMap && typeof sampleMap === 'object' && Object.keys(sampleMap).length) {{
      nextSamplePreprocess[name] = normalizeSamplePreprocessMap({{ [name]: sampleMap }})[name] || {{}};
    }}
  }});
  STATE.classes = Array.isArray(nextClasses) ? nextClasses.slice() : [];
  STATE.counts = nextCounts;
  STATE.sample_previews = nextPreviews;
  STATE.processed_previews = nextProcessed;
  STATE.class_preprocess = nextClassPreprocess;
  STATE.sample_preprocess = nextSamplePreprocess;
}}
async function clearSamples(className) {{
  if (!className) return;
  const currentCount = Number(STATE.counts[className] || 0);
  if (currentCount <= 0) {{
    toast('This class has no samples to clear.');
    return;
  }}
  const ok = await showConfirmDialog(
    'Clear all samples?',
    `All ${{currentCount}} samples in "${{String(className || '')}}" will be deleted. This cannot be undone.`,
    'Clear All'
  );
  if (!ok) return;
  try {{
    const res = await fetch(`${{baseUrl}}/samples/clear`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        class: className
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to clear samples.');
    const next = data.state || {{}};
    STATE.counts[className] = Number(next.count || 0);
    STATE.sample_previews[className] = normalizePreviewList(next.previews);
    STATE.processed_previews[className] = [];
    if (STATE.sample_preprocess && typeof STATE.sample_preprocess === 'object') {{
      try {{ delete STATE.sample_preprocess[className]; }} catch (err) {{}}
    }}
    if (classPreprocessOpen && classPreprocessClass === className) {{
      classPreprocessSamples = [];
      classPreprocessSampleIndex = 0;
      classPreprocessProcessedSrc = '';
      renderClassPreprocessModal();
    }}
    const openHost = document.getElementById(`samplesHost-${{cssSafe(className)}}`);
    if (openHost) {{
      updateOpenSamplesPanel(className);
    }} else {{
      render();  // collapsed card: rebuild the summary strip + count
    }}
    recomputeTrainEnabled();
    refreshTrainRec();
    syncTrainUi();
    toast('All samples cleared.');
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function deleteSample(className, filename) {{
  if (!className || !filename) return;
  try {{
    const res = await fetch(`${{baseUrl}}/samples/delete`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        class: className,
        filename
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to delete sample.');
    const next = data.state || {{}};
    STATE.counts[className] = Number(next.count || 0);
    STATE.sample_previews[className] = normalizePreviewList(next.previews);
    if (classPreprocessOpen && classPreprocessClass === className) {{
      classPreprocessSamples = normalizePreviewList(classPreprocessSamples).filter((item) => String(previewFilename(item) || '') !== String(filename || ''));
    }}
    STATE.processed_previews[className] = normalizePreviewList(
      normalizePreviewList(STATE.processed_previews[className]).filter((item) => String(previewFilename(item) || '') !== String(filename || ''))
    );
    deleteSamplePreprocessConfig(className, filename);
    if (classPreprocessOpen && classPreprocessClass === className) {{
      const remain = normalizePreviewList(STATE.sample_previews[className]);
      if (!remain.length) {{
        classPreprocessSampleIndex = 0;
        classPreprocessProcessedSrc = '';
      }} else if (classPreprocessSampleIndex >= remain.length) {{
        classPreprocessSampleIndex = remain.length - 1;
      }}
      renderClassPreprocessModal();
      refreshClassProcessedPreview();
    }}
    recomputeTrainEnabled();
  refreshTrainRec();
    syncTrainUi();
    const openHost = document.getElementById(`samplesHost-${{cssSafe(className)}}`);
    if (openHost) {{
      updateOpenSamplesPanel(className);
    }} else {{
      render();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
function bindSampleDeleteButtons(scope, className) {{
  if (!scope || !className) return;
  const buttons = scope.querySelectorAll('.sample-delete[data-filename]');
  buttons.forEach((btn) => {{
    if (btn.dataset.bound === '1') return;
    btn.dataset.bound = '1';
    btn.onclick = async (e) => {{
      e.preventDefault();
      e.stopPropagation();
      const filename = String(btn.dataset.filename || '');
      await deleteSample(className, filename);
    }};
  }});
}}
async function renameClass(oldName) {{
  const nextNameRaw = window.prompt('Rename class', oldName);
  if (nextNameRaw == null) return;
  const nextName = String(nextNameRaw).trim();
  if (!nextName || nextName === oldName) return;
  try {{
    const res = await fetch(`${{baseUrl}}/classes/rename`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        old_name: oldName,
        new_name: nextName
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to rename class.');
    applyClassesState(Array.isArray(data.classes) ? data.classes : STATE.classes, STATE.classes);
    if (classPreprocessOpen && classPreprocessClass === oldName) {{
      closeClassPreprocessEditor();
    }}
    if (openSourceClass === oldName) openSourceClass = nextName;
    if (holdRecordClass === oldName) holdRecordClass = nextName;
    persistOpenSourceState();
    recomputeTrainEnabled();
  refreshTrainRec();
    render();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function addClass() {{
  try {{
    const prev = STATE.classes.slice();
    const res = await fetch(`${{baseUrl}}/classes/add`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to add class.');
    applyClassesState(Array.isArray(data.classes) ? data.classes : prev, prev);
    recomputeTrainEnabled();
  refreshTrainRec();
    render();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
// ── OCR character selector ──────────────────────────────────────────────
// In an OCR project the user already knows the target is one of A-Z/0-9, so
// the image-project order of operations — invent a class, then rename it to
// what you actually meant — asks the question backwards.  This panel replaces
// the "Add a class" node for OCR projects only.
//
// It renders no cards and owns no capture state.  Choosing characters writes
// STATE.classes through the same applyClassesState() + render() pair that
// addClass() uses, so the cards, sample strips, webcam panels, drag-reorder
// and per-class editors below are the untouched image-project code with a
// different name in them.  Nothing here knows what a sample is.
const OCR_CHAR_GRID = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.split('');

// Outside the panel because the panel does not control its own lifetime: any
// class edit rebuilds this whole column, and a half-typed entry should not go
// with it.
let ocrCharDraft = '';

// The characters that already have a class, and the class name holding each.
// Derived from STATE.classes rather than tracked alongside it, so there is one
// source of truth: rename a card to "Q" and the Q chip lights up, delete it and
// the chip clears, and a class named "sign" (typed here, or imported) simply
// has no chip while still rendering as an ordinary card.
function ocrTrainedChars() {{
  const out = {{}};
  (Array.isArray(STATE.classes) ? STATE.classes : []).forEach((raw) => {{
    const ch = String(raw == null ? '' : raw).trim().toUpperCase();
    if (ch.length === 1) out[ch] = String(raw);
  }});
  return out;
}}

// One round-trip for the whole selection.  Chaining addClass() instead would
// allocate "Class N" names — not the characters the user picked — and redraw
// the column once per character.
//
// fromDraft: the free-text box sent this.  The draft is cleared *before*
// render(), because render() rebuilds the box from it and would otherwise
// repaint the text the user just submitted.
async function addOcrCharacters(chars, fromDraft) {{
  const want = (Array.isArray(chars) ? chars : [])
    .map((c) => String(c == null ? '' : c).trim())
    .filter((c, i, all) => c && all.indexOf(c) === i);
  if (!want.length) return true;
  try {{
    const prev = STATE.classes.slice();
    const res = await fetch(`${{baseUrl}}/classes/add_named`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      // prune_placeholders: a fresh project seeds "Class 1"/"Class 2" so the
      // image flow always has something to train.  In OCR they are dead
      // weight, so the first character picked clears them out in the same
      // write.  The server only removes ones that are still empty and still
      // auto-named — a placeholder someone has captured into is left alone.
      body: JSON.stringify({{session: STATE.session, names: want, prune_placeholders: true}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to add characters.');
    if (fromDraft) ocrCharDraft = '';
    // The server is authoritative and returns the whole ordered list; the
    // fallback keeps the column honest if it ever answers without one.
    applyClassesState(Array.isArray(data.classes) ? data.classes : prev, prev);
    recomputeTrainEnabled();
    refreshTrainRec();
    render();
    // Newly added characters with no samples yet can be filled with template
    // captures when the developer shortcut is on (Settings -> Advanced).
    autoSeedEmnistForNewClasses(want);
    return true;
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
    return false;
  }}
}}

// The classes still wearing an auto-generated name and holding nothing.  Two
// callers need this exact question answered — the selector below (which
// promises to clear them out) and the empty-state gate in render() (which
// treats "nothing but these" as a project nobody has set up yet) — so the
// definition lives in one place and the two can never drift apart.
function ocrPlaceholderClasses() {{
  return (Array.isArray(STATE.classes) ? STATE.classes : []).filter((c) => {{
    const name = String(c == null ? '' : c);
    const count = Number(STATE.counts ? STATE.counts[name] : 0) || 0;
    return count === 0 && /^Class [0-9]+$/.test(name.trim());
  }});
}}

// True while every class in the project is still a placeholder — i.e. the user
// has not picked a character yet.  Also true for an empty list, which is the
// same situation one step further along.  One real class (a character, or a
// label typed by hand) makes this false and the ordinary card column renders.
function ocrIsFreshProject() {{
  const all = Array.isArray(STATE.classes) ? STATE.classes : [];
  return ocrPlaceholderClasses().length === all.length;
}}

// Mounted in the Edit menu's modal, so it renders no heading of its own: the
// modal head carries the title and the class count.
function renderOcrCharacterSelector(root) {{
  const have = ocrTrainedChars();
  const panel = document.createElement('div');
  panel.className = 'charsel';

  const help = document.createElement('div');
  help.className = 'charsel-help';
  help.textContent = 'Pick the characters this model should recognise. Each one becomes a normal class card in the workspace — capture samples for it the way you would in any project.';
  // A fresh project seeds two "Class N" placeholders so the image flow always
  // has something to train.  They are dead weight in an OCR project, so the
  // add call asks the server to clear them out (see addOcrCharacters) — but
  // only ones that are still empty AND still wearing an auto-generated name,
  // so a placeholder someone has captured into is never touched.  Naming them
  // here sets that expectation before the first click, rather than letting two
  // cards vanish unexplained.
  const placeholders = ocrPlaceholderClasses();
  if (placeholders.length) {{
    const one = placeholders.length === 1;
    help.textContent += ` The empty placeholder${{one ? '' : 's'}} ${{placeholders.join(', ')}} ${{one ? 'is' : 'are'}} still in the list, and will be removed automatically as soon as you pick a character.`;
  }}
  panel.appendChild(help);

  const grid = document.createElement('div');
  grid.className = 'charsel-grid';
  OCR_CHAR_GRID.forEach((ch) => {{
    const on = !!have[ch];
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'charchip';
    chip.textContent = ch;
    // A disabled button cannot fire the click that would add it twice, so the
    // "already selected" rule and its appearance are the same fact.
    chip.disabled = on;
    chip.title = on
      ? `Already training as "${{have[ch]}}" — delete that card to remove it.`
      : `Add ${{ch}} as a class`;
    if (!on) chip.onclick = () => addOcrCharacters([ch]);
    grid.appendChild(chip);
  }});
  panel.appendChild(grid);

  const row = document.createElement('div');
  row.className = 'charsel-row';

  const bulk = (label, chars) => {{
    const todo = chars.filter((c) => !have[c]);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'charsel-btn';
    btn.textContent = label;
    btn.disabled = !todo.length;
    btn.onclick = () => addOcrCharacters(todo);
    return btn;
  }};
  row.appendChild(bulk('+ A–Z', OCR_CHAR_GRID.slice(0, 26)));
  row.appendChild(bulk('+ 0–9', OCR_CHAR_GRID.slice(26)));

  const input = document.createElement('input');
  input.className = 'charsel-input';
  input.type = 'text';
  input.placeholder = 'Others, e.g. A B C 1,2';
  input.value = ocrCharDraft;
  input.oninput = () => {{ ocrCharDraft = input.value; }};
  const commit = () => {{
    // Accepts "ABC", "A B C" and "A,B,C" alike.  Splitting on runs of
    // non-alphanumerics also lets a multi-character label like "STOP" through
    // intact, for OCR work that is not single-glyph.
    const parts = ocrCharDraft.split(/[^0-9A-Za-z]+/).filter(Boolean);
    if (!parts.length) return Promise.resolve(false);
    // Returned, not dropped: the caller can then wait for the class list to
    // actually land before touching the DOM again.
    return addOcrCharacters(parts, true);
  }};
  input.onkeydown = (e) => {{
    if (e.key === 'Enter') {{ e.preventDefault(); commit(); }}
  }};
  row.appendChild(input);

  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.className = 'charsel-btn';
  addBtn.textContent = 'Add';
  addBtn.onclick = commit;
  row.appendChild(addBtn);

  panel.appendChild(row);
  root.appendChild(panel);
}}

// The onboarding hero, shown in place of the card column while an OCR project
// holds nothing but placeholders.  It suppresses those cards rather than
// sitting above them: two empty "Class N" cards are precisely the confusion it
// exists to remove, and rendering them behind the message would leave the
// workspace looking like an Image project with a note taped to it.
function renderOcrEmptyState(root) {{
  const box = document.createElement('div');
  box.className = 'ocr-empty';

  const icon = document.createElement('div');
  icon.className = 'ocr-empty-icon';
  // The same glyph as the Edit menu item, so the button and the menu entry it
  // is a shortcut to read as one action.
  icon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 7 4 4 20 4 20 7"></polyline><line x1="9" y1="20" x2="15" y2="20"></line><line x1="12" y1="4" x2="12" y2="20"></line></svg>`;
  box.appendChild(icon);

  const title = document.createElement('div');
  title.className = 'ocr-empty-title';
  title.textContent = 'No characters selected';
  box.appendChild(title);

  const sub = document.createElement('div');
  sub.className = 'ocr-empty-sub';
  sub.textContent = 'Start by choosing what this model should recognise. Every character you pick becomes a normal class card you can capture samples for.';
  box.appendChild(sub);

  const cta = document.createElement('button');
  cta.type = 'button';
  cta.className = 'ocr-empty-cta';
  cta.id = 'ocrEmptyCta';
  cta.textContent = 'Select Characters to Train';
  cta.onclick = () => openOcrCharModal();
  box.appendChild(cta);

  root.appendChild(box);
}}

// The character grid is a project-level decision rather than a per-card action,
// so it lives in the toolbar's Edit menu.  Inline at the foot of the card
// column it was the far end of a long scroll on any project with more than a
// few characters.
//
// The flag outlives render() for the same reason classPreprocessOpen does: the
// column is rebuilt on every class edit, and the modal has to still be open —
// showing the freshly up-to-date chips — when that rebuild finishes.
let ocrCharModalOpen = false;

function openOcrCharModal() {{
  ocrCharModalOpen = true;
  renderOcrCharModal();
}}
function closeOcrCharModal() {{
  ocrCharModalOpen = false;
  renderOcrCharModal();
}}

function renderOcrCharModal() {{
  const host = document.getElementById('ocrCharModal');
  if (!host) return;
  if (!ocrCharModalOpen) {{
    host.style.display = 'none';
    host.innerHTML = '';
    return;
  }}
  host.style.display = 'flex';
  host.innerHTML = '';
  const card = document.createElement('div');
  card.className = 'modal ocr-char-modal';

  const head = document.createElement('div');
  head.className = 'modal-head';
  const title = document.createElement('strong');
  title.textContent = 'Characters to train';
  head.appendChild(title);
  // The selector panel no longer carries its own heading, so the count that
  // used to sit in it moves here.  It still updates on every pick, because
  // adding a character re-runs render() and render() re-runs this.
  const count = document.createElement('span');
  count.className = 'charsel-count';
  const n = Array.isArray(STATE.classes) ? STATE.classes.length : 0;
  count.textContent = n === 1 ? '1 class' : `${{n}} classes`;
  head.appendChild(count);
  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'iconbtn';
  close.id = 'ocrCharModalClose';
  close.title = 'Close';
  close.textContent = '✕';
  close.onclick = (e) => {{ if (e) e.stopPropagation(); closeOcrCharModal(); }};
  head.appendChild(close);
  card.appendChild(head);

  const body = document.createElement('div');
  body.className = 'modal-body';
  // The selector itself, unchanged: it still writes STATE.classes through
  // applyClassesState() exactly as before.  Only its parent moved.
  renderOcrCharacterSelector(body);
  card.appendChild(body);

  host.appendChild(card);
  // Backdrop closes; the card swallows its own clicks so one that lands on the
  // grid is never read as a click outside the modal.
  host.onclick = (e) => {{ if (e.target === host) closeOcrCharModal(); }};
  card.onclick = (e) => e.stopPropagation();
}}

// ── Target sign sequence (road-sign projects) ──────────────────────────────
//
// A road sign is a SEQUENCE: "A1A" and "AA1" are two different signs built
// from the same two classes.  The classes stay exactly what they are — one
// card per CHARACTER — and the sequence is a separate, ordered piece of
// project state saying which characters, in which order, this project is
// being built to read.  Nothing about training changes; the preview is what
// consumes it, comparing each slot against the model's answer for that slot.
//
// It lives in train_cfg because that is the object the SPA already round-trips
// into the project archive, so the sequence saves, reopens and passes through
// the server's pick_train_cfg with no new plumbing.  It is deliberately NOT
// written to localStorage: that store holds preferences which follow the USER
// between projects, and a target sequence that followed them would silently
// retarget the next sign project.
const SIGN_SEQUENCE_LEN = 3;
// Mirrors ocr_preprocess.SIGN_SEQUENCE_PAD.  A character no sign prints and no
// class is ever named after, so "this slot is not set yet" stays legible
// instead of reading as a literal '?' the model is expected to match.
const SIGN_SEQUENCE_PAD = '?';
// Mirrors record_controller.SIGN_MATCH_TEXT / SIGN_NOMATCH_TEXT.  The server
// sends the rendering it actually decided on, so these are only the fallback
// for a payload that arrived without one — but the banner is the answer the
// whole project exists to give, and it must never be blank because a key was
// missing.
const SIGN_MATCH_TEXT = '🟢 YES — MATCH';
const SIGN_NOMATCH_TEXT = '🔴 NO — NOT MATCH';

// JS mirror of ocr_preprocess.normalize_sign_sequence — the one place the
// shape of a sequence is decided, so the bar, the save payload and the
// preview all agree.  Takes an array or a plain string ("A1A" and "A 1 A"
// both give A,1,A), upper-cases, keeps ONE character per slot (a class card is
// named for a single character, so a two-character slot could never match one)
// and pads short input, so callers always get exactly SIGN_SEQUENCE_LEN
// entries and never have to handle a ragged sequence.
function normalizeSignSequence(value) {{
  const raw = Array.isArray(value)
    ? value.slice()
    : (value == null ? [] : String(value).split(''));
  const chars = [];
  raw.forEach((item) => {{
    const text = String(item == null ? '' : item).trim();
    if (text) chars.push(text.slice(0, 1).toUpperCase());
  }});
  while (chars.length < SIGN_SEQUENCE_LEN) chars.push(SIGN_SEQUENCE_PAD);
  return chars.slice(0, SIGN_SEQUENCE_LEN);
}}

// JS mirror of ocr_preprocess.normalize_sign_sequences: the complete,
// unique 3-character whitelist, first-appearance order.  Incomplete plates
// (a '?' pad) are not targets and are dropped here; the editable primary
// lives on in sign_sequence so the bar can still show a half-typed plate.
function normalizeSignSequences(value) {{
  const raw = Array.isArray(value)
    ? value.slice()
    : (value == null || value === '' ? [] : String(value).split(','));
  const singleCharList = raw.length === SIGN_SEQUENCE_LEN && raw.every(
    (v) => !Array.isArray(v) && String(v == null ? '' : v).trim().length <= 1
  );
  const items = singleCharList ? [raw] : raw;
  const out = [];
  const seen = {{}};
  items.forEach((item) => {{
    if (item == null) return;
    const seq = normalizeSignSequence(item);
    if (seq.some((c) => c === SIGN_SEQUENCE_PAD)) return;
    const text = seq.join('');
    if (!/^[0-9A-Z]{{3}}$/.test(text) || seen[text]) return;
    seen[text] = true;
    out.push(text);
  }});
  return out;
}}

function signSequences() {{
  const cfg = STATE.train_cfg || {{}};
  const seqs = normalizeSignSequences(cfg.sign_sequences);
  const primary = normalizeSignSequence(cfg.sign_sequence);
  if (!primary.some((c) => c === SIGN_SEQUENCE_PAD)) {{
    // sign_sequences[0] is ALWAYS the primary.  A stale first entry (e.g.
    // an older boot payload, or a test/editor that wrote sign_sequence
    // directly) must not survive beside the new primary; any other plate
    // that happens to equal it is deduped too.
    const primaryText = primary.join('');
    const at = seqs.indexOf(primaryText);
    if (at > 0) seqs.splice(at, 1);
    if (seqs.length) seqs[0] = primaryText;
    else seqs.push(primaryText);
  }}
  return seqs;
}}

function signSequence() {{
  const cfg = STATE.train_cfg || {{}};
  return normalizeSignSequence(cfg.sign_sequence);
}}

// True once at least one slot names a character.  A project whose every slot
// is still the pad has no target, and the preview has nothing to answer
// against — the three empty slots show that rather than letting the banner
// read NOT MATCH for a question nobody asked.
function signSequenceIsSet() {{
  return signSequence().some((c) => c !== SIGN_SEQUENCE_PAD);
}}

// The characters the sequence names, deduplicated and in first-appearance
// order: ['A','1','A'] -> ['A','1'].  This is the whole of the "repeated
// character is one class" rule — the sequence keeps both A's, the class list
// keeps one, and neither has to know about the other.
function signSequenceClasses() {{
  const out = [];
  signSequences().forEach((text) => {{
    for (const c of text) {{
      if (c !== SIGN_SEQUENCE_PAD && out.indexOf(c) === -1) out.push(c);
    }}
  }});
  if (!out.length) {{
    signSequence().forEach((c) => {{
      if (c !== SIGN_SEQUENCE_PAD && out.indexOf(c) === -1) out.push(c);
    }});
  }}
  return out;
}}

// A sign slot is one character a model output can name: 0-9 or A-Z.  The
// inline inputs reject everything else on the way in, so the sequence, the
// class list and the device's kTargetSignSequence all speak one alphabet.
const SIGN_SLOT_RE = /^[0-9A-Z]$/;

// The positions one character occupies: ['A','1','A'] gives A -> [1, 3].
// Used for the class card badge, so a repeated character still reads as one
// class without hiding the fact that it is matched twice.
function signSlotPositions(ch) {{
  const out = [];
  signSequences().forEach((text) => {{
    for (let idx = 0; idx < text.length; idx++) {{
      if (text[idx] === ch && out.indexOf(idx + 1) === -1) out.push(idx + 1);
    }}
  }});
  if (!out.length) {{
    signSequence().forEach((c, idx) => {{ if (c === ch && out.indexOf(idx + 1) === -1) out.push(idx + 1); }});
  }}
  out.sort((a, b) => a - b);
  return out;
}}

function signSlotLabel(ch) {{
  const seqs = signSequences();
  if (seqs.length > 1) {{
    const plates = seqs.filter((text) => text.indexOf(ch) !== -1).length;
    if (plates > 1) {{
      const pos = signSlotPositions(ch);
      const posText = pos.length === 1
        ? `Slot ${{pos[0]}}`
        : (pos.length === 2
          ? `Slots ${{pos[0]}} & ${{pos[1]}}`
          : `Slots ${{pos.slice(0, -1).join(', ')}} & ${{pos[pos.length - 1]}}`);
      return `${{plates}} plates · ${{posText}}`;
    }}
  }}
  const pos = signSlotPositions(ch);
  if (pos.length < 2) return '';
  if (pos.length === 2) return `Slots ${{pos[0]}} & ${{pos[1]}}`;
  return `Slots ${{pos.slice(0, -1).join(', ')}} & ${{pos[pos.length - 1]}}`;
}}

// Replace the whole whitelist (the License List modal's Save & Sync path).
// The first plate is the primary, so sign_sequence always stays sequences[0].
function setSignSequences(list) {{
  const seqs = normalizeSignSequences(list);
  if (!seqs.length) return false;
  const next = Object.assign({{}}, STATE.train_cfg || {{}});
  next.sign_sequences = seqs.slice();
  next.sign_sequence = normalizeSignSequence(seqs[0]);
  STATE.train_cfg = next;
  adoptSignSequenceClasses();
  render();
  scheduleSignClassSync(0);
  try {{
    if (previewInputOn && typeof startPreviewPredictLoop === 'function') startPreviewPredictLoop();
  }} catch (e) {{}}
  return true;
}}

// Focus survives the rebuilds a sequence edit causes.  render() clears
// #classes and renderSignSequenceBar() refills the bar; either would drop the
// caret the user is typing through.  A snapshot is taken before the DOM is
// cleared and restored on the new input with the same id.
// ── Allowed License Plates modal (gatekeeper whitelist) ──────────────────
let licenseListOpen = false;
let licenseDraft = [];

function licenseDraftText(row) {{
  return String((licenseDraft[row] == null ? '' : licenseDraft[row]))
    .toUpperCase().replace(/[^0-9A-Z]/g, '').slice(0, SIGN_SEQUENCE_LEN);
}}
function licenseDraftChars(row) {{
  const text = licenseDraftText(row);
  const out = text.split('');
  while (out.length < SIGN_SEQUENCE_LEN) out.push('');
  return out;
}}
function renderLicenseListModal() {{
  const modal = document.getElementById('licenseListModal');
  if (!modal) return;
  modal.classList.toggle('open', !!licenseListOpen);
  if (!licenseListOpen) return;
  const host = document.getElementById('licenseListRows');
  if (!host) return;
  if (!Array.isArray(licenseDraft) || !licenseDraft.length) licenseDraft = [''];
  licenseDraft = licenseDraft.map((_, i) => licenseDraftText(i));
  host.innerHTML = '';
  licenseDraft.forEach((text, row) => {{
    const rowEl = document.createElement('div');
    rowEl.setAttribute('data-license-row', String(row));
    rowEl.style.cssText = 'display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:8px;border:1px solid #E2E8F0;border-radius:10px;background:#F8FAFC;';
    const label = document.createElement('span');
    label.style.cssText = 'min-width:102px;font-weight:700;font-size:13px;color:#0F172A;';
    label.textContent = row === 0 ? 'Plate #1 · Primary' : `Plate #${{row + 1}}`;
    rowEl.appendChild(label);
    const chars = licenseDraftChars(row);
    for (let slot = 0; slot < SIGN_SEQUENCE_LEN; slot++) {{
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.className = 'signseq-slot' + (chars[slot] ? '' : ' empty');
      inp.id = `licenseSlot-${{row}}-${{slot}}`;
      inp.value = chars[slot] || '';
      inp.maxLength = 1;
      inp.placeholder = SIGN_SEQUENCE_PAD;
      inp.autocomplete = 'off';
      inp.autocapitalize = 'characters';
      inp.spellcheck = false;
      inp.title = `Plate #${{row + 1}} position ${{slot + 1}}`;
      inp.onfocus = () => {{ try {{ inp.select(); }} catch (e) {{}} }};
      inp.oninput = () => {{
        const raw = String(inp.value || '').trim().toUpperCase();
        const nextCh = SIGN_SLOT_RE.test(raw.slice(0, 1)) ? raw.slice(0, 1) : '';
        inp.value = nextCh;
        const nextChars = licenseDraftChars(row);
        nextChars[slot] = nextCh;
        licenseDraft[row] = nextChars.join('').replace(/\?+$/, '');
        if (nextCh && slot < SIGN_SEQUENCE_LEN - 1) {{
          const following = document.getElementById(`licenseSlot-${{row}}-${{slot + 1}}`);
          if (following) {{ following.focus(); try {{ following.select(); }} catch (e) {{}} }}
        }}
        const hint = document.getElementById('licenseListHint');
        if (hint) {{
          const seqs = normalizeSignSequences(licenseDraft);
          const union = [];
          seqs.forEach((t) => {{ for (const c of t) if (union.indexOf(c) === -1) union.push(c); }});
          hint.textContent = `${{seqs.length}} complete plate(s) · classes: ${{union.join(' ') || '—'}}`;
        }}
      }};
      rowEl.appendChild(inp);
    }}
    const left = document.createElement('button');
    left.type = 'button';
    left.className = 'signseq-move-left';
    left.textContent = '◀';
    left.title = 'Swap positions 1 and 2';
    left.disabled = !(licenseDraft[row] || '').length;
    left.onclick = () => {{
      const c = licenseDraftChars(row);
      const t = c[0]; c[0] = c[1]; c[1] = t;
      licenseDraft[row] = c.join('').replace(/\?+$/, '');
      renderLicenseListModal();
    }};
    const right = document.createElement('button');
    right.type = 'button';
    right.className = 'signseq-move-right';
    right.textContent = '▶';
    right.title = 'Swap positions 2 and 3';
    right.disabled = !(licenseDraft[row] || '').length;
    right.onclick = () => {{
      const c = licenseDraftChars(row);
      const t = c[1]; c[1] = c[2]; c[2] = t;
      licenseDraft[row] = c.join('').replace(/\?+$/, '');
      renderLicenseListModal();
    }};
    rowEl.appendChild(left);
    rowEl.appendChild(right);
    if (row > 0) {{
      const primaryBtn = document.createElement('button');
      primaryBtn.type = 'button';
      primaryBtn.className = 'tm-btn';
      primaryBtn.textContent = '⭐ Set as Primary';
      primaryBtn.title = 'Make this the plate Preview and the device treat as primary';
      primaryBtn.onclick = () => {{
        const moved = licenseDraft.splice(row, 1)[0];
        licenseDraft.unshift(moved);
        renderLicenseListModal();
      }};
      rowEl.appendChild(primaryBtn);
    }}
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'tm-btn';
    remove.textContent = '🗑️ Remove';
    remove.disabled = licenseDraft.length <= 1;
    remove.onclick = () => {{
      if (licenseDraft.length <= 1) return;
      licenseDraft.splice(row, 1);
      renderLicenseListModal();
    }};
    rowEl.appendChild(remove);
    host.appendChild(rowEl);
  }});
  const hint = document.getElementById('licenseListHint');
  if (hint) {{
    const seqs = normalizeSignSequences(licenseDraft);
    const union = [];
    seqs.forEach((t) => {{ for (const c of t) if (union.indexOf(c) === -1) union.push(c); }});
    hint.textContent = `${{seqs.length}} complete plate(s) · classes: ${{union.join(' ') || '—'}}`;
  }}
}}
function openLicenseListModal() {{
  if (!isSignOcrProject()) return;
  const seqs = signSequences();
  licenseDraft = (seqs.length ? seqs : [signSequence().join('')]).slice();
  licenseListOpen = true;
  renderLicenseListModal();
}}
function closeLicenseListModal() {{
  licenseListOpen = false;
  const modal = document.getElementById('licenseListModal');
  if (modal) modal.classList.remove('open');
}}
function syncLicenseListNav() {{
  const btn = document.getElementById('navLicenseList');
  if (!btn) return;
  const show = isSignOcrProject();
  btn.style.display = show ? '' : 'none';
  if (show) {{
    const count = document.getElementById('navLicenseCount');
    if (count) count.textContent = String(Math.max(1, signSequences().length));
  }}
}}
async function saveLicenseList() {{
  const seqs = normalizeSignSequences(licenseDraft);
  if (!seqs.length) {{
    toast('Add at least one complete 3-character plate.');
    return;
  }}
  const names = [];
  seqs.forEach((text) => {{ for (const c of text) if (names.indexOf(c) === -1) names.push(c); }});
  const next = Object.assign({{}}, STATE.train_cfg || {{}});
  next.sign_sequences = seqs.slice();
  next.sign_sequence = normalizeSignSequence(seqs[0]);
  STATE.train_cfg = next;
  applySignClassesState(names);
  render();
  const autoSeedOn = !!(autoSeedEmnistSamples
    || (STATE.train_cfg && STATE.train_cfg.auto_seed_emnist_samples));
  try {{
    const res = await fetch(`${{baseUrl}}/classes/sync_ocr`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        names: names,
        sign_sequence: normalizeSignSequence(seqs[0]).join(''),
        sign_sequences: seqs,
        reading_order: String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr')),
        auto_seed_emnist_samples: autoSeedOn,
      }})
    }});
    const data = await res.json().catch(() => ({{ok: '0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to sync license plates.');
    const nextClasses = Array.isArray(data.classes) && data.classes.length ? data.classes : names;
    signSyncSavedTarget = `${{names.join('')}}|${{seqs.join(',')}}`;
    if (!signClassesMatch(nextClasses)) applySignClassesState(nextClasses);
    if (data.counts || data.sample_previews || typeof data.train_enabled === 'boolean') {{
      applyRuntimeState(data);
    }}
    if (data.seed_error) toast('Auto-fill failed: ' + String(data.seed_error));
    render();
    syncLicenseListNav();
    closeLicenseListModal();
    toast(`Saved ${{seqs.length}} allowed plate(s).`);
    try {{
      if (previewInputOn && typeof startPreviewPredictLoop === 'function') startPreviewPredictLoop();
    }} catch (e) {{}}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}

function signSlotFocusSnapshot() {{
  const el = document.activeElement;
  const id = String((el && el.id) || '');
  if (!/^signSeqSlot-[0-9]+$/.test(id)) return null;
  let start = null, end = null;
  try {{ start = el.selectionStart; end = el.selectionEnd; }} catch (e) {{}}
  return {{ id: id, start: start, end: end }};
}}

function restoreSignSlotFocus(snapshot) {{
  if (!snapshot) return;
  const el = document.getElementById(snapshot.id);
  if (!el) return;
  try {{
    el.focus();
    if (snapshot.start != null) el.setSelectionRange(snapshot.start, snapshot.end);
  }} catch (e) {{}}
}}

function setSignSequence(seq) {{
  const next = Object.assign({{}}, STATE.train_cfg || {{}});
  next.sign_sequence = normalizeSignSequence(seq);
  const text = next.sign_sequence.join('');
  const seqs = signSequences();
  if (!next.sign_sequence.some((c) => c === SIGN_SEQUENCE_PAD)) {{
    if (seqs.length) seqs[0] = text;
    else seqs.push(text);
    next.sign_sequences = seqs;
  }} else if (Array.isArray(next.sign_sequences)) {{
    // A half-typed primary is not a valid plate; leave the persisted
    // whitelist alone until it is complete (the modal is the guided path).
    next.sign_sequences = seqs;
  }}
  STATE.train_cfg = next;
  // persistTrainCfgStorage() is deliberately NOT called.  See the note above:
  // the sequence is project content, and letting it join the localStorage
  // allow-list would carry it into the next project the user opens.
  //
  // The class cards are part of the sequence, not a second thing to keep in
  // step.  adoptSignSequenceClasses() rewrites STATE.classes to the unique
  // characters here, render() mounts them, and render()'s tail asks the
  // server to persist the same list.  Typing a slot, swapping two slots and
  // opening a saved project therefore take ONE path.
  adoptSignSequenceClasses();
  render();
  // The verdict on screen was computed against the OLD target, which makes it
  // stale the instant this runs — and a stale green YES is the one wrong
  // answer that matters.  The loop is RESTARTED rather than the banner patched
  // from cached slot readings: the match rule lives on the server, so the only
  // honest way to re-answer is to ask again.
  try {{
    if (previewInputOn && typeof startPreviewPredictLoop === 'function') startPreviewPredictLoop();
  }} catch (e) {{}}
}}

// Move one slot's character one position left (-1) or right (+1).  A SWAP of
// neighbours, not a rotation: the user is correcting two characters that are
// in each other's place, and a rotation would also move the third.  Out-of-
// range is a no-op, so the first slot's left arrow can stay bound and simply
// be disabled.
function moveSignSlot(index, delta) {{
  const seq = signSequence();
  const to = index + delta;
  if (index < 0 || index >= seq.length || to < 0 || to >= seq.length) return;
  const tmp = seq[index];
  seq[index] = seq[to];
  seq[to] = tmp;
  setSignSequence(seq);
}}

// ── Automatic class sync ──────────────────────────────────────────────────
// In a sign project the class list is not an independent thing the user
// curates: it is exactly the unique characters of the three slots, in slot
// order.  Two directions are needed and both live here:
//
//   adoptSignSequenceClasses()   local, synchronous, before the cards render
//   syncSignClassesToSequence()  remote, debounced, persists the same list
//
// The local half means the cards are correct on the first frame after an
// edit; the remote half means a reopened project, the trainer and the export
// all see the same order.  No "Add as classes" button exists because there is
// nothing left for it to do.
function signClassesMatch(target) {{
  const have = Array.isArray(STATE.classes) ? STATE.classes : [];
  return have.length === target.length && target.every((c, i) => String(have[i]) === c);
}}

// Rewrite the local class list to the sequence.  Unlike a reorder/rename on
// the ordinary cards, this must NOT fall back to the class that used to sit at
// the same index: inserting "7" before "A" is a NEW empty class, not a rename
// of A, and inheriting A's count and previews would show samples that do not
// exist.  Every record is looked up by its own name; a character that stayed
// keeps its samples, a character that arrived starts at zero, and a character
// that left takes its records with it.
function applySignClassesState(target) {{
  const oldCounts = Object.assign({{}}, STATE.counts || {{}});
  const oldPreviews = Object.assign({{}}, STATE.sample_previews || {{}});
  const oldProcessed = Object.assign({{}}, STATE.processed_previews || {{}});
  const oldClassPreprocess = Object.assign({{}}, STATE.class_preprocess || {{}});
  const oldSamplePreprocess = normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}});
  const nextCounts = {{}};
  const nextPreviews = {{}};
  const nextProcessed = {{}};
  const nextClassPreprocess = {{}};
  const nextSamplePreprocess = {{}};
  target.forEach((name) => {{
    nextCounts[name] = Number(oldCounts[name] || 0);
    nextPreviews[name] = normalizePreviewList(oldPreviews[name]);
    nextProcessed[name] = normalizePreviewList(oldProcessed[name]);
    nextClassPreprocess[name] = normalizeClassPreprocessConfig(oldClassPreprocess[name]);
    const sampleMap = oldSamplePreprocess[name];
    if (sampleMap && typeof sampleMap === 'object' && Object.keys(sampleMap).length) {{
      nextSamplePreprocess[name] = normalizeSamplePreprocessMap({{ [name]: sampleMap }})[name] || {{}};
    }}
  }});
  STATE.classes = target.slice();
  // A class that leaves is no longer "already offered": if a slot names it
  // again later it is a NEW empty class and must be eligible for auto-fill.
  // Without this, A12 -> A1B -> A12 -> A1B leaves B at 0 (the ledger still
  // remembers the first B) whenever the server-side fill did not run.
  for (const name of Array.from(autoSeedEmnistAttempted)) {{
    if (target.indexOf(name) === -1) autoSeedEmnistAttempted.delete(name);
  }}
  STATE.counts = nextCounts;
  STATE.sample_previews = nextPreviews;
  STATE.processed_previews = nextProcessed;
  STATE.class_preprocess = nextClassPreprocess;
  STATE.sample_preprocess = nextSamplePreprocess;
}}

function adoptSignSequenceClasses() {{
  if (!isSignOcrProject()) return false;
  const target = signSequenceClasses();
  if (!target.length) return false;   // all slots cleared: do not wipe the project
  if (signClassesMatch(target)) return false;
  applySignClassesState(target);
  return true;
}}

let signSyncTimer = 0;
let signSyncInFlight = false;
let signSyncQueued = false;
// The target the SERVER has acknowledged.  Kept separate from "the local list
// matches" so the first render of a project (whose payload may already be in
// order) still reconciles once, and so a failed call is retried by the next
// render instead of being forgotten.
let signSyncSavedTarget = '';

function scheduleSignClassSync(delay) {{
  if (!isSignOcrProject()) return;
  if (signSyncTimer) window.clearTimeout(signSyncTimer);
  signSyncTimer = window.setTimeout(() => {{
    signSyncTimer = 0;
    syncSignClassesToSequence();
  }}, delay == null ? 120 : delay);
}}

async function syncSignClassesToSequence() {{
  if (!isSignOcrProject()) return;
  const target = signSequenceClasses();
  if (!target.length) return;           // no target: never delete every class
  const seqText = signSequences().join(',');
  const key = `${{target.join('')}}|${{seqText}}`;
  if (key === signSyncSavedTarget && signClassesMatch(target)) return;
  if (signSyncInFlight) {{ signSyncQueued = true; return; }}
  signSyncInFlight = true;
  // Send the target itself, not just the names derived from it, and send
  // the checkbox state with it.  The flag is browser-local storage on this
  // side but project state on the server; either can be on while the other
  // is not (a project switch, a reopened archive, a cleared profile), so
  // the route ORs the two and heals the persisted copy.  Without this, a
  // new slot character could stay at 0 whenever the server's copy was off.
  const autoSeedOn = !!(autoSeedEmnistSamples
    || (STATE.train_cfg && STATE.train_cfg.auto_seed_emnist_samples));
  try {{
    const res = await fetch(`${{baseUrl}}/classes/sync_ocr`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        names: target,
        sign_sequence: signSequence().join(''),
        sign_sequences: signSequences(),
        reading_order: String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr')),
        auto_seed_emnist_samples: autoSeedOn,
      }})
    }});
    const data = await res.json().catch(() => ({{ok: '0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to sync sign classes.');
    // A newer edit may have landed while this request was in flight.  Its
    // response describes the OLD sequence; applying it would make the cards
    // jump back, so ignore it and let the queued pass send the new one.
    if (signSequenceClasses().join('') !== target.join('')
        || signSequences().join(',') !== seqText) {{
      signSyncQueued = true;
      return;
    }}
    if (data.seed_error) {{
      // The class sync succeeded but the auto-fill did not.  Say so instead
      // of leaving the card silently at 0 so the user can retry.
      toast('Auto-fill failed: ' + String(data.seed_error));
    }}
    const next = Array.isArray(data.classes) && data.classes.length ? data.classes : target;
    signSyncSavedTarget = key;
    if (!signClassesMatch(next)) {{
      applySignClassesState(next);
    }}
    // The response also carries the post-sync runtime state.  Applying it
    // AFTER applySignClassesState (which rebuilds counts from the local
    // snapshot) is what makes a newly added slot character show its 24
    // template samples immediately, without waiting for a second request.
    // One synchronous block: classes, counts, previews and train_enabled
    // are replaced together, so the cards never render a half-applied mix.
    if (data.counts || data.sample_previews || typeof data.train_enabled === 'boolean') {{
      applyRuntimeState(data);
    }}
    render();
    // On a healthy backend the seed above already ran inside this request,
    // so the counts are 24 and this is a no-op.  It remains as the fallback
    // for an older server that answers without counts.
    autoSeedEmnistForNewClasses(next);
  }} catch (e) {{
    // The local list already follows the slots, so the screen stays honest;
    // the next render retries because savedTarget was not advanced.
    toast(String(e && e.message ? e.message : e));
  }} finally {{
    signSyncInFlight = false;
    if (signSyncQueued) {{
      signSyncQueued = false;
      scheduleSignClassSync(0);
    }}
  }}
}}

// A brand-new road-sign project starts as a real, readable sign rather than
// an empty target: three slots and three matching capture cards, with no
// picker and no button.  Only a fresh project (placeholder classes, no
// samples) gets this — an existing project whose slots the user emptied stays
// empty, because its target was a decision, not a default.
const SIGN_DEFAULT_SEQUENCE = ['A', 'B', 'C'];
function ensureSignSequenceDefault() {{
  if (!isSignOcrProject()) return false;
  if (signSequenceIsSet()) return false;
  if (!ocrIsFreshProject()) return false;
  const next = Object.assign({{}}, STATE.train_cfg || {{}});
  next.sign_sequence = SIGN_DEFAULT_SEQUENCE.slice();
  next.sign_sequences = [SIGN_DEFAULT_SEQUENCE.join('')];
  STATE.train_cfg = next;
  return true;
}}

// The bar is the editor.  There is no modal: the three boxes are <input>s and
// every valid keystroke updates the sequence, auto-advances to the next slot
// and syncs the class cards.  render() mounts it as the column's first child;
// setSignSequence() re-renders the column whenever the sequence changes.
function renderSignSequenceBar(root) {{
  const focus = signSlotFocusSnapshot();
  const existing = document.getElementById('signSeqBar');
  if (!isSignOcrProject()) {{
    if (existing) existing.remove();
    return;
  }}
  const mount = root || document.getElementById('classes');
  let box = existing;
  if (!box) {{
    if (!mount) return;
    box = document.createElement('div');
    box.id = 'signSeqBar';
    box.className = 'signseq-bar';
    if (mount.firstChild) mount.insertBefore(box, mount.firstChild);
    else mount.appendChild(box);
  }}
  box.innerHTML = '';

  const head = document.createElement('div');
  head.className = 'signseq-head';
  const title = document.createElement('div');
  title.className = 'signseq-title';
  title.textContent = 'Target sign';
  head.appendChild(title);
  // The direction is the Settings control (#advReadingOrderRow) and stays the
  // single writer for it; echoing it here is a read, so the user can see which
  // way the slots will be ordered without opening Advanced settings.
  const order = String(nullish((STATE.train_cfg || {{}}).reading_order, 'ltr')).toLowerCase();
  const rtl = order === 'rtl';
  const dirTag = document.createElement('span');
  dirTag.className = 'signseq-note';
  dirTag.textContent = rtl ? 'right → left' : 'left → right';
  head.appendChild(dirTag);
  const platesBtn = document.createElement('button');
  platesBtn.type = 'button';
  platesBtn.id = 'signSeqPlatesBtn';
  platesBtn.className = 'tm-btn';
  platesBtn.style.cssText = 'margin-left:auto;padding:4px 10px;font-size:12px;';
  platesBtn.textContent = `+ Plates (${{signSequences().length || 1}})`;
  platesBtn.title = 'Manage allowed 3-character car license plates';
  platesBtn.onclick = (e) => {{ if (e) e.stopPropagation(); openLicenseListModal(); }};
  head.appendChild(platesBtn);
  box.appendChild(head);

  // The one-line instruction is the whole tutorial.  Everything below it is an
  // input; there is no Edit button, no character picker and no class button.
  const steps = document.createElement('div');
  steps.className = 'signseq-note';
  steps.textContent = 'Step 1: Type the 3 sign characters in order below ➔ Step 2: Capture samples for each character card ➔ Step 3: Train & Preview (YES/NO).';
  box.appendChild(steps);

  const row = document.createElement('div');
  row.className = 'signseq-row';
  const seq = signSequence();
  seq.forEach((ch, idx) => {{
    const unset = ch === SIGN_SEQUENCE_PAD;
    const holder = document.createElement('div');
    holder.className = 'signseq-slot-wrap';

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'signseq-slot' + (unset ? ' empty' : '');
    input.id = `signSeqSlot-${{idx}}`;
    input.value = unset ? '' : ch;
    input.maxLength = 1;
    input.placeholder = SIGN_SEQUENCE_PAD;
    input.autocomplete = 'off';
    input.autocapitalize = 'characters';
    input.spellcheck = false;
    input.setAttribute('aria-label', `Sign character ${{idx + 1}}`);
    input.title = `Position ${{idx + 1}}`;
    // Clicking a filled slot selects its character, so the next keystroke
    // replaces it instead of being refused by maxlength.
    input.onfocus = () => {{ try {{ input.select(); }} catch (e) {{}} }};
    input.onclick = () => {{ try {{ input.select(); }} catch (e) {{}} }};
    input.oninput = () => {{
      const raw = String(input.value || '').trim().toUpperCase();
      const nextCh = SIGN_SLOT_RE.test(raw.slice(0, 1)) ? raw.slice(0, 1) : '';
      input.value = nextCh;
      const next = signSequence();
      if (next[idx] !== (nextCh || SIGN_SEQUENCE_PAD)) {{
        next[idx] = nextCh || SIGN_SEQUENCE_PAD;
        setSignSequence(next);
      }}
      // OTP ergonomics: a complete character moves the caret on, and the
      // destination is selected so the next keystroke replaces it.  The last
      // slot stays focused — there is nowhere left to go.
      if (nextCh && idx < SIGN_SEQUENCE_LEN - 1) {{
        const following = document.getElementById(`signSeqSlot-${{idx + 1}}`);
        if (following) {{
          following.focus();
          try {{ following.select(); }} catch (e) {{}}
        }}
      }}
    }};
    input.onkeydown = (e) => {{
      if (e.key === 'Backspace' && !input.value && idx > 0) {{
        e.preventDefault();
        const next = signSequence();
        if (next[idx - 1] !== SIGN_SEQUENCE_PAD) {{
          next[idx - 1] = SIGN_SEQUENCE_PAD;
          setSignSequence(next);
        }}
        const previous = document.getElementById(`signSeqSlot-${{idx - 1}}`);
        if (previous) {{
          previous.focus();
          try {{ previous.select(); }} catch (err) {{}}
        }}
      }} else if (e.key === 'ArrowLeft' && idx > 0) {{
        e.preventDefault();
        const previous = document.getElementById(`signSeqSlot-${{idx - 1}}`);
        if (previous) {{ previous.focus(); try {{ previous.select(); }} catch (err) {{}} }}
      }} else if (e.key === 'ArrowRight' && idx < SIGN_SEQUENCE_LEN - 1) {{
        e.preventDefault();
        const following = document.getElementById(`signSeqSlot-${{idx + 1}}`);
        if (following) {{ following.focus(); try {{ following.select(); }} catch (err) {{}} }}
      }}
    }};
    input.onpaste = (e) => {{
      // Pasting "12A" over slot 1 fills all three slots, which is faster than
      // three clicks and is what a user copying a sign number expects.
      const clip = e.clipboardData || window.clipboardData;
      if (!clip) return;
      const text = String(clip.getData('text') || '');
      const chars = text.toUpperCase().split('')
        .filter((c) => SIGN_SLOT_RE.test(c))
        .slice(0, SIGN_SEQUENCE_LEN - idx);
      if (!chars.length) return;
      e.preventDefault();
      const next = signSequence();
      chars.forEach((c, offset) => {{ next[idx + offset] = c; }});
      setSignSequence(next);
      const last = Math.min(idx + chars.length, SIGN_SEQUENCE_LEN - 1);
      const target = document.getElementById(`signSeqSlot-${{last}}`);
      if (target) {{ target.focus(); try {{ target.select(); }} catch (err) {{}} }}
    }};
    holder.appendChild(input);

    const move = document.createElement('div');
    move.className = 'signseq-move';
    const left = document.createElement('button');
    left.type = 'button';
    left.className = 'signseq-move-left';
    left.id = `signSeqLeft-${{idx}}`;
    left.textContent = '◀';
    left.title = 'Swap with the character on the left';
    left.disabled = idx === 0;
    left.onclick = (e) => {{ if (e) e.stopPropagation(); moveSignSlot(idx, -1); }};
    const right = document.createElement('button');
    right.type = 'button';
    right.className = 'signseq-move-right';
    right.id = `signSeqRight-${{idx}}`;
    right.textContent = '▶';
    right.title = 'Swap with the character on the right';
    right.disabled = idx === SIGN_SEQUENCE_LEN - 1;
    right.onclick = (e) => {{ if (e) e.stopPropagation(); moveSignSlot(idx, 1); }};
    move.appendChild(left);
    move.appendChild(right);
    holder.appendChild(move);

    row.appendChild(holder);
  }});
  box.appendChild(row);

  restoreSignSlotFocus(focus);
}}

// ── Glyph settings (OCR only) ──────────────────────────────────────────────
//
// Convention over configuration.  A live camera capture is dark ink on a
// bright page, the glyph sits somewhere inside a larger frame, and the bbox
// stretches to the grid — so that is what every class gets until someone says
// otherwise.  These are the values the Python normalisers already resolve a
// MISSING key to, so a class with no stored record and a class whose record
// this form wrote train identically; the form only makes the assumption
// visible and overridable rather than changing it.
// Must match what the BACKEND does with a class that carries no glyph keys,
// or this form displays a value the system does not use.  The backend's
// fallbacks are: record_controller's live/preview transform (INK_DARK,
// NORM_CANON, segmentation on) and ocr_trainer._project_convention
// (NORM_CANON when no class states one).  CANON, not STRETCH, because the OCR
// project mixes EMNIST into every training set and CANON is the
// aspect-preserving convention that keeps a camera capture comparable to it —
// STRETCH squashes a tall thin glyph ('1', 'I') into an unreadable blob.
const OCR_GLYPH_DEFAULTS = {{ ink_polarity: 0, glyph_segment: true, glyph_convention: 'canon' }};

const OCR_POLARITY_OPTIONS = [
  [0, 'Dark ink on light (camera captures)'],
  [1, 'Bright ink on dark (EMNIST/MNIST)'],
];
const OCR_CONVENTION_OPTIONS = [
  ['canon', 'MNIST/EMNIST canonical (fit 20 px, centre of mass)'],
  ['stretch', 'Stretch bbox to the full grid'],
];

// The effective settings for one class: what is stored, or the default for
// any key the record does not carry.  Read through the same normalisers the
// save path uses, so the value shown in the control is the value that would
// be stored.
function glyphConfigFor(className) {{
  const stored = (STATE.class_preprocess && STATE.class_preprocess[className]) || {{}};
  // ink_polarity is a single global override (Training card > Advanced), so
  // the effective value per class is the global train_cfg value when no
  // per-class record exists yet (e.g. a class just added).
  const globalInk = normalizeInkPolarity((STATE.train_cfg || {{}}).ink_polarity);
  return {{
    ink_polarity: 'ink_polarity' in stored
      ? normalizeInkPolarity(stored.ink_polarity) : globalInk,
    glyph_segment: 'glyph_segment' in stored
      ? !!stored.glyph_segment : OCR_GLYPH_DEFAULTS.glyph_segment,
    glyph_convention: 'glyph_convention' in stored
      ? normalizeNormConvention(stored.glyph_convention) : OCR_GLYPH_DEFAULTS.glyph_convention,
  }};
}}

// Edits land here, not in STATE, so that closing without applying discards
// them — Apply is the commit.  Kept outside render() for the same reason
// ocrCharDraft is: the column rebuilds on any class edit, and a half-made
// selection should not be thrown away by a rebuild the user did not ask for.
let glyphDraft = null;
let ocrGlyphModalOpen = false;

function openOcrGlyphModal() {{
  const classes = Array.isArray(STATE.classes) ? STATE.classes : [];
  glyphDraft = {{}};
  classes.forEach((name) => {{ glyphDraft[name] = glyphConfigFor(name); }});
  ocrGlyphModalOpen = true;
  renderOcrGlyphModal();
}}
function closeOcrGlyphModal() {{
  ocrGlyphModalOpen = false;
  glyphDraft = null;
  renderOcrGlyphModal();
}}

function renderOcrGlyphModal() {{
  const host = document.getElementById('ocrGlyphModal');
  if (!host) return;
  // The project guard matters here in a way it does not for the character
  // modal: these settings only mean anything to the glyph pipeline, so a
  // lingering open modal after a switch to an image project would be showing
  // controls that no longer apply to anything on screen.
  if (!ocrGlyphModalOpen || !isOcrProject()) {{
    host.style.display = 'none';
    host.innerHTML = '';
    return;
  }}
  const classes = Array.isArray(STATE.classes) ? STATE.classes : [];
  if (!glyphDraft) glyphDraft = {{}};
  classes.forEach((name) => {{ if (!glyphDraft[name]) glyphDraft[name] = glyphConfigFor(name); }});

  host.style.display = 'flex';
  host.innerHTML = '';
  const card = document.createElement('div');
  card.className = 'modal ocr-glyph-modal';

  const head = document.createElement('div');
  head.className = 'modal-head';
  const title = document.createElement('strong');
  title.textContent = 'Glyph preprocessing';
  head.appendChild(title);
  const count = document.createElement('span');
  count.className = 'charsel-count';
  count.textContent = classes.length === 1 ? '1 class' : `${{classes.length}} classes`;
  const side = Number((STATE.train_cfg || {{}}).img_size);
  if (isFinite(side) && side > 0) count.textContent += ` · ${{side}}×${{side}} input`;
  head.appendChild(count);
  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'iconbtn';
  close.id = 'ocrGlyphModalClose';
  close.title = 'Close';
  close.textContent = '✕';
  close.onclick = (e) => {{ if (e) e.stopPropagation(); closeOcrGlyphModal(); }};
  head.appendChild(close);
  card.appendChild(head);

  const body = document.createElement('div');
  body.className = 'modal-body';
  const note = document.createElement('p');
  note.className = 'glyph-note';
  note.textContent = 'Ink polarity is now set once for the whole project via the Ink Polarity dropdown in the '
    + 'Training card\\'s Advanced panel. Per-class segmentation and normalisation are configured here. '
    + 'Every step is integer-exact and mirrored on the device.';
  body.appendChild(note);

  if (!classes.length) {{
    const empty = document.createElement('div');
    empty.className = 'glyph-empty';
    empty.textContent = 'This project has no classes yet. Pick characters first.';
    body.appendChild(empty);
  }} else {{
    const scroll = document.createElement('div');
    scroll.className = 'glyph-scroll';

    const headRow = document.createElement('div');
    headRow.className = 'glyph-row is-head';
    ['Class', 'Segment', 'Normalisation'].forEach((label) => {{
      const cell = document.createElement('span');
      cell.textContent = label;
      headRow.appendChild(cell);
    }});
    scroll.appendChild(headRow);

    const makeSelect = (options, value, onPick) => {{
      const sel = document.createElement('select');
      sel.className = 'glyph-select';
      options.forEach((pair) => {{
        const opt = document.createElement('option');
        opt.value = String(pair[0]);
        opt.textContent = pair[1];
        sel.appendChild(opt);
      }});
      sel.value = String(value);
      sel.onchange = () => onPick(sel.value);
      return sel;
    }};

    classes.forEach((name) => {{
      const row = document.createElement('div');
      row.className = 'glyph-row';

      const nameCell = document.createElement('span');
      nameCell.className = 'glyph-name';
      nameCell.textContent = String(name);
      nameCell.title = String(name);
      row.appendChild(nameCell);

      // Each control writes into the draft only.  Nothing here re-renders the
      // modal — a rebuild mid-edit would take the select out from under the
      // pointer that is using it.
      // ink_polarity is no longer per-class — it is a single global override
      // in the Training card's Advanced panel, applied to every class at
      // train time.

      const segWrap = document.createElement('label');
      segWrap.className = 'glyph-check';
      const seg = document.createElement('input');
      seg.type = 'checkbox';
      seg.checked = !!glyphDraft[name].glyph_segment;
      seg.title = 'On for camera captures (the glyph sits somewhere in a larger frame). '
        + 'Off for an already-isolated imported dataset.';
      seg.onchange = () => {{ glyphDraft[name].glyph_segment = !!seg.checked; }};
      const segLabel = document.createElement('span');
      segLabel.textContent = 'By projection';
      segWrap.appendChild(seg);
      segWrap.appendChild(segLabel);
      row.appendChild(segWrap);

      row.appendChild(makeSelect(
        OCR_CONVENTION_OPTIONS,
        glyphDraft[name].glyph_convention,
        (v) => {{ glyphDraft[name].glyph_convention = normalizeNormConvention(v); }}
      ));

      scroll.appendChild(row);
    }});
    body.appendChild(scroll);
  }}
  card.appendChild(body);

  const actions = document.createElement('div');
  actions.className = 'modal-actions';
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'btn btn-secondary';
  cancel.textContent = 'Cancel';
  cancel.onclick = (e) => {{ if (e) e.stopPropagation(); closeOcrGlyphModal(); }};
  actions.appendChild(cancel);
  const apply = document.createElement('button');
  apply.type = 'button';
  apply.className = 'btn btn-primary';
  apply.id = 'ocrGlyphApply';
  apply.textContent = 'Apply glyph settings';
  apply.disabled = !classes.length;
  apply.onclick = (e) => {{ if (e) e.stopPropagation(); applyGlyphSettings(); }};
  actions.appendChild(apply);
  card.appendChild(actions);

  host.appendChild(card);
  host.onclick = (e) => {{ if (e.target === host) closeOcrGlyphModal(); }};
  card.onclick = (e) => e.stopPropagation();
}}

// Writes every class, not just the ones that were touched: applying the form
// is the user saying "this is how my project is configured", and the defaults
// are worth committing explicitly for exactly that reason.  It is also what
// the Streamlit form this replaces did, so a project configured before and
// after this change ends up with the same records.
async function applyGlyphSettings() {{
  if (!isOcrProject()) return;
  const classes = Array.isArray(STATE.classes) ? STATE.classes : [];
  if (!classes.length) return;
  const next = Object.assign({{}}, STATE.class_preprocess || {{}});
  classes.forEach((name) => {{
    const draft = (glyphDraft && glyphDraft[name]) || glyphConfigFor(name);
    // Through normalizeClassPreprocessConfig so the record written here has
    // the same shape as one written by the per-class preprocess editor —
    // including the mode/threshold defaults this form has no opinion about.
    next[name] = normalizeClassPreprocessConfig(Object.assign({{}}, next[name] || {{}}, {{
      ink_polarity: normalizeInkPolarity(draft.ink_polarity),
      glyph_segment: !!draft.glyph_segment,
      glyph_convention: normalizeNormConvention(draft.glyph_convention),
    }}));
  }});
  try {{
    const res = await fetch(`${{baseUrl}}/classes/save_config`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session, class_preprocess: next}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to save glyph settings.');
    STATE.class_preprocess = next;
    toast('Glyph settings applied — the processed cache was rebuilt.');
    closeOcrGlyphModal();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}

async function deleteClass(name) {{
  if (STATE.classes.length <= 2) {{
    showTrainWarningModal('Cannot delete class.|At least two classes are required — a project cannot have fewer than 2 classes.');
    return;
  }}
  const ok = await showConfirmDialog(
    'Delete this class?',
    `Class "${{String(name || '')}}" and all of its samples will be removed.`,
    'Delete'
  );
  if (!ok) return;
  try {{
    const prev = STATE.classes.slice();
    const res = await fetch(`${{baseUrl}}/classes/delete`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        name
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to delete class.');
    applyClassesState(Array.isArray(data.classes) ? data.classes : prev.filter((x) => x !== name), prev);
    if (classPreprocessOpen && classPreprocessClass === name) {{
      closeClassPreprocessEditor();
    }}
    if (openSourceClass === name) {{
      openSourceClass = '';
      openSourceKind = '';
      clearOpenSourceState();
    }}
    if (holdRecordClass === name) {{
      holdRecording = false;
      holdRecordClass = '';
      holdRecordSource = '';
    }}
    recomputeTrainEnabled();
  refreshTrainRec();
    render();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
function updateFlow() {{
  const wrap = document.querySelector('.wrap');
  const svg = document.querySelector('svg.flow');
  const train = document.getElementById('trainCard');
  const preview = document.getElementById('previewCard');
  const cards = Array.from(document.querySelectorAll('.class-card'));
  if (!wrap || !svg || !train || !preview || cards.length < 1) return;
  const wr = wrap.getBoundingClientRect();
  // #region debug-point E:update-flow
  dbgEvent('E', 'app.py:updateFlow', '[DEBUG] updateFlow geometry snapshot', {{
    wrapWidth: Math.round(wr.width || 0),
    wrapHeight: Math.round(wr.height || 0),
    cardCount: cards.length,
    trainHeight: Math.round((train.getBoundingClientRect().height || 0)),
    previewHeight: Math.round((preview.getBoundingClientRect().height || 0)),
  }});
  // #endregion
  svg.setAttribute('viewBox', `0 0 ${{Math.max(1, Math.round(wr.width))}} ${{Math.max(1, Math.round(wr.height))}}`);
  const t = train.getBoundingClientRect();
  const p = preview.getBoundingClientRect();
  const tx = t.left - wr.left;
  const ty = t.top + t.height * 0.5 - wr.top;
  const tr = t.right - wr.left;
  const py = p.top + p.height * 0.5 - wr.top;
  const px = p.left - wr.left;
  const trainPreviewPath = document.getElementById('flowTrainPreview');
  const existing = Array.from(svg.querySelectorAll('path[id^="flowClass"]'));
  for (const p of existing) {{
    const m = String(p.id || '').match(/^flowClass(\\d+)$/);
    if (!m) continue;
    const idx = Number(m[1] || 0);
    if (idx > cards.length) p.remove();
  }}
  for (let i = 0; i < cards.length; i++) {{
    const id = `flowClass${{i + 1}}`;
    let path = document.getElementById(id);
    if (!path) {{
      path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('id', id);
      path.setAttribute('stroke', 'rgba(0,0,0,0.16)');
      path.setAttribute('stroke-width', '2');
      path.setAttribute('fill', 'none');
      if (trainPreviewPath && trainPreviewPath.parentNode === svg) {{
        svg.insertBefore(path, trainPreviewPath);
      }} else {{
        svg.appendChild(path);
      }}
    }}
    const cr = cards[i].getBoundingClientRect();
    const cx = cr.right - wr.left;
    const cy = cr.top + cr.height * 0.5 - wr.top;
    const offset = (i - (cards.length - 1) / 2) * 14;
    setFlowPath(id, `M ${{cx}} ${{cy}} C ${{cx + 44}} ${{cy}}, ${{tx - 44}} ${{ty + offset}}, ${{tx}} ${{ty}}`);
  }}
  setFlowPath('flowTrainPreview', `M ${{tr}} ${{ty}} C ${{tr + 34}} ${{ty}}, ${{px - 34}} ${{py}}, ${{px}} ${{py}}`);
}}
let trainAdvancedOpen = false;
let recHintsForced = false;
function manualRoiOrDefault(cfg) {{
  const roi = cfg && Array.isArray(cfg.manual_roi) && cfg.manual_roi.length === 4 ? cfg.manual_roi : [0.00, 0.00, 1.00, 1.00];
  return roi.map((v, idx) => {{
    const n = Number(v);
    if (!isFinite(n)) return idx < 2 ? 0 : 1;
    return Math.max(0, Math.min(1, n));
  }});
}}
async function refreshTrainRec() {{
  try {{
    const res = await fetch(`${{baseUrl}}/train/recommend`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{ session: STATE.session, cfg: STATE.train_cfg || {{}} }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (res.ok && data.ok === '1' && data.train_rec) {{
      STATE.train_rec = data.train_rec;
      refreshRecHints();
    }}
  }} catch (err) {{}}
}}
function refreshRecHints() {{
  const cfg = STATE.train_cfg || {{}};
  const rec = STATE.train_rec || null;
  const FIELDS = [
    ['recBatchHint', 'batch_size', 'batch'],
    ['recEpochsHint', 'epochs', 'epochs'],
    ['recLrHint', 'learning_rate', 'lr'],
    ['recValHint', 'validation_split', 'val_split']
  ];
  let anyDiffers = false;
  for (let i = 0; i < FIELDS.length; i++) {{
    const id = FIELDS[i][0];
    const key = FIELDS[i][1];
    const fmt = FIELDS[i][2];
    const el = document.getElementById(id);
    if (!el) continue;
    if (!rec || typeof rec[key] === 'undefined' || rec[key] === null) {{ el.textContent = ''; continue; }}
    const cur = Number(cfg[key]);
    const recV = Number(rec[key]);
    const differs = isFinite(cur) && isFinite(recV) && Math.abs(cur - recV) >= 1e-9;
    if (differs) anyDiffers = true;
    const reason = rec.reasons && rec.reasons[fmt] ? String(rec.reasons[fmt]) : '';
    if (differs) {{
      el.textContent = `💡 Recommended: ${{recV}} — ${{reason}}`;
    }} else if (recHintsForced) {{
      el.textContent = `✓ Recommended: ${{recV}} — ${{reason}}`;
    }} else {{
      el.textContent = '';
    }}
  }}
  const imgSizeHint = document.getElementById('imgSizeHint');
  if (imgSizeHint) {{
    const cfg0 = STATE.train_cfg || {{}};
    const side = Number(nullish(cfg0.img_size, 96));
    const cur = Number(currentSerialFrameSide || 0);
    imgSizeHint.textContent = (cur === side)
      ? `Device sketch: IMG_SIZE=${{side}} ✓`
      : `Device sketch must use IMG_SIZE=${{side}} (capture side now ${{cur}})`;
  }}
  const recBtn = document.getElementById('advUseRec');
  if (recBtn) {{
    const hasRec = !!(rec && rec.total_samples >= 2);
    recBtn.style.display = hasRec ? '' : 'none';
    if (!hasRec) return;
    if (anyDiffers) {{
      recBtn.textContent = '💡 Use recommended';
    }} else if (recHintsForced) {{
      recBtn.textContent = 'Hide recommendation';
    }} else {{
      recBtn.textContent = '💡 Show recommendation';
    }}
  }}
}}
function syncAdvancedInlineInputs() {{
  const cfg = STATE.train_cfg || {{}};
  const set = (id, value) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.value = nullish(value, nullish(el.value, ''));
  }};
  set('advImgSizeInline', nullish(cfg.img_size, 96));
  set('advBatchInline', nullish(cfg.batch_size, 32));
  set('advEpochsInline', nullish(cfg.epochs, 20));
  set('advValInline', nullish(cfg.validation_split, 0.25));
  set('advLrInline', nullish(cfg.learning_rate, 0.0016));
  set('advConv1Inline', nullish(cfg.conv1_filters, 8));
  set('advConv2Inline', nullish(cfg.conv2_filters, 16));
  set('advDenseInline', nullish(cfg.dense_units, 32));
  // Global ink polarity override (OCR projects). Defaults to 0 (dark ink on
  // a bright background), matching TrainConfig.ink_polarity.
  set('advInkPolarity', nullish(cfg.ink_polarity, 0));
  // Reading direction for multi-glyph projects (road signs). Defaults MUST
  // match TrainConfig.reading_order in trainer.py.
  set('advReadingOrder', nullish(cfg.reading_order, 'ltr'));
  // OCR mixing / transfer learning.  Defaults MUST match the TrainConfig
  // field defaults in trainer.py — the server sends the same values, so a
  // drift here would show one number and train with another.
  const setBool = (id, value) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.checked = !!value;
  }};
  set('advOcrSplit', nullish(cfg.ocr_emnist_split, 'balanced'));
  set('advOcrPerClass', nullish(cfg.ocr_emnist_per_class, 200));
  set('advOcrBackgroundN', nullish(cfg.ocr_background_samples, 0));
  set('advOcrPretrainN', nullish(cfg.ocr_pretrain_per_class, 1200));
  set('advOcrPretrainEpochs', nullish(cfg.ocr_pretrain_epochs, 4));
  set('advOcrHeadEpochs', nullish(cfg.ocr_head_epochs, 12));
  set('advOcrHeadLr', nullish(cfg.ocr_head_lr, 0.0015));
  set('advOcrFinetuneEpochs', nullish(cfg.ocr_finetune_epochs, 25));
  set('advOcrFinetuneLr', nullish(cfg.ocr_finetune_lr, 0.0002));
  set('advOcrUnfreeze', nullish(cfg.ocr_unfreeze_blocks, 1));
  setBool('advOcrEmnist', nullish(cfg.ocr_emnist_enabled, true));
  // One toggle, two backing fields: a Car License project owns the explicit
  // "use _background INSTEAD of OOD" setting (default OFF); a basic OCR
  // project keeps its existing background-class switch (default ON).
  setBool(
    'settingsUseBackgroundOod',
    isSignOcrProject()
      ? !!nullish(cfg.sign_background_replace_ood, false)
      : !!nullish(cfg.ocr_background_enabled, true)
  );
  setBool('advOcrTransfer', nullish(cfg.ocr_transfer_learning, true));
  syncOcrPanelVisibility();
  refreshRecHints();
}}
const OCR_BOOL_KEYS = ['ocr_emnist_enabled', 'ocr_background_enabled', 'sign_background_replace_ood', 'ocr_transfer_learning'];
const OCR_STRING_KEYS = ['ocr_emnist_split', 'reading_order'];
const OCR_FLOAT_KEYS = ['ocr_head_lr', 'ocr_finetune_lr'];
function setTrainCfgField(key, rawValue) {{
  const cfg = STATE.train_cfg || {{}};
  let v = rawValue;
  if (key === 'preprocess_mode') {{
    v = String(rawValue || 'auto_by_label');
    if (!['auto_by_label', 'manual_roi', 'none'].includes(v)) return;
  }} else if (OCR_BOOL_KEYS.includes(key)) {{
    // Checkboxes are strictly boolean — the trainer's dataclass fields are
    // typed bool, and a stringified "false" would read as truthy server-side.
    v = !!rawValue;
  }} else if (OCR_STRING_KEYS.includes(key)) {{
    v = String(rawValue || '');
    if (!v) return;
  }} else if (key === 'validation_split' || key === 'learning_rate' || OCR_FLOAT_KEYS.includes(key)) {{
    v = Number(v);
    if (!isFinite(v)) return;
  }} else {{
    v = parseInt(String(v), 10);
    if (!isFinite(v)) return;
  }}
  const next = Object.assign({{}}, cfg);
  next[key] = v;
  STATE.train_cfg = next;
  persistTrainCfgStorage();
  renderTrainStatus();
  if (OCR_BOOL_KEYS.includes(key)) syncOcrPanelVisibility();
  refreshRecHints();
}}
function setManualRoiField(index, rawValue) {{
  const cfg = Object.assign({{}}, STATE.train_cfg || {{}});
  const roi = manualRoiOrDefault(cfg);
  const n = Number(rawValue);
  if (!isFinite(n)) return;
  roi[index] = Math.max(0, Math.min(1, n));
  roi[2] = Math.max(roi[2], roi[0] + 0.01);
  roi[3] = Math.max(roi[3], roi[1] + 0.01);
  roi[0] = Math.min(roi[0], roi[2] - 0.01);
  roi[1] = Math.min(roi[1], roi[3] - 0.01);
  cfg.manual_roi = roi;
  STATE.train_cfg = cfg;
  persistTrainCfgStorage();
}}
// Road-sign OCR: a multi-glyph sequence read as one label ("L 50", "STOP").
// It shares the character pipeline with basic OCR but is the only project that
// reads glyphs in a direction, so the Sign Reading Direction control is gated
// on this rather than on isOcrProject().
function isSignOcrProject() {{
  return String((STATE && STATE.project_type) || '').toLowerCase() === 'sign_ocr';
}}
// Any glyph-based project. Both basic OCR (one character per crop) and road
// sign OCR draw from the same character preprocessing pipeline, so the glyph
// editors, the per-class preprocess records and the ink-polarity override
// apply to each.
function isOcrProject() {{
  const pt = String((STATE && STATE.project_type) || '').toLowerCase();
  return pt === 'ocr' || pt === 'sign_ocr';
}}
function syncOcrPanelVisibility() {{
  const cfg = STATE.train_cfg || {{}};
  const ocr = isOcrProject();
  // The OCR extras now live in the global Settings modal; show the section
  // only for OCR projects, and a hint otherwise.
  const modalSection = document.getElementById('modalOcrSection');
  if (modalSection) modalSection.style.display = ocr ? 'block' : 'none';
  const modalEmpty = document.getElementById('modalOcrEmpty');
  if (modalEmpty) modalEmpty.style.display = ocr ? 'none' : 'block';
  // Template-sample seeding only makes sense for a project whose classes are
  // characters (basic OCR and road signs share the glyph pipeline).
  const emnistGroup = document.getElementById('settingsEmnistGroup');
  if (emnistGroup) emnistGroup.style.display = ocr ? 'block' : 'none';
  const ocrPanel = document.getElementById('advOcrPanel');
  if (ocrPanel) ocrPanel.style.display = ocr ? 'grid' : 'none';
  // The global ink-polarity override applies to every glyph-based project:
  // basic OCR and road sign OCR alike.
  const inkRow = document.getElementById('advInkPolarityRow');
  if (inkRow) inkRow.style.display = ocr ? 'flex' : 'none';
  // Direction only means something when a label is a SEQUENCE of glyphs, which
  // is the road-sign case.  Basic OCR reads one character per crop, so the
  // control is hidden there rather than offering a setting with no effect.
  const readingRow = document.getElementById('advReadingOrderRow');
  if (readingRow) readingRow.style.display = isSignOcrProject() ? 'flex' : 'none';
  // Under transfer learning these three numbers are not built into the model
  // at all — ocr_trainer attaches a fresh N+1 head to the cached backbone —
  // so leaving them on screen would invite tuning a knob that does nothing.
  // They come back the moment transfer learning is switched off.
  const arch = document.getElementById('advArchRows');
  if (arch) {{
    const transfer = !!nullish(cfg.ocr_transfer_learning, true);
    arch.style.display = (ocr && transfer) ? 'none' : 'contents';
  }}
  // Same reasoning one level down: with mixing off there is no synthetic half
  // to size, and with the background class off there are no junk crops to
  // count.  Leaving the numbers live would imply they still do something.
  const show = (id, on) => {{
    const el = document.getElementById(id);
    if (el) el.style.display = on ? '' : 'none';
  }};
  // Edit ▸ Edit characters belongs to BASIC OCR only.  A road-sign project's
  // characters are edited in the three inline slots on the target bar — there
  // is nothing a second character editor could add, and two places writing the
  // same list is exactly how they drift apart.  The separator is toggled
  // alongside the item rather than wrapping the pair in a container, because
  // the dropdown's children are laid out by flex-direction: column and an
  // extra div would need its own display rule to stay in the flow.
  const basicOcr = String((STATE && STATE.project_type) || '').toLowerCase() === 'ocr';
  show('navEditCharacters', basicOcr);
  // Glyph settings are PER CLASS, so the item only makes sense once there is a
  // class to configure.  In a fresh OCR project the only classes are the ones
  // the empty state has already promised to delete on the first pick — opening
  // a configuration form for them would invite the user to tune settings that
  // are about to stop existing.
  show('navGlyphSettings', ocr && !ocrIsFreshProject());
  show('navEditCharactersSep', basicOcr);
  const mix = !!nullish(cfg.ocr_emnist_enabled, true);
  show('advOcrSplitRow', ocr && mix);
  show('advOcrPerClassRow', ocr && mix);
  show('advOcrMixHelp', ocr && mix);
  const useBackgroundOod = isSignOcrProject()
    ? !!nullish(cfg.sign_background_replace_ood, false)
    : !!nullish(cfg.ocr_background_enabled, true);
  show('advOcrBackgroundNRow', ocr && useBackgroundOod);
}}
function renderAdvancedPanel() {{
  const panel = document.getElementById('trainAdvPanel');
  if (!panel) return;
  // The advanced panel lives inline in the Training card and is toggled by
  // the "Advanced" button. Standard hyperparameters stay here for quick
  // tuning; only OCR-specific extras live in the global Settings modal.
  panel.style.display = trainAdvancedOpen ? 'block' : 'none';
  const chev = document.getElementById('advChevron');
  if (chev) chev.textContent = trainAdvancedOpen ? '▾' : '▸';
  const btn = document.getElementById('advBtn');
  if (btn) btn.setAttribute('aria-expanded', trainAdvancedOpen ? 'true' : 'false');
  syncOcrPanelVisibility();
  syncAdvancedInlineInputs();
  updateFlow();
  setTimeout(() => {{ if (typeof syncFrameHeight === 'function') syncFrameHeight(); }}, 50);
}}
async function applyRecommendedTrainCfg() {{
  const rec = STATE.train_rec || null;
  if (!rec) return;
  const cfg = Object.assign({{}}, STATE.train_cfg || {{}});
  let differs = false;
  const applyKey = (key) => {{
    if (!isFinite(Number(rec[key]))) return;
    const cur = Number(cfg[key]);
    const recV = Number(rec[key]);
    if (!isFinite(cur) || Math.abs(cur - recV) >= 1e-9) differs = true;
    cfg[key] = recV;
  }};
  applyKey('batch_size');
  applyKey('epochs');
  applyKey('learning_rate');
  applyKey('validation_split');
  if (differs) {{
    STATE.train_cfg = cfg;
    persistTrainCfgStorage();
    recHintsForced = false;
    syncAdvancedInlineInputs();
    // Persist to backend so the applied values survive page reruns
    try {{
      await fetch(`${{baseUrl}}/train/config`, {{
        method: 'POST',
        headers: {{'Content-Type':'application/json'}},
        body: JSON.stringify({{ session: STATE.session, cfg: STATE.train_cfg }})
      }});
      toast('Recommended settings applied.');
    }} catch (err) {{
      toast('Applied in this session (backend save failed).');
    }}
  }} else {{
    recHintsForced = !recHintsForced;
    refreshRecHints();
    toast(recHintsForced ? 'Showing recommendations.' : 'Hiding recommendations.');
  }}
}}
function toggleAdvancedInline() {{
  trainAdvancedOpen = !trainAdvancedOpen;
  renderAdvancedPanel();
}}
function bindAdvancedInlineHandlers() {{
  const panel = document.getElementById('trainAdvPanel');
  if (!panel || panel.dataset.bound === '1') return;
  panel.dataset.bound = '1';
  const bindNum = (id, key) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', (e) => setTrainCfgField(key, e.target.value));
    el.addEventListener('change', (e) => setTrainCfgField(key, e.target.value));
  }};
  bindNum('advImgSizeInline', 'img_size');
  bindNum('advBatchInline', 'batch_size');
  bindNum('advEpochsInline', 'epochs');
  bindNum('advValInline', 'validation_split');
  bindNum('advLrInline', 'learning_rate');
  bindNum('advConv1Inline', 'conv1_filters');
  bindNum('advConv2Inline', 'conv2_filters');
  bindNum('advDenseInline', 'dense_units');
  // Global ink polarity: an int (0 or 1) select. Stored as an int on
  // STATE.train_cfg.ink_polarity and applied to every class at train time.
  // On change it also immediately writes ink_polarity into every entry of
  // STATE.class_preprocess and persists via /classes/save_config, so live
  // preview / inference and newly added classes pick it up right away.
  const inkSel = document.getElementById('advInkPolarity');
  if (inkSel) {{
    inkSel.addEventListener('change', (e) => {{
      const v = parseInt(e.target.value, 10);
      setTrainCfgField('ink_polarity', v);
      applyGlobalInkPolarityToAllClasses(v);
      // ... and the OPEN DRAFT.  Writing the class records is not enough on
      // its own: the preview sends `sample_config` built from
      // classPreprocessDraft, and _preprocess_image_png_ocr prefers
      // sample_config over class_config outright — so a draft snapshotted
      // before this click kept rendering the previous polarity until the
      // modal was reopened.  No-op when nothing is open.  Kept alongside the
      // class write rather than instead of it: that one is what new classes
      // and the next modal-open read from.
      if (classPreprocessDraft && typeof classPreprocessDraft === 'object') {{
        classPreprocessDraft.ink_polarity = v;
      }}
      // The open preprocess preview renders with the class's ink_polarity, so
      // it would keep showing the previous polarity until the modal is
      // reopened.  No-op when nothing is open.
      refreshClassProcessedPreview();
    }});
  }}
  // 'change', not 'input': a checkbox mid-click reports its OLD checked state
  // on `input` in some engines, which would store the opposite of what the
  // user just clicked.
  const bindCheck = (id, key) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', (e) => setTrainCfgField(key, !!e.target.checked));
  }};
  const bindSelect = (id, key) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', (e) => setTrainCfgField(key, String(e.target.value)));
  }};
  bindCheck('advOcrEmnist', 'ocr_emnist_enabled');
  bindSelect('advOcrSplit', 'ocr_emnist_split');
  // Road-sign reading direction ("ltr" / "rtl"), a plain string on
  // STATE.train_cfg. Hidden for image and basic-OCR projects by
  // syncOcrPanelVisibility(); harmless when hidden because a hidden select
  // still carries its value into the train payload.
  bindSelect('advReadingOrder', 'reading_order');
  bindNum('advOcrPerClass', 'ocr_emnist_per_class');
  // The toggle writes BOTH fields.  `sign_background_replace_ood` is what a
  // Car License training/export reads; `ocr_background_enabled` keeps a
  // basic-OCR project's background switch in sync and is what the mixed-set
  // builder ultimately consumes.  Writing both here is what makes one
  // checkbox mean the same thing in either workspace.
  const bgOodToggle = document.getElementById('settingsUseBackgroundOod');
  if (bgOodToggle) {{
    bgOodToggle.addEventListener('change', (e) => {{
      const on = !!e.target.checked;
      setTrainCfgField('sign_background_replace_ood', on);
      setTrainCfgField('ocr_background_enabled', on);
      syncAdvancedInlineInputs();
    }});
  }}
  bindNum('advOcrBackgroundN', 'ocr_background_samples');
  bindCheck('advOcrTransfer', 'ocr_transfer_learning');
  bindNum('advOcrPretrainN', 'ocr_pretrain_per_class');
  bindNum('advOcrPretrainEpochs', 'ocr_pretrain_epochs');
  bindNum('advOcrHeadEpochs', 'ocr_head_epochs');
  bindNum('advOcrHeadLr', 'ocr_head_lr');
  bindNum('advOcrFinetuneEpochs', 'ocr_finetune_epochs');
  bindNum('advOcrFinetuneLr', 'ocr_finetune_lr');
  bindNum('advOcrUnfreeze', 'ocr_unfreeze_blocks');
}}
function resetTrainCfg() {{
  // The baseline is the server's TrainConfig dataclass, sent in the boot
  // payload — NOT a list here.  The literal that used to live here had
  // drifted to the old 8/16/32 architecture (the comment in trainer.py
  // records that stack collapsing to one class), so "Reset to defaults"
  // reset the model to something smaller than either the current default or
  // anything the user had chosen.  Anything the server adds to TrainConfig
  // from now on arrives here without this function changing.
  const defaults = STATE.train_cfg_defaults;
  if (!defaults || typeof defaults !== 'object') {{
    toast('Defaults unavailable — reload the page.');
    return;
  }}
  // Read the target BEFORE the reset overwrites it: signSequence() reads
  // STATE.train_cfg, and the defaults carry null there.
  const keptSignSequence = isSignOcrProject() ? signSequence() : null;
  const keptSignSequences = isSignOcrProject() ? signSequences() : null;
  STATE.train_cfg = Object.assign({{}}, defaults);
  // The target sign sequence is project CONTENT, not a hyperparameter: it says
  // what the project is about, the way the class list does.  A reset that
  // silently dropped it would leave a trained model pointed at nothing, with
  // no way to notice.  Carried across, like the classes are.
  if (isSignOcrProject()) {{
    STATE.train_cfg.sign_sequence = keptSignSequence;
    STATE.train_cfg.sign_sequences = keptSignSequences;
  }}
  persistTrainCfgStorage();
  syncAdvancedInlineInputs();
  renderTrainStatus();
  refreshRecHints();
  renderSignSequenceBar();
  toast('Advanced settings reset.');
}}
let classDragState = null;
function bindClassDragHandle(handle, index) {{
  handle.addEventListener('pointerdown', (e) => {{
    if (e.button !== 0) return;
    const card = handle.closest('.card.class-card');
    if (!card) return;
    const col = document.getElementById('classes');
    const gap = col ? (parseFloat(getComputedStyle(col).gap) || 24) : 24;
    classDragState = {{
      index, card, startY: e.clientY, target: index, moved: false,
      step: card.offsetHeight + gap,
    }};
    card.classList.add('drag-live');
    document.body.classList.add('class-dragging');
    try {{ handle.setPointerCapture(e.pointerId); }} catch (err) {{}}
    try {{ document.body.style.userSelect = 'none'; }} catch (err) {{}}
    queueFrameHeightSync();  // padding headroom grows the frame
    e.preventDefault();
  }});
  handle.addEventListener('pointermove', (e) => {{
    const s = classDragState;
    if (!s || s.index !== index) return;
    const dy = e.clientY - s.startY;
    if (!s.moved && Math.abs(dy) < 5) return;
    s.moved = true;
    s.card.style.transform = `translateY(${{dy}}px)`;
    s.card.style.zIndex = '40';
    s.card.style.opacity = '0.9';
    const cards = Array.from(document.querySelectorAll('#classes .card.class-card'));
    let target = s.index;
    for (let i = 0; i < cards.length; i++) {{
      const r = cards[i].getBoundingClientRect();
      if (e.clientY > r.top + r.height / 2) target = i;
    }}
    if (cards.length && e.clientY > cards[cards.length - 1].getBoundingClientRect().bottom) {{
      target = cards.length;
    }}
    s.target = target;
    // Squeeze: cards between the dragged slot and the target slot slide out
    // of the way (animated by the .card transition), leaving a visible gap
    // where the card will land.
    for (let i = 0; i < cards.length; i++) {{
      if (i === s.index) continue;
      let shift = 0;
      if (target > s.index && i > s.index && i <= target) shift = -s.step;
      else if (target < s.index && i >= target && i < s.index) shift = s.step;
      cards[i].style.transform = shift ? `translateY(${{shift}}px)` : '';
    }}
    updateFlow();  // class→training connector curves follow the cards live
    queueFrameHeightSync();
    // Auto-scroll the PARENT Streamlit page while dragging near the
    // viewport edges, so classes off-screen (5+ cards) stay reachable.
    try {{
      const pwin = window.parent;
      const doc = pwin ? pwin.document : null;
      if (doc) {{
        let scroller = doc.scrollingElement || doc.documentElement;
        if (!scroller || scroller.scrollHeight <= scroller.clientHeight) {{
          const main = doc.querySelector('section.stMain, main');
          if (main && main.scrollHeight > main.clientHeight) scroller = main;
        }}
        if (scroller) {{
          const edge = 70;
          const speed = 20;
          if (e.clientY > window.innerHeight - edge) {{
            scroller.scrollTop += Math.min(speed, Math.max(1, (e.clientY - (window.innerHeight - edge)) / 2));
          }} else if (e.clientY < edge) {{
            scroller.scrollTop -= Math.min(speed, Math.max(1, (edge - e.clientY) / 2));
          }}
        }}
      }}
    }} catch (err) {{}}
  }});
  const finishDrag = () => {{
    const s = classDragState;
    if (!s || s.index !== index) return;
    classDragState = null;
    try {{
      s.card.style.transform = '';
      s.card.style.zIndex = '';
      s.card.style.opacity = '';
      s.card.classList.remove('drag-live');
    }} catch (err) {{}}
    try {{ document.body.style.userSelect = ''; }} catch (err) {{}}
    if (s.moved && s.target !== s.index) {{
      // render() rebuilds the column in the new order — no need to animate
      // the others back; just release the headroom padding.
      document.body.classList.remove('class-dragging');
      moveClass(s.index, s.target);
      updateFlow();
    }} else {{
      // Canceled or dropped in place: animate the squeeze back, then shrink
      // the frame again.  The 0.16s card transition means the connector
      // curves must be re-drawn a few times to follow the return animation.
      document.querySelectorAll('#classes .card.class-card').forEach((c) => {{
        c.style.transform = '';
      }});
      document.body.classList.remove('class-dragging');
      queueFrameHeightSync();
      updateFlow();
      window.setTimeout(() => updateFlow(), 80);
      window.setTimeout(() => updateFlow(), 200);
    }}
  }};
  handle.addEventListener('pointerup', finishDrag);
  handle.addEventListener('pointercancel', finishDrag);
}}
async function moveClass(fromIndex, toIndex) {{
  const classes = Array.isArray(STATE.classes) ? STATE.classes.slice() : [];
  if (fromIndex < 0 || fromIndex >= classes.length || toIndex < 0 || toIndex > classes.length) return;
  if (fromIndex === toIndex) return;
  const prev = classes.slice();
  const [moved] = classes.splice(fromIndex, 1);
  classes.splice(toIndex, 0, moved);
  STATE.classes = classes;
  render();
  scheduleLayoutResync();
  try {{
    const res = await fetch(`${{baseUrl}}/classes/reorder`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{session: STATE.session, classes: classes}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to reorder classes.');
    syncTrainUi();
    refreshTrainRec();
  }} catch (err) {{
    STATE.classes = prev;
    render();
    scheduleLayoutResync();
    toast(String(err && err.message ? err.message : err));
  }}
}}
function render() {{
  // #region debug-point B:render-entry
  dbgEvent('B', 'app.py:render', '[DEBUG] render start', {{classes: STATE.classes, openSourceClass, openSourceKind, initial_open_source_class: STATE.initial_open_source_class, initial_open_source_kind: STATE.initial_open_source_kind}});
  // #endregion
  const signFocus = signSlotFocusSnapshot();
  // A fresh sign project starts with a real target, and every project's class
  // list is rewritten to its slots BEFORE the cards are built — so the
  // column's first frame already shows the three character cards, never an
  // auto-named placeholder and never the OCR picker.
  try {{
    ensureSignSequenceDefault();
    adoptSignSequenceClasses();
  }} catch (e) {{ console.error('[render] sign class adopt', e); }}
  const root = document.getElementById('classes');
  root.innerHTML = '';
  // The target sequence goes above the cards, before the loop builds them.
  // Mounted here rather than in the tail because it is the first thing in the
  // column: the sign a project is being built to read should be readable
  // before any of the characters it is built from.
  try {{
    renderSignSequenceBar(root);
    restoreSignSlotFocus(signFocus);
    syncLicenseListNav();
  }} catch (e) {{ console.error('[render] sign sequence bar', e); }}
  // A fresh OCR project presents onboarding instead of two untouched
  // placeholders.  Decided once, before the loop, because the loop's input is
  // what changes: with nothing picked there are no cards at all and the hero
  // takes the whole column.  renderList is STATE.classes verbatim in every
  // other case, so the card indices below still line up with it.
  const ocrEmptyState = isOcrProject() && ocrIsFreshProject();
  const renderList = ocrEmptyState ? [] : STATE.classes;
  for (let i=0; i<renderList.length; i++) {{
    const name = renderList[i];
    try {{
      const card = document.createElement('div');
      card.className = 'card class-card';
      const head = document.createElement('div');
      head.className = 'class-head';
      // A sign project's cards follow the three slots: there is no manual
      // add, delete, rename or drag, because every one of those would be
      // undone by the next sync.  The card stays a capture surface for its
      // character, which is the only thing the sequence does not decide.
      const signMode = isSignOcrProject();
      if (!signMode) {{
        const dragHandle = document.createElement('span');
        dragHandle.className = 'class-drag-handle';
        dragHandle.textContent = '⠿';
        dragHandle.title = 'Drag to reorder classes';
        bindClassDragHandle(dragHandle, i);
        head.appendChild(dragHandle);
      }}
      const title = document.createElement('div');
      title.className = 'class-title';
      title.textContent = name;
      if (signMode) {{
        // A character used in more than one slot is still ONE class; the
        // badge says which positions it answers for ("Slots 1 & 3").
        const slots = signSlotLabel(String(name == null ? '' : name).toUpperCase());
        if (slots) {{
          const badge = document.createElement('span');
          badge.className = 'signslot-badge';
          badge.textContent = slots;
          title.appendChild(badge);
        }}
      }} else {{
        const edit = document.createElement('button');
        edit.className = 'iconbtn tm-rename-btn';
        edit.title = 'Rename this class';
        edit.innerHTML = createIcon('rename');
        edit.onclick = () => renameClass(name);
        title.appendChild(edit);
      }}
      head.appendChild(title);
      const more = document.createElement('button');
      more.className = 'iconbtn more tm-delete-class';
      // Trash can, not a cross: the cross now always means "fold this panel
      // away", so the card-level destructive action needs its own shape.
      more.innerHTML = createIcon('trash');
      more.title = 'Delete Class';
      more.onclick = async (e) => {{
        if (e) {{
          e.preventDefault();
          e.stopPropagation();
        }}
        await deleteClass(name);
      }};
      // Edit (preprocess) sits immediately left of Clear Samples, separated
      // from the Rename icon by the title's flex spacer so the two edits
      // can't be misclicked into each other.
      const preprocessBtn = document.createElement('button');
      preprocessBtn.className = 'preprocess-chip class-preprocess-chip';
      preprocessBtn.type = 'button';
      preprocessBtn.textContent = 'Edit';
      preprocessBtn.onclick = (e) => {{
        if (e) {{
          e.preventDefault();
          e.stopPropagation();
        }}
        openClassPreprocessEditor(name);
      }};
      const clearBtn = document.createElement('button');
      clearBtn.className = 'iconbtn clear-samples';
      clearBtn.title = 'Clear all samples';
      clearBtn.innerHTML = createIcon('clear');
      clearBtn.onclick = async (e) => {{
        if (e) {{
          e.preventDefault();
          e.stopPropagation();
        }}
        await clearSamples(name);
      }};
      head.appendChild(preprocessBtn);
      head.appendChild(clearBtn);
      if (!signMode) head.appendChild(more);  // delete ⋮ sits at the head's right end
      card.appendChild(head);
      const div = document.createElement('div');
      div.className = 'divider';
      card.appendChild(div);
      const c = Number(STATE.counts[name] || 0);
      const sh = document.createElement('div');
      sh.className = 'summary-title';
      sh.id = `summaryTitle-${{cssSafe(name)}}`;
      sh.textContent = `${{c}} ${{c === 1 ? 'Image Sample' : 'Image Samples'}}`;
      card.appendChild(sh);

      const row = document.createElement('div');
      row.className = 'tm-action-toolbar';

      const upWebcam = document.createElement('button');
      upWebcam.className = 'sample';
      upWebcam.innerHTML = `${{createIcon('webcam')}}<span>Webcam</span>`;
      upWebcam.onclick = () => openSourcePanel('webcam', name);

      const upUpload = document.createElement('button');
      upUpload.className = 'sample';
      upUpload.innerHTML = `${{createIcon('upload')}}<span>Upload</span>`;
      upUpload.onclick = () => openSourcePanel('upload', name);

      const upDevice = document.createElement('button');
      upDevice.className = 'sample';
      upDevice.innerHTML = `${{createIcon('device')}}<span>Device</span>`;
      upDevice.onclick = () => openSourcePanel('device', name);

      row.appendChild(upWebcam);
      row.appendChild(upUpload);
      row.appendChild(upDevice);

      if (openSourceClass === name) {{
        // Highlight the tile whose drawer is currently open: clicking it
        // again folds the drawer back up (openSourcePanel toggles).
        const activeTile = {{ webcam: upWebcam, upload: upUpload, device: upDevice }}[openSourceKind];
        if (activeTile) activeTile.classList.add('active');
      }}

      if (openSourceClass === name) {{
        // #region debug-point B:render-open-panel
        dbgEvent('B', 'app.py:render', '[DEBUG] rendering expanded source panel', {{name, openSourceKind, sampleCount: c}});
        // #endregion
        card.appendChild(row);
        const selectedPort = currentSerialPort || '';
        const sampleCount = c;
        const panel = document.createElement('div');
        panel.className = 'source-panel';
        let bodyMarkup;
        try {{
          const buildChip = (label, icon, kind) => `
            <span class="tm-source-chip tm-source-chip-${{kind}}" title="${{escapeHtml(String(label || ''))}}">
              <span class="tm-chip-icon">${{icon}}</span>
              <span class="tm-chip-text">${{escapeHtml(String(label || '—'))}}</span>
            </span>
          `;
          const currentPortLabel = (() => {{
            const opts = Array.isArray(STATE.serial_ports) ? STATE.serial_ports : [];
            const m = opts.find(p => p && String(p.device || p.path || p) === String(currentSerialPort || ''));
            if (m) return String(m.description || m.product || m.device || m.path || currentSerialPort);
            return currentSerialPort || 'No device selected';
          }})();
          const currentCamLabel = (() => {{
            const idx = Number(currentWebcamIndex);
            const opts = Array.isArray(STATE.webcam_options) ? STATE.webcam_options : [];
            const m = opts.find(w => Number(w.index) === idx || String(w.uid || '') === String(currentWebcamUid || ''));
            return (m && String(m.label || m.device || m.title || '')) || (opts[idx] && String(opts[idx].label || opts[idx].device || `Camera ${{idx}}`)) || `Camera ${{idx}}`;
          }})();
          const jumpChip = `<button type="button" class="tm-source-chip tm-source-chip-settings" id="sourceSettingsToggle-${{cssSafe(name)}}" title="Open global input settings">
            <span class="tm-chip-icon">${{createIcon('settings')}}</span><span class="tm-chip-text">Settings</span>
          </button>`;
          const sourceChipRow = (chips) => `<div class="tm-source-chip-row">${{chips}}</div>`;
          bodyMarkup =
            openSourceKind === 'upload'
              ? `<button class="upload-pick" type="button" id="uploadPick-${{cssSafe(name)}}">Choose images from your files</button>
                 <div class="upload-hint">Images are added to this class and appear immediately on the right.</div>`
              : (
                openSourceKind === 'device'
                  ? `${{sourceChipRow(buildChip(currentPortLabel, createIcon('device'), 'port') + jumpChip)}}
                     <div class="device-help">${{deviceHelpText()}}</div>
                     <div class="source-preview-wrap">
                       <img id="sourcePreview-${{cssSafe(name)}}" class="preview-frame" alt="Preview"/>
                     </div>
                     <div class="source-note" id="sourceNote-${{cssSafe(name)}}"></div>
                     <div class="source-actions">
                       <button class="btn btn-primary" type="button" id="sourceCapture-${{cssSafe(name)}}">Capture</button>
                       <button class="btn" type="button" id="sourceHold-${{cssSafe(name)}}">${{holdRecording && holdRecordClass === name ? 'Recording...' : 'Hold to Capture'}}</button>
                     </div>`
                  : `${{sourceChipRow(buildChip(currentCamLabel, createIcon('webcam'), 'cam') + jumpChip)}}
                     <div class="device-help">${{webcamHelpText()}}</div>
                     <div class="source-preview-wrap">
                       <img id="sourcePreview-${{cssSafe(name)}}" class="preview-frame" alt="Preview"/>
                     </div>
                     <div class="source-note" id="sourceNote-${{cssSafe(name)}}"></div>
                     <div class="source-actions">
                       <button class="btn btn-primary" type="button" id="sourceCapture-${{cssSafe(name)}}">Capture</button>
                       <button class="btn" type="button" id="sourceHold-${{cssSafe(name)}}">${{holdRecording && holdRecordClass === name ? 'Recording...' : 'Hold to Capture'}}</button>
                     </div>`
              );
        }} catch (markupErr) {{
          console.error('[render] source-panel innerHTML build failed for', name, markupErr);
          bodyMarkup = `<div class="render-error" style="padding:12px;background:#331010;color:#ffaaaa;border-radius:8px;font-family:monospace;white-space:pre-wrap;font-size:12px;"><b>Render error (${{escapeHtml(name)}}):</b>\n${{escapeHtml(String(markupErr && markupErr.stack || markupErr.message || markupErr))}}</div>`;
        }}
        try {{
          panel.innerHTML = `
            <div class="source-left">
              <div class="source-head">
                <span>${{sourceLabel(openSourceKind)}}</span>
                <div class="source-tools">
                  <button class="source-close-pill" type="button" title="Collapse panel (Esc)" id="sourceClose-${{cssSafe(name)}}">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"></polyline></svg>
                    <span>Close</span>
                  </button>
                </div>
              </div>
              ${{bodyMarkup}}
            </div>
            <div class="source-right">
              <h4>${{sourceSamplesTitle(openSourceKind)}}</h4>
              <div class="source-count" id="sourceCount-${{cssSafe(name)}}">${{sampleCount}}<small>${{sampleCount === 1 ? 'Image Sample' : 'Image Samples'}}</small></div>
              <div id="samplesHost-${{cssSafe(name)}}">${{buildSamplesMarkup(name)}}</div>
            </div>
          `;
        }} catch (htmlErr) {{
          console.error('[render] panel.innerHTML assignment failed for', name, htmlErr);
          panel.innerHTML = `<div class="render-error" style="padding:12px;background:#331010;color:#ffaaaa;border-radius:8px;font-family:monospace;white-space:pre-wrap;font-size:12px;"><b>Preview render error:</b>\n${{escapeHtml(String(htmlErr && htmlErr.stack || htmlErr.message || htmlErr))}}</div>`;
        }}
        card.appendChild(panel);
      }} else {{
        const summary = document.createElement('div');
        summary.className = 'summary-row';
        summary.appendChild(row);
        const samples = document.createElement('div');
        samples.className = 'summary-samples';
        try {{
          samples.innerHTML = buildSamplesStripMarkup(name);
        }} catch (stripErr) {{
          console.error('[render] samples strip failed for', name, stripErr);
          samples.innerHTML = `<div class="render-error" style="padding:8px;color:#ffaaaa;font-family:monospace;font-size:12px;">Samples render failed: ${{escapeHtml(String(stripErr && stripErr.message || stripErr))}}</div>`;
        }}
        summary.appendChild(samples);
        card.appendChild(summary);
      }}
      root.appendChild(card);
    }} catch (classErr) {{
      // Swallow individual class failures so a single broken preview/state
      // doesn't abort the entire render and leave *all* classes blank.
      // Also show the error in-DOM so the user doesn't need devtools.
      console.error('[render] class card build FAILED for class:', name, classErr);
      const errCard = document.createElement('div');
      errCard.className = 'card class-card';
      const errHead = document.createElement('div');
      errHead.className = 'class-head';
      const errTitle = document.createElement('div');
      errTitle.className = 'class-title';
      errTitle.textContent = name;
      errHead.appendChild(errTitle);
      errCard.appendChild(errHead);
      const errDivider = document.createElement('div');
      errDivider.className = 'divider';
      errCard.appendChild(errDivider);
      const errBody = document.createElement('div');
      errBody.style.cssText = 'padding:12px 16px;background:#331010;color:#ffaaaa;border-radius:8px;font-family:monospace;white-space:pre-wrap;font-size:12px;margin:8px;';
      errBody.innerHTML = `<b>Render failed for class "${{escapeHtml(name)}}":</b>\n${{escapeHtml(String(classErr && classErr.stack || classErr.message || classErr))}}\n\nPlease paste this error text when asking for help.`;
      errCard.appendChild(errBody);
      root.appendChild(errCard);
      // Don't rethrow — keep looping so other classes still render.
    }}
  }}
  try {{
    if (ocrEmptyState) {{
      // Nothing picked yet: the hero replaces the column rather than sitting
      // above it (the loop above rendered no cards to sit above).  Its button
      // opens the same modal the Edit menu does.
      renderOcrEmptyState(root);
    }} else if (!isOcrProject()) {{
      const add = document.createElement('div');
      add.className = 'addclass';
      add.innerHTML = `<span class="addclass-icon">${{createIcon('plus')}}</span><span>Add a class</span>`;
      add.onclick = () => addClass();
      root.appendChild(add);
    }}
    // An OCR project past the empty state renders its cards and nothing else:
    // the selector now lives in the Edit menu's modal, which is rendered in
    // the tail below so it survives this rebuild.

    // The cards above are showing the sequence's characters; make the
    // server's class list agree.  Deferred to a timer so this render always
    // finishes first — a response can itself trigger a render when the
    // server normalises a name.
    if (isSignOcrProject()) scheduleSignClassSync(0);

    const trainBtn = document.getElementById('trainBtn');
    recomputeTrainEnabled();
    refreshTrainRec();
    syncTrainUi();
    if (trainBtn) trainBtn.onclick = () => {{
      if (!STATE.train_enabled) return;
      startTrain();
    }};

    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {{
      exportBtn.disabled = !STATE.export_enabled;
      exportBtn.onclick = async () => {{
        if (!STATE.export_enabled) return;
        exportBtn.disabled = true;
        try {{
          const pickRes = await fetch(`${{baseUrl}}/export/pick_dir?session=${{encodeURIComponent(STATE.session)}}`);
          const pick = await pickRes.json().catch(() => ({{ok:'0'}}));
          if (!pickRes.ok || pick.ok !== '1') {{
            if (pick && pick.canceled === '1') return;
            throw new Error(pick.error || 'Unable to choose folder.');
          }}
          const exportDir = String(pick.export_dir || '').trim();
          if (!exportDir) return;
          const runData = await exportRunWithOverwriteConfirm(
            exportDir,
            String(exportModelName || 'tm'),
            String(exportArrayName || '')
          );
          if (!runData || runData.canceled === '1') return;
          toast(`Exported to: ${{String(runData.export_dir || exportDir)}}`);
        }} catch (e) {{
          toast(String(e && e.message ? e.message : e));
        }} finally {{
          exportBtn.disabled = !STATE.export_enabled;
        }}
      }};
    }}
    try {{
      renderPreviewCard();
      renderTrainStatus();
      bindPreviewControls();
      renderClassPreprocessModal();
      // Rebuilt on every render so the chips stay in step with the cards: the
      // modal stays open across a pick, and addOcrCharacters() re-renders.
      renderOcrCharModal();
      renderOcrGlyphModal();
    }} catch (previewErr) {{
      console.error('[render] preview/train binding failed', previewErr);
      toast(`Preview init failed: ${{String(previewErr && previewErr.message || previewErr)}}`, 8000);
    }}

    const advBtn = document.getElementById('advBtn');
    if (advBtn) advBtn.onclick = toggleAdvancedInline;
    const advReset = document.getElementById('advReset');
    if (advReset) advReset.onclick = () => resetTrainCfg();
    const advUseRec = document.getElementById('advUseRec');
    if (advUseRec) advUseRec.onclick = () => applyRecommendedTrainCfg();
    try {{ bindAdvancedInlineHandlers(); }} catch (e) {{ console.error('[render] bindAdvancedInlineHandlers', e); }}
    try {{ renderAdvancedPanel(); }} catch (e) {{ console.error('[render] renderAdvancedPanel', e); }}
    const nav = document.getElementById('goHome');
    const navOpen = document.getElementById('navOpenProject');
    const navSave = document.getElementById('navSaveProject');
    const navExportDataset = document.getElementById('navExportDataset');
    const navReturn = document.getElementById('navReturn');
    const navReset = document.getElementById('navResetProject');
    if (nav) nav.onclick = (e) => {{
      if (e) e.stopPropagation();
    }};
    // Toolbar segmented dropdowns (Problem 2: File / Edit menus)
    const menuFileBtn = document.getElementById('menuFileBtn');
    if (menuFileBtn) menuFileBtn.onclick = (e) => {{
      if (e) e.stopPropagation();
      const d = document.getElementById('menuFileDropdown');
      const isOpen = d && d.classList.contains('open');
      closeToolbarDropdowns();
      if (!isOpen && d) d.classList.add('open');
    }};
    const menuEditBtn = document.getElementById('menuEditBtn');
    if (menuEditBtn) menuEditBtn.onclick = (e) => {{
      if (e) e.stopPropagation();
      const d = document.getElementById('menuEditDropdown');
      const isOpen = d && d.classList.contains('open');
      closeToolbarDropdowns();
      if (!isOpen && d) d.classList.add('open');
    }};
    // OCR only, like the item itself — syncOcrPanelVisibility() hides it
    // everywhere else, so this binding is dormant in an image project.
    const navEditCharacters = document.getElementById('navEditCharacters');
    if (navEditCharacters) navEditCharacters.onclick = (e) => {{
      if (e) e.stopPropagation();
      closeToolbarDropdowns();
      openOcrCharModal();
    }};
    const navGlyphSettings = document.getElementById('navGlyphSettings');
    if (navGlyphSettings) navGlyphSettings.onclick = (e) => {{
      if (e) e.stopPropagation();
      closeToolbarDropdowns();
      openOcrGlyphModal();
    }};
    // Settings button (Problem 3.2 SSOT modal) + settings close/save bindings
    const navSettings = document.getElementById('navSettings');
    if (navSettings) navSettings.onclick = (e) => {{
      if (e) e.stopPropagation();
      closeToolbarDropdowns();
      openGlobalSettings();
    }};
    const navLicenseList = document.getElementById('navLicenseList');
    if (navLicenseList) navLicenseList.onclick = (e) => {{
      if (e) e.stopPropagation();
      closeToolbarDropdowns();
      openLicenseListModal();
    }};
    const licClose = document.getElementById('licenseListClose');
    if (licClose) licClose.onclick = (e) => {{ if (e) e.stopPropagation(); closeLicenseListModal(); }};
    const licCancel = document.getElementById('licenseListCancel');
    if (licCancel) licCancel.onclick = (e) => {{ if (e) e.stopPropagation(); closeLicenseListModal(); }};
    const licSave = document.getElementById('licenseListSave');
    if (licSave) licSave.onclick = (e) => {{ if (e) e.stopPropagation(); saveLicenseList(); }};
    const licAdd = document.getElementById('licenseListAdd');
    if (licAdd) licAdd.onclick = (e) => {{
      if (e) e.stopPropagation();
      licenseDraft.push('');
      renderLicenseListModal();
      const row = licenseDraft.length - 1;
      const first = document.getElementById(`licenseSlot-${{row}}-0`);
      if (first) first.focus();
    }};
    const licModal = document.getElementById('licenseListModal');
    if (licModal) licModal.onclick = (e) => {{
      if (e && e.target === licModal) closeLicenseListModal();
    }};
    const setCancel = document.getElementById('settingsCancel');
    if (setCancel) setCancel.onclick = (e) => {{ if (e) e.stopPropagation(); closeGlobalSettings(); }};
    const setSave = document.getElementById('settingsSave');
    if (setSave) setSave.onclick = (e) => {{ if (e) e.stopPropagation(); saveGlobalSettings(); }};
    const setModal = document.getElementById('globalSettingsModal');
    if (setModal) setModal.onclick = (e) => {{
      if (e && e.target === setModal) closeGlobalSettings();
    }};
    // Advanced Settings expand/collapse toggle
    const advToggle = document.getElementById('settingsAdvancedToggle');
    const advBody = document.getElementById('settingsAdvancedBody');
    const advChevron = document.getElementById('settingsAdvancedChevron');
    if (advToggle && advBody) {{
      advToggle.onclick = (e) => {{
        if (e) e.stopPropagation();
        const expanded = advToggle.getAttribute('aria-expanded') === 'true';
        advToggle.setAttribute('aria-expanded', String(!expanded));
        advBody.style.display = expanded ? 'none' : 'flex';
      }};
    }}
    // Template-sample auto-fill (Settings -> Advanced, OCR projects only).
    // The checkbox does BOTH things at once: that is the point of "existing
    // classes update immediately".  Turning it on persists the flag, then
    // the controller backfills every character class that already exists
    // with zero samples and returns the full state — which is applied to
    // STATE and re-rendered right here, with no reload.
    const autoSeedBox = document.getElementById('settingsAutoSeedEmnist');
    if (autoSeedBox) {{
      autoSeedBox.onchange = (e) => {{
        if (e) e.stopPropagation();
        applyAutoSeedEmnistSetting(!!autoSeedBox.checked);
      }};
    }}
    const seedBtn = document.getElementById('settingsSeedEmnistNowBtn');
    if (seedBtn) {{
      seedBtn.onclick = async (e) => {{
        if (e) e.stopPropagation();
        seedBtn.disabled = true;
        try {{
          const data = await fillEmptyClassesWithEmnist();
          if (data) {{
            const seeded = Array.isArray(data.seeded) ? data.seeded : [];
            toast(seeded.length
              ? `Filled ${{seeded.length}} empty character class${{seeded.length === 1 ? '' : 'es'}} with 24 template samples each.`
              : 'Every character class already has samples — nothing to fill.');
          }}
        }} finally {{
          seedBtn.disabled = false;
        }}
      }};
    }}
    // Preferences button + modal bindings
    const navPrefs = document.getElementById('navPreferences');
    if (navPrefs) navPrefs.onclick = (e) => {{
      if (e) e.stopPropagation();
      closeToolbarDropdowns();
      openPreferences();
    }};
    const prefCancel = document.getElementById('prefCancel');
    if (prefCancel) prefCancel.onclick = (e) => {{ if (e) e.stopPropagation(); closePreferences(); }};
    const prefSave = document.getElementById('prefSave');
    if (prefSave) prefSave.onclick = (e) => {{ if (e) e.stopPropagation(); savePreferences(); }};
    const prefModal = document.getElementById('preferencesModal');
    if (prefModal) prefModal.onclick = (e) => {{
      if (e && e.target === prefModal) closePreferences();
    }};
    if (navOpen) navOpen.onclick = async (e) => {{
      if (e) e.stopPropagation();
      closeNavMenu();
      await openProject();
    }};
    if (navSave) navSave.onclick = async (e) => {{
      if (e) e.stopPropagation();
      closeNavMenu();
      await saveProjectQuick();
    }};
    const navSaveAs = document.getElementById('navSaveAsProject');
    if (navSaveAs) navSaveAs.onclick = async (e) => {{
      if (e) e.stopPropagation();
      closeNavMenu();
      await saveProjectAs();
    }};
    const navNew = document.getElementById('navNewProject');
    if (navNew) navNew.onclick = async (e) => {{
      if (e) e.stopPropagation();
      closeNavMenu();
      await newProject();
    }};
    if (navExportDataset) navExportDataset.onclick = async (e) => {{
      if (e) e.stopPropagation();
      closeNavMenu();
      await exportDataset();
    }};
    if (navReturn) navReturn.onclick = async (e) => {{
      if (e) e.stopPropagation();
      closeNavMenu();
      await returnHome();
    }};
    if (navReset) navReset.onclick = async (e) => {{
      if (e) e.stopPropagation();
      closeNavMenu();
      await resetProject();
    }};
    if (!navMenuBound) {{
      document.addEventListener('click', () => closeNavMenu());
      navMenuBound = true;
    }}
    if (openSourceClass) {{
      const safe = cssSafe(openSourceClass);
      const closeBtn = document.getElementById(`sourceClose-${{safe}}`);
      const capBtn = document.getElementById(`sourceCapture-${{safe}}`);
      const holdBtn = document.getElementById(`sourceHold-${{safe}}`);
      const devSel = document.getElementById(`deviceSelect-${{safe}}`);
      const uploadPick = document.getElementById(`uploadPick-${{safe}}`);
      const settingsToggle = document.getElementById(`sourceSettingsToggle-${{safe}}`);
      const settingsSave = document.getElementById(`sourceSettingsSave-${{safe}}`);
      const settingsCancel = document.getElementById(`sourceSettingsCancel-${{safe}}`);
      if (closeBtn) closeBtn.onclick = () => closeSourcePanel();
      if (capBtn) capBtn.onclick = captureSource;
      if (holdBtn) {{
        holdBtn.onpointerdown = (e) => {{
          try {{ holdBtn.setPointerCapture(e.pointerId); }} catch (err) {{}}
          e.preventDefault();
          startHoldCapture();
        }};
        holdBtn.onpointerup = (e) => {{
          e.preventDefault();
          stopHoldCapture();
        }};
        holdBtn.onpointercancel = () => stopHoldCapture();
        holdBtn.onpointerleave = () => stopHoldCapture();
        holdBtn.onlostpointercapture = () => stopHoldCapture();
      }}
      if (devSel) devSel.onchange = (e) => changeDevicePort(openSourceClass, e.target.value || '');
      if (devSel) {{
        const selectId = `deviceSelect-${{safe}}`;
        devSel.onpointerdown = () => refreshSerialPorts(false, selectId);
        devSel.onmousedown = () => refreshSerialPorts(false, selectId);
      }}
      if (settingsToggle) settingsToggle.onclick = (e) => {{
        if (e) {{ e.preventDefault(); e.stopPropagation(); }}
        openGlobalSettings();
      }};
      if (settingsSave) settingsSave.onclick = () => applySourceSettings(openSourceClass);
      if (settingsCancel) settingsCancel.onclick = () => {{
        sourceSettingsOpen = false;
        render();
      }};
      if (uploadPick) uploadPick.onclick = async () => {{
        if (window.pywebview && window.pywebview.api && window.pywebview.api.pick_upload_files) {{
          // Packaged app: the SPA <input type="file"> does not open a dialog
          // inside pywebview on macOS — use the desktop shell's native picker.
          try {{
            const sess = String(STATE.session || '');
            if (!sess) {{
              toast('Upload not ready yet — wait for the project to load, then try again.', 6000);
              return;
            }}
            const results = await window.pywebview.api.pick_upload_files(baseUrl, sess, openSourceClass);
            const total = Array.isArray(results) ? results.length : 0;
            if (!results || !total) {{
              // Either user Cancelled the dialog, or picker hit a hard error.
              const err = (results && results[0] && results[0].error) ? String(results[0].error) : '';
              if (err) toast(`Upload aborted: ${{err}}`, 6000);
              return;
            }}
            let uploaded = 0;
            const failed = [];
            for (const r of (results || [])) {{
              const fname = String((r && r.filename) || 'file');
              if (r && r.ok) {{
                uploaded += 1;
                const thumb = String((r && r.thumb_b64) || (r && r.image_b64) || '');
                if (thumb) prependSamplePreview(openSourceClass, {{src: `data:image/png;base64,${{thumb}}`, filename: String(r.saved_filename || r.filename || fname)}});
                incrementSampleCount(openSourceClass, 1);
              }} else {{
                const why = String((r && r.error) || '');
                failed.push(why ? `${{fname}} (${{why}})` : fname);
                toast(`Upload failed: ${{fname}}${{why ? ' — ' + why : ''}}`, 5000);
              }}
            }}
            recomputeTrainEnabled();
            syncTrainUi();
            updateOpenSamplesPanel(openSourceClass);
            const lines = [`Uploaded ${{uploaded}}/${{total}} sample(s) to ${{openSourceClass}}.`];
            if (failed.length) lines.push(`Failed ${{failed.length}}: ${{failed.slice(0, 8).join('; ')}}${{failed.length > 8 ? '; ...' : ''}}`);
            toast(lines.join(' '), Math.max(4000, 2000 + Math.min(8000, failed.length * 600)));
          }} catch (e) {{
            toast(String(e && e.message ? e.message : e), 6000);
          }}
          return;
        }}
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.multiple = true;
        input.onchange = () => uploadFiles(openSourceClass, input.files);
        input.click();
      }};
      syncSourceActionButtons(openSourceClass);
    }}
    root.querySelectorAll('.class-card').forEach((card, idx) => {{
      const className = STATE.classes[idx];
      if (className) bindSampleDeleteButtons(card, className);
    }});
    bindLayoutImageObservers(root);

    // The very first open of a freshly-created project can race Streamlit's
    // own STATE.session hydration — so first call is silent if session is
    // still empty, then we retry once 2 s later.
    const sessAtLaunch = String(STATE.session || '').trim();
    if (sessAtLaunch) {{
      ensureOpenSourceLive({{silentFail: true}});
    }} else {{
      ensureOpenSourceLive({{silent: true}});
      setTimeout(() => {{
        const sess = String(STATE.session || '').trim();
        if (sess) {{
          ensureOpenSourceLive({{silentFail: true}});
          if (openSourceClass && openSourceKind && (openSourceKind === 'webcam' || openSourceKind === 'device')) {{
            startSourceLiveLoop();
          }}
        }}
      }}, 2000);
    }}
    updateFlow();
    scheduleLayoutResync();
    if (!resizeBound) {{
      window.addEventListener('resize', () => {{
        // #region debug-point C:window-resize
        dbgEvent('C', 'app.py:window.resize', '[DEBUG] window resize event', {{
          session: String(STATE.session || ''),
          innerWidth: window.innerWidth || 0,
          innerHeight: window.innerHeight || 0,
          docClientHeight: document.documentElement ? document.documentElement.clientHeight : 0,
          docScrollHeight: document.documentElement ? document.documentElement.scrollHeight : 0,
        }});
        // #endregion
        scheduleLayoutResync();
      }});
      resizeBound = true;
    }}
    window.onpointerup = () => stopHoldCapture();
    window.onblur = () => stopHoldCapture();
    document.onvisibilitychange = () => {{
      if (document.hidden) stopHoldCapture();
    }};
    // Everything above this point has just been rebuilt.  If the interactive
    // guide is open, the spotlight and the click-to-advance binding were
    // built on nodes that no longer exist, so ask it to re-seat itself on
    // the fresh DOM (it re-measures and re-binds its own listeners).
    if (window.tmTour && typeof window.tmTour.sync === 'function') {{
      try {{ window.tmTour.sync(); }} catch (e) {{ console.warn('[render] tour sync failed', e); }}
    }}
    toast(STATE.notice);
  }} catch (tailErr) {{
    // Last-resort safety net: class cards are already in the DOM at this
    // point, so even if the preview/train binding code blows up we still
    // show a usable UI + tell the user *exactly* what exploded.
    console.error('[render] post-classes binding FAILED', tailErr);
    toast(`UI bind error: ${{String(tailErr && tailErr.message || tailErr)}} (see red error boxes for class-level details)`, 12000);
  }}
}}
window.addEventListener('load', () => {{
  // #region debug-point A:window-load
  dbgEvent('A', 'app.py:window.load', '[DEBUG] window load event', {{
    session: String(STATE.session || ''),
    innerWidth: window.innerWidth || 0,
    innerHeight: window.innerHeight || 0,
    href: parentUrl().toString(),
  }});
  // #endregion
  queueFrameHeightSync();
  requestShellLayoutRefresh('image-project-mount:load');
  const isMac = /Mac|iPhone|iPad|iPod/i.test(navigator.platform || '') ||
                (navigator.userAgentData && navigator.userAgentData.platform &&
                 /mac/i.test(navigator.userAgentData.platform));

  // ── Global keyboard router ──────────────────────────────────────────
  // One place decides what "the topmost layer" is, so Escape, the Space
  // capture shortcut and the Cmd/Ctrl shortcuts can never disagree about
  // which overlay is live.
  const isTypingTarget = (el) => {{
    if (!el) return false;
    const tag = String(el.tagName || '').toUpperCase();
    return !!(el.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT');
  }};
  // A field that belongs to a modal does not swallow Escape: pressing it in
  // the ROI editor still closes the editor (Priority 3) rather than merely
  // blurring the box the user is typing in.
  const insideModal = (el) => !!(el && el.closest && el.closest('.modal-backdrop, .tm-settings-modal'));
  const settingsModalOpen = () => {{
    const m = document.getElementById('globalSettingsModal');
    return !!(m && m.classList.contains('open'));
  }};
  const preferencesModalOpen = () => {{
    const m = document.getElementById('preferencesModal');
    return !!(m && m.classList.contains('open'));
  }};
  const dropdownOpen = () => {{
    const a = document.getElementById('menuFileDropdown');
    const b = document.getElementById('menuEditDropdown');
    return !!((a && a.classList.contains('open')) || (b && b.classList.contains('open')) || navMenuOpen);
  }};
  const anyModalOpen = () => !!(
    classPreprocessOpen || settingsModalOpen() || preferencesModalOpen() ||
    ocrCharModalOpen || ocrGlyphModalOpen
  );
  const sourceDrawerOpen = () => !!(openSourceClass && openSourceKind);

  // Escape dismissal stack, highest layer first.  Each modal is closed
  // through its own function so the flags it owns are reset exactly the way
  // a click on its own ✕ would reset them.
  const runEscapeStack = () => {{
    const active = document.activeElement;
    if (isTypingTarget(active) && !insideModal(active)) {{ active.blur(); return; }}
    if (dropdownOpen()) {{ closeNavMenu(); return; }}
    if (classPreprocessOpen) {{ closeClassPreprocessEditor(); return; }}
    if (typeof licenseListOpen !== 'undefined' && licenseListOpen) {{ closeLicenseListModal(); return; }}
    if (settingsModalOpen()) {{ closeGlobalSettings(); return; }}
    if (preferencesModalOpen()) {{ closePreferences(); return; }}
    if (ocrCharModalOpen) {{ closeOcrCharModal(); return; }}
    if (ocrGlyphModalOpen) {{ closeOcrGlyphModal(); return; }}
    if (sourceDrawerOpen()) {{ closeSourcePanel(); return; }}
  }};

  // Space = hands-free capture.  A tap saves one frame; holding the key past
  // SPACE_HOLD_MS starts a hold capture and releasing stops it — so a road
  // sign can be held in one hand while the other taps or holds Space.
  const SPACE_HOLD_MS = 220;
  let spaceDown = false, spaceHoldTimer = null, spaceHoldStarted = false;
  const spaceShortcutEligible = () => (
    !anyModalOpen() && !dropdownOpen() &&
    sourceDrawerOpen() && (openSourceKind === 'webcam' || openSourceKind === 'device')
  );
  const endSpaceShortcut = () => {{
    if (spaceHoldTimer) {{ clearTimeout(spaceHoldTimer); spaceHoldTimer = null; }}
    if (!spaceDown) return;
    spaceDown = false;
    if (spaceHoldStarted) {{
      spaceHoldStarted = false;
      stopHoldCapture();
    }} else {{
      // Too short to be a hold: treat it as a single-frame capture.
      captureSource();
    }}
  }};

  window.addEventListener('keydown', (e) => {{
    const key = String(e.key || '');
    const lower = key.toLowerCase();

    if (key === 'Escape' || key === 'Esc') {{
      if (e.defaultPrevented) return;
      runEscapeStack();
      e.preventDefault();
      e.stopImmediatePropagation();
      return;
    }}

    if (key === ' ' || key === 'Spacebar') {{
      // Space on a focused control (or in a text field) belongs to that
      // control — only the bare workspace drives capture.
      if (isTypingTarget(e.target)) return;
      if (e.target && e.target.closest && e.target.closest('button, a, [role="button"]')) return;
      if (!spaceShortcutEligible()) return;
      e.preventDefault();
      if (e.repeat || spaceDown) return;
      spaceDown = true;
      spaceHoldTimer = setTimeout(async () => {{
        spaceHoldTimer = null;
        if (!spaceDown) return;
        spaceHoldStarted = true;
        await startHoldCapture();
        // Released while /start was still in flight: fold the just-opened
        // hold back down so a fast tap can never leak a running recording.
        if (!spaceDown) {{
          spaceHoldStarted = false;
          await stopHoldCapture();
        }}
      }}, SPACE_HOLD_MS);
      return;
    }}

    const mod = isMac ? (e.metaKey && !e.ctrlKey && !e.altKey) : (e.ctrlKey && !e.metaKey && !e.altKey);
    if (!mod) return;

    if (key === ',') {{
      e.preventDefault(); e.stopImmediatePropagation();
      if (settingsModalOpen()) closeGlobalSettings(); else openGlobalSettings();
    }} else if (lower === 's' && !e.shiftKey) {{
      e.preventDefault(); e.stopImmediatePropagation();
      if (classPreprocessOpen) {{
        if (classPreprocessClass && selectedClassSampleFilename(classPreprocessClass) && currentClassPreprocessDirty()) {{
          saveCurrentClassPreprocessSample(false);
        }}
      }} else {{
        saveProjectQuick();
      }}
    }} else if (lower === 's' && e.shiftKey) {{
      e.preventDefault(); e.stopImmediatePropagation();
      saveProjectAs();
    }} else if (key === 'Enter') {{
      e.preventDefault(); e.stopImmediatePropagation();
      if (STATE.train_enabled && !trainInFlight) startTrain();
    }} else if (lower === 'o') {{
      e.preventDefault(); e.stopImmediatePropagation();
      openProject();
    }} else if (lower === 'n') {{
      e.preventDefault(); e.stopImmediatePropagation();
      newProject();
    }} else if (lower === 'e') {{
      e.preventDefault(); e.stopImmediatePropagation();
      exportModel();
    }}
  }}, true);

  window.addEventListener('keyup', (e) => {{
    const key = String(e.key || '');
    if (key === ' ' || key === 'Spacebar') endSpaceShortcut();
  }});
}});
window.addEventListener('pagehide', () => {{
  cleanupWorkspaceFrameBeforeNavigate('pagehide');
}});
window.addEventListener('beforeunload', () => {{
  cleanupWorkspaceFrameBeforeNavigate('beforeunload');
}});
window.addEventListener('orientationchange', () => {{
  // #region debug-point C:orientation-change
  dbgEvent('C', 'app.py:orientationchange', '[DEBUG] orientationchange event', {{
    session: String(STATE.session || ''),
    innerWidth: window.innerWidth || 0,
    innerHeight: window.innerHeight || 0,
  }});
  // #endregion
  queueFrameHeightSync();
  logScrollLayers('window-load');
}});
if (window.__tmStageMark) window.__tmStageMark('before-render');
render();
if (window.__tmStageMark) window.__tmStageMark('after-render');
logScrollLayers('after-render');
requestShellLayoutRefresh('image-project-mount');
mountReflowTimers.push(window.setTimeout(() => {{
  if (window.__tmNavigatingAway) return;
  requestShellLayoutRefresh('image-project-mount:t120');
  logScrollLayers('mount-t120');
}}, 120));
mountReflowTimers.push(window.setTimeout(() => {{
  if (window.__tmNavigatingAway) return;
  requestShellLayoutRefresh('image-project-mount:t360');
  logScrollLayers('mount-t360');
}}, 360));
</script>

<script>
/* ── Built-in Interactive Coach-Mark Tour (Image Project) ─────────────
   A zero-dependency, state-aware onboarding guide: no CDN, no library,
   so the app keeps working fully offline.  Steps may drive the *real*
   UI through onEnter / onLeave hooks — that is how the spotlight can
   land on live controls inside the Sample Worktable, the Advanced
   hyper-parameter panel and the OOD settings group. */
(function () {{
  'use strict';

  /* One seen/autoskip pair PER PROJECT TYPE.  The three curricula document
     three genuinely different workflows, so dismissing the Image tour must
     not silence the OCR one the first time that project is opened — and
     re-opening a workspace the user has already toured should not re-teach
     it.  selectCurriculum() rewrites both keys to name the active type. */
  var SEEN_KEY = 'tm_tour_seen_image_v1';
  var AUTOSKIP_KEY = 'tm_tour_autoskip_image_v1';
  var WELCOME_DELAY_MS = 1200;
  var TOUR_TYPE = 'image';
  var PHASES = [];
  var STEPS = [];

  function byId(id) {{ return document.getElementById(id); }}

  function isVisibleNode(el) {{
    if (!el || !el.getClientRects().length) return false;
    var st = window.getComputedStyle(el);
    return st.display !== 'none' && st.visibility !== 'hidden' && Number(st.opacity) !== 0;
  }}

  function firstVisible(selectors) {{
    for (var i = 0; i < selectors.length; i++) {{
      if (!selectors[i]) continue;
      var el = null;
      try {{ el = document.querySelector(selectors[i]); }} catch (err) {{ el = null; }}
      if (isVisibleNode(el)) return el;
    }}
    return null;
  }}

  function runHook(fn, label) {{
    if (typeof fn !== 'function') return;
    try {{ fn(); }} catch (err) {{ console.warn('[tour] ' + label + ' failed', err); }}
  }}

  function projectClasses() {{
    try {{ return Array.isArray(STATE.classes) ? STATE.classes.filter(Boolean) : []; }} catch (err) {{ return []; }}
  }}

  /* The workspace script declares its state with top-level `let`, which
     never lands on `window` — a sibling classic script shares the same
     global lexical scope, so the hooks below reach those bindings by
     bare name.  Each access is guarded with `typeof` so the tour still
     degrades gracefully if a project type renders without them. */
  function setAdvancedPanel(wantOpen) {{
    if (typeof trainAdvancedOpen === 'undefined') return;
    trainAdvancedOpen = !!wantOpen;
    if (typeof renderAdvancedPanel === 'function') renderAdvancedPanel();
  }}

  function openWorktable() {{
    if (typeof openClassPreprocessEditor !== 'function') return;
    if (typeof classPreprocessOpen !== 'undefined' && classPreprocessOpen) return;
    var names = projectClasses();
    if (!names.length) return;
    try {{ openClassPreprocessEditor(names[0]); }} catch (err) {{ console.warn('[tour] open worktable failed', err); }}
  }}

  function closeWorktable() {{
    if (typeof classPreprocessOpen === 'undefined' || !classPreprocessOpen) return;
    if (typeof closeClassPreprocessEditor !== 'function') return;
    try {{ closeClassPreprocessEditor(true); }} catch (err) {{ console.warn('[tour] close worktable failed', err); }}
  }}

  function closeSettingsModal() {{
    if (typeof closeGlobalSettings !== 'function') return;
    try {{ closeGlobalSettings(); }} catch (err) {{ console.warn('[tour] close settings failed', err); }}
  }}

  /* Idempotent, and only ever used as a FALLBACK: the curriculum's own Open
     checkpoint is the user pressing ⚙ Settings.  This is what puts the
     modal back when the guide is entered from somewhere else — a chapter
     pill, `Skip step →`, or a step resumed out of order — so the inner
     checkpoints still have a visible screen to land on. */
  function openSettingsModal() {{
    var modal = byId('globalSettingsModal');
    if (modal && modal.classList.contains('open')) return;
    if (typeof openGlobalSettings !== 'function') return;
    try {{ openGlobalSettings(); }} catch (err) {{ console.warn('[tour] open settings failed', err); }}
  }}

  /* Folding the sampling drawer away.  The drawer's own ▴ Close is the
     checkpoint; this only guarantees the workspace is drawer-free when a
     later step is reached by skipping past that checkpoint. */
  function closeSourceDrawer() {{
    if (typeof openSourceClass === 'undefined' || !openSourceClass) return;
    if (typeof closeSourcePanel !== 'function') return;
    try {{
      var pending = closeSourcePanel();
      if (pending && typeof pending.catch === 'function') pending.catch(function () {{}});
    }} catch (err) {{ console.warn('[tour] close drawer failed', err); }}
  }}

  /* ---- which screen is on screen right now? --------------------------
     Every Open / Inner / Close checkpoint asks one of these.  A step
     whose screen is shut falls back to the button that OPENS it rather
     than spotlighting a control the modal is covering, which is what
     keeps a chapter jump or a `Skip step →` from stranding the user on a
     slide with nothing to click. */
  function settingsOpen() {{
    var modal = byId('globalSettingsModal');
    return !!(modal && modal.classList.contains('open'));
  }}

  function drawerOpen() {{
    return typeof openSourceClass !== 'undefined' && !!openSourceClass;
  }}

  function worktableOpen() {{
    return typeof classPreprocessOpen !== 'undefined' && !!classPreprocessOpen;
  }}

  function advancedOpen() {{
    return typeof trainAdvancedOpen !== 'undefined' && !!trainAdvancedOpen;
  }}

  function charModalOpen() {{
    return typeof ocrCharModalOpen !== 'undefined' && !!ocrCharModalOpen;
  }}

  function editMenuOpen() {{
    var d = byId('menuEditDropdown');
    return !!(d && d.classList.contains('open'));
  }}

  function fileMenuOpen() {{
    var d = byId('menuFileDropdown');
    return !!(d && d.classList.contains('open'));
  }}

  function preferencesOpen() {{
    var m = byId('preferencesModal');
    return !!(m && m.classList.contains('open'));
  }}

  function glyphModalOpen() {{
    return typeof ocrGlyphModalOpen !== 'undefined' && !!ocrGlyphModalOpen;
  }}

  function navMenuOpenNow() {{
    return typeof navMenuOpen !== 'undefined' && !!navMenuOpen;
  }}

  /* Is any workspace layer sitting above the tour card?  Escape belongs to
     the topmost layer, and while a menu, modal or the source drawer is
     open that is NOT the tour: handing the key straight to the tour used to
     kill the guide instead of closing the thing the user was looking at. */
  function workspaceLayerOpen() {{
    try {{
      if (fileMenuOpen() || editMenuOpen() || navMenuOpenNow()) return true;
      if (worktableOpen() || settingsOpen() || preferencesOpen()) return true;
      if (typeof licenseListOpen !== 'undefined' && licenseListOpen) return true;
      if (charModalOpen() || glyphModalOpen()) return true;
      if (drawerOpen()) return true;
    }} catch (err) {{ return false; }}
    return false;
  }}

  /* EMNIST mixing lives inside the Settings modal's *collapsed* Advanced
     disclosure, so opening the modal alone still leaves that group at
     `display:none`.  A step that documents it resumes onto the group, so
     this hook unfolds the disclosure first — the same thing the user's
     own press of the ▸ toggle does. */
  function unfoldSettingsAdvanced() {{
    var toggle = byId('settingsAdvancedToggle');
    var body = byId('settingsAdvancedBody');
    if (!toggle || !body) return;
    if (toggle.getAttribute('aria-expanded') === 'true') return;
    toggle.setAttribute('aria-expanded', 'true');
    body.style.display = 'flex';
  }}

  /* The targets that live behind that collapsed disclosure.  A step that
     documents any of them has to unfold it, or `firstVisible()` finds
     nothing and the ring has no control left to sit on. */
  var ADVANCED_DISCLOSURE_TARGETS = ['#settingsAdvancedToggle', '#settingsEmnistGroup',
    '#settingsAutoSeedEmnist', '#settingsSeedEmnistNowBtn'];

  function ensureStepDisclosure(step) {{
    if (!step || !settingsOpen()) return;
    var sels = [step.target].concat(step.fallback || []);
    if (step.advanceTarget) sels.push(step.advanceTarget);
    (step.advanceAlso || []).forEach(function (s) {{ sels.push(s); }});
    for (var i = 0; i < sels.length; i++) {{
      if (ADVANCED_DISCLOSURE_TARGETS.indexOf(sels[i]) >= 0) {{ unfoldSettingsAdvanced(); return; }}
    }}
  }}

  /* ═══ Curriculum 1 · Image Project (20 steps) ═══════════════════════
     The original comprehensive walkthrough: the three-column workspace,
     the capture sources, ROI + brightness thresholds, the 9 CNN
     hyperparameters, live preview and the 3-layer OOD gate, then the
     5-file MCU export. */
  var IMAGE_STEPS = [
    {{
      chapter: 1,
      title: 'Your Image Project, End to End',
      /* Step 0 is the big picture rather than a lesson, so it carries
         `overview` and a `stages` list instead of a spotlight: the card
         floats centred over a dimmed workspace (target: null — cutting
         the whole .layout box out left only a hairline rim dimmed, which
         read as a bug, not a welcome) and its one button starts the
         walkthrough. */
      target: null,
      fallback: [],
      placement: 'auto',
      overview: true,
      stages: [
        {{ t: 'Collect Samples', d: 'Add your classes, then capture labelled images from the webcam, a folder or the ESP32-P4 serial link.' }},
        {{ t: 'Clean &amp; Crop', d: 'Open the Sample Worktable to set the ROI box and the Dark / Lum thresholds for each class.' }},
        {{ t: 'Train', d: 'Set the CNN hyperparameters, train the int8 model and read the per-class accuracy.' }},
        {{ t: 'Test Live', d: 'Watch the preview verdict, and the 3-layer OOD gate throwing out frames that are not your classes.' }},
        {{ t: 'Export &amp; Deploy', d: 'Build the 5-file C/C++ package that the ESP32-P4 flashes.' }}
      ],
      body: '<p>This workspace builds a tiny embedded image classifier and deploys it to your <b>ESP32-P4</b>. Work left to right, exactly like the lab manual.</p>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">💡</span><span>The guide <b>points</b> at the real control for every step — and the moment you start working it shrinks into a floating orb in the corner, so it never covers what you are using.</span></div>' +
        '<p class="tm-tour-muted">The whole trip, in five stages:</p>'
    }},
    {{
      chapter: 1,
      title: 'Save, Open &amp; Export from the File Menu',
      target: '#menuFileBtn',
      fallback: ['.topnav'],
      placement: 'bottom',
      advanceTarget: '#menuFileBtn',
      checkpoint: 'Click "File ▾" to see the project commands',
      body: '<p><b>File ▾</b> holds everything that acts on the project as a whole:</p>' +
        '<ul>' +
        '<li><b>New project</b> (<kbd>⌘N</kbd>) — start a fresh, empty workspace.</li>' +
        '<li><b>Open project</b> (<kbd>⌘O</kbd>) — load a saved <code>.tmproj</code> archive, including its samples and settings.</li>' +
        '<li><b>Save</b> (<kbd>⌘S</kbd>) and <b>Save as</b> (<kbd>⇧⌘S</kbd>) — write the project to disk under a new name.</li>' +
        '<li><b>Export dataset</b> — dump the raw sample images as files, for a backup or an offline experiment.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">💾</span><span>Save before every training run. A <code>.tmproj</code> keeps the samples, the ROI boxes and the brightness thresholds — not just the class names.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Preferences &amp; Return Home',
      target: '#navPreferences',
      fallback: ['#navReturn', '.topnav'],
      placement: 'bottom',
      advanceTarget: '#navPreferences',
      checkpoint: 'Click "Preferences" to see the display settings',
      body: '<ul>' +
        '<li><b>Preferences</b> opens a small modal whose one setting, <b>Max sample thumbnails</b> (<code>#prefMaxSamples</code>), caps how many sample thumbnails each class card caches and draws — 12 by default.</li>' +
        '<li>Raise it while you are eyeballing a set for bad captures; lower it on a slow machine or a project with thousands of samples.</li>' +
        '<li><b>← Return Home</b> leaves the workspace for the start page. It is navigation, not a reset: the project on disk is untouched.</li>' +
        '</ul>'
    }},
    {{
      chapter: 1,
      title: 'Open Input Settings',
      target: '#navSettings',
      fallback: ['.topnav'],
      placement: 'bottom',
      advanceTarget: '#navSettings',
      checkpoint: 'Click "⚙ Settings" in the top bar to open Input Settings',
      body: '<p><b>⚙ Settings</b> is the single source of truth for the capture hardware — the webcam, the ESP32-P4 serial link and the frame geometry — applied to every class card at once.</p>' +
        '<ul>' +
        '<li>MCU defaults: frame <b>96×96</b>, <b>Grayscale</b>, baud <b>115200</b> (or <b>921600</b>), sync header <b>AA 55 AA</b>.</li>' +
        '<li>The next three checkpoints walk the modal from top to bottom, then close it again.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">⚠️</span><span>Close the Arduino IDE <b>Serial Monitor</b> before connecting — only one program can own the ESP32-P4 serial port.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Camera &amp; MCU Serial Fields',
      target: '#settingsCameraGroup',
      fallback: ['#settingsMcuGroup', '#settingsWebcam', '#globalSettingsModal'],
      placement: 'left',
      /* Inside the modal now.  The guard is only for the ways in that skip
         the Open checkpoint — a chapter pill, or `Skip step →` — where the
         modal may still be shut; there the step re-doors onto ⚙ instead of
         spotlighting a group the modal is covering. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      checkpoint: 'Click the highlighted ⚙ Settings button to open Input Settings',
      body: '<ul>' +
        '<li><b>Webcam</b> — the capture device; <b>Capture FPS</b> throttles the live preview stream.</li>' +
        '<li><b>Serial Device (port)</b> — the ESP32-P4 USB device. Re-scan after plugging it in.</li>' +
        '<li><b>Frame side</b> / <b>Channels</b> must match the firmware — <b>96×96</b> and <b>Grayscale</b> by default.</li>' +
        '<li><b>Baud rate</b>: <b>115200</b> for reliability, <b>921600</b> for faster frame streaming.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>Deep dive · what each serial field does</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table"><tr><th>Field</th><th>Meaning</th></tr>' +
        '<tr><td><code>Port</code></td><td>USB serial device of the ESP32-P4 (re-scan after plugging in).</td></tr>' +
        '<tr><td><code>Frame side</code></td><td>Square frame the device streams; must match its firmware.</td></tr>' +
        '<tr><td><code>Channels</code></td><td><b>1</b> = grayscale (default), <b>3</b> = RGB colour.</td></tr>' +
        '<tr><td><code>Baud</code></td><td>115200 for reliability, 921600 for faster frame streaming.</td></tr>' +
        '<tr><td><code>Sync header</code></td><td>Byte pattern marking the start of a frame (AA 55 AA) — under <b>▸ Advanced Settings</b>.</td></tr>' +
        '</table></div></details>'
    }},
    {{
      chapter: 1,
      title: 'Preview Source &amp; the OOD Gates',
      target: '#settingsOodGroup',
      fallback: ['#settingsPreviewSource', '#globalSettingsModal'],
      placement: 'left',
      /* Still inside the modal.  A chapter pill can land here with the modal
         shut, so the step re-doors onto ⚙ exactly like the group above. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      advanceTarget: '#settingsPreviewSource',
      checkpoint: 'Pick the preview source, then check the four OOD gates',
      body: '<ul>' +
        '<li><b>Preview input source</b> (<code>#settingsPreviewSource</code>) decides what the preview pane reads: <b>Webcam</b>, <b>Device (MCU)</b> over serial, or a <b>Local Image File</b>. The preview card draws its own source row from this value.</li>' +
        '<li><b>OOD sign min %</b> / <b>max %</b> — Layer 1, the share of dark target pixels. Below the min (empty scene) or above the max (covered lens) the frame is rejected <b>before inference</b>.</li>' +
        '<li><b>OOD max prob</b> — Layer 2, the winning class must clear this floor.</li>' +
        '<li><b>OOD entropy max</b> — Layer 3, the probabilities must not be spread too evenly, i.e. the model must not be guessing.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>🎛️ Tuning cheat-sheet</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table"><tr><th>Symptom</th><th>Fix</th></tr>' +
        '<tr><td>Real signs are missed</td><td><b>Loosen</b>: lower <code>sign min %</code>, raise <code>sign max %</code>, lower <code>max prob</code>, raise <code>entropy max</code>.</td></tr>' +
        '<tr><td>Empty wall is classified</td><td><b>Tighten</b>: raise <code>sign min %</code>, lower <code>sign max %</code>, raise <code>max prob</code>, lower <code>entropy max</code>.</td></tr>' +
        '</table></div></details>' +
        '<p class="tm-tour-muted">Defaults: sign <b>0.3 % – 70 %</b>, max prob <b>≥ 0.60</b>, entropy <b>≤ 0.70</b>. These four <code>constexpr</code> values travel into the exported <code>model_settings.h</code>.</p>'
    }},
    {{
      chapter: 1,
      title: 'Leave Input Settings',
      target: '#settingsCancel',
      fallback: ['#settingsSave', '.tm-settings-card'],
      placement: 'left',
      /* Inside the modal.  If the guide is entered from a chapter pill or
         `Skip step →` the modal may be shut, so the step re-doors onto ⚙
         rather than spotlighting a Close button nobody can see. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      advanceTarget: '#settingsCancel',
      /* Save applies the edits too, so it finishes this step just as
         legitimately as Cancel — both used to be one unbound button. */
      advanceAlso: ['#settingsSave'],
      onLeave: closeSettingsModal,
      checkpoint: 'Click "Cancel" (or "Save changes") to exit Settings and return to the workspace',
      body: '<p><b>Cancel</b> closes the modal without applying the edits on screen; <b>Save changes</b> applies them to every class card at once and writes the OOD gates into the host preview.</p>' +
        '<p class="tm-tour-muted">The modal can also be dismissed with <kbd>Esc</kbd>, but the tour follows you out either way.</p>'
    }},
    {{
      chapter: 2,
      title: 'Organize &amp; Order Your Classes',
      target: '.class-card',
      fallback: ['.col-classes'],
      placement: 'right',
      body: '<p>Every icon on a class card header does one thing:</p>' +
        '<ul>' +
        '<li><b>⠿</b> (drag handle) — drag the card to reorder. The top-to-bottom order <b>is</b> the class index (<b>0, 1, 2…</b>) the ESP32-P4 sends to the ESP32-S3.</li>' +
        '<li><b>✏️</b> (<code>.tm-rename-btn</code>) — rename the class, e.g. <code>LEFT</code>, <code>STOP</code>, <code>RIGHT</code>.</li>' +
        '<li><b>🧹</b> (<code>.clear-samples</code>) — clear every sample in that class without deleting the class itself.</li>' +
        '<li><b>🗑</b> (<code>.tm-delete-class</code>) — delete the class, samples and all. This one cannot be undone.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🔧</span><span>Reordering changes the on-device index mapping — re-export the model after you reorder.</span></div>'
    }},
    {{
      chapter: 2,
      title: 'Add a Class',
      target: '.addclass',
      fallback: ['.col-classes'],
      placement: 'right',
      advanceTarget: '.addclass',
      checkpoint: 'Click "+ Add a class" to add another label',
      body: '<p><b>+ Add a class</b> at the foot of the column appends a new label with an empty sample set. Rename it straight away, then capture its samples.</p>' +
        '<ul>' +
        '<li>Two classes is the minimum a classifier can learn; most projects want three or more.</li>' +
        '<li>A brand-new class arrives with zero samples, so <b>Train</b> stays disabled until every class has some.</li>' +
        '</ul>'
    }},
    {{
      chapter: 2,
      title: 'Pick a Sample Source',
      target: '.tm-action-toolbar',
      fallback: ['.class-card'],
      placement: 'right',
      advanceTarget: '.tm-action-toolbar button.sample',
      checkpoint: 'Click "Webcam" to open the camera drawer',
      body: '<p>Three sources feed a class. Pick the one that matches your hardware:</p>' +
        '<ul>' +
        '<li><b>Webcam</b> — the laptop or an external USB camera.</li>' +
        '<li><b>Upload</b> — batch-import existing <code>.png</code> / <code>.jpg</code> images from disk.</li>' +
        '<li><b>Device</b> — stream live frames from the ESP32-P4 camera over USB serial.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">👆</span><span>Clicking the tile you are already on folds the drawer back up — these are <b>toggles</b>, not one-way doors.</span></div>'
    }},
    {{
      chapter: 2,
      title: 'Capture Labeled Image Samples',
      live: 'samples',
      target: '[id^="sourceCapture-"]',
      fallback: ['.source-actions', '.source-panel'],
      placement: 'right',
      /* The capture controls only exist while the drawer is open.  Coming
         in from the Open checkpoint they are already there; coming in from
         a chapter pill the step re-doors onto the tile that opens them. */
      phaseIf: function () {{ return !drawerOpen(); }},
      phaseTarget: '.tm-action-toolbar button.sample',
      phaseName: 'the camera drawer',
      advanceTarget: '[id^="sourceCapture-"]',
      /* Holding the button is the burst gesture, and Upload swaps the pair
         for a file picker: all three finish the same checkpoint, so all
         three are bound. */
      advanceAlso: ['[id^="sourceHold-"]', '[id^="uploadPick-"]'],
      checkpoint: 'Click "Capture" for one frame, or hold "Hold to Capture" / <kbd>Space</kbd> to burst-capture',
      body: '<p>The drawer opens under the class card. Its preview is <b>live</b> for Webcam and Device, and empty for Upload.</p>' +
        '<ul>' +
        '<li><b>Capture</b> grabs a single frame and files it under this class.</li>' +
        '<li><b>Hold to Capture</b> — hold the button (or hold <kbd>Space</kbd>) to <b>burst-capture</b> while you move the sign around.</li>' +
        '<li><b>Upload</b> instead offers <b>Choose images from your files</b>.</li>' +
        '<li>A <b>⚙</b> chip in the drawer head jumps straight to this class&rsquo;s settings without leaving the drawer.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🎯</span><span>Aim for <b>50–100+ samples per class</b>, varying distance, angle and lighting — the model only learns what it sees.</span></div>'
    }},
    {{
      chapter: 2,
      title: 'Collapse the Camera Drawer',
      target: '[id^="sourceClose-"]',
      fallback: ['.source-close-pill', '.source-panel'],
      placement: 'right',
      phaseIf: function () {{ return !drawerOpen(); }},
      phaseTarget: '.tm-action-toolbar button.sample',
      phaseName: 'the camera drawer',
      advanceTarget: '[id^="sourceClose-"]',
      onLeave: closeSourceDrawer,
      checkpoint: 'Click "Close" (▴) to collapse the camera drawer',
      body: '<p>The <b>▴ Close</b> pill folds the drawer back into the class card; the tile stays highlighted until you collapse it, so you always know which class you were shooting.</p>' +
        '<p class="tm-tour-muted">Press <kbd>Esc</kbd> to collapse the drawer too — the keyboard router gives an open drawer priority over closing the tour.</p>'
    }},
    {{
      chapter: 3,
      title: 'Open the Sample Worktable',
      target: '.class-preprocess-chip',
      fallback: ['.class-card'],
      placement: 'right',
      advanceTarget: '.class-preprocess-chip',
      checkpoint: 'Click "Edit" to open the Sample Worktable',
      body: '<p>Real backgrounds carry <b>extraneous variables</b> — shadows, hands, floor tiles. Feeding those to the CNN teaches it to recognise the room instead of the sign.</p>' +
        '<ul>' +
        '<li><b>Edit</b> on a class card opens the <b>Sample Worktable</b> — the per-sample inspector.</li>' +
        '<li>There you crop the <b>Region of Interest (ROI)</b> and mask the background with brightness thresholds.</li>' +
        '</ul>' +
        '<p>The next three checkpoints walk the worktable and then close it again.</p>'
    }},
    {{
      chapter: 3,
      title: 'Auto ROI vs. Manual ROI Cropping',
      live: 'thresholds',
      target: '.class-preprocess-mode-row',
      fallback: ['#classPreprocessMode', '.class-preprocess-shell'],
      placement: 'left',
      /* Inside the worktable now.  The guard only covers the ways in that
         skip the Open checkpoint — a chapter pill, or `Skip step →`. */
      phaseIf: function () {{ return !worktableOpen(); }},
      phaseTarget: '.class-preprocess-chip',
      phaseName: 'the Sample Worktable',
      onEnter: openWorktable,
      advanceTarget: '#classPreprocessMode',
      checkpoint: 'Keep "Auto (find sign)", or pick "Manual ROI" and drag the crop box',
      body: '<ul>' +
        '<li><b>Original Sample (left)</b> — the raw frame with the green / blue <b>ROI</b> crop box drawn over it.</li>' +
        '<li><b>Processed Preview (right)</b> — the exact filtered and resized <code>Image Size × Image Size</code> tensor the network will be handed.</li>' +
        '</ul>' +
        '<p><b>Preprocess Mode</b>:</p>' +
        '<ul>' +
        '<li><code>Auto (find sign)</code> — searches for the darkest target region and crops it. The best default.</li>' +
        '<li><code>Manual ROI</code> — drag on the original image to lock a fixed crop box (<code>ROI X1, Y1, X2, Y2</code>). Use it when the sign is always in the same spot.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">⏱️</span><span>Preprocessing runs bit-identically on the ESP32-P4 and in this app, so what you see here is what the MCU sees.</span></div>'
    }},
    {{
      chapter: 3,
      title: 'Dark Thr &amp; Lum Thr Noise Filtering',
      live: 'thresholds',
      target: '#classPreprocessDark',
      fallback: ['.class-preprocess-right', '#classPreprocessModal'],
      placement: 'left',
      phaseIf: function () {{ return !worktableOpen(); }},
      phaseTarget: '.class-preprocess-chip',
      phaseName: 'the Sample Worktable',
      onEnter: openWorktable,
      advanceTarget: '#classPreprocessDark',
      /* Save Sample finishes this step too, and it is not a descendant of
         the slider being spotlit, so it needs its own binding.  Closing the
         worktable is now a checkpoint of its own, right after this one. */
      advanceAlso: ['#classPreprocessSave'],
      checkpoint: 'Drag "Dark Thr" / "Lum Thr" until the preview is clean, then click "Save Sample"',
      body: '<ul>' +
        '<li><b>Dark Thr</b> — pixels darker than the threshold become <b>white (255)</b>, i.e. shadow.</li>' +
        '<li><b>Lum Thr</b> — pixels brighter than the threshold become <b>white (255)</b>, i.e. bright background.</li>' +
        '</ul>' +
        '<p>Together they isolate the dark sign body from everything else. Tune both until the right-hand preview is mostly white with a crisp black glyph.</p>' +
        '<ul>' +
        '<li>The <b>thumbnail strip</b> (<code>.class-preprocess-thumb</code>) jumps between this class&rsquo;s samples.</li>' +
        '<li><b>Delete Sample</b> throws the current frame away for good — keep it for blur, glare or a hand in the way.</li>' +
        '<li><b>Save Sample</b> writes the crop and the thresholds onto this sample, then returns to the workspace.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>Deep dive · worktable shortcuts</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table">' +
        '<tr><td><kbd>F1</kbd></td><td>Switch to Auto ROI</td></tr>' +
        '<tr><td><kbd>F2</kbd></td><td>Switch to Manual ROI</td></tr>' +
        '<tr><td><kbd>←</kbd> <kbd>→</kbd> <kbd>↑</kbd> <kbd>↓</kbd></td><td>Previous / next sample</td></tr>' +
        '<tr><td><kbd>Shift</kbd> + arrows</td><td>Nudge the ROI box by one pixel</td></tr>' +
        '<tr><td><kbd>D</kbd></td><td>Delete the current sample</td></tr>' +
        '<tr><td><kbd>S</kbd></td><td>Save the sample</td></tr>' +
        '</table></div></details>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">💾</span><span>Always click <b>Save Sample</b> (or press <kbd>S</kbd>) after tuning — otherwise your thresholds are discarded when the worktable closes.</span></div>'
    }},
    {{
      chapter: 3,
      title: 'Close the Sample Worktable',
      target: '#classPreprocessCloseTop',
      fallback: ['#classPreprocessCancel', '.class-preprocess-actions'],
      placement: 'left',
      /* Still inside the worktable: Save Sample finishes the previous step
         but leaves the worktable open, so the exit is its own checkpoint
         rather than something the tour does behind the user's back. */
      phaseIf: function () {{ return !worktableOpen(); }},
      phaseTarget: '.class-preprocess-chip',
      phaseName: 'the Sample Worktable',
      advanceTarget: '#classPreprocessCloseTop',
      /* The footer's Close leaves the worktable too, and it is not a
         descendant of the ✕ in the worktable's head. */
      advanceAlso: ['#classPreprocessCancel'],
      onLeave: closeWorktable,
      checkpoint: 'Click "Close" (×) to exit the Sample Worktable',
      body: '<p><b>Close</b> (<b>×</b> in the worktable head, or the <b>Close</b> button in the footer) returns to the three-column workspace. Everything you saved stays on its class card.</p>' +
        '<ul>' +
        '<li><b>Save Sample</b> writes the crop and the brightness thresholds onto that one sample.</li>' +
        '<li><b>Close</b> is safe to press at any time — it only discards edits you have not saved.</li>' +
        '<li><kbd>Esc</kbd> closes it too, and so does clicking the dimmed backdrop.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">➡️</span><span>With every class filtered, the middle column is next: train the embedded model.</span></div>'
    }},
    {{
      chapter: 4,
      title: 'Reveal the Advanced Training Options',
      target: '.train-card',
      fallback: ['.col-train'],
      placement: 'left',
      /* The Advanced panel starts collapsed on a fresh workspace.  Rather
         than swinging it open behind the user's back, the step asks them to
         press Advanced — the same button they will use — and only then
         lands on the rows inside it. */
      phaseIf: function () {{ return !advancedOpen(); }},
      phaseTarget: '#advBtn',
      phaseName: 'the <b>Advanced</b> training options',
      advanceTarget: '#advBtn',
      checkpoint: 'Click "Advanced" to unfold the hyperparameters',
      body: '<p>The middle column owns the training run. Two controls matter here:</p>' +
        '<ul>' +
        '<li><b>Advanced</b> unfolds the hyperparameters. It starts collapsed because the defaults are already tuned for a small embedded model.</li>' +
        '<li><b>Train Embedded Model</b> starts the run, and stays disabled until every class has samples.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🧠</span><span>Training happens on this machine and streams per-epoch progress into the card, so you can watch accuracy climb.</span></div>'
    }},
    {{
      chapter: 4,
      title: 'Configure Hyperparameters',
      target: '#trainAdvPanel',
      fallback: ['.train-card', '.col-train'],
      placement: 'left',
      /* The rows only exist while the panel is unfolded.  Coming in from
         the Open checkpoint they are already there; coming in from a
         chapter pill the step re-doors onto Advanced first. */
      phaseIf: function () {{ return !advancedOpen(); }},
      phaseTarget: '#advBtn',
      phaseName: 'the <b>Advanced</b> training options',
      advanceTarget: '#advUseRec',
      /* Reset restores the defaults, and the inline rows are the numbers
         the recommended button fills in: touching any of them settles this
         checkpoint too. */
      advanceAlso: ['#advReset', '#advInkPolarity', '#advEpochsInline', '#advBatchInline', '#advLrInline'],
      checkpoint: 'Tune the rows, or click "💡 Use recommended" to fill them in',
      body: '<ul>' +
        '<li>Click <b>💡 Use recommended</b> to auto-calculate <code>Epochs</code> and <code>Batch Size</code> for roughly <b>800 gradient updates</b>; <b>Reset</b> puts the shipped defaults back.</li>' +
        '<li><b>Speed vs. accuracy:</b> larger <code>Image Size</code> (96 / 160 / 384), <code>Conv1/Conv2 Filters</code> and <code>Dense Units</code> raise capacity but <b>slow on-device inference</b> and grow the flash footprint.</li>' +
        '<li>Keep <code>Image Size</code> equal to the MCU <code>IMG_SIZE</code>, or the exported model will reject every frame.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>📖 View all 9 hyperparameters</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table"><tr><th>Parameter</th><th>What it controls</th></tr>' +
        '<tr><td><code>Image Size</code></td><td>Square input resolution; must equal the MCU <code>IMG_SIZE</code>.</td></tr>' +
        '<tr><td><code>Color Mode</code></td><td>1 channel is roughly 3× cheaper; license plates need no colour.</td></tr>' +
        '<tr><td><code>Batch Size</code></td><td>How many samples make up one gradient step.</td></tr>' +
        '<tr><td><code>Epochs</code></td><td>Full passes over the training data.</td></tr>' +
        '<tr><td><code>Validation Split</code></td><td>Fraction held out to produce an honest accuracy score.</td></tr>' +
        '<tr><td><code>Learning Rate</code></td><td>Size of each weight update.</td></tr>' +
        '<tr><td><code>Conv1 Filters</code></td><td>Feature detectors in the first convolution layer.</td></tr>' +
        '<tr><td><code>Conv2 Filters</code></td><td>Feature detectors in the second convolution layer.</td></tr>' +
        '<tr><td><code>Dense Units</code></td><td>Width of the final classifier head.</td></tr>' +
        '</table></div></details>'
    }},
    {{
      chapter: 4,
      title: 'Train the Embedded Model',
      target: '#trainBtn',
      fallback: ['.train-card', '.col-train'],
      placement: 'left',
      advanceTarget: '#trainBtn',
      live: 'train',
      checkpoint: 'Click "Train Embedded Model" (or press ⌘/Ctrl + Enter) to start',
      body: '<p>Press <b>Train Embedded Model</b> (<kbd>⌘</kbd><kbd>Enter</kbd> / <kbd>Ctrl</kbd><kbd>Enter</kbd>). The run streams per-epoch progress into the card.</p>' +
        '<ul>' +
        '<li>When it finishes, the status pill (<code>#trainStatus</code>) turns into <b>Model Trained · tap for accuracy</b> — click it for the per-class score.</li>' +
        '<li>The pill is disabled until <b>every</b> class has at least one sample; it is the app&rsquo;s way of saying the training set is not ready.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">📊</span><span>If one class lags far behind the others, it usually needs more samples or a tighter ROI — go back to <b>Edit</b> before re-training.</span></div>'
    }},
    {{
      chapter: 5,
      title: 'Real-Time Preview (Input · ROI · Orig)',
      target: '.preview-card',
      fallback: ['.col-preview'],
      placement: 'left',
      body: '<p>The preview card runs the <b>same int8</b> graph on the host CPU that the MCU will run, so accuracy here predicts on-device accuracy.</p>' +
        '<ul>' +
        '<li><b>Input</b> — streaming on/off. While it is off the pane holds the last frame.</li>' +
        '<li><b>ROI</b> — draw the crop rectangle over the scene.</li>' +
        '<li><b>Orig</b> — show the full masked frame. With both on, the pane shows the processed crop the network is handed.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">📁</span><span>The row underneath picks where the pane reads from — a <b>Webcam</b>, the <b>Device</b> port plus <b>Refresh</b>, or <b>Upload Image…</b> — independently of the capture drawer.</span></div>'
    }},
    {{
      chapter: 5,
      title: 'Live Dark / Lum Sliders &amp; the Confidence Bars',
      target: '.preview-card',
      fallback: ['.col-preview'],
      placement: 'left',
      body: '<ul>' +
        '<li>The live <b>Dark</b> / <b>Lum</b> sliders under the image re-tune the brightness mask while you watch — they should match the thresholds you saved in the worktable.</li>' +
        '<li>The <b>confidence bars</b> underneath are the per-class scores for the frame on screen; the empty-space guard reads <code>No Sign</code> when a gate rejects it.</li>' +
        '</ul>' +
        '<p class="tm-tour-muted">Sanity-check the preview before exporting: a mask that looks wrong here will look exactly as wrong on the device.</p>'
    }},
    {{
      chapter: 6,
      title: 'Export the Quantized INT8 Model for the ESP32-P4',
      target: '#exportBtn',
      fallback: ['.preview-card'],
      placement: 'bottom',
      advanceTarget: '#exportBtn',
      checkpoint: 'Click "Export Model" to build the 5-file deployment package',
      body: '<p>Click <b>Export Model</b> to generate the deployment package, then copy the <b>5 files</b> into the folder holding your <code>TFLite.ino</code> sketch:</p>' +
        '<ul>' +
        '<li><code>tm_model_data.cpp</code> · <code>tm_model_data.h</code> — the quantized weights as a C array.</li>' +
        '<li><code>model_resolver.h</code> — the MicroInterpreter op resolver.</li>' +
        '<li><code>model_settings.cpp</code> · <code>model_settings.h</code> — <code>constexpr</code> thresholds, labels, image size and the OOD gates.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>🖥️ Serial-Monitor live tuning (no reflash)</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table">' +
        '<tr><td><code>get mask</code></td><td>Print the current Dark / Lum thresholds.</td></tr>' +
        '<tr><td><code>set mask &lt;dark&gt; &lt;lum&gt;</code></td><td>Change the brightness mask live.</td></tr>' +
        '<tr><td><code>get ood</code></td><td>Print the four OOD gates.</td></tr>' +
        '<tr><td><code>set ood &lt;spmin&gt; &lt;spmax&gt; &lt;mpmin&gt; &lt;ermax&gt;</code></td><td>Retune the OOD gate live.</td></tr>' +
        '<tr><td><code>mode infer</code> / <code>mode gray</code></td><td>Switch between inference and raw grayscale streaming.</td></tr>' +
        '</table></div></details>'
    }},
    {{
      chapter: 6,
      title: 'You are ready to build 🎉',
      target: null,
      fallback: [],
      placement: 'auto',
      body: '<p>That is the whole <b>Image Project</b> loop:</p>' +
        '<ul>' +
        '<li>Collect samples per class, then <b>Edit</b> each class to crop the ROI and threshold the noise away.</li>' +
        '<li><b>Train</b>, check per-class accuracy, then tune the OOD gates in <b>⚙ Settings</b> until empty frames report <code>No Sign</code>.</li>' +
        '<li><b>Export</b> the 5 files into <code>TFLite.ino</code> and re-tune live over the Serial Monitor.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">📖</span><span>Re-open this guide at any time from <b>📖 Tutorial</b> in the top toolbar.</span></div>'
    }}
  ];

  /* ── Overlays the glyph curricula drive ──────────────────────────────
     Basic OCR reaches its character picker through Edit ▾ in the top
     toolbar; a road-sign project edits its three characters on the
     inline target bar and never uses that menu at all.  These wrappers
     keep the step definitions declarative, and every one is guarded so a
     project that does not render the control still tours cleanly. */
  function openEditMenu() {{
    if (typeof openToolbarDropdown !== 'function') return;
    try {{ openToolbarDropdown('menuEditDropdown'); }} catch (err) {{ console.warn('[tour] open Edit menu failed', err); }}
  }}

  function closeMenus() {{
    if (typeof closeToolbarDropdowns !== 'function') return;
    try {{ closeToolbarDropdowns(); }} catch (err) {{}}
  }}

  function closeCharOverlays() {{
    closeMenus();
    if (typeof closeOcrCharModal === 'function') {{ try {{ closeOcrCharModal(); }} catch (err) {{}} }}
    if (typeof closeOcrGlyphModal === 'function') {{ try {{ closeOcrGlyphModal(); }} catch (err) {{}} }}
  }}

  /* ═══ Curriculum 2 · Character OCR (16 steps) ════════════════════════
     One glyph per crop: choose the character set, collect samples,
     binarize with the integer Otsu the MCU runs, transfer-learn from
     EMNIST, preview a single glyph, then export the OCR firmware. */
  var OCR_STEPS = [
    {{
      chapter: 1,
      title: 'Your Character OCR Project, End to End',
      /* Step 0 is the big picture rather than a lesson, so it carries
         `overview` and a `stages` list instead of a spotlight: the card
         floats centred over a dimmed workspace and its one button starts
         the walkthrough. */
      target: null,
      fallback: [],
      placement: 'auto',
      overview: true,
      stages: [
        {{ t: 'Set Your Character Set', d: 'Pick which characters this project must read — digits, letters, or the punctuation the firmware knows.' }},
        {{ t: 'Collect &amp; Crop Samples', d: 'Capture one glyph per crop, then open the Sample Worktable to set the ROI box and the Dark / Lum thresholds.' }},
        {{ t: 'Binarize the Glyph', d: 'Apply the same integer Otsu the MCU runs, at 28×28, so host and device see byte-identical input.' }},
        {{ t: 'Transfer-Learn from EMNIST', d: 'Train the int8 classifier, warm-starting from EMNIST with the ink polarity matching your captures.' }},
        {{ t: 'Preview &amp; Export', d: 'Preview a single glyph, watch the _background rejection class, then export the OCR firmware.' }}
      ],
      body: '<p>This workspace trains a <b>single-character</b> classifier — one glyph per crop — and ships it to the <b>ESP32-P4</b> as <code>TFLite_OCR</code>.</p>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">💡</span><span>Nothing here is read as a <b>sequence</b>: one crop in, one character out. The <b>🚗 Car License Project</b> is a separate project type. The guide <b>points</b> at the real control for every step, and shrinks into a floating orb the moment you start working.</span></div>' +
        '<p class="tm-tour-muted">The whole trip, in five stages:</p>'
    }},
    {{
      chapter: 1,
      title: 'Save, Open &amp; Export from the File Menu',
      target: '#menuFileBtn',
      fallback: ['.topnav'],
      placement: 'bottom',
      advanceTarget: '#menuFileBtn',
      checkpoint: 'Click "File ▾" to see the project commands',
      body: '<p><b>File ▾</b> holds everything that acts on the project as a whole — including the character set you are about to pick:</p>' +
        '<ul>' +
        '<li><b>New project</b> (<kbd>⌘N</kbd>) — start a fresh, empty workspace.</li>' +
        '<li><b>Open project</b> (<kbd>⌘O</kbd>) — load a saved <code>.tmproj</code> archive, samples, character set and thresholds included.</li>' +
        '<li><b>Save</b> (<kbd>⌘S</kbd>) and <b>Save as</b> (<kbd>⇧⌘S</kbd>) — write the project to disk under a new name.</li>' +
        '<li><b>Export dataset</b> — dump the raw glyph images as files, for a backup or an offline experiment.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">💾</span><span>Save before every training run. A <code>.tmproj</code> keeps the glyphs, the crop boxes and the brightness thresholds — not just the class names.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Open the Edit Menu to Pick Characters',
      target: '#navEditCharacters',
      fallback: ['#menuEditBtn', '#ocrEmptyCta', '.col-classes'],
      placement: 'bottom',
      /* The menu item lives inside the Edit ▾ dropdown, which is shut until
         the user opens it.  Rather than popping the menu open for them, the
         step spotlights the Edit button and follows their press in. */
      phaseIf: function () {{ return !editMenuOpen(); }},
      phaseTarget: '#menuEditBtn',
      phaseName: 'the <b>Edit</b> menu',
      advanceTarget: '#navEditCharacters',
      onLeave: closeMenus,
      checkpoint: 'Click "Edit ▾", then "Edit characters"',
      body: '<p>Every character you select becomes a <b>normal class card</b> you can capture samples for — nothing else changes in the workflow.</p>' +
        '<ul>' +
        '<li><b>Edit ▾</b> in the top bar holds <b>Edit characters</b>.</li>' +
        '<li>In a brand-new project you can also click <b>Select Characters to Train</b> in the empty column.</li>' +
        '<li>Pick only what the device really has to read. A tighter set trains faster and scores higher.</li>' +
        '</ul>' +
        '<p>Leave <b>_background</b> enabled unless you are certain the camera will only ever see a glyph.</p>'
    }},
    {{
      chapter: 1,
      title: 'Preferences &amp; Return Home',
      target: '#navPreferences',
      fallback: ['#navReturn', '.topnav'],
      placement: 'bottom',
      advanceTarget: '#navPreferences',
      checkpoint: 'Click "Preferences" to see the display settings',
      body: '<ul>' +
        '<li><b>Preferences</b> opens a small modal whose one setting, <b>Max sample thumbnails</b> (<code>#prefMaxSamples</code>), caps how many glyph thumbnails each character card caches and draws — 12 by default.</li>' +
        '<li>Raise it while you are scanning a character&rsquo;s set for bad crops; lower it on a slow machine or a project with thousands of glyphs.</li>' +
        '<li><b>← Return Home</b> leaves the workspace for the start page. It is navigation, not a reset: the project on disk is untouched.</li>' +
        '</ul>'
    }},
    {{
      chapter: 1,
      title: 'Open Input Settings',
      target: '#navSettings',
      fallback: ['.topnav'],
      placement: 'bottom',
      advanceTarget: '#navSettings',
      checkpoint: 'Click "⚙ Settings" in the top bar to open Input Settings',
      body: '<p><b>⚙ Settings</b> is the single source of truth for the capture hardware — the webcam, the ESP32-P4 serial link and the frame geometry — applied to every character card at once.</p>' +
        '<ul>' +
        '<li>The four checkpoints after this one walk the modal from top to bottom: the camera and serial fields, the preview source and the OOD gates, the EMNIST mixing panel, and the zero-hardware sample filler.</li>' +
        '<li>It ends by asking you to close the modal yourself, so nothing vanishes from under your hands.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">⚠️</span><span>Close the Arduino IDE <b>Serial Monitor</b> before connecting — only one program can own the ESP32-P4 serial port.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Camera &amp; MCU Serial Fields',
      target: '#settingsCameraGroup',
      fallback: ['#settingsMcuGroup', '#settingsWebcam', '#globalSettingsModal'],
      placement: 'left',
      /* Inside the modal now.  The guard only covers the ways in that skip
         the Open checkpoint — a chapter pill, or `Skip step →` — where the
         modal may still be shut; there the step re-doors onto ⚙ instead of
         spotlighting a group the modal is covering. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      checkpoint: 'Click the highlighted ⚙ Settings button to open Input Settings',
      body: '<ul>' +
        '<li><b>Webcam</b> — the camera you hold the characters up to; <b>Capture FPS</b> throttles the live stream.</li>' +
        '<li><b>Serial Device (port)</b> — the ESP32-P4 USB device. Re-scan after plugging it in.</li>' +
        '<li><b>Frame side</b> / <b>Channels</b> must match the firmware — a glyph pipeline usually runs <b>28×28</b> or <b>96×96</b>, <b>Grayscale</b>.</li>' +
        '<li><b>Baud rate</b>: <b>115200</b> for reliability, <b>921600</b> for faster frame streaming.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>Deep dive · what each serial field does</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table"><tr><th>Field</th><th>Meaning</th></tr>' +
        '<tr><td><code>Port</code></td><td>USB serial device of the ESP32-P4 (re-scan after plugging in).</td></tr>' +
        '<tr><td><code>Frame side</code></td><td>Square frame the device streams; must match its firmware.</td></tr>' +
        '<tr><td><code>Channels</code></td><td><b>1</b> = grayscale (default), <b>3</b> = RGB colour.</td></tr>' +
        '<tr><td><code>Baud</code></td><td>115200 for reliability, 921600 for faster frame streaming.</td></tr>' +
        '<tr><td><code>Sync header</code></td><td>Byte pattern marking the start of a frame (AA 55 AA) — under <b>▸ Advanced Settings</b>.</td></tr>' +
        '</table></div></details>'
    }},
    {{
      chapter: 1,
      title: 'Preview Source &amp; the OOD Gates',
      target: '#settingsOodGroup',
      fallback: ['#settingsPreviewSource', '#globalSettingsModal'],
      placement: 'left',
      /* Still inside the modal.  A chapter pill can land here with the modal
         shut, so the step re-doors onto ⚙ exactly like the group above. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      advanceTarget: '#settingsPreviewSource',
      checkpoint: 'Pick the preview source, then check the four OOD gates',
      body: '<ul>' +
        '<li><b>Preview input source</b> (<code>#settingsPreviewSource</code>) decides what the preview pane reads: <b>Webcam</b>, <b>Device (MCU)</b> over serial, or a <b>Local Image File</b>. The preview card draws its own source row from this value.</li>' +
        '<li><b>OOD sign min %</b> / <b>max %</b> — Layer 1, the share of dark target pixels. Below the min (empty frame) or above the max (covered lens) the frame is rejected <b>before inference</b>, so a blank page never scores.</li>' +
        '<li><b>OOD max prob</b> — Layer 2, the winning character must clear this floor.</li>' +
        '<li><b>OOD entropy max</b> — Layer 3, the glyph probabilities must not be spread too evenly, i.e. the model must not be guessing.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>🎛️ Tuning cheat-sheet</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table"><tr><th>Symptom</th><th>Fix</th></tr>' +
        '<tr><td>A real glyph reads <code>No Sign</code></td><td><b>Loosen</b>: lower <code>sign min %</code>, lower <code>max prob</code>, raise <code>entropy max</code>.</td></tr>' +
        '<tr><td>An empty frame is classified</td><td><b>Tighten</b>: raise <code>sign min %</code>, raise <code>max prob</code>, lower <code>entropy max</code>.</td></tr>' +
        '</table></div></details>' +
        '<p class="tm-tour-muted">These four <code>constexpr</code> values travel into the exported <code>model_settings.h</code>.</p>'
    }},
    {{
      chapter: 1,
      title: 'OCR Dataset Mixing &amp; Transfer Learning',
      target: '#advOcrPanel',
      fallback: ['#modalOcrSection', '#globalSettingsModal'],
      placement: 'left',
      /* Still inside the modal, and this panel is NOT behind the Advanced
         disclosure: it sits in the modal's own Training Hyperparameters
         group, so the ⚙ door is the only thing that has to be open. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      advanceTarget: '#advOcrEmnist',
      checkpoint: 'Decide how much EMNIST is mixed into your character set',
      body: '<p>This is the OCR-only panel. It decides what the model trains on beyond your own crops:</p>' +
        '<ul>' +
        '<li><b>Mix EMNIST into every class</b> (<code>#advOcrEmnist</code>) — augments each character with public handwriting glyphs so a digit you captured 60 times is backed by thousands more.</li>' +
        '<li><b>EMNIST split</b> (<code>#advOcrSplit</code>) — which corpus to sample from: <code>ByClass</code>, <code>ByMerge</code>, <code>Letters</code>, <code>Digits</code> or <code>MNIST</code>. Match your character set.</li>' +
        '<li><b>Synthetic per class</b> (<code>#advOcrPerClass</code>) — how many extra EMNIST glyphs are added per character. <code>0</code> mirrors your own per-class count.</li>' +
        '<li><b>Use <code>_background</code> noise class to replace OOD rejection</b> (<code>#settingsUseBackgroundOod</code>) — ON trains an extra synthetic <code>_background</code> class that becomes the only per-slot rejector; OFF keeps the target characters as the sole outputs and leans on the OOD gates.</li>' +
        '<li><b>Background crops</b> (<code>#advOcrBackgroundN</code>) — how many noise crops train that rejector class; used only when the toggle above is ON.</li>' +
        '<li><b>Fine-tune pre-trained backbone</b> (<code>#advOcrTransfer</code>) — keeps the EMNIST backbone and attaches a fresh head: the fastest route to a high score on a few hundred glyphs. The pre-train / head / fine-tune rows below it appear only while it is on.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">⚠️</span><span>Mixing only helps when the fonts agree. Hand-written glyphs next to a clean printed font can pull accuracy down — watch the per-character score.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Zero-Hardware EMNIST Auto-Fill',
      target: '#settingsEmnistGroup',
      fallback: ['#settingsAdvancedToggle', '#globalSettingsModal'],
      placement: 'left',
      /* This row lives under the collapsed ▸ Advanced Settings disclosure,
         so `onEnter` unfolds it once the modal is on screen — and
         `resolveStepTargets` re-unfolds on every re-place, so a chapter
         pill jump cannot land on a `display:none` row either. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      onEnter: unfoldSettingsAdvanced,
      advanceTarget: '#settingsAutoSeedEmnist',
      advanceAlso: ['#settingsSeedEmnistNowBtn'],
      checkpoint: 'Tick "Auto-fill empty classes", or click "Fill Empty Classes with EMNIST Samples Now"',
      body: '<p><b>Advanced Settings</b> (<code>#settingsAdvancedToggle</code>) holds the frame sync word and this developer tool. It exists so the whole pipeline can be exercised <b>before any camera is wired up</b>:</p>' +
        '<ul>' +
        '<li><b>Auto-fill empty classes with EMNIST template samples</b> (<code>#settingsAutoSeedEmnist</code>) — the automatic version: every <b>empty</b> character class gets <b>24</b> dark-ink-on-light EMNIST captures, so <b>Train</b> unlocks with no hardware at all.</li>' +
        '<li><b>Fill Empty Classes with EMNIST Samples Now</b> (<code>#settingsSeedEmnistNowBtn</code>) — runs the fill once, immediately, and leaves the auto-fill choice as it is.</li>' +
        '<li>Classes that already hold real camera samples are <b>kept untouched</b> — the filler only touches the gaps.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🧪</span><span>Useful for a first rehearsal of capture → preprocess → train → preview. Delete the synthetic samples again before you judge real accuracy.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Leave Input Settings',
      target: '#settingsCancel',
      fallback: ['#settingsSave', '.tm-settings-card'],
      placement: 'left',
      /* The EMNIST checkpoint left the modal open on purpose, so the exit is
         the user's own press rather than the modal vanishing under them. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      advanceTarget: '#settingsCancel',
      advanceAlso: ['#settingsSave'],
      onLeave: closeSettingsModal,
      checkpoint: 'Click "Cancel" (or "Save changes") to exit Settings and return to the workspace',
      body: '<p><b>Save changes</b> keeps the EMNIST mix and the OOD gates for the next training run; <b>Cancel</b> leaves the previous values in place. Either closes the modal.</p>' +
        '<p class="tm-tour-muted"><kbd>Esc</kbd> closes it as well — the tour follows you back out to the workspace.</p>'
    }},
    {{
      chapter: 2,
      title: 'Organize &amp; Order Your Character Cards',
      target: '.class-card',
      fallback: ['.col-classes'],
      placement: 'right',
      /* Entering this chapter from a chapter pill would otherwise leave the
         Edit ▾ dropdown or a character picker covering the class column. */
      onEnter: closeCharOverlays,
      body: '<p>Every character you picked is a class card. Every icon on its header does one thing:</p>' +
        '<ul>' +
        '<li><b>⠿</b> (drag handle) — drag the card to reorder. The top-to-bottom order <b>is</b> the class index the firmware reports.</li>' +
        '<li><b>✏️</b> (<code>.tm-rename-btn</code>) — rename the class, e.g. <code>A</code>, <code>1</code>, <code>%</code>.</li>' +
        '<li><b>🧹</b> (<code>.clear-samples</code>) — clear every glyph in that class without deleting the class itself.</li>' +
        '<li><b>🗑</b> (<code>.tm-delete-class</code>) — delete the class, samples and all. This one cannot be undone.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🔧</span><span>Reordering changes the on-device index mapping — re-export the model after you reorder.</span></div>'
    }},
    {{
      chapter: 2,
      title: 'Add a Character Class',
      target: '.addclass',
      fallback: ['.col-classes'],
      placement: 'right',
      advanceTarget: '.addclass',
      checkpoint: 'Click "+ Add a class" to add another character',
      body: '<p><b>+ Add a class</b> at the foot of the column appends a label with an empty sample set — the way to add a character the picker did not offer.</p>' +
        '<ul>' +
        '<li>Rename it straight away with the <b>✏️</b> pencil, so the label matches the glyph it holds.</li>' +
        '<li>A brand-new class arrives with zero samples, so <b>Train</b> stays disabled until every class has some.</li>' +
        '</ul>'
    }},
    {{
      chapter: 2,
      title: 'Pick a Sample Source',
      target: '.tm-action-toolbar',
      fallback: ['.class-card'],
      placement: 'right',
      advanceTarget: '.tm-action-toolbar button.sample',
      checkpoint: 'Click "Webcam" to open the camera drawer',
      body: '<p>Three sources feed a character card:</p>' +
        '<ul>' +
        '<li><b>Webcam</b> — hold a printed or hand-written character up to the camera.</li>' +
        '<li><b>Upload</b> — import an existing folder of <code>.png</code> / <code>.jpg</code> glyphs.</li>' +
        '<li><b>Device</b> — stream frames from the ESP32-P4 camera over USB serial.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">👆</span><span>Clicking the tile you are already on folds the drawer back up — these are <b>toggles</b>, not one-way doors.</span></div>'
    }},
    {{
      chapter: 2,
      title: 'Capture Character Samples',
      live: 'samples',
      target: '[id^="sourceCapture-"]',
      fallback: ['.source-actions', '.source-panel'],
      placement: 'right',
      /* The capture controls only exist while the drawer is open: coming in
         from the Open checkpoint they are already there, coming in from a
         chapter pill the step re-doors onto the tile that opens them. */
      phaseIf: function () {{ return !drawerOpen(); }},
      phaseTarget: '.tm-action-toolbar button.sample',
      phaseName: 'the camera drawer',
      advanceTarget: '[id^="sourceCapture-"]',
      /* Holding the button is the burst gesture, and Upload swaps the pair
         for a file picker: all three finish the same checkpoint. */
      advanceAlso: ['[id^="sourceHold-"]', '[id^="uploadPick-"]'],
      checkpoint: 'Click "Capture" for one glyph, or hold "Hold to Capture" / <kbd>Space</kbd> to burst-capture',
      body: '<p>Capture <b>one character per class</b>. Vary the size, the stroke thickness and the lighting a little, but keep the character centred.</p>' +
        '<ul>' +
        '<li><b>Capture</b> files the current frame under this character.</li>' +
        '<li><b>Hold to Capture</b> — hold the button or <kbd>Space</kbd> for a burst while you move the sheet.</li>' +
        '<li><b>Upload</b> instead offers <b>Choose images from your files</b>.</li>' +
        '<li>A <b>⚙</b> chip in the drawer head jumps straight to this class&rsquo;s settings without leaving the drawer.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🎯</span><span>Aim for <b>60–100+ samples per character</b>. Crops are auto-tightened later, so loose framing is fine here.</span></div>'
    }},
    {{
      chapter: 2,
      title: 'Collapse the Camera Drawer',
      target: '[id^="sourceClose-"]',
      fallback: ['.source-close-pill', '.source-panel'],
      placement: 'right',
      phaseIf: function () {{ return !drawerOpen(); }},
      phaseTarget: '.tm-action-toolbar button.sample',
      phaseName: 'the camera drawer',
      advanceTarget: '[id^="sourceClose-"]',
      onLeave: closeSourceDrawer,
      checkpoint: 'Click "Close" (▴) to collapse the camera drawer',
      body: '<p>The <b>▴ Close</b> pill folds the drawer back into the class card, leaving the captured thumbnails in the summary strip.</p>' +
        '<p class="tm-tour-muted">Press <kbd>Esc</kbd> to collapse the drawer too — an open drawer outranks closing the tour.</p>'
    }},
    {{
      chapter: 2,
      title: 'Open the Sample Worktable',
      target: '.class-preprocess-chip',
      fallback: ['.class-card'],
      placement: 'right',
      advanceTarget: '.class-preprocess-chip',
      checkpoint: 'Click "Edit" to open the Sample Worktable',
      body: '<p>Click <b>Edit</b> on a character card to open the <b>Sample Worktable</b> — the per-sample inspector.</p>' +
        '<ul>' +
        '<li>The left pane shows the raw frame with the crop box over it.</li>' +
        '<li>The right pane shows the exact binarized <b>28×28</b> tensor the network will receive.</li>' +
        '<li>Use it to throw out a bad capture (<kbd>D</kbd>) or to nudge the crop (<kbd>Shift</kbd> + arrows).</li>' +
        '</ul>' +
        '<p>The next three checkpoints walk the worktable and then close it again.</p>'
    }},
    {{
      chapter: 3,
      title: 'Integer Otsu Binarization &amp; Vertical Projection',
      live: 'thresholds',
      target: '.class-preprocess-mode-row',
      fallback: ['#classPreprocessMode', '.class-preprocess-shell'],
      placement: 'left',
      /* Inside the worktable now.  The guard only covers the ways in that
         skip the Open checkpoint — a chapter pill, or `Skip step →`. */
      phaseIf: function () {{ return !worktableOpen(); }},
      phaseTarget: '.class-preprocess-chip',
      phaseName: 'the Sample Worktable',
      onEnter: openWorktable,
      advanceTarget: '#classPreprocessMode',
      checkpoint: 'Keep "Auto (glyph)", or pick "Manual ROI" and pin the crop box',
      body: '<ul>' +
        '<li><b>Auto (glyph)</b> — recomputes <b>Otsu&rsquo;s threshold</b> per sample with integer arithmetic, then centres the ink by <b>vertical projection</b>.</li>' +
        '<li><b>Manual ROI</b> — drag a fixed crop box (<code>ROI X1, Y1, X2, Y2</code>) when every capture frames the character the same way.</li>' +
        '<li><b>Dark Thr / Lum Thr</b> clamp the histogram before binarization, which tames a bright page or a shadowed one.</li>' +
        '</ul>' +
        '<p>The right-hand pane is the exact tensor the ESP32-P4 will infer on, so tune until it looks like a clean printed glyph.</p>' +
        '<details class="tm-tour-details"><summary>Deep dive · why integer arithmetic matters</summary><div class="tm-tour-details-body">' +
        '<p>Float Otsu on the host and float Otsu on the MCU disagree in the last bits, and a single flipped pixel rescales the whole 28×28 tensor. The pipeline therefore runs the <b>same fixed-point</b> implementation on both sides — what you see in the right-hand pane is literally the byte array the ESP32-P4 infers on.</p>' +
        '</div></details>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">💾</span><span>Press <kbd>S</kbd> (or <b>Save Sample</b>) after tuning — unsaved thresholds are discarded when the worktable closes.</span></div>'
    }},
    {{
      chapter: 3,
      title: 'Dark Thr, Lum Thr &amp; Save Sample',
      live: 'thresholds',
      target: '#classPreprocessDark',
      fallback: ['.class-preprocess-right', '#classPreprocessModal'],
      placement: 'left',
      /* Still inside the worktable.  The guard covers the ways in that skip
         the Open checkpoint. */
      phaseIf: function () {{ return !worktableOpen(); }},
      phaseTarget: '.class-preprocess-chip',
      phaseName: 'the Sample Worktable',
      onEnter: openWorktable,
      advanceTarget: '#classPreprocessDark',
      /* Save Sample finishes this checkpoint too, and it is not a descendant
         of the slider being spotlit, so it needs its own binding. */
      advanceAlso: ['#classPreprocessSave'],
      checkpoint: 'Drag "Dark Thr" / "Lum Thr" until the glyph is clean, then click "Save Sample"',
      body: '<ul>' +
        '<li><b>Dark Thr</b> — pixels darker than the threshold become <b>white (255)</b>, i.e. shadow.</li>' +
        '<li><b>Lum Thr</b> — pixels brighter than the threshold become <b>white (255)</b>, i.e. page.</li>' +
        '</ul>' +
        '<p>Together they leave only the dark strokes of the glyph on a pure-white field. Tune both until the right-hand preview is mostly white with a crisp black character.</p>' +
        '<ul>' +
        '<li>The <b>thumbnail strip</b> (<code>.class-preprocess-thumb</code>) jumps between this class&rsquo;s samples.</li>' +
        '<li><b>Delete Sample</b> (<code>#classPreprocessDeleteSample</code>) throws the current frame away for good — keep it for blur, glare or a hand in the way.</li>' +
        '<li><b>Save Sample</b> (<code>#classPreprocessSave</code>) writes the crop and the thresholds onto this sample, then returns to the workspace.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>Deep dive · worktable shortcuts</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table">' +
        '<tr><td><kbd>F1</kbd></td><td>Switch to Auto ROI</td></tr>' +
        '<tr><td><kbd>F2</kbd></td><td>Switch to Manual ROI</td></tr>' +
        '<tr><td><kbd>←</kbd> <kbd>→</kbd> <kbd>↑</kbd> <kbd>↓</kbd></td><td>Previous / next sample</td></tr>' +
        '<tr><td><kbd>Shift</kbd> + arrows</td><td>Nudge the ROI box by one pixel</td></tr>' +
        '<tr><td><kbd>D</kbd></td><td>Delete the current sample</td></tr>' +
        '<tr><td><kbd>S</kbd></td><td>Save the sample</td></tr>' +
        '</table></div></details>'
    }},
    {{
      chapter: 3,
      title: 'Close the Sample Worktable',
      target: '#classPreprocessCloseTop',
      fallback: ['#classPreprocessCancel', '.class-preprocess-actions'],
      placement: 'left',
      /* Save Sample finishes the previous step but leaves the worktable
         open, so the exit is a checkpoint of its own. */
      phaseIf: function () {{ return !worktableOpen(); }},
      phaseTarget: '.class-preprocess-chip',
      phaseName: 'the Sample Worktable',
      advanceTarget: '#classPreprocessCloseTop',
      advanceAlso: ['#classPreprocessCancel'],
      onLeave: closeWorktable,
      checkpoint: 'Click "Close" (×) to exit the Sample Worktable',
      body: '<p><b>Close</b> (<b>×</b> in the worktable head, or the footer button) returns to the class column. Saved crops and thresholds stay on the sample.</p>' +
        '<p class="tm-tour-muted"><kbd>Esc</kbd> closes it too, and so does clicking the dimmed backdrop.</p>'
    }},
    {{
      chapter: 4,
      title: 'Reveal the Advanced Training Options',
      target: '.train-card',
      fallback: ['.col-train'],
      placement: 'left',
      /* The Advanced panel is collapsed on a fresh workspace, and the ink
         polarity the next checkpoint talks about lives in it — so the step
         asks the user to press Advanced rather than unfolding it for them. */
      phaseIf: function () {{ return !advancedOpen(); }},
      phaseTarget: '#advBtn',
      phaseName: 'the <b>Advanced</b> training options',
      advanceTarget: '#advBtn',
      checkpoint: 'Click "Advanced" to unfold the training options',
      body: '<p>The middle column trains the glyph classifier. <b>Advanced</b> unfolds the hyperparameters, and <b>Train Embedded Model</b> starts the run.</p>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🧠</span><span>One crop in, one character out — the card reports per-character accuracy when the run ends.</span></div>'
    }},
    {{
      chapter: 4,
      title: 'Transfer Learning &amp; Global Ink Polarity',
      target: '#advInkPolarityRow',
      fallback: ['#trainAdvPanel', '.train-card'],
      placement: 'left',
      /* The row only exists while the panel is unfolded. */
      phaseIf: function () {{ return !advancedOpen(); }},
      phaseTarget: '#advBtn',
      phaseName: 'the <b>Advanced</b> training options',
      advanceTarget: '#advInkPolarity',
      /* Recommended / Reset / the inline numbers are the rest of the panel,
         and touching any of them settles this checkpoint too. */
      advanceAlso: ['#advUseRec', '#advReset', '#advEpochsInline', '#advBatchInline', '#advLrInline'],
      checkpoint: 'Set "Ink Polarity" to match how your glyphs are drawn',
      body: '<ul>' +
        '<li><b>Transfer learning</b> (on by default) keeps an EMNIST-pretrained backbone and attaches a fresh head for <i>your</i> character set — the fastest route to a high score on a few hundred crops.</li>' +
        '<li><b>Ink Polarity</b> says which way the ink sits: <code>dark on light</code> for camera captures, <code>light on dark</code> for MNIST/EMNIST-style images. Get it wrong and the model learns the inverse of your data.</li>' +
        '<li>The architecture rows are hidden while transfer learning is on: those numbers belong to the frozen backbone, not to your head.</li>' +
        '<li><b>💡 Use recommended</b> fills the inline numbers in for roughly <b>800 gradient updates</b>; <b>Reset</b> puts the shipped defaults back.</li>' +
        '</ul>'
    }},
    {{
      chapter: 4,
      title: 'Train the Embedded Model',
      target: '#trainBtn',
      fallback: ['.train-card', '.col-train'],
      placement: 'left',
      advanceTarget: '#trainBtn',
      live: 'train',
      checkpoint: 'Click "Train Embedded Model" (or press ⌘/Ctrl + Enter) to start',
      body: '<p>Press <b>Train Embedded Model</b> (<kbd>⌘</kbd><kbd>Enter</kbd> / <kbd>Ctrl</kbd><kbd>Enter</kbd>), then open the green accuracy pill (<code>#trainStatus</code>) for the per-character breakdown.</p>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">📊</span><span>A character that lags behind the rest usually needs more samples — go back to <b>Edit</b> and re-train.</span></div>'
    }},
    {{
      chapter: 5,
      title: 'Live Single-Glyph Preview',
      target: '.preview-card',
      fallback: ['.col-preview'],
      placement: 'left',
      body: '<ul>' +
        '<li>Toggle <b>Input</b> ON to push live frames through the quantized model.</li>' +
        '<li>Toggle <b>ROI</b> to draw the crop rectangle over the frame and <b>Orig</b> to see the full masked frame; with both on you get the cropped, masked tensor handed to the model.</li>' +
        '<li>The row underneath (<code>#previewContextBar</code>) picks where the pane reads from — a <b>Webcam</b>, the <b>Device</b> port plus <b>Refresh</b>, or <b>📁 Upload Image…</b> — independently of the capture drawer.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🔁</span><span>Preview runs the <b>same int8</b> graph on the host CPU the MCU will run, so what you score here is what the device scores.</span></div>'
    }},
    {{
      chapter: 5,
      title: 'Live Dark / Lum Sliders &amp; the Confidence Bars',
      target: '.preview-card',
      fallback: ['.col-preview'],
      placement: 'left',
      body: '<ul>' +
        '<li>The live <b>Dark</b> / <b>Lum</b> sliders under the image (<code>#previewDarkSlider</code>, <code>#previewLumSlider</code>) re-tune the brightness mask while you watch — they should match the thresholds you saved in the worktable.</li>' +
        '<li>The <b>confidence bars</b> in <code>#previewOutput</code> are the top-<i>k</i> characters for the frame on screen; a rejected frame reads <code>No Sign</code> instead.</li>' +
        '</ul>' +
        '<p class="tm-tour-muted">Sanity-check the preview before exporting: a mask that looks wrong here will look exactly as wrong on the device.</p>'
    }},
    {{
      chapter: 5,
      title: 'Export the OCR Firmware Package',
      target: '#exportBtn',
      fallback: ['.preview-card'],
      placement: 'bottom',
      advanceTarget: '#exportBtn',
      checkpoint: 'Click "Export Model" to build the 5-file deployment package',
      body: '<p>Click <b>Export Model</b>, then drop the <b>5 files</b> next to your <code>TFLite_OCR.ino</code> sketch:</p>' +
        '<ul>' +
        '<li><code>tm_model_data.cpp</code> · <code>tm_model_data.h</code> — the quantized weights.</li>' +
        '<li><code>model_resolver.h</code> — the MicroInterpreter op resolver.</li>' +
        '<li><code>model_settings.cpp</code> · <code>model_settings.h</code> — labels, geometry and the OOD gates as <code>constexpr</code>.</li>' +
        '</ul>' +
        '<p class="tm-tour-muted">Re-export after any change to the character set, the image size or the OOD thresholds.</p>'
    }},
    {{
      chapter: 5,
      title: 'You are ready to build 🎉',
      target: null,
      fallback: [],
      placement: 'auto',
      body: '<p>That is the whole <b>Character OCR</b> loop:</p>' +
        '<ul>' +
        '<li>Pick the characters → collect samples → crop and binarize each glyph.</li>' +
        '<li>Transfer-learn from EMNIST, with the ink polarity matching your captures.</li>' +
        '<li>Preview a single glyph, then export the 5 files into <code>TFLite_OCR.ino</code>.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">📖</span><span>Re-open this guide any time from <b>📖 Tutorial</b>. The Car License workflow has its own tour.</span></div>'
    }}
  ];

  /* ═══ Curriculum 3 · 🚗 Car License Project (29 steps) ═══════════════
     A labelled SEQUENCE: find the plate, split it into three slots, read
     each slot as a glyph, then verify the whole reading — against one or
     more allowed plates. */
  var SIGN_OCR_STEPS = [
    {{
      chapter: 1,
      title: 'Your 🚗 Car License Project, End to End',
      /* Step 0 is the big picture rather than a lesson, so it carries
         `overview` and a `stages` list instead of a spotlight: the card
         floats centred over a dimmed workspace and its one button starts
         the walkthrough. */
      target: null,
      fallback: [],
      placement: 'auto',
      overview: true,
      stages: [
        {{ t: 'Set the Target Plate', d: 'Type the sign&rsquo;s three characters into the slots on the left — that target is what the model verifies, and 🪪 License List holds one or more allowed plates.' }},
        {{ t: 'Collect &amp; Crop Samples', d: 'Capture one glyph per crop for each character, then open the Sample Worktable to set the ROI box and the Dark / Lum thresholds.' }},
        {{ t: 'Isolate the Glyphs', d: 'Each class learns a single character; the 3-character plate is split into its slots at Preview time, never during training.' }},
        {{ t: 'Order &amp; Train', d: 'Set the reading direction, train the tiny int8 classifier and read the per-character accuracy.' }},
        {{ t: 'Verdict &amp; Export', d: 'Read the three slots and the 🟢 YES / 🔴 NO banner, then export the Car License firmware package.' }}
      ],
      body: '<p>The <b>🚗 Car License Project</b> does not classify a whole plate. It <b>verifies a known target</b>, one character at a time, and answers <b>🟢 YES (MATCH)</b> or <b>🔴 NO (NOT MATCH)</b>.</p>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">💡</span><span>Because the target is fixed, the model only ever has to learn the handful of characters that appear on it — a very small, very accurate classifier. The guide <b>points</b> at the real control for every step, and shrinks into a floating orb the moment you start working.</span></div>' +
        '<p class="tm-tour-muted">The whole trip, in five stages:</p>'
    }},
    {{
      chapter: 1,
      title: 'Save, Open &amp; Export from the File Menu',
      target: '#menuFileBtn',
      fallback: ['.topnav'],
      placement: 'bottom',
      advanceTarget: '#menuFileBtn',
      checkpoint: 'Click "File ▾" to see the project commands',
      body: '<p><b>File ▾</b> holds everything that acts on the project as a whole — including the target sign you are about to declare:</p>' +
        '<ul>' +
        '<li><b>New project</b> (<kbd>⌘N</kbd>) — start a fresh, empty workspace.</li>' +
        '<li><b>Open project</b> (<kbd>⌘O</kbd>) — load a saved <code>.tmproj</code> archive, its sign sequence, samples and thresholds included.</li>' +
        '<li><b>Save</b> (<kbd>⌘S</kbd>) and <b>Save as</b> (<kbd>⇧⌘S</kbd>) — write the project to disk under a new name.</li>' +
        '<li><b>Export dataset</b> — dump the raw sample images as files, for a backup or an offline experiment.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">💾</span><span>Save before every training run. A <code>.tmproj</code> keeps the three-character target, the crop boxes and the brightness thresholds — not just the character names.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Preferences &amp; Return Home',
      target: '#navPreferences',
      fallback: ['#navReturn', '.topnav'],
      placement: 'bottom',
      advanceTarget: '#navPreferences',
      checkpoint: 'Click "Preferences" to see the display settings',
      body: '<ul>' +
        '<li><b>Preferences</b> opens a small modal whose one setting, <b>Max samples displayed per class</b> (<code>#prefMaxSamples</code>), caps how many sample thumbnails each character card caches and draws — 12 by default.</li>' +
        '<li>Raise it while you are eyeballing a glyph set for bad captures; lower it on a slow machine or a project with thousands of samples.</li>' +
        '<li><b>← Return Home</b> leaves the workspace for the start page. It is navigation, not a reset: the project on disk is untouched.</li>' +
        '</ul>'
    }},
    {{
      chapter: 1,
      title: 'Open Input Settings',
      target: '#navSettings',
      fallback: ['.topnav'],
      placement: 'bottom',
      advanceTarget: '#navSettings',
      checkpoint: 'Click "⚙ Settings" in the top bar to open Input Settings',
      body: '<p><b>⚙ Settings</b> is the single source of truth for the capture hardware — the webcam, the ESP32-P4 serial link and the frame geometry — applied to every character card at once. It also carries the sign-only options: the preview source, the OOD gates, the EMNIST mixing panel and the template auto-fill.</p>' +
        '<ul>' +
        '<li>MCU defaults: frame <b>96×96</b>, <b>Grayscale</b>, baud <b>115200</b> (or <b>921600</b>), sync header <b>AA 55 AA</b>.</li>' +
        '<li>The next five checkpoints walk the modal from top to bottom, then close it again.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">⚠️</span><span>Close the Arduino IDE <b>Serial Monitor</b> before connecting — only one program can own the ESP32-P4 serial port.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Camera &amp; MCU Serial Fields',
      target: '#settingsCameraGroup',
      fallback: ['#settingsMcuGroup', '#settingsWebcam', '#globalSettingsModal'],
      placement: 'left',
      /* Inside the modal now.  The guard is only for the ways in that skip
         the Open checkpoint — a chapter pill, or `Skip step →` — where the
         modal may still be shut; there the step re-doors onto ⚙ instead of
         spotlighting a group the modal is covering. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      checkpoint: 'Click the highlighted ⚙ Settings button to open Input Settings',
      body: '<ul>' +
        '<li><b>Webcam</b> — the capture device; <b>Capture FPS</b> throttles the live preview stream.</li>' +
        '<li><b>Serial Device (port)</b> — the ESP32-P4 USB device. Re-scan after plugging it in.</li>' +
        '<li><b>Frame side</b> (<code>#settingsFrameSide</code>) / <b>Channels</b> must match the firmware — <b>96×96</b> and <b>Grayscale</b> by default. Raise Frame side to <b>160×160</b> for the sharpest capture: it is the <b>camera&rsquo;s</b> resolution, not the model&rsquo;s.</li>' +
        '<li><b>Image Size</b> is the model&rsquo;s own square, set in the middle column under ▸ <b>Advanced</b> (<code>#advImgSizeInline</code>) — <b>96</b>, or <b>28</b> for a single glyph. Capture high and train compact: the app scales every incoming frame down to the model&rsquo;s <code>img_size</code>, and the exported firmware carries that size in <code>model_settings.h</code>, so the device scales its camera frames the same way.</li>' +
        '<li><b>Baud rate</b>: <b>115200</b> for reliability, <b>921600</b> for faster frame streaming.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>Deep dive · what each serial field does</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table"><tr><th>Field</th><th>Meaning</th></tr>' +
        '<tr><td><code>Port</code></td><td>USB serial device of the ESP32-P4 (re-scan after plugging in).</td></tr>' +
        '<tr><td><code>Frame side</code></td><td>Square frame the device streams; must match its firmware. It is separate from the model&rsquo;s <b>Image Size</b> — frames are scaled down to it.</td></tr>' +
        '<tr><td><code>Channels</code></td><td><b>1</b> = grayscale (default), <b>3</b> = RGB colour.</td></tr>' +
        '<tr><td><code>Baud</code></td><td>115200 for reliability, 921600 for faster frame streaming.</td></tr>' +
        '<tr><td><code>Sync header</code></td><td>Byte pattern marking the start of a frame (AA 55 AA) — under <b>▸ Advanced Settings</b>.</td></tr>' +
        '</table></div></details>'
    }},
    {{
      chapter: 1,
      title: 'Preview Source &amp; the OOD Gates',
      target: '#settingsOodGroup',
      fallback: ['#settingsPreviewSource', '#globalSettingsModal'],
      placement: 'left',
      /* Still inside the modal.  A chapter pill can land here with the modal
         shut, so the step re-doors onto ⚙ exactly like the group above. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      advanceTarget: '#settingsPreviewSource',
      checkpoint: 'Pick the preview source, then check the four OOD gates',
      body: '<ul>' +
        '<li><b>Preview Image Input Source</b> (<code>#settingsPreviewSource</code>) decides what the preview pane reads: <b>Webcam</b>, <b>Device (MCU)</b> over serial, or a <b>Local Image File</b>. The preview card draws its own source row from this value — and the sign-only synth pills live in the <b>Local Image File</b> source.</li>' +
        '<li><b>OOD sign min %</b> / <b>max %</b> — Layer 1, the share of dark target pixels. Below the min (empty scene) or above the max (covered lens) the frame is rejected <b>before inference</b>.</li>' +
        '<li><b>OOD max prob</b> — Layer 2, the winning character must clear this floor.</li>' +
        '<li><b>OOD entropy max</b> — Layer 3, the probabilities must not be spread too evenly, i.e. the model must not be guessing.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>🎛️ Tuning cheat-sheet</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table"><tr><th>Symptom</th><th>Fix</th></tr>' +
        '<tr><td>Real signs are missed</td><td><b>Loosen</b>: lower <code>sign min %</code>, raise <code>sign max %</code>, lower <code>max prob</code>, raise <code>entropy max</code>.</td></tr>' +
        '<tr><td>Empty wall is accepted</td><td><b>Tighten</b>: raise <code>sign min %</code>, lower <code>sign max %</code>, raise <code>max prob</code>, lower <code>entropy max</code>.</td></tr>' +
        '</table></div></details>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🎯</span><span>On the Car License path each slot also has its own confidence floor, and by default (<code>_background</code> OFF) the four gates above plus that floor are the whole rejector — so start permissive and tighten until a blank frame reliably reads red. The next checkpoint covers the optional <code>_background</code> class.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Dataset Mixing, _background &amp; Transfer Learning',
      target: '#advOcrPanel',
      fallback: ['#modalOcrSection', '#globalSettingsModal'],
      placement: 'left',
      /* Still inside the modal, and this panel is NOT behind the Advanced
         disclosure: it sits in the modal's own Training Hyperparameters
         group, so the ⚙ door is the only thing that has to be open. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      advanceTarget: '#advOcrEmnist',
      /* Flipping the `_background` toggle is this checkpoint's own
         decision, so touching it counts as doing the step. */
      advanceAlso: ['#settingsUseBackgroundOod'],
      checkpoint: 'Decide how much EMNIST is mixed in, and whether the model trains a _background class',
      body: '<p>This is the glyph-project panel. It decides what the model trains on beyond your own captures:</p>' +
        '<ul>' +
        '<li><b>Mix EMNIST into every class</b> (<code>#advOcrEmnist</code>) — augments each character with public handwriting glyphs, so a character you captured 60 times is backed by thousands more.</li>' +
        '<li><b>EMNIST split</b> (<code>#advOcrSplit</code>) — which corpus to sample from: <code>ByClass</code>, <code>ByMerge</code>, <code>Letters</code>, <code>Digits</code> or <code>MNIST</code>. Match the characters on your sign.</li>' +
        '<li><b>Synthetic per class</b> (<code>#advOcrPerClass</code>) — how many extra EMNIST glyphs are added per character. <code>0</code> mirrors your own per-class count.</li>' +
        '<li><b>Use <code>_background</code> noise class to replace OOD rejection</b> (<code>#settingsUseBackgroundOod</code>) — OFF (the car-license default) trains only the target characters and rejects with the OOD gates plus the per-slot probability floor; ON trains an extra synthetic <code>_background</code> class that becomes the only per-slot rejector.</li>' +
        '<li><b>Background crops</b> (<code>#advOcrBackgroundN</code>) — how many noise crops train that rejector class; used only when the toggle above is ON.</li>' +
        '<li><b>Fine-tune pre-trained backbone</b> (<code>#advOcrTransfer</code>) — keeps the EMNIST backbone and attaches a fresh head: the fastest route to a high score on a few hundred glyphs. The pre-train / head / fine-tune rows below it appear only while it is on.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">⚠️</span><span>Mixing only helps when the fonts agree. Hand-written glyphs next to a clean printed font can pull accuracy down — watch the per-character score.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Zero-Hardware EMNIST Auto-Fill',
      target: '#settingsEmnistGroup',
      fallback: ['#settingsAdvancedToggle', '#globalSettingsModal'],
      placement: 'left',
      /* This row lives under the collapsed ▸ Advanced Settings disclosure,
         so `onEnter` unfolds it once the modal is on screen — and
         `resolveStepTargets` re-unfolds on every re-place, so a chapter
         pill jump cannot land on a `display:none` row either. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      onEnter: unfoldSettingsAdvanced,
      advanceTarget: '#settingsAutoSeedEmnist',
      advanceAlso: ['#settingsSeedEmnistNowBtn'],
      checkpoint: 'Tick "Auto-fill empty classes", or click "Fill Empty Classes with EMNIST Samples Now"',
      body: '<p><b>Advanced Settings</b> (<code>#settingsAdvancedToggle</code>) holds the frame sync word and this developer tool. It exists so the whole pipeline can be exercised <b>before any camera is wired up</b>:</p>' +
        '<ul>' +
        '<li><b>Auto-fill empty classes with EMNIST template samples</b> (<code>#settingsAutoSeedEmnist</code>) — the automatic version: every <b>empty</b> character class gets <b>24</b> dark-ink-on-light EMNIST captures, so <b>Train</b> unlocks with no hardware at all.</li>' +
        '<li><b>Fill Empty Classes with EMNIST Samples Now</b> (<code>#settingsSeedEmnistNowBtn</code>) — runs the fill once, immediately, and leaves the auto-fill choice as it is.</li>' +
        '<li>Classes that already hold real camera samples are <b>kept untouched</b> — the filler only touches the gaps.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🧪</span><span>Useful for a first rehearsal of capture → preprocess → train → preview on a sign you have not photographed yet. Delete the synthetic samples again before you judge real accuracy.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Leave Input Settings',
      target: '#settingsCancel',
      fallback: ['#settingsSave', '.tm-settings-card'],
      placement: 'left',
      /* The EMNIST checkpoint left the modal open on purpose, so the exit is
         the user's own press rather than the modal vanishing under them. */
      phaseIf: function () {{ return !settingsOpen(); }},
      phaseTarget: '#navSettings',
      phaseName: 'Input Settings',
      advanceTarget: '#settingsCancel',
      advanceAlso: ['#settingsSave'],
      onLeave: closeSettingsModal,
      checkpoint: 'Click "Cancel" (or "Save changes") to exit Settings and return to the workspace',
      body: '<p><b>Save changes</b> keeps the EMNIST mix, the preview source and the OOD gates for the next run and writes the gates into the exported <code>model_settings</code> constants; <b>Cancel</b> keeps the previous values. Either closes the modal.</p>' +
        '<p class="tm-tour-muted"><kbd>Esc</kbd> closes it as well — the tour follows you back out to the workspace.</p>'
    }},
    {{
      chapter: 1,
      title: 'Set the 3 Target Sign Characters, in Order',
      live: 'signseq',
      target: '#signSeqBar',
      fallback: ['.signseq-bar', '.col-classes'],
      placement: 'right',
      advanceTarget: '#signSeqSlot-0',
      checkpoint: 'Type each character into the three slots, left to right',
      body: '<ul>' +
        '<li>Type the sign&rsquo;s characters into the <b>three slots</b>, left to right, exactly as they are printed — for example <b>A ▸ 1 ▸ 2</b>.</li>' +
        '<li>Each slot becomes a <b>class</b> the model must learn, so a repeated character (<b>A ▸ 1 ▸ A</b>) is trained once — its card carries a <b>Slots 1 &amp; 3</b> badge instead of a second card.</li>' +
        '<li>The <b>left → right</b> / <b>right → left</b> tag in the bar echoes the reading direction in force.</li>' +
        '</ul>' +
        '<p>The three slots are the <b>single writer</b> of the target: change them and the class list, the training set and the preview verdict all follow.</p>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">⌨️</span><span>Focus a slot and type, or <b>paste the whole sign</b> into the first slot to fill all three at once. The tour advances once the bar has settled, so you can fill the sign without leaving the step.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Allow More Than One Plate: the 🪪 License List',
      target: '#navLicenseList',
      fallback: ['#signSeqBar', '.topnav'],
      placement: 'bottom',
      advanceTarget: '#navLicenseList',
      /* The whitelist is its own modal, and the next checkpoint lives on the
         slot bar behind it — so leaving the step closes what it opened,
         exactly like the Settings and drawer exits. */
      onLeave: closeLicenseListModal,
      checkpoint: 'Click "🪪 License List" beside ⚙ Settings to add a second valid plate',
      body: '<p>Think of the gatekeeper at a car park, who is allowed to remember <b>several</b> cars rather than only one. <b>🪪 License List</b> (<code>#navLicenseList</code>) sits next to <b>⚙ Settings</b> and holds <b>one or more valid 3-character plates</b> — for example <b>A12</b> and <b>B18</b>.</p>' +
        '<ul>' +
        '<li>Every plate on the list is a reading the firmware accepts; a plate that is not listed stays <b>🔴 NO</b>.</li>' +
        '<li><b>+ Add Allowed License Plate</b> (<code>#licenseListAdd</code>) appends another row, and <b>Save &amp; Sync Classes</b> merges the <b>unique characters of every plate</b> into the left-hand Character Cards — a character shared by two plates (such as <code>1</code>) has its samples collected <b>once</b> and serves both.</li>' +
        '<li>The badge on the button (<code>#navLicenseCount</code>) counts the plates on the list, and the first plate is the one the slot bar above shows.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🅿️</span><span>A single plate — the default — is enough for this walkthrough. Add a second only when one camera has to recognise more than one car.</span></div>'
    }},
    {{
      chapter: 1,
      title: 'Swap the Slots with the ◀ ▶ Arrows',
      target: '.signseq-move',
      fallback: ['#signSeqBar', '.signseq-row'],
      placement: 'right',
      advanceTarget: '#signSeqRight-1',
      checkpoint: 'Click the ▶ arrow to swap a character with its right-hand neighbour',
      body: '<p>Each slot carries a tiny <b>◀ ▶</b> pair. They move that character one position left or right, which is the fast way to fix a sign you typed in the wrong order — far quicker than retyping it.</p>' +
        '<ul>' +
        '<li>The <b>left</b> arrow of slot 1 and the <b>right</b> arrow of slot 3 are disabled: there is nowhere to go.</li>' +
        '<li>Swapping rewrites the <b>reading direction note</b> in the bar, so you can see immediately which end the firmware will call slot 1.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🔁</span><span>A swap does not lose samples: they belong to the character, not to the position — the same class simply answers a different slot.</span></div>'
    }},
    {{
      chapter: 2,
      title: 'Your Character Cards &amp; Their Icons',
      target: '.class-card',
      fallback: ['.col-classes'],
      placement: 'right',
      body: '<p>Every character in the sign has one card. In a Car License project the cards follow the slots, so the header carries a different set of affordances from an image project:</p>' +
        '<ul>' +
        '<li><b>Slots 1 &amp; 3</b> (<code>.signslot-badge</code>) — which positions of the sign this character answers for. One class can serve several slots.</li>' +
        '<li><b>Edit</b> (<code>.class-preprocess-chip</code>) — open the <b>Sample Worktable</b> for this character.</li>' +
        '<li><b>🧹</b> (<code>.clear-samples</code>) — clear every sample in that class without deleting the class.</li>' +
        '<li><b>🗑</b> (<code>.tm-delete-class</code>) — delete the class, samples and all. This one cannot be undone.</li>' +
        '</ul>' +
        '<p>There is no drag handle and no rename pencil here: the slots own the order and the names, and a manual edit would be undone by the next sync.</p>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🎯</span><span>Each card collects <b>single-character glyphs</b>. Splitting the 3-character plate into its slots happens in <b>Preview</b>, never during training — a whole plate filed under <code>A</code> would teach the model the plate, not the letter.</span></div>'
    }},
    {{
      chapter: 2,
      title: 'Pick a Sample Source',
      target: '.tm-action-toolbar',
      fallback: ['.class-card'],
      placement: 'right',
      advanceTarget: '.tm-action-toolbar button.sample',
      checkpoint: 'Click "Webcam" to open the camera drawer',
      body: '<p>Three sources feed a character card. Pick the one that matches your hardware:</p>' +
        '<ul>' +
        '<li><b>Webcam</b> — the laptop or an external USB camera.</li>' +
        '<li><b>Upload</b> — batch-import existing <code>.png</code> / <code>.jpg</code> glyphs.</li>' +
        '<li><b>Device</b> — stream live frames from the ESP32-P4 camera over USB serial.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">👆</span><span>Clicking the tile you are already on folds the drawer back up — these are <b>toggles</b>, not one-way doors.</span></div>'
    }},
    {{
      chapter: 2,
      title: 'Capture Character Samples',
      live: 'samples',
      target: '[id^="sourceCapture-"]',
      fallback: ['.source-actions', '.source-panel'],
      placement: 'right',
      /* The capture controls only exist while the drawer is open.  Coming
         in from the Open checkpoint they are already there; coming in from
         a chapter pill the step re-doors onto the tile that opens them. */
      phaseIf: function () {{ return !drawerOpen(); }},
      phaseTarget: '.tm-action-toolbar button.sample',
      phaseName: 'the camera drawer',
      advanceTarget: '[id^="sourceCapture-"]',
      /* Holding the button is the burst gesture, and Upload swaps the pair
         for a file picker: all three finish the same checkpoint. */
      advanceAlso: ['[id^="sourceHold-"]', '[id^="uploadPick-"]'],
      checkpoint: 'Click "Capture" for one glyph, or hold "Hold to Capture" / <kbd>Space</kbd> to burst-capture',
      body: '<p>Fill the frame with <b>one character</b> and capture it under that character&rsquo;s card, then switch cards for the next one.</p>' +
        '<ul>' +
        '<li><b>Capture</b> files the current frame under this character.</li>' +
        '<li><b>Hold to Capture</b> — hold the button (or hold <kbd>Space</kbd>) to <b>burst-capture</b> while you move the sign, which is the quickest way to get fifty frames.</li>' +
        '<li><b>Upload</b> instead offers <b>Choose images from your files</b>, and a <b>⚙</b> chip in the drawer head jumps to this class&rsquo;s settings.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🎯</span><span>Aim for <b>50–100+ samples per character</b>, varying distance, angle and lighting. The plate split happens later, in Preview.</span></div>'
    }},
    {{
      chapter: 2,
      title: 'Collapse the Camera Drawer',
      target: '[id^="sourceClose-"]',
      fallback: ['.source-close-pill', '.source-panel'],
      placement: 'right',
      phaseIf: function () {{ return !drawerOpen(); }},
      phaseTarget: '.tm-action-toolbar button.sample',
      phaseName: 'the camera drawer',
      advanceTarget: '[id^="sourceClose-"]',
      onLeave: closeSourceDrawer,
      checkpoint: 'Click "Close" (▴) to collapse the camera drawer',
      body: '<p>The <b>▴ Close</b> pill folds the drawer back into the class card, keeping the captured thumbnails in the summary strip.</p>' +
        '<p class="tm-tour-muted">Press <kbd>Esc</kbd> to collapse the drawer instead of leaving the tour — an open drawer outranks the tour in the keyboard stack.</p>'
    }},
    {{
      chapter: 2,
      title: 'Open the Sample Worktable',
      target: '.class-preprocess-chip',
      fallback: ['.class-card'],
      placement: 'right',
      advanceTarget: '.class-preprocess-chip',
      checkpoint: 'Click "Edit" to open the Sample Worktable',
      body: '<p>Click <b>Edit</b> on a character card to open the <b>Sample Worktable</b> for that character.</p>' +
        '<ul>' +
        '<li>A license plate is a <b>fixed target</b>, so pinning this character&rsquo;s glyph with <b>Manual ROI</b> is often more reliable than auto-search.</li>' +
        '<li><b>Dark Thr</b> / <b>Lum Thr</b> isolate the dark glyph from the plate and the background.</li>' +
        '</ul>' +
        '<p>The next three checkpoints walk the worktable and then close it again.</p>'
    }},
    {{
      chapter: 3,
      title: 'Isolate a Single Character in the Worktable',
      live: 'thresholds',
      target: '.class-preprocess-mode-row',
      fallback: ['#classPreprocessMode', '.class-preprocess-shell'],
      placement: 'left',
      /* Inside the worktable now.  The guard only covers the ways in that
         skip the Open checkpoint — a chapter pill, or `Skip step →`. */
      phaseIf: function () {{ return !worktableOpen(); }},
      phaseTarget: '.class-preprocess-chip',
      phaseName: 'the Sample Worktable',
      onEnter: openWorktable,
      advanceTarget: '#classPreprocessMode',
      checkpoint: 'Pick "Manual ROI" to pin the glyph, or keep "Auto" and drag the box',
      body: '<ul>' +
        '<li><b>Original Sample (left)</b> — the raw frame with the green / blue <b>ROI</b> crop box drawn over it.</li>' +
        '<li><b>Processed Preview (right)</b> — the exact filtered and resized glyph this character&rsquo;s sample becomes: the tensor the glyph model receives.</li>' +
        '</ul>' +
        '<p><b>Preprocess Mode</b>:</p>' +
        '<ul>' +
        '<li><code>Auto (find sign)</code> — searches for the darkest target region and crops it. The best default.</li>' +
        '<li><code>Manual ROI</code> — drag on the original image to lock a fixed crop box (<code>ROI X1, Y1, X2, Y2</code>). Use it when the character is always in the same spot on the plate.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">⏱️</span><span>The 3-slot split is a separate <b>inference</b> step in Preview — it is never part of training. This worktable sees one character per crop.</span></div>'
    }},
    {{
      chapter: 3,
      title: 'Dark Thr, Lum Thr &amp; Save Sample',
      live: 'thresholds',
      target: '#classPreprocessDark',
      fallback: ['.class-preprocess-right', '#classPreprocessModal'],
      placement: 'left',
      /* Save Sample finishes this step too, and it is not a descendant of
         the slider being spotlit, so it needs its own binding.  Closing the
         worktable is a checkpoint of its own, right after this one. */
      phaseIf: function () {{ return !worktableOpen(); }},
      phaseTarget: '.class-preprocess-chip',
      phaseName: 'the Sample Worktable',
      onEnter: openWorktable,
      advanceTarget: '#classPreprocessDark',
      advanceAlso: ['#classPreprocessSave'],
      checkpoint: 'Drag "Dark Thr" / "Lum Thr" until the glyph is crisp, then click "Save Sample"',
      body: '<ul>' +
        '<li><b>Dark Thr</b> — pixels darker than the threshold become pure <b>white (255)</b>, i.e. shadow.</li>' +
        '<li><b>Lum Thr</b> — pixels brighter than the threshold become pure <b>white (255)</b>, i.e. bright background.</li>' +
        '</ul>' +
        '<p>Together they keep the dark strokes of the character and erase everything else. Tune both until the right-hand preview is mostly white with a crisp black glyph.</p>' +
        '<ul>' +
        '<li>The <b>thumbnail strip</b> (<code>.class-preprocess-thumb</code>) jumps between this character&rsquo;s samples.</li>' +
        '<li><b>Delete Sample</b> throws the current frame away for good — keep it for blur, glare or a hand in the way.</li>' +
        '<li><b>Save Sample</b> writes the crop and the thresholds onto this sample, then returns to the workspace.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>Deep dive · worktable shortcuts</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table">' +
        '<tr><td><kbd>F1</kbd> / <kbd>F2</kbd></td><td>Auto ROI / Manual ROI</td></tr>' +
        '<tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>Previous / next sample</td></tr>' +
        '<tr><td><kbd>Shift</kbd> + arrows</td><td>Nudge the ROI box by one pixel</td></tr>' +
        '<tr><td><kbd>D</kbd> / <kbd>S</kbd></td><td>Delete / save the sample</td></tr>' +
        '</table></div></details>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">💾</span><span>Always click <b>Save Sample</b> (or press <kbd>S</kbd>) after tuning — otherwise your thresholds are discarded when the worktable closes.</span></div>'
    }},
    {{
      chapter: 3,
      title: 'Close the Sample Worktable',
      target: '#classPreprocessCloseTop',
      fallback: ['#classPreprocessCancel', '.class-preprocess-actions'],
      placement: 'left',
      /* Still inside the worktable: Save Sample finishes the previous step
         but leaves the worktable open, so the exit is its own checkpoint
         rather than something the tour does behind the user's back. */
      phaseIf: function () {{ return !worktableOpen(); }},
      phaseTarget: '.class-preprocess-chip',
      phaseName: 'the Sample Worktable',
      advanceTarget: '#classPreprocessCloseTop',
      advanceAlso: ['#classPreprocessCancel'],
      onLeave: closeWorktable,
      checkpoint: 'Click "Close" (×) to exit the Sample Worktable',
      body: '<p><b>Close</b> (<b>×</b> in the worktable head, or the <b>Close</b> button in the footer) returns to the three-column workspace. Everything you saved stays on its character card.</p>' +
        '<ul>' +
        '<li><b>Save Sample</b> writes the crop and the brightness thresholds onto that one sample.</li>' +
        '<li><b>Close</b> is safe to press at any time — it only discards edits you have not saved.</li>' +
        '<li><kbd>Esc</kbd> closes it too, and so does clicking the dimmed backdrop.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">➡️</span><span>With every character filtered, the middle column is next: order and train.</span></div>'
    }},
    {{
      chapter: 4,
      title: 'Reveal the Advanced Training Options',
      target: '.train-card',
      fallback: ['.col-train'],
      placement: 'left',
      /* The Advanced panel starts collapsed on a fresh workspace, and the
         reading order and ink polarity the next two checkpoints talk about
         live inside it — so the step asks the user to press Advanced rather
         than unfolding it for them. */
      phaseIf: function () {{ return !advancedOpen(); }},
      phaseTarget: '#advBtn',
      phaseName: 'the <b>Advanced</b> training options',
      advanceTarget: '#advBtn',
      checkpoint: 'Click "Advanced" to unfold the training options',
      body: '<p>The middle column trains the tiny glyph classifier the verification loop uses. Two controls matter here:</p>' +
        '<ul>' +
        '<li><b>Advanced</b> unfolds the reading order, the ink polarity and the hyperparameters. It starts collapsed because the defaults already suit a small embedded model.</li>' +
        '<li><b>Train Embedded Model</b> starts the run, and stays disabled until every character has samples.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🧠</span><span>The sign itself is never the training input — each card trains on <b>single-character</b> glyphs.</span></div>'
    }},
    {{
      chapter: 4,
      title: 'Reading Direction L➔R vs R➔L',
      target: '#advReadingOrderRow',
      fallback: ['#trainAdvPanel', '.train-card'],
      placement: 'left',
      /* Reading order is an Advanced row, and the panel is collapsed by
         default — so the step re-doors onto Advanced if it is shut. */
      phaseIf: function () {{ return !advancedOpen(); }},
      phaseTarget: '#advBtn',
      phaseName: 'the <b>Advanced</b> reading-order options',
      advanceTarget: '#advReadingOrder',
      checkpoint: 'Set "Reading order" to match the way the sign is printed',
      body: '<ul>' +
        '<li><b>Reading order</b> (<code>#advReadingOrder</code>) tells the verification which end of the sign is slot 1. <code>ltr</code> reads left to right, <code>rtl</code> right to left.</li>' +
        '<li>Set it to match the way the sign is printed — it decides the order of the three slots <i>and</i> the direction the preview reports.</li>' +
        '<li>The choice is echoed as a <b>left → right</b> / <b>right → left</b> tag on the target bar, so you can check it without re-opening this panel.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🔁</span><span>If the preview reads your sign backwards, this row is the fix — not the slot values.</span></div>'
    }},
    {{
      chapter: 4,
      title: 'Global Ink Polarity &amp; Hyperparameters',
      target: '#advInkPolarityRow',
      fallback: ['#trainAdvPanel', '.train-card'],
      placement: 'left',
      phaseIf: function () {{ return !advancedOpen(); }},
      phaseTarget: '#advBtn',
      phaseName: 'the <b>Advanced</b> training options',
      advanceTarget: '#advInkPolarity',
      /* Recommended / Reset / the inline numbers are the rest of the panel,
         and touching any of them settles this checkpoint too. */
      advanceAlso: ['#advUseRec', '#advReset', '#advEpochsInline', '#advBatchInline', '#advLrInline'],
      checkpoint: 'Set "Ink Polarity" to match how your glyphs are drawn',
      body: '<ul>' +
        '<li><b>Ink Polarity</b> (<code>#advInkPolarity</code>) says which way the glyphs sit: <code>dark on light</code> for camera captures, <code>light on dark</code> for MNIST/EMNIST-style images. Get it wrong and the model learns the inverse of your data.</li>' +
        '<li><b>Transfer learning</b> keeps an EMNIST-pretrained backbone and attaches a fresh head for the characters on your sign — the fastest route to a high score on a few hundred crops. While it is on, the architecture rows are hidden: those numbers belong to the frozen backbone.</li>' +
        '<li><b>💡 Use recommended</b> fills the inline numbers in for roughly <b>800 gradient updates</b>; <b>Reset</b> puts the shipped defaults back.</li>' +
        '</ul>'
    }},
    {{
      chapter: 4,
      title: 'Train the Embedded Model',
      target: '#trainBtn',
      fallback: ['.train-card', '.col-train'],
      placement: 'left',
      advanceTarget: '#trainBtn',
      live: 'train',
      checkpoint: 'Click "Train Embedded Model" (or press ⌘/Ctrl + Enter) to start',
      body: '<p>Press <b>Train Embedded Model</b> (<kbd>⌘</kbd><kbd>Enter</kbd> / <kbd>Ctrl</kbd><kbd>Enter</kbd>). The run streams per-epoch progress into the card.</p>' +
        '<ul>' +
        '<li>When it finishes, the status pill (<code>#trainStatus</code>) turns into <b>Model Trained · tap for accuracy</b> — click it for the per-character breakdown.</li>' +
        '<li>The pill is disabled until <b>every</b> character has at least one sample.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">📊</span><span>A character that keeps coming back as <code>_background</code> needs more of that character, or a tighter ROI.</span></div>'
    }},
    {{
      chapter: 5,
      title: '3-Slot Split Preview &amp; the YES / NO Verdict',
      target: '.preview-card',
      fallback: ['.col-preview'],
      placement: 'left',
      body: '<ul>' +
        '<li>Toggle <b>Input</b> ON and point the camera at the plate: the preview splits the frame into the <b>three slots</b> and reads each one through the same int8 graph the MCU runs.</li>' +
        '<li>A <b>🟢 YES — MATCH</b> banner (<code>.signmatch.yes</code>) means all three slots agreed with the target, each cleared its confidence floor, and none came back as <code>_background</code>.</li>' +
        '<li>A <b>🔴 NO — NOT MATCH</b> banner names which slot disagreed and why: wrong character, <code>_background</code>, no glyph, or below the confidence floor.</li>' +
        '<li>Under the banner, one card per slot (<code>.signmatch-slot</code>) shows <b>WANT</b>, the character that was <b>got</b>, the 28×28 glyph, and the <code>.prob</code> percentage.</li>' +
        '<li>Toggle <b>ROI</b> to draw the crop rectangle and <b>Orig</b> to see the full masked frame.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🔁</span><span>The banner is computed from the <b>same</b> quantized graph the MCU runs, so a green verdict here predicts a green verdict on the device.</span></div>'
    }},
    {{
      chapter: 5,
      title: 'Synth Quick-Test: Match · Reversed · Mismatch',
      target: '#previewSynthMatchBtn',
      fallback: ['#previewSynthRevBtn', '.preview-synth-group', '.preview-card'],
      placement: 'left',
      /* These pills live in the preview's Local Image File source, because
         that is the input they replace. */
      advanceTarget: '#previewSynthMatchBtn',
      advanceAlso: ['#previewSynthRevBtn', '#previewSynthMismatchBtn'],
      checkpoint: 'Click "🧪 Synth Match" to build a synthetic plate from your own samples',
      body: '<p>No license plate on your desk? The three synth pills in the preview&rsquo;s <b>Local Image File</b> source build a plate out of the project&rsquo;s <b>own samples</b>, so the verdict is testable with no files on disk and no camera:</p>' +
        '<ul>' +
        '<li><b>🧪 Synth Match</b> (<code>#previewSynthMatchBtn</code>) — combine the three target characters into a plate the model should call <b>🟢 YES</b>.</li>' +
        '<li><b>🔀 Synth Reversed</b> (<code>#previewSynthRevBtn</code>) — combine them in reverse order. If it also reads green, your <b>reading order</b> is set the wrong way round.</li>' +
        '<li><b>❌ Synth Mismatch</b> (<code>#previewSynthMismatchBtn</code>) — swap in a wrong character, which must produce <b>🔴 NO</b> and name the slot that disagreed.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">🧪</span><span>These pills are disabled until the sign is set and a model has been trained — they need a target and a verdict to test.</span></div>'
    }},
    {{
      chapter: 5,
      title: 'Live Dark / Lum Sliders &amp; the Slot Probabilities',
      target: '.preview-card',
      fallback: ['.col-preview'],
      placement: 'left',
      body: '<ul>' +
        '<li>The live <b>Dark</b> / <b>Lum</b> sliders under the image (<code>#previewDarkSlider</code>, <code>#previewLumSlider</code>) re-tune the brightness mask while you watch — they should match the thresholds you saved in the worktable.</li>' +
        '<li>Each <code>.signmatch-slot</code> shows the winning character&rsquo;s <code>.prob</code> percentage, so you can see <b>which</b> slot is marginal rather than only that the plate failed.</li>' +
        '<li>The row above the image (<code>#previewContextBar</code>) picks where the pane reads from — a <b>Webcam</b>, the <b>Device</b> port plus <b>Refresh</b>, or <b>📁 Upload Image…</b> — independently of the capture drawer.</li>' +
        '</ul>' +
        '<p class="tm-tour-muted">Sanity-check the preview before exporting: a mask that looks wrong here will look exactly as wrong on the device.</p>'
    }},
    {{
      chapter: 5,
      title: 'Export the Car License Firmware Package',
      target: '#exportBtn',
      fallback: ['.preview-card'],
      placement: 'bottom',
      advanceTarget: '#exportBtn',
      checkpoint: 'Click "Export Model" to build the 5-file deployment package',
      body: '<p>Click <b>Export Model</b>, then copy the <b>5 files</b> next to your <code>TFLite_Sign.ino</code> sketch:</p>' +
        '<ul>' +
        '<li><code>tm_model_data.cpp</code> · <code>tm_model_data.h</code> — the quantized weights as a C array.</li>' +
        '<li><code>model_resolver.h</code> — the MicroInterpreter op resolver.</li>' +
        '<li><code>model_settings.cpp</code> · <code>model_settings.h</code> — the labels, the geometry, the <b>reading order</b>, the target sign and the OOD gates as <code>constexpr</code>.</li>' +
        '</ul>' +
        '<details class="tm-tour-details"><summary>🖥️ Serial-Monitor live tuning (no reflash)</summary><div class="tm-tour-details-body">' +
        '<table class="tm-tour-table">' +
        '<tr><td><code>get mask</code></td><td>Print the current Dark / Lum thresholds.</td></tr>' +
        '<tr><td><code>set mask &lt;dark&gt; &lt;lum&gt;</code></td><td>Change the brightness mask live.</td></tr>' +
        '<tr><td><code>get ood</code></td><td>Print the four OOD gates.</td></tr>' +
        '<tr><td><code>set ood &lt;spmin&gt; &lt;spmax&gt; &lt;mpmin&gt; &lt;ermax&gt;</code></td><td>Retune the OOD gate live.</td></tr>' +
        '<tr><td><code>mode infer</code> / <code>mode gray</code></td><td>Switch between inference and raw grayscale streaming.</td></tr>' +
        '</table></div></details>' +
        '<p class="tm-tour-muted">Re-export after any change to the target sign, the reading order or the thresholds.</p>'
    }},
    {{
      chapter: 5,
      title: 'You are ready to build 🎉',
      target: null,
      fallback: [],
      placement: 'auto',
      body: '<p>That is the whole <b>3-character car license</b> loop:</p>' +
        '<ul>' +
        '<li>Declare the sign in the three slots, then capture <b>one glyph per crop</b> for each character.</li>' +
        '<li>Pin the plate with a Manual ROI, threshold the noise away, and set the reading order.</li>' +
        '<li>Train the tiny glyph classifier, preview until the verdict is reliably green, then export.</li>' +
        '</ul>' +
        '<div class="tm-tour-tip"><span class="tm-tour-tip-ico">📖</span><span>Re-open this guide any time from <b>📖 Tutorial</b>.</span></div>'
    }}
  ];

  /* ── Curriculum selection ────────────────────────────────────────────
     Each project type documents a genuinely different workflow, so each
     gets its own step list AND its own "already seen" flag: dismissing
     the Image tour must not silence the OCR tour the first time that
     project is opened.  The type is read from the live STATE at start
     time, so opening a .tmproj of another type re-arms the right guide. */
  var PHASES_BY_TYPE = {{
    image: [
      {{ n: 1, label: 'Workspace & Setup' }},
      {{ n: 2, label: 'Classes & Samples' }},
      {{ n: 3, label: 'Preprocessing' }},
      {{ n: 4, label: 'Training' }},
      {{ n: 5, label: 'Preview & OOD' }},
      {{ n: 6, label: 'Export & Deploy' }}
    ],
    ocr: [
      {{ n: 1, label: 'Setup & Characters' }},
      {{ n: 2, label: 'Samples & Cropping' }},
      {{ n: 3, label: 'Glyph Preprocessing' }},
      {{ n: 4, label: 'Training' }},
      {{ n: 5, label: 'Preview & Export' }}
    ],
    sign_ocr: [
      {{ n: 1, label: 'Welcome & Target Sign' }},
      {{ n: 2, label: 'Samples & ROI' }},
      {{ n: 3, label: 'Glyph Isolation' }},
      {{ n: 4, label: 'Order & Training' }},
      {{ n: 5, label: 'Preview & Export' }}
    ]
  }};

  var CURRICULA = {{
    image: IMAGE_STEPS,
    ocr: OCR_STEPS,
    sign_ocr: SIGN_OCR_STEPS
  }};

  function activeProjectType() {{
    var t = '';
    try {{ t = String((STATE && STATE.project_type) || '').toLowerCase(); }} catch (err) {{ t = ''; }}
    /* A classified import is an image project in every way that matters
       to the tour, and an unknown value must still land somewhere sane. */
    if (t === 'ocr' || t === 'sign_ocr') return t;
    return 'image';
  }}

  function selectCurriculum() {{
    TOUR_TYPE = activeProjectType();
    STEPS = CURRICULA[TOUR_TYPE] || IMAGE_STEPS;
    PHASES = PHASES_BY_TYPE[TOUR_TYPE] || PHASES_BY_TYPE.image;
    SEEN_KEY = 'tm_tour_seen_' + TOUR_TYPE + '_v1';
    AUTOSKIP_KEY = 'tm_tour_autoskip_' + TOUR_TYPE + '_v1';
    return TOUR_TYPE;
  }}

  var tourOpen = false;
  var tourIndex = 0;
  var uiBefore = null;
  var placeTimer = null;
  /* The step the ring currently on screen belongs to.  Placement waits on
     a frame, so this is the only way to tell a ring that has been
     re-seated for the new step from the old one still sitting where the
     last step left it — the headless walk reads it through `place()`. */
  var placedIndex = -1;
  /* Click-to-advance state.  `advState` mirrors what the card should say
     about the spotlighted control: nothing, "click it", or "it is
     disabled — use Next". */
  var advState = 'off';
  /* `advTarget` is the node the listeners sit on; `advControl` is the
     control inside it that actually carries the affordance.  Both are kept
     so `armTargetAdvance` can tell a binding that is still valid from one
     `render()` has orphaned by replacing the card it lived in. */
  var advTarget = null;
  var advControl = null;
  var advClick = null;
  var advChange = null;
  var advPress = null;
  var advTimer = null;
  /* Further controls that also complete the step (see `advanceAlso`). */
  var advExtra = [];
  /* Geometry of the last `placeTour`, kept so the arrow and the beacon can
     be re-aimed without re-measuring the whole workspace. */
  var placeMark = null;
  /* Phase B — "hands-on" mode.  `orbMode` means the card and the dim layer
     are OFF screen and the floating orb is on: the user is working, and
     from that moment on nothing in the workspace may move the tour.
     `orbSnapshot` is what the live state looked like the moment they
     started, so "you have done the step's own work" is a comparison
     rather than a guess.  `orbFlash*` shows a short acknowledgement
     ("🛠️ Hands-on mode") before the live badge takes over. */
  var orbMode = false;
  var orbSnapshot = null;
  /* The hand-over froze nothing yet: the step's own screen is still shut,
     so its live values are not on screen to be the baseline. */
  var orbBaselineDue = false;
  var orbWatch = null;
  var orbSettle = null;
  var orbFlashText = '';
  var orbFlashUntil = 0;
  var orbDragged = false;
  var orbEventsBound = false;
  /* Set the moment the user touches the control a step is about — pressing
     it to hand the workspace over, or working it once the pill is on.  It
     is what makes the pill turn green on steps whose work produces no
     readable live value (a closed modal, a collapsed disclosure), so no
     step can leave the user staring at a pill that never acknowledges
     them.  `liveSignature` can only say "the numbers moved"; this says
     "the user did the thing". */
  var orbActionSatisfied = false;

  function autoskipOn() {{
    try {{ return window.localStorage.getItem(AUTOSKIP_KEY) === '1'; }} catch (err) {{ return false; }}
  }}

  function markSeen() {{
    try {{ window.localStorage.setItem(SEEN_KEY, '1'); }} catch (err) {{}}
  }}

  function snapshotUi() {{
    var snap = {{ advanced: false, settings: false, worktable: false, previewCfg: false }};
    try {{
      if (typeof trainAdvancedOpen !== 'undefined') snap.advanced = !!trainAdvancedOpen;
      if (typeof classPreprocessOpen !== 'undefined') snap.worktable = !!classPreprocessOpen;
      if (typeof previewSettingsOpen !== 'undefined') snap.previewCfg = !!previewSettingsOpen;
      var modal = byId('globalSettingsModal');
      snap.settings = !!(modal && modal.classList.contains('open'));
    }} catch (err) {{}}
    return snap;
  }}

  function restoreUi(snap) {{
    if (!snap) return;
    if (!snap.settings) closeSettingsModal();
    if (!snap.worktable) closeWorktable();
    if (!snap.advanced) setAdvancedPanel(false);
    if (!snap.previewCfg && typeof previewSettingsOpen !== 'undefined') {{
      try {{
        previewSettingsOpen = false;
        if (typeof renderPreviewSettings === 'function') renderPreviewSettings();
      }} catch (err) {{}}
    }}
  }}

  /* ── Explain & Point ─────────────────────────────────────────────────
     Phase A of every step: the card spotlights one real control and says
     what to do there.  Using that control is the answer to the card, so
     goStep() arms a listener on the spotlight target — but the answer is
     NOT "skip ahead":
       · a button / chip / toolbar keeps its native click (which opens the
         drawer or modal it belongs to) and, on a 180 ms settle beat, the
         card COLLAPSES into the floating orb;
       · a field, select or slider does the same on `change`, debounced
         350 ms, because a text box fires an event per keystroke and the
         user may keep dragging for a while;
       · a control that cannot be used YET (Train before samples, Export
         before training) says so in the card instead of silently
         swallowing the click — a disabled button never dispatches one, so
         nothing else could tell the user why it did nothing.
     Nothing here moves the tour on.  The step is left when the user says
     so — by clicking the orb — which is what keeps the tutorial from
     teleporting out from under someone taking their twentieth photo. */
  var ADV_CLICK_SEL = 'button, [role="button"], a[href], input, select, textarea, ' +
    '.class-preprocess-chip, .chip, .tm-toolbar-btn, .tm-action-btn, .ocr-empty-cta';
  /* Gestures that are a press rather than a click: the Hold-to-Capture
     button captures on pointerdown, and the ROI box is dragged from one. */
  var ADV_PRESS_SEL = '[id^="sourceHold-"], .roi-overlay, #classPreprocessStage';
  var ADV_SETTLE_MS = 180;
  var ADV_INPUT_MS = 350;
  /* Beat between the orb being clicked and the next card appearing: long
     enough to read the acknowledgement, short enough not to feel laggy. */
  var ORB_ADVANCE_MS = 420;

  function unbindTargetAdvance() {{
    if (advTimer) {{ window.clearTimeout(advTimer); advTimer = null; }}
    var drop = function (el, fn, type) {{
      if (!el || !fn) return;
      try {{ el.removeEventListener(type, fn, true); }} catch (err) {{}}
    }};
    if (advClick) drop(advTarget, advClick, 'click');
    if (advChange) {{
      drop(advTarget, advChange, 'change');
      drop(advTarget, advChange, 'input');
    }}
    if (advPress) {{
      drop(advTarget, advPress, 'pointerdown');
      drop(advTarget, advPress, 'mousedown');
    }}
    for (var i = 0; i < advExtra.length; i++) {{
      var binding = advExtra[i];
      drop(binding.el, binding.fn, binding.type || 'click');
    }}
    advExtra = [];
    advTarget = null; advControl = null;
    advClick = null; advChange = null; advPress = null;
  }}

  /* The user just used the control the card was pointing at.  That is the
     answer to "explain and point": stop explaining, get the card and the
     dim layer off the screen, and start carrying the live state instead.
     The settle beat lets the control finish its own job — opening a
     drawer, committing a value — before the orb reads the result. */
  function collapseIntent(delay) {{
    if (!tourOpen || orbMode) return;
    /* The user just used the control the card pointed at.  That is the step
       done as far as the pill is concerned, whatever the step's live value
       happens to be — record it before the hand-over so the pill can say so
       the moment it appears. */
    orbActionSatisfied = true;
    if (advTimer) window.clearTimeout(advTimer);
    advTimer = window.setTimeout(function () {{ advTimer = null; enterOrbMode(); }}, delay || ADV_SETTLE_MS);
  }}

  /* The one way a step is ever left.  Called from the orb, from Next →,
     from Skip step →, from a chapter chip and from the arrow keys — never
     from a workspace event. */
  function advanceLater(delay) {{
    if (advTimer) window.clearTimeout(advTimer);
    flashOrb('✅ Moving on&hellip;');
    advTimer = window.setTimeout(function () {{
      advTimer = null;
      if (!tourOpen) return;
      if (tourIndex >= STEPS.length - 1) endTour('finish');
      else goStep(tourIndex + 1);
    }}, delay);
  }}

  /* A step names what to spotlight and — when the spotlight is a
     container — the control inside it that really carries the affordance
     (`#trainBtn` inside the training card).  Both the advance binding and
     the click beacon read the pair from here, so the mark can never point
     at something other than what a click there would do.

     A step may also declare that the screen it lives in is SHUT — a
     closed modal, a collapsed panel (`phaseTarget`, optionally with
     `phaseIf`).  Nothing is forced open in that case: the step points at
     the control that opens the screen, and the click carries the user in
     and then hands the workspace over (see `collapseIntent`).  Same step
     index, two phases, no teleporting. */
  function resolveStepTargets(step) {{
    var out = {{ target: null, control: null, phase: false }};
    if (!step) return out;
    /* A step that documents something inside Settings' collapsed Advanced
       disclosure must unfold it before anything is measured — this runs on
       every arm and every re-place, so a re-render cannot leave the group
       shut while the card points into it. */
    ensureStepDisclosure(step);
    var shut = (typeof step.phaseIf === 'function') && !!step.phaseIf();
    var el = shut ? null : firstVisible([step.target].concat(step.fallback || []));
    if (!el && step.phaseTarget) {{
      var door = firstVisible([step.phaseTarget]);
      if (door) return {{ target: door, control: door, phase: true }};
    }}
    if (!el) el = firstVisible([step.target].concat(step.fallback || []));
    if (!el) return out;
    out.target = el;
    out.control = el;
    if (step.advanceTarget) {{
      var named = null;
      try {{ named = document.querySelector(step.advanceTarget); }} catch (err) {{ named = null; }}
      if (named && isVisibleNode(named)) out.control = named;
    }}
    return out;
  }}

  /* ── Phase B · the floating orb ("hands-on" mode) ─────────────────────
     The card above explains a step and points at one control.  The moment
     the user touches it, this takes over.  Three things it exists for:
       · the dim layer and the card come OFF screen, so every button,
         slider and modal is reachable while the guide is still running;
       · the pill carries the LIVE state of the job — sample counts,
         thresholds, the target sign, training — instead of a static
         sentence the user has already read;
       · it never advances anything by itself.  Clicking it is the user
         saying "done", and `?` brings the card back to re-read.
     Live state is read through two channels, because the workspace
     changes in two ways: DOM events for what the user does (input,
     change, click), and a poll for what the backend pushes back
     (STATE.counts after a capture, a finished training run) which arrives
     with no event at all. */
  var ORB_READY_BADGE = '✅ Step Ready!';
  var LIVE_SEL = '#classPreprocessDark, #classPreprocessDarkNum, #classPreprocessLum, ' +
    '#classPreprocessLumNum, #classPreprocessMode, #signSeqSlot-0, #signSeqSlot-1, ' +
    '#signSeqSlot-2, #advReadingOrder, #advInkPolarity, #previewDarkSlider, ' +
    '#previewDarkNum, #previewLumSlider, #previewLumNum';

  function orbNode() {{ return byId('tmTourOrb'); }}

  function orbEsc(s) {{
    return String(s === undefined || s === null ? '' : s).replace(/[&<>"]/g, function (c) {{
      return {{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }}[c];
    }});
  }}

  /* Every class's sample count in one number.  The rejection class the
     export adds is not something the user collects, so it never shows up
     in the count the orb reads back to them. */
  function samplesTotal() {{
    var total = 0;
    try {{
      var counts = (STATE && STATE.counts) || {{}};
      Object.keys(counts).forEach(function (k) {{
        if (k === '{BACKGROUND_CLASS_NAME}') return;
        total += Number(counts[k]) || 0;
      }});
    }} catch (err) {{}}
    return total;
  }}

  function trainDone() {{
    var el = byId('trainStatus');
    if (el && /Model Trained/.test(String(el.textContent || ''))) return true;
    return !!(STATE && STATE.trainClassAccuracies);
  }}

  /* The values a step of kind `thresholds` / `signseq` is about, as one
     comparable string — every slider, select and slot the tour watches. */
  function liveValues() {{
    var out = [];
    try {{
      Array.prototype.forEach.call(document.querySelectorAll(LIVE_SEL), function (el) {{
        out.push(String(el.id || el.className || '?') + '=' + String(el.value === undefined ? '' : el.value));
      }});
    }} catch (err) {{}}
    return out.join('|');
  }}

  /* What "this step's own work" looks like as a value.  A step opts in by
     declaring `live`: `samples` (its sample count), `train` (whether a run
     has finished), `thresholds` (every slider / select the worktable and
     the preview own) or `signseq` (the three target-sign slots).  Only
     those kinds can flip the badge green — that is what stops an unrelated
     click from congratulating the user. */
  function liveSignature(step) {{
    var kind = (step && step.live) || '';
    if (kind === 'samples') return 'n=' + samplesTotal();
    if (kind === 'train') {{
      /* A model that already exists reads "Model Trained" the moment the
         workspace loads, so the status line alone cannot say whether the
         run the user just started has finished — the per-class accuracies
         the backend pushes at the end of a run can. */
      var accs = (STATE && STATE.trainClassAccuracies) || null;
      return 'trained=' + (trainDone() ? '1' : '0') + ';acc=' +
        (accs ? JSON.stringify(accs) : '');
    }}
    if (kind === 'thresholds' || kind === 'signseq') return liveValues();
    return '';
  }}

  /* What "the user has done nothing yet" looked like when the workspace
     was handed over.  A step whose screen is still shut cannot be frozen
     yet: the card presses its door on the next tick, and the panel's own
     arrival would otherwise read as the user's work — a green badge
     before a single slider has been touched.  So the baseline waits for
     the screen the step documents, and takes its values then. */
  function armOrbBaseline() {{
    var step = STEPS[tourIndex] || {{}};
    if (typeof step.phaseIf === 'function') {{
      var open = true;
      try {{ open = !step.phaseIf(); }} catch (err) {{ open = true; }}
      if (!open) {{ orbBaselineDue = true; return; }}
    }}
    orbBaselineDue = false;
    orbSnapshot = liveSignature(step);
  }}

  /* The step's live headline plus whether its own work looks done.
     "Done" has two independent sources, and either is enough: the user
     touched the control the step is about (`orbActionSatisfied`), or the
     step's live values moved off the baseline frozen at hand-over.  Steps
     whose work leaves no readable value — opening a modal, unfolding a
     disclosure — have no signature at all, so without the first source
     their pill could never turn green. */
  function orbState() {{
    var step = STEPS[tourIndex] || {{}};
    var kind = step.live || '';
    var out = {{ text: '', ready: false }};
    var sig = liveSignature(step);
    var moved = !!sig && orbSnapshot !== null && sig !== orbSnapshot;
    out.ready = orbActionSatisfied || moved;
    if (kind === 'samples') {{
      var counts = (STATE && STATE.counts) || {{}};
      var parts = [];
      Object.keys(counts).forEach(function (k) {{
        if (k === '{BACKGROUND_CLASS_NAME}') return;
        parts.push(orbEsc(k) + ': ' + (Number(counts[k]) || 0));
      }});
      out.text = 'Collecting Samples' + (parts.length ? ' (' + parts.join(', ') + ')' : '');
    }} else if (kind === 'thresholds') {{
      out.text = 'Adjusting ROI &amp; Thresholds';
    }} else if (kind === 'signseq') {{
      var seq = [];
      for (var i = 0; i < 3; i++) {{
        var slot = byId('signSeqSlot-' + i);
        if (slot) seq.push(orbEsc(slot.value || '·'));
      }}
      out.text = 'Target Sign: ' + (seq.join(' ') || 'not set');
    }} else if (kind === 'train') {{
      out.text = trainDone() ? 'Model Trained' : 'Train Your Model';
    }} else {{
      out.text = orbEsc(stripTourTags(step.title || 'This step'));
    }}
    return out;
  }}

  function stripTourTags(html) {{
    return String(html).replace(/<[^>]*>/g, '').replace(/&amp;/g, '&');
  }}

  /* Steps that document a screen — Settings, the drawer, the worktable —
     keep that screen open while the user works, so the pill has to say so
     instead of pretending the step is over. */
  function orbScreenHint(step) {{
    if (!step || typeof step.phaseIf !== 'function') return '';
    var open = false;
    try {{ open = !step.phaseIf(); }} catch (err) {{ open = false; }}
    if (!open) return '';
    var name = stripTourTags(step.phaseName || 'the panel');
    return 'Close ' + orbEsc(name) + ' &amp; click here for Next Step &rarr;';
  }}

  /* A short-lived acknowledgement (hands-on mode, moving on).  It wins
     over the live badge for a moment, then gets out of the way — the pill
     must not be a place where text piles up. */
  function flashOrb(text) {{
    orbFlashText = String(text || '');
    orbFlashUntil = orbFlashText ? (Date.now() + 1800) : 0;
    renderOrb();
  }}

  function renderOrb() {{
    var orb = orbNode();
    if (!orb) return;
    if (!tourOpen || !orbMode) {{
      orb.classList.remove('on', 'ready', 'dragging');
      return;
    }}
    var step = STEPS[tourIndex] || {{}};
    /* A baseline that was still waiting for its screen now has one. */
    if (orbBaselineDue) armOrbBaseline();
    var st = orbState();
    var flashing = !!orbFlashText && Date.now() < orbFlashUntil;
    var textNode = byId('tmTourOrbText');
    var goNode = byId('tmTourOrbGo');
    if (textNode) {{
      if (flashing) {{
        textNode.innerHTML = orbFlashText;
      }} else {{
        var hint = st.ready ? 'click here to continue &rarr;' : (orbScreenHint(step) || 'Click here for Next Step &rarr;');
        textNode.innerHTML = (st.ready ? ORB_READY_BADGE + ' ' : '') +
          '<b>Step ' + (tourIndex + 1) + '/' + STEPS.length + ':</b> ' + st.text + ' · ' + hint;
      }}
    }}
    if (goNode) goNode.innerHTML = st.ready ? '&#9989; Done &mdash; Next Step &rarr;' : 'Next Step &rarr;';
    orb.classList.add('on');
    orb.classList.toggle('ready', !!st.ready && !flashing);
  }}

  /* Does this event target belong to the control the current step is
     about?  Every selector the step mentions counts — the spotlight, its
     fallbacks, the named inner control and the `advanceAlso` group — so
     "the user worked the thing this step documents" is decided the same
     way the Phase-A binding decided it. */
  function matchesStepControl(target) {{
    if (!target || !target.closest) return false;
    var step = STEPS[tourIndex] || {{}};
    var sels = [];
    if (step.target) sels.push(step.target);
    (step.fallback || []).forEach(function (s) {{ sels.push(s); }});
    if (step.advanceTarget) sels.push(step.advanceTarget);
    (step.advanceAlso || []).forEach(function (s) {{ sels.push(s); }});
    for (var i = 0; i < sels.length; i++) {{
      try {{ if (target.closest(sels[i])) return true; }} catch (err) {{}}
    }}
    return false;
  }}

  function orbEventHook(ev) {{
    if (!orbMode) return;
    var t = ev && ev.target;
    /* The pill's own buttons are not workspace adjustments. */
    if (t && t.closest && t.closest('#tmTourOrb')) return;
    /* Working the control the step is about is the step being done, even
       when the work produces no live value to compare against. */
    if (matchesStepControl(t)) orbActionSatisfied = true;
    /* Let the workspace finish reacting to the event first (opening the
       drawer, re-rendering the worktable) — then read the result. */
    if (orbSettle) window.clearTimeout(orbSettle);
    orbSettle = window.setTimeout(function () {{ orbSettle = null; renderOrb(); }}, 150);
  }}

  function startOrbWatch() {{
    if (!orbEventsBound) {{
      orbEventsBound = true;
      document.addEventListener('input', orbEventHook, true);
      document.addEventListener('change', orbEventHook, true);
      document.addEventListener('click', orbEventHook, true);
    }}
    if (!orbWatch) {{
      orbWatch = window.setInterval(function () {{
        if (!orbMode) {{ stopOrbWatch(); return; }}
        renderOrb();
      }}, 900);
    }}
  }}

  function stopOrbWatch() {{
    if (orbWatch) {{ window.clearInterval(orbWatch); orbWatch = null; }}
    if (orbSettle) {{ window.clearTimeout(orbSettle); orbSettle = null; }}
    if (orbEventsBound) {{
      orbEventsBound = false;
      document.removeEventListener('input', orbEventHook, true);
      document.removeEventListener('change', orbEventHook, true);
      document.removeEventListener('click', orbEventHook, true);
    }}
  }}

  /* Every step begins in Phase A — card and ring on, pill off — so leaving
     a step's hands-on mode is a hard reset rather than a style toggle. */
  function resetOrb() {{
    orbMode = false;
    orbSnapshot = null;
    orbBaselineDue = false;
    orbActionSatisfied = false;
    orbFlashText = '';
    orbFlashUntil = 0;
    orbDragged = false;
    stopOrbWatch();
    var orb = orbNode();
    if (orb) {{
      orb.classList.remove('on', 'ready', 'dragging');
      orb.style.left = '';
      orb.style.top = '';
      orb.style.right = '';
      orb.style.bottom = '';
    }}
  }}

  function enterOrbMode() {{
    if (!tourOpen || orbMode) return;
    orbMode = true;
    /* From here on nothing in the workspace may move the tour: the user is
       working, and only the pill has a say. */
    unbindTargetAdvance();
    advState = 'off';
    hideTourMarks();
    var card = byId('tmTourCard');
    var spot = byId('tmTourSpotlight');
    if (card) card.classList.remove('on');
    if (spot) spot.classList.remove('on', 'dim');
    armOrbBaseline();
    flashOrb('🛠️ Hands-on mode — the card is out of your way. Work at your own pace.');
    startOrbWatch();
    if (typeof syncFrameHeight === 'function') {{
      try {{ syncFrameHeight(); }} catch (err) {{}}
    }}
  }}

  /* `?` on the pill: back to Phase A for the SAME step, card and all. */
  function leaveOrbMode() {{
    if (!tourOpen || !orbMode) return;
    orbMode = false;
    stopOrbWatch();
    orbFlashText = '';
    orbFlashUntil = 0;
    var orb = orbNode();
    if (orb) orb.classList.remove('on', 'ready', 'dragging');
    var card = byId('tmTourCard');
    if (card) card.classList.add('on');
    armTargetAdvance(true);
    schedulePlace();
  }}

  /* Bound once: the pill's own controls, and dragging it out of the way.
     The whole pill advances the tour, so a stray click can never do
     anything else. */
  function bindOrb() {{
    var orb = orbNode();
    if (!orb || orb.__tmOrbBound) return;
    orb.__tmOrbBound = true;
    orb.addEventListener('click', function (ev) {{
      if (orbDragged) {{ orbDragged = false; return; }}
      var t = ev && ev.target;
      if (t && t.closest && t.closest('#tmTourOrbHelp')) {{ leaveOrbMode(); return; }}
      advanceLater(ORB_ADVANCE_MS);
    }});
    var drag = null;
    orb.addEventListener('pointerdown', function (ev) {{
      if (ev.button !== 0) return;
      var t = ev.target;
      if (t && t.closest && t.closest('button')) return;
      var r = orb.getBoundingClientRect();
      drag = {{ x: ev.clientX, y: ev.clientY, left: r.left, top: r.top }};
      try {{ orb.setPointerCapture(ev.pointerId); }} catch (err) {{}}
    }});
    orb.addEventListener('pointermove', function (ev) {{
      if (!drag) return;
      var dx = ev.clientX - drag.x;
      var dy = ev.clientY - drag.y;
      if (!orbDragged && (Math.abs(dx) + Math.abs(dy)) < 5) return;
      orbDragged = true;
      orb.classList.add('dragging');
      var r = orb.getBoundingClientRect();
      var m = 8;
      var left = Math.max(m, Math.min(window.innerWidth - r.width - m, drag.left + dx));
      var top = Math.max(m, Math.min(window.innerHeight - r.height - m, drag.top + dy));
      orb.style.left = Math.round(left) + 'px';
      orb.style.top = Math.round(top) + 'px';
      orb.style.right = 'auto';
      orb.style.bottom = 'auto';
    }});
    var endDrag = function (ev) {{
      if (!drag) return;
      drag = null;
      orb.classList.remove('dragging');
      try {{ orb.releasePointerCapture(ev.pointerId); }} catch (err) {{}}
    }};
    orb.addEventListener('pointerup', endDrag);
    orb.addEventListener('pointercancel', endDrag);
  }}

  function armTargetAdvance(force) {{
    var step = STEPS[tourIndex];
    if (!force) {{
      /* Already armed, or an advance is already on the settle beat:
         re-arming would drop the user's click on the floor. */
      if (advTimer) return;
      if (advTarget) {{
        /* `render()` rebuilds the class column from scratch
           (`#classes.innerHTML = ''`), which replaces the very node the
           listeners sit on — so a binding that still looks armed is in fact
           dead.  A detached node, or one that no longer matches what the
           step resolves to now, is stale: fall through and bind the fresh
           node instead of returning. */
        var now = resolveStepTargets(step);
        if (advTarget.isConnected && now.target === advTarget && now.control === advControl) return;
      }}
      if (!step) return;
    }}
    unbindTargetAdvance();
    /* Hands-on mode owns the tour from here: the card is off screen, so
       there is nothing left to arm. */
    if (!tourOpen || orbMode) {{ setAdvState('off'); return; }}
    if (!step) {{ setAdvState('off'); return; }}
    var found = resolveStepTargets(step);
    var el = found.target;
    if (!el) {{ setAdvState('off'); return; }}
    var control = found.control;
    var tag = String(control.tagName || '').toUpperCase();

    advTarget = el;
    advControl = control;
    if (control.disabled) {{ setAdvState('disabled'); return; }}

    advChange = function (ev) {{
      var t = ev && ev.target;
      if (t && t.disabled) return;
      collapseIntent(ADV_INPUT_MS);
    }};
    advClick = function (ev) {{
      var t = ev && ev.target;
      var hit = (t && t.closest) ? t.closest(ADV_CLICK_SEL) : null;
      /* Only a real control counts: a click on label text or on the
         card's padding is not an instruction to start working. */
      if (!hit || hit.disabled) return;
      collapseIntent(ADV_SETTLE_MS);
    }};
    /* Manual ROI drag and Hold-to-Capture never fire a click: the gesture
       that means "I am working in here" is the press. */
    advPress = function (ev) {{
      var t = ev && ev.target;
      if (!t || !t.closest) return;
      var hit = t.closest(ADV_PRESS_SEL);
      if (!hit || hit.disabled) return;
      collapseIntent(ADV_SETTLE_MS);
    }};

    if (found.phase) {{
      /* The screen this step documents is still shut.  Landing on the
         control that opens it and pressing it is the lesson: the press
         opens the screen and the card collapses onto the pill, leaving
         the user inside the screen they just opened — same step, no
         teleport, and nothing to read while they work. */
      var door = function (ev) {{
        var t = ev && ev.target;
        var hit = (t && t.closest) ? t.closest(ADV_CLICK_SEL) : null;
        if (!hit || hit.disabled) return;
        collapseIntent(ADV_SETTLE_MS);
      }};
      el.addEventListener('click', door, true);
      advClick = door;
      setAdvState('phase');
      return;
    }}

    /* Bind click AND input/change on the spotlighted node whatever it is.
       The spotlight is usually a container (`#settingsCameraGroup`,
       `.class-preprocess-mode-row`, `.tm-action-toolbar`) whose real control
       is a descendant, so gating on `control.tagName` left every one of
       those silent; the cue still reads as "input" for a real form field,
       which is the only place the distinction is visible to the user. */
    el.addEventListener('click', advClick, true);
    el.addEventListener('input', advChange, true);
    el.addEventListener('change', advChange, true);
    el.addEventListener('pointerdown', advPress, true);
    el.addEventListener('mousedown', advPress, true);
    setAdvState((tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') ? 'input' : 'click');

    /* A step may name further controls that also mean "working inside this
       screen" — the worktable's Save Sample and Close, or a drawer's three
       capture affordances.  Each entry is a selector, possibly a
       comma-separated group (`'[id^="sourceCapture-"], [id^="uploadPick-"]'`),
       so every match is bound rather than only the first. */
    (step.advanceAlso || []).forEach(function (sel) {{
      var nodes = [];
      try {{ nodes = Array.prototype.slice.call(document.querySelectorAll(sel)); }}
      catch (err) {{ nodes = []; }}
      nodes.forEach(function (extra) {{
        if (!extra || extra.disabled || !isVisibleNode(extra)) return;
        try {{
          extra.addEventListener('click', advClick, true);
          extra.addEventListener('input', advChange, true);
          extra.addEventListener('change', advChange, true);
          advExtra.push({{ el: extra, fn: advClick, type: 'click' }});
          advExtra.push({{ el: extra, fn: advChange, type: 'input' }});
          advExtra.push({{ el: extra, fn: advChange, type: 'change' }});
        }} catch (err) {{ return; }}
      }});
    }});
  }}

  function setAdvState(state) {{
    if (advState === state) return;
    advState = state;
    updateTourCue();
  }}

  /* ---- directional marks: the arrow and the click beacon -------------
     The spotlight says WHICH control, but on a three-column workspace a
     ring alone still leaves the eye hunting for it.  Two pure-decoration
     marks close that gap: an arrow sitting in the gap between the card
     and the ring, its tip facing the ring, and — while the step is
     waiting for a click — a pulsing beacon beside the very control the
     step names.  Both are `pointer-events: none`, both are hidden for
     steps with no target and when the tour closes. */
  var ARROW_HALF = 17;
  var BEACON_HALF = 13;

  function hideTourMarks() {{
    var arrow = byId('tmTourArrow');
    var beacon = byId('tmTourBeacon');
    if (arrow) arrow.classList.remove('on');
    if (beacon) beacon.classList.remove('on');
  }}

  function paintTourMarks() {{
    var arrow = byId('tmTourArrow');
    var beacon = byId('tmTourBeacon');
    if (!arrow || !beacon) return;
    var m = placeMark;
    if (!tourOpen || !m || !m.target) {{ hideTourMarks(); return; }}

    /* Aim from whichever side the card ended up on, and sit midway in the
       gap so the arrow reads as the link between the two — never closer
       than 22px to the ring, in case the card was clamped on top of it. */
    var deg = 0;
    var ax = 0;
    var ay = 0;
    if (m.mode === 'right') {{
      deg = -90;
      ax = Math.max(m.sLeft + m.sw + 22, (m.sLeft + m.sw + m.cardLeft) / 2);
      ay = m.sTop + (m.sh / 2);
    }} else if (m.mode === 'left') {{
      deg = 90;
      ax = Math.min(m.sLeft - 22, (m.cardLeft + m.cw + m.sLeft) / 2);
      ay = m.sTop + (m.sh / 2);
    }} else if (m.mode === 'bottom') {{
      deg = 0;
      ay = Math.max(m.sTop + m.sh + 22, (m.sTop + m.sh + m.cardTop) / 2);
      ax = m.sLeft + (m.sw / 2);
    }} else {{
      deg = 180;
      ay = Math.min(m.sTop - 22, (m.cardTop + m.ch + m.sTop) / 2);
      ax = m.sLeft + (m.sw / 2);
    }}
    ax = Math.max(m.margin + ARROW_HALF, Math.min(m.vw - m.margin - ARROW_HALF, ax));
    ay = Math.max(m.margin + ARROW_HALF, Math.min(m.vh - m.margin - ARROW_HALF, ay));
    arrow.style.left = Math.round(ax) + 'px';
    arrow.style.top = Math.round(ay) + 'px';
    arrow.style.transform = 'translate(-50%, -50%) rotate(' + deg + 'deg)';
    arrow.classList.add('on');

    if (advState !== 'click' && advState !== 'input' && advState !== 'phase') {{ beacon.classList.remove('on'); return; }}
    var control = resolveStepTargets(STEPS[tourIndex]).control;
    if (!control || !isVisibleNode(control)) {{ beacon.classList.remove('on'); return; }}
    var r = control.getBoundingClientRect();
    var bx = r.left + (r.width / 2);
    /* The hand points up, so it sits under the control it is asking for;
       when the card is already below, it moves above instead of landing
       on the card. */
    var by = (m.mode === 'bottom') ? (r.top - BEACON_HALF - 2) : (r.bottom + BEACON_HALF + 2);
    if (r.width > m.vw - (m.margin * 2)) bx = m.vw / 2;
    bx = Math.max(m.margin + BEACON_HALF, Math.min(m.vw - m.margin - BEACON_HALF, bx));
    by = Math.max(m.margin + BEACON_HALF, Math.min(m.vh - m.margin - BEACON_HALF, by));
    beacon.style.left = Math.round(bx - BEACON_HALF) + 'px';
    beacon.style.top = Math.round(by - BEACON_HALF) + 'px';
    beacon.classList.add('on');
  }}

  /* Phase A's cue lives on the card rather than on a floating pill: the
     user is already reading the card, and a badge there cannot be mistaken
     for part of the controls it is describing.  Phase B has no card, so the
     same call keeps the pill's live badge in step with the state. */
  function updateTourCue() {{
    var box = byId('tmTourCue');
    paintTourMarks();
    renderOrb();
    if (!box) return;
    if (advState === 'off') {{
      box.className = 'tm-tour-cue';
      box.innerHTML = '';
      box.style.display = 'none';
      return;
    }}
    var step = STEPS[tourIndex] || {{}};
    if (advState === 'disabled') {{
      box.className = 'tm-tour-cue hint';
      box.innerHTML = '<span class="tm-tour-cue-ico">💡</span><span>Collect samples first to enable this button, or click <b>Next &rarr;</b> to continue the tour</span>';
    }} else if (advState === 'phase') {{
      /* The screen this step documents is still shut, so the door is the
         checkpoint: name the thing that opens, and what to press. */
      box.className = 'tm-tour-cue phase';
      box.innerHTML = '<span class="tm-tour-cue-ico">🎯</span><span><b>Checkpoint:</b> ' +
        (step.checkpoint || ('click the highlighted control to open <b>' + (step.phaseName || 'the next panel') + '</b>')) +
        '</span>';
    }} else {{
      box.className = 'tm-tour-cue';
      box.innerHTML = '<span class="tm-tour-cue-ico">🎯</span><span><b>Checkpoint:</b> ' +
        (step.checkpoint || 'click the highlighted control, or press <b>Next &rarr;</b> to continue') +
        '</span>';
    }}
    box.style.display = 'flex';
  }}

  function placeTour(step) {{
    var spot = byId('tmTourSpotlight');
    var card = byId('tmTourCard');
    /* A placement is queued on the next frame, so one can still be in
       flight when the guide closes: re-seating here would put the ring
       back on screen over a workspace nobody is being guided through. */
    if (!tourOpen) return;
    if (!spot || !card || !step) return;
    /* Hands-on mode: the dim layer, the ring and the card are deliberately
       OFF screen so every control is reachable.  There is nothing to place
       — and re-placing would put the card back in front of the user. */
    if (orbMode) return;
    placedIndex = tourIndex;
    var vw = window.innerWidth || document.documentElement.clientWidth || 0;
    var vh = window.innerHeight || document.documentElement.clientHeight || 0;
    var margin = 14;
    var gap = 16;
    var cw = card.offsetWidth || 392;
    var ch = card.offsetHeight || 300;
    /* `resolveStepTargets` also decides the trigger phase, so the ring, the
       arrow and the click binding can never disagree about the target. */
    var target = resolveStepTargets(step).target;

    if (!target) {{
      /* Nothing to spotlight (welcome / finish): dim the entire viewport
         and float the card dead-centre. */
      placeMark = null;
      hideTourMarks();
      spot.classList.add('on', 'dim');
      spot.style.top = '0px';
      spot.style.left = '0px';
      spot.style.width = '0px';
      spot.style.height = '0px';
      card.style.left = Math.max(margin, Math.round((vw - cw) / 2)) + 'px';
      card.style.top = Math.max(margin, Math.round((vh - ch) / 2)) + 'px';
      return;
    }}

    var r = target.getBoundingClientRect();
    var pad = (step.pad === undefined) ? 6 : step.pad;
    var sTop = Math.max(margin, r.top - pad);
    var sLeft = Math.max(margin, r.left - pad);
    var sRight = Math.min(vw - margin, r.left + r.width + pad);
    var sBottom = Math.min(vh - margin, r.top + r.height + pad);
    var sw = Math.max(10, sRight - sLeft);
    var sh = Math.max(10, sBottom - sTop);
    spot.style.top = Math.round(sTop) + 'px';
    spot.style.left = Math.round(sLeft) + 'px';
    spot.style.width = Math.round(sw) + 'px';
    spot.style.height = Math.round(sh) + 'px';
    spot.classList.add('on');
    spot.classList.remove('dim');

    var space = {{
      right: vw - (sLeft + sw) - margin,
      left: sLeft - margin,
      bottom: vh - (sTop + sh) - margin,
      top: sTop - margin
    }};
    var fits = {{
      right: space.right >= cw + gap,
      left: space.left >= cw + gap,
      bottom: space.bottom >= ch + gap,
      top: space.top >= ch + gap
    }};
    var order = [];
    if (step.placement && step.placement !== 'auto') order.push(step.placement);
    ['right', 'left', 'bottom', 'top'].forEach(function (side) {{
      if (order.indexOf(side) < 0) order.push(side);
    }});
    var mode = null;
    for (var i = 0; i < order.length; i++) {{ if (fits[order[i]]) {{ mode = order[i]; break; }} }}
    if (!mode) {{
      /* Nothing fits (a big three-column workspace): fall back to the
         roomiest side and let the clamp below keep the card on screen. */
      mode = order[0];
      for (var j = 1; j < order.length; j++) {{ if (space[order[j]] > space[mode]) mode = order[j]; }}
    }}

    var top = 0;
    var left = 0;
    if (mode === 'right') {{ left = sLeft + sw + gap; top = r.top + (r.height / 2) - (ch / 2); }}
    else if (mode === 'left') {{ left = sLeft - gap - cw; top = r.top + (r.height / 2) - (ch / 2); }}
    else if (mode === 'bottom') {{ top = sTop + sh + gap; left = r.left + (r.width / 2) - (cw / 2); }}
    else {{ top = sTop - gap - ch; left = r.left + (r.width / 2) - (cw / 2); }}

    left = Math.max(margin, Math.min(Math.max(margin, vw - cw - margin), left));
    top = Math.max(margin, Math.min(Math.max(margin, vh - ch - margin), top));
    card.style.left = Math.round(left) + 'px';
    card.style.top = Math.round(top) + 'px';

    placeMark = {{
      mode: mode, target: target,
      sLeft: sLeft, sTop: sTop, sw: sw, sh: sh,
      cardLeft: left, cardTop: top, cw: cw, ch: ch,
      vw: vw, vh: vh, margin: margin
    }};
    paintTourMarks();
  }}

  function schedulePlace() {{
    if (!tourOpen) return;
    if (placeTimer) window.clearTimeout(placeTimer);
    window.requestAnimationFrame(function () {{ placeTour(STEPS[tourIndex]); }});
    /* Panels opened by onEnter animate/measure lazily, so re-seat the
       spotlight once the layout has settled — and give click-to-advance
       a second chance to bind, in case its target only became visible
       when the panel finished opening. */
    placeTimer = window.setTimeout(function () {{
      if (!tourOpen || orbMode) return;
      placeTour(STEPS[tourIndex]);
      /* The workspace may have replaced the spotlighted card with a fresh
         one since this step was armed (`render()` rebuilds the column).
         `armTargetAdvance()` unbinds and re-binds whenever the armed node
         no longer matches what the step resolves to now, so the new card
         is just as clickable as the old one. */
      armTargetAdvance();
    }}, 280);
  }}

  /* Press a door on the beat AFTER the click that asked for it.  The
     workspace closes its toolbar menus from a document-level click
     listener, and a nested click fired from inside the card's own button
     would be undone by that listener the moment the outer event finished
     bubbling — leaving the user in hands-on mode in front of the very
     panel the card promised to open. */
  function pressDoor(el) {{
    if (!el) return;
    window.setTimeout(function () {{ try {{ el.click(); }} catch (err) {{}} }}, 0);
  }}

  function bindTourCard() {{
    var card = byId('tmTourCard');
    if (!card) return;
    var closeBtn = byId('tmTourClose');
    if (closeBtn) closeBtn.onclick = function () {{ endTour('dismiss'); }};
    var skipBtn = byId('tmTourSkip');
    /* Secondary escape hatch, not the exit: `Skip step →` steps over the one
       checkpoint the user cannot or does not want to perform and leaves the
       guide running.  Dropping out entirely is the × in the card's head, or
       Esc.  Either way the step's onEnter/onLeave hooks put the workspace
       back in the state the next slide expects. */
    if (skipBtn) skipBtn.onclick = function () {{ goStep(tourIndex + 1); }};
    var backBtn = byId('tmTourBack');
    if (backBtn) backBtn.onclick = function () {{ goStep(tourIndex - 1); }};
    var startBtn = byId('tmTourStart');
    if (startBtn) startBtn.onclick = function () {{
      /* "Start Doing This" is the same intent as touching the spotlighted
         control, expressed in words.  A step that lives in a shut screen
         (Settings, the drawer, the worktable) opens its own door first, so
         the user is not dropped into hands-on mode staring at a closed
         panel the card never mentioned. */
      var door = resolveStepTargets(STEPS[tourIndex]);
      if (door.phase && door.target) pressDoor(door.target);
      enterOrbMode();
    }};
    var nextBtn = byId('tmTourNext');
    if (nextBtn) nextBtn.onclick = function () {{
      var step = STEPS[tourIndex];
      var res = resolveStepTargets(step);
      if (res.phase && res.target) {{
        /* The step's own screen is still shut and the card is pointing at
           the button that opens it.  Next → must not walk past that door,
           so it presses it for the user and hands the workspace over —
           which is what the card promises.  Only if the door refuses to
           open does the next press advance for real, so the user can never
           get stuck on this step. */
        pressDoor(res.target);
        collapseIntent(ADV_SETTLE_MS);
        return;
      }}
      if (tourIndex >= STEPS.length - 1) endTour('finish');
      else advanceLater(ORB_ADVANCE_MS);
    }};
    var auto = byId('tmTourAutoskip');
    if (auto) auto.onchange = function () {{
      try {{
        if (auto.checked) window.localStorage.setItem(AUTOSKIP_KEY, '1');
        else window.localStorage.removeItem(AUTOSKIP_KEY);
      }} catch (err) {{}}
    }};
    Array.prototype.forEach.call(card.querySelectorAll('[data-chapter]'), function (chip) {{
      chip.onclick = function () {{
        var n = Number(chip.getAttribute('data-chapter')) || 1;
        for (var i = 0; i < STEPS.length; i++) {{ if (STEPS[i].chapter === n) {{ goStep(i); return; }} }}
      }};
    }});
  }}

  function renderTourCard() {{
    var card = byId('tmTourCard');
    var step = STEPS[tourIndex];
    if (!card || !step) return;
    var phase = PHASES[Math.max(0, Math.min(PHASES.length - 1, (step.chapter || 1) - 1))];
    var last = tourIndex === STEPS.length - 1;
    var chips = PHASES.map(function (p) {{
      var cls = 'tm-tour-chapter' + (p.n === step.chapter ? ' active' : '');
      return '<button type="button" class="' + cls + '" data-chapter="' + p.n + '">' + p.n + '. ' + p.label + '</button>';
    }}).join('');
    var html = '';
    html += '<div class="tm-tour-head">';
    html += '<div class="tm-tour-head-top">';
    html += '<span class="tm-tour-phase">Phase ' + phase.n + ' of ' + PHASES.length + ' · ' + phase.label + '</span>';
    html += '<span class="tm-tour-stepcount">Step ' + (tourIndex + 1) + ' / ' + STEPS.length + '</span>';
    html += '<button type="button" class="tm-tour-close" id="tmTourClose" title="Close the tour (Esc)">&times;</button>';
    html += '</div>';
    html += '<div class="tm-tour-title">' + step.title + '</div>';
    html += '</div>';
    html += '<div class="tm-tour-cue" id="tmTourCue" style="display:none"></div>';
    html += '<div class="tm-tour-body">' + step.body + '</div>';
    if (step.overview && step.stages && step.stages.length) {{
      /* Step 0 is the map: the whole trip as numbered stages, so the user
         knows where the guide is taking them before it starts pointing. */
      html += '<ol class="tm-tour-stages">';
      step.stages.forEach(function (s, i) {{
        html += '<li><span class="tm-tour-stage-num">' + (i + 1) + '</span>' +
          '<span class="tm-tour-stage-txt"><span class="tm-tour-stage-t">' + s.t + '</span>' +
          '<span class="tm-tour-stage-d">' + s.d + '</span></span></li>';
      }});
      html += '</ol>';
    }}
    html += '<div class="tm-tour-foot">';
    html += '<div class="tm-tour-chapters">' + chips + '</div>';
    html += '<label class="tm-tour-autoskip"><input type="checkbox" id="tmTourAutoskip"' + (autoskipOn() ? ' checked' : '') + '/><span>Don&rsquo;t show automatically on startup</span></label>';
    if (step.overview) {{
      /* One door out of the overview: start the guide. */
      html += '<div class="tm-tour-actions">';
      html += '<button type="button" class="tm-tour-btn primary wide" id="tmTourNext">Start Step-by-Step Guide &rarr;</button>';
      html += '</div>';
    }} else {{
      /* Phase A.  The wide button below the prose is the point of the whole
         redesign: it is how the user says "let me work" and gets the card
         and the dim layer out of the way.  Next Step → shares the row with
         Skip / Back for the user who has already done this step.  A step
         with nothing to point at (the closing slide) has no hands-on mode
         to offer, so it keeps the plain row. */
      if (step.target || step.phaseTarget) {{
        html += '<div class="tm-tour-actions">';
        html += '<button type="button" class="tm-tour-btn primary wide" id="tmTourStart">🛠️ Start Doing This (Minimize to Orb)</button>';
        html += '</div>';
      }}
      html += '<div class="tm-tour-actions">';
      html += '<button type="button" class="tm-tour-btn ghost" id="tmTourSkip"' + (last ? ' disabled' : '') + '>Skip step &rarr;</button>';
      html += '<span class="tm-tour-spacer"></span>';
      html += '<button type="button" class="tm-tour-btn" id="tmTourBack"' + (tourIndex === 0 ? ' disabled' : '') + '>&larr; Back</button>';
      html += '<button type="button" class="tm-tour-btn" id="tmTourNext">' + (last ? 'Finish &#127881;' : 'Next Step &rarr;') + '</button>';
      html += '</div>';
    }}
    html += '</div>';
    card.innerHTML = html;
    bindTourCard();
    updateTourCue();
  }}

  function goStep(next) {{
    if (!tourOpen) return;
    var clamped = Math.max(0, Math.min(STEPS.length - 1, next));
    var current = STEPS[tourIndex];
    if (current && clamped !== tourIndex) runHook(current.onLeave, 'onLeave');
    tourIndex = clamped;
    /* Drop the previous step's trigger before the new one is armed: a
       stale listener would advance the tour off a control that belongs
       to a slide the user has already left. */
    unbindTargetAdvance();
    advState = 'off';
    resetOrb();
    runHook(STEPS[tourIndex].onEnter, 'onEnter');
    /* Every step starts in Phase A.  Coming out of hands-on mode the card
       was taken off screen, so the next step's card has to be put back —
       otherwise the pill would move the user on to a step nothing
       explains. */
    var card = byId('tmTourCard');
    if (card) card.classList.add('on');
    renderTourCard();
    armTargetAdvance(true);
    schedulePlace();
    if (typeof syncFrameHeight === 'function') {{
      window.setTimeout(function () {{ try {{ syncFrameHeight(); }} catch (err) {{}} }}, 320);
    }}
  }}

  function startTour(force) {{
    /* Which curriculum runs is decided HERE rather than at parse time:
       the workspace mounts once and never reloads, so the project type
       can change under the tour's feet (File ▾ → Open project). */
    selectCurriculum();
    if (tourOpen) {{ goStep(0); return; }}
    if (force !== true && autoskipOn()) return;
    uiBefore = snapshotUi();
    tourOpen = true;
    tourIndex = 0;
    var card = byId('tmTourCard');
    if (card) card.classList.add('on');
    unbindTargetAdvance();
    advState = 'off';
    resetOrb();
    runHook(STEPS[0].onEnter, 'onEnter');
    renderTourCard();
    armTargetAdvance(true);
    schedulePlace();
  }}

  function endTour(reason) {{
    if (!tourOpen) return;
    var current = STEPS[tourIndex];
    runHook(current && current.onLeave, 'onLeave');
    tourOpen = false;
    unbindTargetAdvance();
    advState = 'off';
    resetOrb();
    if (placeTimer) {{ window.clearTimeout(placeTimer); placeTimer = null; }}
    restoreUi(uiBefore);
    uiBefore = null;
    var card = byId('tmTourCard');
    var spot = byId('tmTourSpotlight');
    if (card) {{ card.classList.remove('on'); card.innerHTML = ''; }}
    if (spot) spot.classList.remove('on', 'dim');
    placeMark = null;
    hideTourMarks();
    markSeen();
    window.__tmTourLastEnd = String(reason || 'dismiss');
    if (typeof syncFrameHeight === 'function') {{
      try {{ syncFrameHeight(); }} catch (err) {{}}
    }}
  }}

  /* Registered at parse time, so this runs before the workspace keyboard
     router (which binds on `load`): that is what lets the tour keep the
     arrows to itself while it is open.  Escape is different, though — the
     workspace owns an Escape priority stack, and while a menu, modal or
     the source drawer is open the TOPMOST layer is that layer, not the
     guide.  So Escape is handed back (`return` with no preventDefault)
     whenever a workspace layer is up, and only acts on the tour once
     nothing is covering it. */
  window.addEventListener('keydown', function (e) {{
    if (!tourOpen) return;
    var key = String(e.key || '');
    if (key === 'Escape' || key === 'Esc') {{
      if (workspaceLayerOpen()) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      /* First Escape folds the card into the pill (the user is working,
         not leaving); a second one, from the pill, ends the guide. */
      if (!orbMode) {{ enterOrbMode(); return; }}
      endTour('escape');
      return;
    }}
    var el = document.activeElement;
    var tag = el ? String(el.tagName || '').toUpperCase() : '';
    if (el && (el.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT')) return;
    /* Hands-on mode hands the workspace over completely: the arrows belong
       to the slider or the carousel the user is working, never to a step
       they cannot see.  Only the pill leaves a step from here. */
    if (orbMode) return;
    if (key === 'ArrowRight') {{
      e.preventDefault();
      e.stopImmediatePropagation();
      if (tourIndex >= STEPS.length - 1) endTour('finish'); else goStep(tourIndex + 1);
    }} else if (key === 'ArrowLeft') {{
      e.preventDefault();
      e.stopImmediatePropagation();
      goStep(tourIndex - 1);
    }}
  }}, true);

  window.addEventListener('resize', function () {{ if (tourOpen) schedulePlace(); }}, {{ passive: true }});
  window.addEventListener('scroll', function () {{ if (tourOpen) schedulePlace(); }}, {{ passive: true, capture: true }});

  /* Small debug/automation surface (used by the headless UI test and
     handy while developing); it does not affect normal behaviour. */
  window.tmTour = {{
    start: function () {{ startTour(true); }},
    next: function () {{ if (tourIndex >= STEPS.length - 1) endTour('finish'); else goStep(tourIndex + 1); }},
    back: function () {{ goStep(tourIndex - 1); }},
    goTo: function (i) {{ goStep(i); }},
    close: function () {{ endTour('dismiss'); }},
    isOpen: function () {{ return tourOpen; }},
    current: function () {{ return tourIndex; }},
    total: function () {{ return STEPS.length; }},
    titles: function () {{ return STEPS.map(function (s) {{ return s.title; }}); }},
    projectType: function () {{ return TOUR_TYPE; }},
    /* The curriculum is re-read from STATE on every start, so a test can
       switch projects and call this without reloading the workspace. */
    select: function () {{ return selectCurriculum(); }},
    /* What the card is telling the user about the spotlighted control. */
    cue: function () {{ return advState; }},
    /* Phase A, for the headless test: placement waits on a frame, so this
       says which step the ring on screen was measured for, and whether it
       is a cut-out ring (`dim` is false) or the whole viewport dimmed. */
    place: function () {{
      var spot = byId('tmTourSpotlight');
      return {{
        index: placedIndex,
        on: !!(spot && spot.classList.contains('on')),
        dim: !!(spot && spot.classList.contains('dim'))
      }};
    }},
    /* Phase B, for the headless test: whether the floating orb has taken
       over, whether the step's own work looks done (the green badge), and
       what the pill is currently saying. */
    orb: function () {{
      var node = orbNode();
      var card = byId('tmTourCard');
      var spot = byId('tmTourSpotlight');
      var text = byId('tmTourOrbText');
      return {{
        on: !!orbMode,
        ready: !!(node && node.classList.contains('ready')),
        live: orbState(),
        text: text ? String(text.textContent || '') : '',
        goText: (function () {{ var g = byId('tmTourOrbGo'); return g ? String(g.textContent || '') : ''; }})(),
        visible: !!isVisibleNode(node),
        cardOn: !!(card && card.classList.contains('on')),
        spotOn: !!(spot && spot.classList.contains('on'))
      }};
    }},
    /* Same intent as the wide button on the card. */
    handsOn: function () {{ enterOrbMode(); }},
    /* Click the pill, exactly as a user would. */
    clickOrb: function () {{ var n = orbNode(); if (n) n.click(); }},
    /* `?` on the pill: bring the instruction card back for the same step. */
    askOrb: function () {{ var n = byId('tmTourOrbHelp'); if (n) n.click(); }},
    stages: function () {{ return (STEPS[0] && STEPS[0].stages) ? STEPS[0].stages.length : 0; }},
    overview: function () {{ return !!(STEPS[0] && STEPS[0].overview); }},
    /* The directional marks, for the headless test: whether the arrow is
       aimed at the ring and whether the click beacon is on it. */
    marks: function () {{
      var arrow = byId('tmTourArrow');
      var beacon = byId('tmTourBeacon');
      return {{
        aimed: !!(arrow && arrow.classList.contains('on') && placeMark),
        arrow: !!(arrow && arrow.classList.contains('on')),
        beacon: !!(beacon && beacon.classList.contains('on')),
        angle: arrow ? String(arrow.style.transform || '') : ''
      }};
    }},
    seen: function () {{ try {{ return window.localStorage.getItem(SEEN_KEY) === '1'; }} catch (err) {{ return false; }} }},
    /* Called by the workspace at the end of `render()`.  That function
       rebuilds the class column (`#classes.innerHTML = ''`), so the node
       the ring was measured against and the node the click binding sat on
       are both gone; re-seating and re-arming here is what keeps the guide
       pointing at something that exists. */
    sync: function () {{ if (tourOpen && !orbMode) schedulePlace(); }}
  }};

  function bootTour() {{
    var btn = byId('navTutorial');
    if (btn) btn.onclick = function () {{ startTour(true); }};
    /* The pill is bound once for the page's lifetime: it comes and goes by
       class, so re-binding it on every step would only stack listeners. */
    bindOrb();
    /* Pick the curriculum before touching the seen flag: each project
       type keeps its own, so the answer here depends on which guide
       this workspace would show. */
    selectCurriculum();
    var seen = false;
    try {{ seen = window.localStorage.getItem(SEEN_KEY) === '1'; }} catch (err) {{ seen = true; }}
    if (!seen && !autoskipOn()) {{
      window.setTimeout(function () {{ startTour(true); }}, WELCOME_DELAY_MS);
    }}
  }}

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bootTour);
  else bootTour();
}})();
</script>
</body>
</html>
        ''',
        height=1600,
        scrolling=False,
    )


def _render_image_project() -> None:
    # #region debug-point A:render-image-project
    _dbg_open_project_layout("A", "pre-fix", "app.py:_render_image_project", "[DEBUG] render image project page", {"session": str(st.session_state.get("session_id", "")), "project_type": str(st.session_state.get("project_type", "")), "query": dict(st.query_params) if hasattr(st, "query_params") else {}, "workspace": str(_session_workspace())})
    # #endregion
    inject_teachable_style()
    controller = _ensure_fresh_record_controller()
    webcam_options = _list_camera_options()
    preferred_webcam_index, preferred_webcam_uid = _preferred_webcam(webcam_options)
    # ── Pull the SPA's AJAX-written live config BEFORE anything writes to
    # the controller.  The SPA persists serial_port / webcam_index / baud /
    # frame side / channels ONLY via /live/config (the set_serial_port /
    # set_webcam_index tm_action handlers are dead — setAction has no call
    # sites).  Without this read-back, every rerun clobbered the user's
    # choices with stale session state (serial_port='' and an auto-derived
    # webcam index) — the "selecting a device port seems to be treated as
    # an index" bug.
    _live_cfg = controller.get_config(str(st.session_state.session_id))
    if _live_cfg is not None:
        st.session_state.tm_serial_port = str(_live_cfg.serial_port or "")
        st.session_state.tm_serial_baud = int(_live_cfg.serial_baud)
        st.session_state.tm_serial_sync = str(_live_cfg.serial_sync or "")
        st.session_state.tm_serial_frame_side = int(_live_cfg.serial_frame_side)
        st.session_state.tm_serial_channels = int(_live_cfg.serial_channels)
        st.session_state.tm_webcam_index = int(_live_cfg.webcam_index)
        st.session_state.tm_webcam_id = str(_live_cfg.webcam_id or "")
        st.session_state.tm_webcam_user_selected = True
    else:
        # Fresh session (no controller config yet): restore the last-used
        # camera from the workspace stamp (uniqueID → current index), and
        # only fall back to the preferred non-virtual camera when nothing
        # was ever chosen.  Auto-selecting "the first camera" here was the
        # bug — for Iriun users that default is the wrong feed.
        restored_index = None
        persisted = _load_persisted_webcam()
        if persisted and str(persisted.get("webcam_id") or ""):
            from record_controller import _resolve_webcam_index_from_unique_id

            persisted_uid = str(persisted["webcam_id"])
            # The persisted uniqueID is AUTHORITATIVE (the camera is opened
            # by uid on macOS).  The resolved index only preselects the
            # dropdown — when the camera is currently disconnected the uid
            # is still kept so the open fails with a clear message instead
            # of silently falling back to a positional probe of index 0
            # (which opened the virtual camera).
            restored_index = _resolve_webcam_index_from_unique_id(persisted_uid)
            st.session_state.tm_webcam_id = persisted_uid
            st.session_state.tm_webcam_user_selected = True
            if restored_index is not None:
                st.session_state.tm_webcam_index = int(restored_index)
        if restored_index is None:
            current_webcam_index = int(getattr(st.session_state, "tm_webcam_index", preferred_webcam_index) or preferred_webcam_index)
            current_webcam_label = next(
                (str(item.get("label", "")) for item in webcam_options if int(item.get("index", 0)) == int(current_webcam_index)),
                "",
            )
            # Helper: look up the unique_id for a given camera-index from the
            # freshly-enumerated webcam_options (same index could be a
            # different camera across boots if virtuals are inserted).
            def _uid_for_index(camera_index: int) -> str:
                for item in webcam_options:
                    try:
                        if int(item.get("index", -1)) == int(camera_index):
                            return str(
                                item.get("unique_id") or item.get("uid") or item.get("id") or ""
                            )
                    except Exception:
                        continue
                return ""

            if (not st.session_state.get("tm_webcam_user_selected", False)) and webcam_options:
                # Auto-select the preferred non-virtual camera (index + UID
                # together — we MUST set both, otherwise `_open_working_camera`
                # sees an empty UID and falls back to positional index 0,
                # which on macOS with Iriun installed is the Iriun virtual
                # feed even when index=preferred is correct).
                st.session_state.tm_webcam_index = int(preferred_webcam_index)
                st.session_state.tm_webcam_id = str(preferred_webcam_uid or _uid_for_index(preferred_webcam_index))
                st.session_state.tm_webcam_user_selected = False
            elif webcam_options and _is_virtual_camera_label(current_webcam_label):
                # Heal an auto-selected virtual camera: swap to the preferred
                # physical one, and remember to also swap the unique_id.
                st.session_state.tm_webcam_index = int(preferred_webcam_index)
                st.session_state.tm_webcam_id = str(preferred_webcam_uid or _uid_for_index(preferred_webcam_index))
                st.session_state.tm_webcam_user_selected = False
            else:
                # Keep the user's current index, but ALWAYS look up the
                # matching unique_id from the fresh enumeration so the next
                # controller open uses a unique_id-based open (no drift).
                current_uid = str(getattr(st.session_state, "tm_webcam_id", "") or "")
                if not current_uid:
                    st.session_state.tm_webcam_id = _uid_for_index(int(current_webcam_index))
                st.session_state.tm_webcam_index = int(current_webcam_index)
    serial_ports = [
        {
            "device": p.device,
            "label": f"{p.device} - {p.description}" if p.description else p.device,
        }
        for p in list_serial_ports()
        if _is_likely_user_serial_port(p.device, p.description)
    ]
    # Build the config from the MERGED values above (controller read-back
    # wins for the SPA-controlled fields; session state supplies the rest).
    controller.set_config(st.session_state.session_id, _tm_session_config())

    classes = list(st.session_state.tm_classes) if st.session_state.tm_classes else ["Class 1", "Class 2"]
    if len(classes) == 1:
        classes = classes + ["Class 2"]
    disk_classes = _tm_load_classes_meta()
    if disk_classes:
        classes = disk_classes
    else:
        _tm_save_classes_meta(classes)
    st.session_state.tm_classes = classes

    action = _tm_get_query_param("tm_action").strip()
    notice = ""
    #region debug-point B:action-entry
    _dbg_capture_webcam_source("B", "pre-fix", "app.py:_render_image_project", "[DEBUG] tm_action received", {"action": action, "query": dict(st.query_params) if hasattr(st, "query_params") else {}, "tm_open_source_class": st.session_state.get("tm_open_source_class", ""), "tm_open_source_kind": st.session_state.get("tm_open_source_kind", ""), "tm_webcam_index": st.session_state.get("tm_webcam_index", 0)})
    #endregion
    if action:
        if action == "home":
            # #region debug-point B:action-home
            _dbg_open_project_layout("B", "pre-fix", "app.py:_render_image_project", "[DEBUG] action home triggered", {"session": str(st.session_state.get("session_id", "")), "query_before_clear": dict(st.query_params) if hasattr(st, "query_params") else {}, "project_type_before": str(st.session_state.get("project_type", ""))})
            # #endregion
            # Full teardown: stop live + record, drop the session from the
            # record controller, reset session state, clear the URL (the
            # address bar reset happens here via st.query_params, applied by
            # Streamlit's own frontend JS), then rerun to the home page.
            _tm_handle_return_home(controller)
        if action == "addclass":
            st.session_state.tm_classes = classes + [_next_class_name(classes)]
            _tm_clear_query_params()
            st.rerun()
        if action == "delete":
            try:
                idx = int(_tm_get_query_param("idx") or "-1")
            except Exception:
                idx = -1
            if 0 <= idx < len(classes) and len(classes) > 1:
                _remove_tm_class(classes[idx])
                st.session_state.tm_classes = [c for i, c in enumerate(classes) if i != idx]
            _tm_clear_query_params()
            st.rerun()
        if action == "rename":
            try:
                idx = int(_tm_get_query_param("idx") or "-1")
            except Exception:
                idx = -1
            new_name = _tm_get_query_param("name").strip()
            if 0 <= idx < len(classes) and new_name:
                edited = list(classes)
                edited[idx] = new_name
                updated = _apply_tm_class_names(classes, edited)
                st.session_state.tm_classes = updated
            _tm_clear_query_params()
            st.rerun()
        if action == "set_serial_port":
            st.session_state.tm_serial_port = _tm_get_query_param("serial_port").strip()
            st.session_state.tm_open_source_class = _tm_get_query_param("open_class").strip()
            st.session_state.tm_open_source_kind = _tm_get_query_param("open_kind").strip()
            _tm_clear_query_params()
            st.rerun()
        if action == "set_webcam_index":
            try:
                st.session_state.tm_webcam_index = int(float(_tm_get_query_param("webcam_index") or st.session_state.tm_webcam_index))
                st.session_state.tm_webcam_user_selected = True
            except Exception:
                pass
            st.session_state.tm_open_source_class = _tm_get_query_param("open_class").strip()
            st.session_state.tm_open_source_kind = _tm_get_query_param("open_kind").strip()
            #region debug-point D:set-webcam-index
            _dbg_capture_webcam_source("D", "pre-fix", "app.py:_render_image_project", "[DEBUG] webcam index updated from action", {"tm_webcam_index": st.session_state.tm_webcam_index, "tm_open_source_class": st.session_state.tm_open_source_class, "tm_open_source_kind": st.session_state.tm_open_source_kind})
            #endregion
            _tm_clear_query_params()
            st.rerun()
        if action == "open_source":
            st.session_state.tm_open_source_class = _tm_get_query_param("open_class").strip()
            st.session_state.tm_open_source_kind = _tm_get_query_param("open_kind").strip()
            st.session_state.tm_frontend_notice = _tm_get_query_param("notice").strip()
            #region debug-point B:open-source-action
            _dbg_capture_webcam_source("B", "pre-fix", "app.py:_render_image_project", "[DEBUG] open_source action applied", {"tm_open_source_class": st.session_state.tm_open_source_class, "tm_open_source_kind": st.session_state.tm_open_source_kind, "tm_frontend_notice": st.session_state.tm_frontend_notice})
            #endregion
            _tm_clear_query_params()
            st.rerun()
        if action == "close_source":
            st.session_state.tm_open_source_class = ""
            st.session_state.tm_open_source_kind = ""
            _tm_clear_query_params()
            st.rerun()
        if action == "advanced":
            cfg = st.session_state.train_cfg
            try:
                batch_size = max(1, int(float(_tm_get_query_param("batch_size") or cfg.batch_size)))
                epochs = max(1, int(float(_tm_get_query_param("epochs") or cfg.epochs)))
                validation_split = float(_tm_get_query_param("validation_split") or cfg.validation_split)
                learning_rate = float(_tm_get_query_param("learning_rate") or cfg.learning_rate)
                conv1_filters = max(1, int(float(_tm_get_query_param("conv1_filters") or cfg.conv1_filters)))
                conv2_filters = max(1, int(float(_tm_get_query_param("conv2_filters") or cfg.conv2_filters)))
                dense_units = max(1, int(float(_tm_get_query_param("dense_units") or cfg.dense_units)))
            except Exception:
                notice = "Invalid advanced settings."
            else:
                st.session_state.train_cfg = TrainConfig(
                    img_size=cfg.img_size,
                    color_mode=cfg.color_mode,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=max(0.05, min(0.5, validation_split)),
                    seed=cfg.seed,
                    optimizer=cfg.optimizer,
                    learning_rate=max(0.00001, learning_rate),
                    conv1_filters=conv1_filters,
                    conv2_filters=conv2_filters,
                    dense_units=dense_units,
                    representative_samples=cfg.representative_samples,
                    preprocess_mode=str(getattr(cfg, "preprocess_mode", "auto_by_label")),
                    manual_roi=getattr(cfg, "manual_roi", None),
                )
                notice = "Advanced settings updated."
            _tm_clear_query_params()
            st.rerun()

    counts: Dict[str, int] = {}
    empty_classes: List[str] = []
    total_samples = 0
    for name in classes:
        class_dir = _tm_dataset_dir() / sanitize_class_name(name)
        n = len(_tm_class_image_files(class_dir)) if class_dir.exists() else 0
        counts[name] = n
        total_samples += n
        if n == 0:
            empty_classes.append(name)
    train_ready = len([c for c in classes if counts.get(c, 0) > 0]) >= 2 and not empty_classes and total_samples > 0
    st.session_state.tm_train_ready = train_ready
    if not train_ready:
        if len(classes) < 2:
            st.session_state.tm_train_block_reason = "Training requires at least 2 classes."
        elif empty_classes:
            st.session_state.tm_train_block_reason = "Each class needs at least 1 sample image."
        else:
            st.session_state.tm_train_block_reason = "Add sample images before training."
    else:
        st.session_state.tm_train_block_reason = ""

    if action == "train":
        st.session_state.tm_open_source_class = _tm_get_query_param("open_class").strip()
        st.session_state.tm_open_source_kind = _tm_get_query_param("open_kind").strip()
        _tm_clear_query_params()
        notice = "Training starts in-page."

    if action == "export":
        _tm_clear_query_params()
        meta = _tm_load_train_latest()
        tflite_path = Path(str(meta.get("tflite_path") or "")).expanduser().resolve() if isinstance(meta, dict) else Path()
        labels = list(meta.get("labels") or []) if isinstance(meta, dict) else []
        if (not meta) or (not tflite_path.exists()):
            notice = "Train a model first."
            st.rerun()
        else:
            model_name = str(st.session_state.get("tm_model_name", "model")).strip() or "model"
            array_name = str(st.session_state.get("tm_array_name", "g_model")).strip() or "g_model"
            export_dir = Path(st.session_state.last_export_dir).expanduser().resolve() if st.session_state.last_export_dir.strip() else _default_export_dir()
            errors = _validate_export_inputs(export_dir, model_name, array_name, tflite_path)
            if errors:
                notice = errors[0]
            else:
                export_dir.mkdir(parents=True, exist_ok=True)
                source_bytes = tflite_path.read_bytes()
                from trainer import export_tflite_c_sources

                src, hdr = export_tflite_c_sources(source_bytes, array_name=array_name)
                (export_dir / f"{model_name}.tflite").write_bytes(source_bytes)
                (export_dir / "model.h").write_text(hdr, encoding="utf-8")
                (export_dir / "model.cpp").write_text('#include "model.h"\n\n' + src, encoding="utf-8")
                (export_dir / "labels.txt").write_text("\n".join([str(x) for x in labels]) + "\n", encoding="utf-8")
                # Stamp the deployed pointer so Preview keeps validating the
                # exported (device) model instead of silently following later
                # training runs.
                try:
                    shutil.copyfile(_tm_train_latest_path(), _tm_dataset_dir().parent / "deployed.json")
                except Exception:
                    pass
                notice = f"Exported to: {export_dir}"
            st.rerun()

    if action == "export_browse":
        _tm_clear_query_params()
        meta = _tm_load_train_latest()
        tflite_path = Path(str(meta.get("tflite_path") or "")).expanduser().resolve() if isinstance(meta, dict) else Path()
        labels = list(meta.get("labels") or []) if isinstance(meta, dict) else []
        if (not meta) or (not tflite_path.exists()):
            notice = "Train a model first."
            st.rerun()
        picked = _pick_directory_dialog(initial_dir=str(st.session_state.last_export_dir or ""))
        if picked:
            st.session_state.last_export_dir = str(picked)
        model_name = str(st.session_state.get("tm_model_name", "model")).strip() or "model"
        array_name = str(st.session_state.get("tm_array_name", "g_model")).strip() or "g_model"
        export_dir = Path(st.session_state.last_export_dir).expanduser().resolve() if st.session_state.last_export_dir.strip() else _default_export_dir()
        errors = _validate_export_inputs(export_dir, model_name, array_name, tflite_path)
        if errors:
            notice = errors[0]
        else:
            export_dir.mkdir(parents=True, exist_ok=True)
            source_bytes = tflite_path.read_bytes()
            from trainer import export_tflite_c_sources

            src, hdr = export_tflite_c_sources(source_bytes, array_name=array_name)
            (export_dir / f"{model_name}.tflite").write_bytes(source_bytes)
            (export_dir / "model.h").write_text(hdr, encoding="utf-8")
            (export_dir / "model.cpp").write_text('#include "model.h"\n\n' + src, encoding="utf-8")
            (export_dir / "labels.txt").write_text("\n".join([str(x) for x in labels]) + "\n", encoding="utf-8")
            notice = f"Exported to: {export_dir}"
        st.rerun()

    export_enabled = _tm_load_train_latest() is not None
    if not notice:
        notice = str(st.session_state.get("tm_frontend_notice", "") or "")
    st.session_state.tm_frontend_notice = ""
    sample_previews = _tm_sample_previews(classes, limit_per_class=12)
    initial_open_source_class = str(st.session_state.get("tm_open_source_class", "") or "")
    initial_open_source_kind = str(st.session_state.get("tm_open_source_kind", "") or "")
    # NOTE: the live-config read-back now happens at the TOP of this render
    # (before set_config) so SPA choices survive reruns — see the
    # "Pull the SPA's AJAX-written live config" block above.

    # Hidden "Return Home" button: the SPA iframe (components.html) cannot
    # navigate the parent (sandbox has no allow-top-navigation) and custom
    # postMessages are ignored by Streamlit's frontend, so returnHome() in the
    # SPA clicks this button through same-origin parent DOM access. The click
    # triggers a Streamlit rerun where the handler below performs the teardown
    # (stop live + record, drop the session, reset session state, rewrite the
    # parent URL via st.query_params).
    st.markdown('<style>div[data-testid="stButton"]:has(button[kind="secondary"]) { display: none; }</style>', unsafe_allow_html=True)
    if st.button("Return Home", key="tm_return_home", type="secondary"):
        _tm_handle_return_home(controller)

    _render_tm_old_frontend_html(
        port=int(controller.port),
        session_id=str(st.session_state.session_id),
        classes=classes,
        counts=counts,
        train_enabled=train_ready,
        export_enabled=export_enabled,
        notice=notice,
        train_cfg=st.session_state.train_cfg,
        serial_ports=serial_ports,
        current_serial_port=str(st.session_state.tm_serial_port),
        current_serial_baud=int(st.session_state.tm_serial_baud),
        current_serial_sync=str(st.session_state.tm_serial_sync),
        current_serial_frame_side=int(st.session_state.tm_serial_frame_side),
        current_serial_channels=int(st.session_state.tm_serial_channels),
        webcam_options=webcam_options,
        current_webcam_index=int(st.session_state.tm_webcam_index),
        current_webcam_id=str(st.session_state.get("tm_webcam_id") or ""),
        sample_previews=sample_previews,
        initial_open_source_class=initial_open_source_class,
        initial_open_source_kind=initial_open_source_kind,
    )


def _tm_classes_meta_path() -> Path:
    return _session_workspace() / "tm_dataset" / "tm_classes.json"


def _render_ocr_project_page() -> None:
    """OCR / Character workspace.

    Deliberately NOT a second frontend.  Capture, class editing, the sample
    strip, training, quantisation, export and the OOD preview are all
    project-agnostic: what differs is the sample transform (dispatched
    server-side on `project_type`) and the model-input size (28).  So this
    page renders the shared workspace and adds nothing but the glyph-specific
    explanation — the per-class glyph controls live in the workspace itself
    (Edit ▸ Glyph settings), so nothing here grows with the class count.
    """
    inject_teachable_style()
    st.markdown(
        '''
        <div class="tm-start-panel" style="background-color:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:24px 32px;margin-bottom:1.5rem;margin-top:1rem;">
            <h1 class="tm-start-panel-title" style="margin:0 0 10px 0;font-size:2.2rem;font-weight:800;color:#0F172A;">🔠 OCR / Character Project</h1>
            <p class="tm-start-panel-sub" style="margin:0;max-width:860px;line-height:1.6;color:#475569;font-size:15px;">
                Character recognition for the ESP32-P4. Samples are captured exactly like the image project,
                then each one is reduced to a glyph tensor by an integer-exact pipeline whose Python and C
                implementations agree bit-for-bit. Training, INT8 quantisation, export and the OOD gate are
                the same code the image project uses.
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    # No glyph controls here: they are per-class and would be 3 Streamlit
    # widgets each, rendered into the main document flow (108 of them at 36
    # characters).  They now live in the workspace SPA's OCR-only "Glyph
    # settings" modal — Edit ▸ Glyph settings — which is also the single
    # writer for them (/classes/save_config).
    _render_image_project()


def _render_sign_ocr_project_page() -> None:
    """Car License (3-character) workspace.

    The third glyph project, and structurally the second one: the same 28x28
    character front-end, the same per-character class cards, the same
    training and export.  What it adds lives entirely in the workspace SPA —
    the Target Sign Sequence bar, the character editor, and a preview that
    splits one frame into three slots and answers MATCH / NOT MATCH.  So this
    page is the OCR page with a different explanation above it, exactly as
    the OCR page is the image page with a different explanation above it.
    """
    inject_teachable_style()
    st.markdown(
        '''
        <div class="tm-start-panel" style="background-color:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:24px 32px;margin-bottom:1.5rem;margin-top:1rem;">
            <h1 class="tm-start-panel-title" style="margin:0 0 10px 0;font-size:2.2rem;font-weight:800;color:#0F172A;">🚗 Car License Project</h1>
            <p class="tm-start-panel-sub" style="margin:0;max-width:860px;line-height:1.6;color:#475569;font-size:15px;">
                A license plate is one frame holding three characters in a fixed
                reading order. Set the target sequence, train one class per CHARACTER
                — a repeated character is one class, not two — then point a camera at
                the plate and the preview splits it into three slots and answers
                🟢 YES or 🔴 NO.
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    _render_image_project()


def _render_classified_import_page() -> None:
    # #region debug-point B:render-classified-import
    _dbg_open_project_layout("B", "pre-fix", "app.py:_render_classified_import_page", "[DEBUG] render classified import page", {"session": str(st.session_state.get("session_id", "")), "project_type": str(st.session_state.get("project_type", "")), "query": dict(st.query_params) if hasattr(st, "query_params") else {}, "local_import_path": str(st.session_state.get("local_import_path", ""))})
    # #endregion
    inject_teachable_style()

    # Breadcrumb back navigation
    if st.button("← Back to Home", key="tm_back_from_classified", type="secondary"):
        _tm_teardown_controller_session(_ensure_fresh_record_controller(), str(st.session_state.get("session_id", "")))
        _reset_session_workspace()
        _tm_clear_query_params()
        st.rerun()

    # Full-width header panel (matches Start Page design system)
    st.markdown(
        '''
        <div class="tm-start-panel" style="background-color:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:32px 40px;margin-bottom:2.5rem;margin-top:1rem;">
            <h1 class="tm-start-panel-title" style="margin:0 0 12px 0;font-size:2.2rem;font-weight:800;color:#0F172A;">Import a classified image folder</h1>
            <p class="tm-start-panel-sub" style="margin:0;max-width:800px;line-height:1.6;color:#475569;font-size:15px;">
                Import pre-classified image folders. Supported layouts include <code>label/image</code> and common <code>train|val|test/label/image</code> datasets. Each folder name becomes a class label.
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # Form inputs flow full-width to match the header panel edges
    input_left, input_right = st.columns([5, 1], vertical_alignment="bottom")
    prior_path = str(st.session_state.local_import_path or "")
    with input_left:
        st.markdown('<div class="tm-classified-browse-marker"></div>', unsafe_allow_html=True)
        path_str = st.text_input(
            "Classified folder",
            value=st.session_state.local_import_path,
            placeholder="Choose a folder like label/image or train/label/image",
        )
        st.session_state.local_import_path = path_str
    path_changed = str(path_str or "").strip() != str(prior_path or "").strip()
    if path_changed:
        st.session_state.imported = None
        st.session_state.class_rename = {}
    with input_right:
        if st.button("Browse...", key="tm_pick_classified_dir"):
            picked = _pick_directory_dialog(initial_dir=st.session_state.local_import_path)
            if picked:
                st.session_state.local_import_path = picked
                st.session_state.imported = None
                st.session_state.class_rename = {}
                st.rerun()

    path_value = str(st.session_state.local_import_path or "").strip()
    if st.button("Load folder", type="primary", key="tm_read_classified_dir", disabled=(not path_value), use_container_width=True):
        p = Path(path_value).expanduser()
        if not p.exists() or not p.is_dir():
            st.error("Path does not exist or is not a folder.")
            st.session_state.imported = None
        else:
            imported = infer_imported_data(p)
            if not imported.class_to_images:
                st.error("No dataset detected. Put images under label folders, or use train|val|test/label/image.")
                st.session_state.imported = None
            elif not imported.classified:
                st.error("This folder is not a supported classified dataset. Use label/image or train|val|test/label/image.")
                st.session_state.imported = None
            else:
                st.session_state.imported = imported
                st.session_state.class_rename = {name: name for name in imported.class_to_images.keys()}
                st.rerun()

    imported = st.session_state.imported
    if imported is None or not imported.classified:
        return

    _render_overview(imported)
    st.markdown("### Rename classes")
    rename: Dict[str, str] = {}
    for class_name, images in imported.class_to_images.items():
        c1, c2 = st.columns([4, 1])
        with c1:
            rename[class_name] = st.text_input(
                f"{class_name} name",
                value=st.session_state.class_rename.get(class_name, class_name),
                key=f"tm_classified_rename_{class_name}",
                label_visibility="collapsed",
            )
        with c2:
            st.markdown(
                f'<div class="{_status_style("ready")}">{len(images)} samples</div>',
                unsafe_allow_html=True,
            )
    st.session_state.class_rename = rename

    if st.button("Import to workspace", type="primary", key="tm_import_classified_workspace", use_container_width=True):
        try:
            class_names = _import_classified_into_workspace(imported, rename)
        except Exception as e:
            st.error(str(e))
            return
        # #region debug-point B:classified-import-to-workspace
        _dbg_open_project_layout("B", "pre-fix", "app.py:_render_classified_import_page", "[DEBUG] classified import to workspace", {"session": str(st.session_state.get("session_id", "")), "class_names": list(class_names or []), "query_before": dict(st.query_params) if hasattr(st, "query_params") else {}})
        # #endregion
        st.session_state.tm_classes = class_names or ["Class 1"]
        st.session_state.dataset_dir = str(_tm_dataset_dir())
        st.session_state.imported = None
        st.session_state.project_type = "image"
        _tm_set_query_params(tm_project="image", tm_session=st.session_state.session_id)
        st.rerun()


def _tm_dataset_dir() -> Path:
    ws = _session_workspace()
    return (ws / "tm_dataset").resolve()


@st.cache_resource
def _get_record_controller() -> RecordController:
    c = RecordController()
    c.start()
    try:
        c._hot_mtime = _hot_module_mtimes().get("record_controller")
    except Exception:
        pass
    if not getattr(c, "_exit_hooked", False):
        c._exit_hooked = True
        import atexit as _atexit_mod

        def _release_controller_at_exit() -> None:
            # App-close cleanup: stop live workers (their finally blocks
            # release the camera / serial handles), release every bridge
            # AVCaptureSession, and shut the HTTP server down.  Hot-reload
            # creates a second instance whose handler is also registered —
            # release_all_resources is idempotent, so both firing is safe.
            try:
                c.release_all_resources()
            except Exception:
                pass
            try:
                from mac_camera import shutdown_all_caps as _shutdown_caps

                _shutdown_caps(wait_s=0.2)
            except Exception:
                pass

        _atexit_mod.register(_release_controller_at_exit)
    return c


# Hot-reload support: the cached controller instance must be replaced when
# record_controller.py changed.  The instance itself carries the module
# mtime it was built from, so this is a no-op on every normal rerun.
def _ensure_fresh_record_controller() -> RecordController:
    c = _get_record_controller()
    want = _hot_module_mtimes().get("record_controller")
    if want is None:
        return c
    if getattr(c, "_hot_mtime", None) == want:
        return c
    c._hot_mtime = want
    try:
        c.stop()  # shut down the old HTTP server (frees its port)
    except Exception:
        pass
    _get_record_controller.clear()
    c = _get_record_controller()
    c._hot_mtime = want
    return c


def _save_sample_png(class_name: str, png_bytes: bytes) -> Path:
    out_dir = _tm_dataset_dir() / sanitize_class_name(class_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + ".png"
    out_path = out_dir / name
    out_path.write_bytes(png_bytes)
    return out_path


def _img_to_png_bytes(img) -> bytes:
    import io
    from PIL import Image

    if isinstance(img, Image.Image):
        im = img
    else:
        im = Image.open(io.BytesIO(img))
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    return bio.getvalue()


def _preprocess_image_to_96x96_gray(png_bytes: bytes, crop_box: Optional[Tuple[int, int, int, int]] = None, img_size: int = 96) -> bytes:
    import io
    from PIL import Image

    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        im = im.crop((x1, y1, x2, y2))
    im = im.resize((int(img_size), int(img_size)))
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def _render_crop_ui() -> None:
    if st.session_state.tm_pending_image is None:
        return
    try:
        from streamlit_drawable_canvas import st_canvas
    except Exception:
        st.error("Missing dependency streamlit-drawable-canvas. ROI crop mode is unavailable.")
        return

    import io
    from PIL import Image

    st.subheader("Crop region (ROI)")
    png_bytes = st.session_state.tm_pending_image
    class_name = st.session_state.tm_pending_class
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = im.size
    scale = min(520 / max(w, 1), 520 / max(h, 1), 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)
    im_disp = im.resize((disp_w, disp_h))

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="rgba(0, 122, 255, 1)",
        background_image=im_disp,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        key="crop_canvas",
    )

    rect = None
    if canvas.json_data and canvas.json_data.get("objects"):
        obj = canvas.json_data["objects"][-1]
        if obj and obj.get("type") == "rect":
            rect = obj

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        if st.button("Cancel", key="crop_cancel"):
            st.session_state.tm_pending_image = None
            st.session_state.tm_pending_class = None
            st.rerun()
    with col_b:
        if st.button("Save", type="primary", key="crop_confirm"):
            if rect is None:
                st.error("Draw a rectangle first.")
                return
            left = int(rect.get("left", 0))
            top = int(rect.get("top", 0))
            rw = int(rect.get("width", 0))
            rh = int(rect.get("height", 0))
            if rw <= 1 or rh <= 1:
                st.error("ROI is too small.")
                return
            x1 = int(left / scale)
            y1 = int(top / scale)
            x2 = int((left + rw) / scale)
            y2 = int((top + rh) / scale)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            out_png = _preprocess_image_to_96x96_gray(png_bytes, crop_box=(x1, y1, x2, y2), img_size=int(getattr(st.session_state.get("train_cfg"), "img_size", 96) or 96))
            _save_sample_png(class_name, out_png)
            st.session_state.tm_pending_image = None
            st.session_state.tm_pending_class = None
            st.rerun()

    _out_side = int(getattr(st.session_state.get("train_cfg"), "img_size", 96) or 96)
    st.caption(f"Input: {w}x{h}  →  Output: {_out_side}x{_out_side}x1")


def _render_tm_class_panel() -> None:
    _render_crop_ui()
    classes = list(st.session_state.tm_classes) or ["Class 1"]
    edited_names: List[str] = []
    total_samples = 0
    empty_classes: List[str] = []
    if "tm_edit_class_idx" not in st.session_state:
        st.session_state.tm_edit_class_idx = -1

    controller = _ensure_fresh_record_controller()
    controller.set_config(st.session_state.session_id, _tm_session_config())

    for idx, name in enumerate(classes):
        class_dir = _tm_dataset_dir() / sanitize_class_name(name)
        samples = _tm_class_image_files(class_dir)
        total_samples += len(samples)
        if len(samples) == 0:
            empty_classes.append(name)
        with st.container():
            st.markdown('<div class="tm-class-card-marker"></div>', unsafe_allow_html=True)
            editing = int(st.session_state.tm_edit_class_idx) == idx
            head_a, head_b, head_c = st.columns([5.6, 0.65, 0.75])
            with head_a:
                if editing:
                    st.markdown('<div class="tm-class-title-field">', unsafe_allow_html=True)
                    edited_name = st.text_input(
                        f"class-name-{idx}",
                        value=name,
                        key=f"tm_class_name_{idx}",
                        label_visibility="collapsed",
                        placeholder=f"Class {idx + 1}",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="tm-class-title-text"><h3>{html_escape(name)}</h3></div>',
                        unsafe_allow_html=True,
                    )
                    edited_name = name
                edited_names.append(edited_name)
            with head_b:
                if editing:
                    if st.button("✓", key=f"tm_done_edit_{idx}", use_container_width=True):
                        st.session_state.tm_edit_class_idx = -1
                        st.rerun()
                else:
                    if st.button("✎", key=f"tm_edit_{idx}", use_container_width=True):
                        st.session_state.tm_edit_class_idx = idx
                        st.rerun()
            with head_c:
                delete_disabled = len(classes) <= 1
                st.markdown('<div class="tm-class-menu">', unsafe_allow_html=True)
                with st.popover("⋮", use_container_width=True):
                    if st.button("Delete class", key=f"tm_delete_class_{idx}", use_container_width=True, disabled=delete_disabled):
                        _remove_tm_class(name)
                        st.session_state.tm_classes = [item for item in classes if item != name]
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="tm-class-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="tm-class-subhead">Add Image Samples:</div>', unsafe_allow_html=True)

            btn_a, btn_b, btn_c = st.columns(3, gap="small")
            with btn_a:
                with st.popover("▢\nWebcam", use_container_width=True):
                    permission = ensure_camera_access(int(st.session_state.tm_webcam_index or 0), probe_open=False, unique_id=str(st.session_state.get("tm_webcam_id") or ""))
                    st.session_state.tm_camera_permission_status = permission.status
                    st.session_state.tm_camera_permission_note = permission.message
                    st.session_state.tm_camera_permission_class = name
                    if not permission.allowed:
                        st.markdown(
                            f'<div class="tm-camera-note">{html_escape(permission.message)}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        cam = st.camera_input("Webcam", key=f"tm_cam_{idx}", label_visibility="collapsed")
                        if cam is not None:
                            png = _img_to_png_bytes(cam.getvalue())
                            out_png = _preprocess_image_to_96x96_gray(png, crop_box=None, img_size=int(getattr(st.session_state.get("train_cfg"), "img_size", 96) or 96))
                            _save_sample_png(name, out_png)
                            st.rerun()
            with btn_b:
                with st.popover("▦\nDevice", use_container_width=True):
                    up = st.file_uploader(
                        "Device image",
                        type=["png", "jpg", "jpeg", "bmp", "webp"],
                        accept_multiple_files=True,
                        key=f"tm_device_up_{idx}",
                        label_visibility="collapsed",
                    )
                    if up:
                        for f in up:
                            png = _img_to_png_bytes(f.getvalue())
                            out_png = _preprocess_image_to_96x96_gray(png, crop_box=None, img_size=int(getattr(st.session_state.get("train_cfg"), "img_size", 96) or 96))
                            _save_sample_png(name, out_png)
                        st.rerun()
            with btn_c:
                st.markdown('<div class="tm-class-upload">', unsafe_allow_html=True)
                with st.popover("⇧\nUpload", use_container_width=True):
                    upload = st.file_uploader(
                        "Upload",
                        type=["png", "jpg", "jpeg", "bmp", "webp"],
                        accept_multiple_files=True,
                        key=f"tm_up_{idx}",
                        label_visibility="collapsed",
                    )
                    if upload:
                        for f in upload:
                            png = _img_to_png_bytes(f.getvalue())
                            if st.session_state.tm_crop_mode == "roi":
                                st.session_state.tm_pending_image = png
                                st.session_state.tm_pending_class = name
                                st.rerun()
                            out_png = _preprocess_image_to_96x96_gray(png, crop_box=None, img_size=int(getattr(st.session_state.get("train_cfg"), "img_size", 96) or 96))
                            _save_sample_png(name, out_png)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            if (
                st.session_state.tm_camera_permission_note
                and st.session_state.tm_camera_permission_class == name
                and st.session_state.tm_camera_permission_status
                not in {"granted", "not_required"}
            ):
                st.markdown(
                    f'<div class="tm-camera-note">{html_escape(st.session_state.tm_camera_permission_note)}</div>',
                    unsafe_allow_html=True,
                )

    if edited_names != classes:
        try:
            updated_names = _apply_tm_class_names(classes, edited_names)
        except Exception as e:
            st.error(str(e))
        else:
            if updated_names != classes:
                name_map = {old: new for old, new in zip(classes, updated_names)}
                st.session_state.tm_classes = updated_names
                for key in ("tm_capture_class", "tm_camera_permission_class", "tm_pending_class"):
                    current = st.session_state.get(key)
                    if current in name_map:
                        st.session_state[key] = name_map[current]
                st.rerun()

    with st.container():
        st.markdown('<div class="tm-add-class-marker"></div>', unsafe_allow_html=True)
        if st.button("⊞ Add a class", key="tm_add_class", use_container_width=True):
            st.session_state.tm_classes = classes + [_next_class_name(classes)]
            st.rerun()
    st.session_state.tm_crop_mode = "full"

    non_empty_classes = [c for c in classes if c not in empty_classes]
    train_ready = len(non_empty_classes) >= 2 and not empty_classes and total_samples > 0
    st.session_state.tm_total_classes = len(classes)
    st.session_state.tm_total_samples = total_samples
    st.session_state.tm_train_ready = train_ready
    if not train_ready:
        if len(classes) < 2:
            st.session_state.tm_train_block_reason = "Training requires at least 2 classes."
        elif empty_classes:
            st.session_state.tm_train_block_reason = "Each class needs at least 1 sample image."
        else:
            st.session_state.tm_train_block_reason = "Add sample images before training."
    else:
        st.session_state.tm_train_block_reason = ""


def _render_hold_capture_panel(controller: RecordController, class_name: str, samples: List[Path]) -> None:
    from urllib.parse import quote

    source = st.session_state.tm_capture_source
    st.markdown('<div class="tm-panel tm-capture-panel">', unsafe_allow_html=True)
    st.markdown(
        f'''
<div class="tm-capture-head">
  <strong>{html_escape(source.title())}</strong>
  <span class="{_status_style('ready')}">Live</span>
</div>
        ''',
        unsafe_allow_html=True,
    )

    preview: Optional[bytes] = None
    if source == "device":
        preview = controller.preview_serial_png(
            st.session_state.tm_serial_port,
            int(st.session_state.tm_serial_baud),
            str(st.session_state.tm_serial_sync),
            frame_side=int(st.session_state.tm_serial_frame_side),
            channels=int(st.session_state.tm_serial_channels),
        )
        if preview is None:
            st.warning("Unable to preview device stream. Check serial port and baudrate.")
    elif source == "webcam":
        permission = ensure_camera_access(int(st.session_state.tm_webcam_index or 0), probe_open=False, unique_id=str(st.session_state.get("tm_webcam_id") or ""))
        st.session_state.tm_camera_permission_status = permission.status
        st.session_state.tm_camera_permission_note = permission.message
        if not permission.allowed:
            st.error(permission.message)
            st.markdown("</div>", unsafe_allow_html=True)
            return
        preview = controller.preview_webcam_png(
            int(st.session_state.tm_webcam_index or 0),
            webcam_id=str(st.session_state.get("tm_webcam_id") or ""),
            session_id=str(st.session_state.get("session_id") or ""),
        )
        if preview is None:
            st.warning("Unable to preview webcam. Check system permission or whether the camera is in use.")

    crop_box: Optional[Tuple[int, int, int, int]] = None
    if st.session_state.tm_crop_mode == "roi" and preview is not None:
        try:
            from streamlit_drawable_canvas import st_canvas
        except Exception:
            st.error("Missing dependency streamlit-drawable-canvas. ROI crop mode is unavailable.")
        else:
            import io
            from PIL import Image

            im = Image.open(io.BytesIO(preview)).convert("RGB")
            w, h = im.size
            scale = min(520 / max(w, 1), 520 / max(h, 1), 1.0)
            disp_w, disp_h = int(w * scale), int(h * scale)
            im_disp = im.resize((disp_w, disp_h))
            canvas = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=3,
                stroke_color="rgba(0, 122, 255, 1)",
                background_image=im_disp,
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                drawing_mode="rect",
                key="tm_hold_roi_canvas",
            )
            if canvas.json_data and canvas.json_data.get("objects"):
                obj = canvas.json_data["objects"][-1]
                if obj and obj.get("type") == "rect":
                    left = int(obj.get("left", 0))
                    top = int(obj.get("top", 0))
                    rw = int(obj.get("width", 0))
                    rh = int(obj.get("height", 0))
                    if rw > 1 and rh > 1:
                        x1 = int(left / scale)
                        y1 = int(top / scale)
                        x2 = int((left + rw) / scale)
                        y2 = int((top + rh) / scale)
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        x2 = max(x1 + 1, min(w, x2))
                        y2 = max(y1 + 1, min(h, y2))
                        crop_box = (x1, y1, x2, y2)
                        st.session_state.tm_record_crop_box = crop_box

    capture_left, capture_right = st.columns([1.05, 1.05], gap="small")
    with capture_left:
        st.markdown('<div class="tm-capture-stage">', unsafe_allow_html=True)
        if preview is not None:
            st.image(preview, use_container_width=True)
        st.session_state.tm_record_fps = st.slider("FPS", min_value=1.0, max_value=20.0, value=float(st.session_state.tm_record_fps), step=1.0)

        base = f"http://127.0.0.1:{controller.port}"
        q_class = quote(class_name)
        q_sess = quote(st.session_state.session_id)
        q_source = quote(source)
        start_url = f"{base}/start?session={q_sess}&source={q_source}&class={q_class}"
        stop_url = f"{base}/stop?session={q_sess}"
        html = make_hold_button_html("Hold to Record", start_url=start_url, stop_url=stop_url)
        components.html(html, height=88)
        st.markdown("</div>", unsafe_allow_html=True)
    with capture_right:
        st.markdown('<div class="tm-capture-side-head">Samples</div>', unsafe_allow_html=True)
        if samples:
            st.image([str(p) for p in samples[-6:]], width=72)
        else:
            st.markdown('<div class="tm-preview-note">No samples yet.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_tm_train_panel() -> None:
    dataset_dir = _tm_dataset_dir()
    train_ready = bool(st.session_state.get("tm_train_ready", False))
    train_reason = str(st.session_state.get("tm_train_block_reason", "")).strip()
    with st.container():
        st.markdown('<div class="tm-train-card-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="tm-card-head"><h3>Training</h3></div>', unsafe_allow_html=True)
        if st.button("Train Embedded Model", type="primary", key="tm_train_primary", use_container_width=True, disabled=not train_ready):
            if not train_ready:
                return
            st.session_state.export_validated_token = ""
            runs_dir = APP_DATA_DIR / "runs"
            run_dir = new_run_dir(runs_dir)
            cfg = st.session_state.train_cfg
            cfg = TrainConfig(
                img_size=96,
                color_mode="grayscale",
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                validation_split=cfg.validation_split,
                seed=cfg.seed,
                optimizer=cfg.optimizer,
                learning_rate=cfg.learning_rate,
                conv1_filters=cfg.conv1_filters,
                conv2_filters=cfg.conv2_filters,
                dense_units=cfg.dense_units,
                representative_samples=cfg.representative_samples,
                preprocess_mode=str(getattr(cfg, "preprocess_mode", "auto_by_label")),
                manual_roi=getattr(cfg, "manual_roi", None),
            )
            st.session_state.train_cfg = cfg
            with st.spinner("Training and exporting int8 TFLite..."):
                result = train_and_export(
                    dataset_dir=dataset_dir,
                    run_dir=run_dir,
                    cfg=cfg,
                    model_base_name="model",
                    array_name="g_model",
                )
            st.session_state.train_result = result
        with st.popover("Advanced ▾", use_container_width=True):
            st.session_state.train_cfg = _render_train_config(st.session_state.train_cfg)

def _render_tm_preview_export_panel() -> None:
    result = st.session_state.train_result
    with st.container():
        st.markdown('<div class="tm-preview-card-marker"></div>', unsafe_allow_html=True)
        head_left, head_right = st.columns([1.55, 1.05])
        with head_left:
            st.markdown('<div class="tm-card-head"><h3>Preview</h3></div>', unsafe_allow_html=True)
        with head_right:
            export_clicked = st.button(
                "⇪ Export Model",
                key="tm_export_primary",
                disabled=result is None,
                use_container_width=True,
            )

        if result is None:
            st.markdown(
                '<div class="tm-card-note">You must train a model on the left before you can preview it here.</div>',
                unsafe_allow_html=True,
            )
        elif st.session_state.tm_last_device_frame:
            st.image(st.session_state.tm_last_device_frame, use_container_width=True)
        else:
            st.markdown('<div class="tm-card-note">Model ready.</div>', unsafe_allow_html=True)

    if result is None:
        return

    model_name = str(st.session_state.get("tm_model_name", "model")).strip() or "model"
    array_name = str(st.session_state.get("tm_array_name", "g_model")).strip() or "g_model"
    export_dir = Path(st.session_state.last_export_dir).expanduser().resolve() if st.session_state.last_export_dir.strip() else _default_export_dir()
    current_token = f"{export_dir}|{model_name}|{array_name}"
    if export_clicked:
        errors = _validate_export_inputs(export_dir, model_name, array_name, result.tflite_path)
        if errors:
            st.session_state.export_validated_token = ""
            st.error("\n".join(errors))
            return

        export_dir.mkdir(parents=True, exist_ok=True)
        source_bytes = result.tflite_path.read_bytes()
        from trainer import export_tflite_c_sources

        src, hdr = export_tflite_c_sources(source_bytes, array_name=array_name)
        (export_dir / f"{model_name}.tflite").write_bytes(source_bytes)
        (export_dir / "model.h").write_text(hdr, encoding="utf-8")
        (export_dir / "model.cpp").write_text('#include "model.h"\n\n' + src, encoding="utf-8")
        (export_dir / "labels.txt").write_text("\n".join(result.labels) + "\n", encoding="utf-8")
        st.success(f"Exported to: {export_dir}")


def _tm_eula_flag_path() -> Path:
    """Persistent first-launch acceptance flag (app-data, not the repo)."""
    try:
        from dataset_io import app_data_dir

        return Path(app_data_dir()) / "eula_accepted.flag"
    except Exception:
        return Path(tempfile.gettempdir()) / "TFLiteTraining_eula_accepted.flag"


def _tm_eula_accepted() -> bool:
    if bool(st.session_state.get("tm_eula_accepted")):
        return True
    try:
        if _tm_eula_flag_path().exists():
            st.session_state.tm_eula_accepted = True
            return True
    except Exception:
        pass
    return False


def _render_eula_gate() -> None:
    """First-launch license gate: the EULA must be accepted before the app
    runs.  Acceptance is persisted so it is asked exactly once per machine."""
    inject_teachable_style()
    st.markdown("## End User License Agreement")
    st.markdown(
        "Please read the license agreement below. You must accept it to use this software."
    )
    try:
        _eula_path = os.path.join(os.path.dirname(__file__) or ".", "EULA.md")
        if os.path.exists(_eula_path):
            with open(_eula_path, encoding="utf-8") as _efh:
                st.markdown(_efh.read())
        else:
            st.warning("EULA.md not found in this build.")
    except Exception:
        pass
    _col_ok, _col_no = st.columns(2)
    with _col_ok:
        if st.button("I Agree", key="tm_eula_agree", type="primary", use_container_width=True):
            try:
                _tm_eula_flag_path().parent.mkdir(parents=True, exist_ok=True)
                _tm_eula_flag_path().write_text("1", encoding="utf-8")
            except Exception:
                pass
            st.session_state.tm_eula_accepted = True
            st.rerun()
    with _col_no:
        if st.button("I Disagree", key="tm_eula_disagree", type="secondary", use_container_width=True):
            st.session_state.tm_eula_declined = True
            st.rerun()


def main() -> None:
    _persist_session_id()
    st.set_page_config(page_title="TF Lite Training", layout="wide", page_icon="AIoScouts.png")
    _init_session()

    # ── First-launch license gate ────────────────────────────────────────
    # The EULA is part of the commercial distribution contract: without a
    # recorded acceptance the app must not run (and must not let a user
    # drift into the workspace through a stale URL either).
    if st.session_state.get("tm_eula_declined"):
        st.warning(
            "You declined the license agreement, so the application cannot "
            "run. Restart the application to review the agreement again."
        )
        st.stop()
    if not _tm_eula_accepted():
        _render_eula_gate()
        st.stop()

    if st.session_state.project_type is None:
        _render_new_project()
        return

    if st.session_state.project_type == "image":
        _render_image_project()
    elif st.session_state.project_type == "image_classified_import":
        _render_classified_import_page()
    elif st.session_state.project_type == "ocr":
        _render_ocr_project_page()
    elif st.session_state.project_type == PROJECT_TYPE_SIGN_OCR:
        _render_sign_ocr_project_page()
    else:
        st.warning("This project type is not implemented yet.")
        # Escape hatch: an unrecognised tm_project value (hand-edited or stale
        # URL) otherwise strands the app with no in-app way back.
        if st.button("← Back to Home", key="tm_back_from_unknown_type", type="secondary"):
            st.session_state.project_type = None
            st.session_state.step = 0
            _tm_clear_query_params()
            st.rerun()


if __name__ == "__main__":
    main()
