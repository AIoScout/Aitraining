from __future__ import annotations

import base64
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
    PROJECT_TYPE_CLASSIFIED_IMPORT,
    PROJECT_TYPE_IMAGE,
    PROJECT_TYPE_OCR,
    ImportedData,
    app_data_dir,
    default_img_size,
    export_classified_dataset,
    export_from_assignments,
    infer_imported_data,
    sanitize_class_name,
    materialize_files,
    materialize_zip_bytes,
    new_output_dir,
)
from image_preprocess import normalize_class_preprocess_map
from ocr_dataset import BACKGROUND_CLASS_NAME, DEFAULT_EMNIST_SPLIT, EMNIST_SPLITS
from ocr_preprocess import GLYPH_SIZE
from serial_device import list_serial_ports
from record_controller import RecordController, SessionConfig, make_hold_button_html


APP_NAME = "TFLiteTraining"


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
    if "ocr" in name or "glyph" in name or "char" in name:
        return PROJECT_TYPE_OCR
    if "image" in ptype or "image" in name:
        return PROJECT_TYPE_IMAGE
    return PROJECT_TYPE_IMAGE


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


def _tm_set_img_size(size: int) -> None:
    """Rebuild the frozen TrainConfig with a new model-input side."""
    prev_cfg = st.session_state.train_cfg
    if int(getattr(prev_cfg, "img_size", 0) or 0) == int(size):
        return
    st.session_state.train_cfg = TrainConfig(
        img_size=int(size),
        color_mode=str(getattr(prev_cfg, "color_mode", "grayscale")),
        batch_size=int(getattr(prev_cfg, "batch_size", 16)),
        epochs=int(getattr(prev_cfg, "epochs", 10)),
        validation_split=float(getattr(prev_cfg, "validation_split", 0.2)),
        seed=int(getattr(prev_cfg, "seed", 42)),
        optimizer=str(getattr(prev_cfg, "optimizer", "adam")),
        learning_rate=float(getattr(prev_cfg, "learning_rate", 0.001)),
        conv1_filters=int(getattr(prev_cfg, "conv1_filters", 32)),
        conv2_filters=int(getattr(prev_cfg, "conv2_filters", 64)),
        dense_units=int(getattr(prev_cfg, "dense_units", 128)),
        representative_samples=int(getattr(prev_cfg, "representative_samples", 200)),
        preprocess_mode=str(getattr(prev_cfg, "preprocess_mode", "auto_by_label")),
        manual_roi=getattr(prev_cfg, "manual_roi", None),
    )


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


def _tm_open_project_into_session(p: Path, ptype: str) -> None:
    """Open a .tmproj into a FRESH workspace and route to its project type.

    Shared by every Start-Page card — an OCR .tmproj restores exactly the
    same way an image one does, because everything that differs (the sample
    transform, the model input size, the training labels) is already keyed
    off `project_type` server-side.  Adding a card must never mean copying
    this block.
    """
    _begin_fresh_tm_session()
    st.session_state.tm_return_target = "home"
    controller = _ensure_fresh_record_controller()
    controller.set_config(st.session_state.session_id, _tm_session_config(project_type=str(ptype)))
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
    if isinstance(train_cfg_state, dict):
        prev_cfg = st.session_state.train_cfg
        st.session_state.train_cfg = TrainConfig(
            img_size=int(effective_size),
            color_mode=str(getattr(prev_cfg, "color_mode", "grayscale")),
            batch_size=int(train_cfg_state.get("batch_size", getattr(prev_cfg, "batch_size", 16))),
            epochs=int(train_cfg_state.get("epochs", getattr(prev_cfg, "epochs", 10))),
            validation_split=float(train_cfg_state.get("validation_split", getattr(prev_cfg, "validation_split", 0.2))),
            seed=int(getattr(prev_cfg, "seed", 42)),
            optimizer=str(getattr(prev_cfg, "optimizer", "adam")),
            learning_rate=float(train_cfg_state.get("learning_rate", getattr(prev_cfg, "learning_rate", 0.001))),
            conv1_filters=int(train_cfg_state.get("conv1_filters", getattr(prev_cfg, "conv1_filters", 32))),
            conv2_filters=int(train_cfg_state.get("conv2_filters", getattr(prev_cfg, "conv2_filters", 64))),
            dense_units=int(train_cfg_state.get("dense_units", getattr(prev_cfg, "dense_units", 128))),
            representative_samples=int(getattr(prev_cfg, "representative_samples", 200)),
            preprocess_mode=str(train_cfg_state.get("preprocess_mode", getattr(prev_cfg, "preprocess_mode", "auto_by_label"))),
            manual_roi=train_cfg_state.get("manual_roi", getattr(prev_cfg, "manual_roi", None)),
        )
    st.session_state.project_type = effective_type
    st.session_state.tm_current_project_path = str(p)
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
        <div style="background-color:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:40px;margin-bottom:2rem;display:flex;flex-direction:row;justify-content:space-between;align-items:center;gap:40px;position:relative;overflow:hidden;">
        <div style="flex:1;min-width:500px;z-index:2;">
        <h2 style="margin:0 0 8px 0;font-size:14px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;">Desktop AI Training</h2>
        <h1 style="margin:0 0 16px 0;font-size:2.8rem;font-weight:800;color:#0F172A;letter-spacing:-1px;line-height:1.1;">TFLite Training</h1>
        <p style="margin:0 0 28px 0;max-width:540px;line-height:1.6;color:#475569;font-size:15px;">Build classroom-friendly image models without touching code. Inspired by Google Teachable Machine, with a desktop-first workflow for importing data, training, and exporting TFLite + C sources.</p>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:40px;">
        <span class="tm-chip">Import local images</span>
        <span class="tm-chip">Capture from webcam</span>
        <span class="tm-chip">Train lightweight CNN</span>
        <span class="tm-chip">Export .tflite + model.cpp</span>
        </div>
        <div style="border-top:1px solid #E2E8F0;padding-top:32px;display:flex;flex-direction:column;align-items:flex-start;">
        <div style="font-size:11px !important;text-transform:uppercase;letter-spacing:1.5px;color:#64748B;font-weight:700;margin-bottom:16px;line-height:1;">Recommended Workflow</div>
        <div style="display:flex;flex-direction:row;align-items:center;gap:14px;">
        <div style="display:flex;align-items:center;gap:8px;"><span style="background:#E0E7FF;color:#3169FF;border-radius:50%;width:24px;height:24px;display:flex;justify-content:center;align-items:center;font-size:12px;font-weight:800;line-height:1;flex-shrink:0;text-align:center;padding:0;margin:0;letter-spacing:normal;">1</span><span style="font-weight:600;font-size:14px;color:#1E293B;line-height:1;">Collect</span></div>
        <span style="display:flex;align-items:center;color:#CBD5E1;font-size:14px;line-height:1;margin-top:-1px;">➔</span>
        <div style="display:flex;align-items:center;gap:8px;"><span style="background:#E0E7FF;color:#3169FF;border-radius:50%;width:24px;height:24px;display:flex;justify-content:center;align-items:center;font-size:12px;font-weight:800;line-height:1;flex-shrink:0;text-align:center;padding:0;margin:0;letter-spacing:normal;">2</span><span style="font-weight:600;font-size:14px;color:#1E293B;line-height:1;">Label</span></div>
        <span style="display:flex;align-items:center;color:#CBD5E1;font-size:14px;line-height:1;margin-top:-1px;">➔</span>
        <div style="display:flex;align-items:center;gap:8px;"><span style="background:#E0E7FF;color:#3169FF;border-radius:50%;width:24px;height:24px;display:flex;justify-content:center;align-items:center;font-size:12px;font-weight:800;line-height:1;flex-shrink:0;text-align:center;padding:0;margin:0;letter-spacing:normal;">3</span><span style="font-weight:600;font-size:14px;color:#1E293B;line-height:1;">Train</span></div>
        <span style="display:flex;align-items:center;color:#CBD5E1;font-size:14px;line-height:1;margin-top:-1px;">➔</span>
        <div style="display:flex;align-items:center;gap:8px;"><span style="background:#E0E7FF;color:#3169FF;border-radius:50%;width:24px;height:24px;display:flex;justify-content:center;align-items:center;font-size:12px;font-weight:800;line-height:1;flex-shrink:0;text-align:center;padding:0;margin:0;letter-spacing:normal;">4</span><span style="font-weight:600;font-size:14px;color:#1E293B;line-height:1;">Export</span></div>
        </div>
        </div>
        </div>
        <div style="position:absolute;right:20px;top:40px;width:280px;height:280px;opacity:0.04;z-index:1;pointer-events:none;">
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
            <h3 style="margin:0;font-size:20px;font-weight:700;color:#1E293B;letter-spacing:-0.5px;">
                ✨ Choose a project type
            </h3>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    card_col1, card_col2, card_col3 = st.columns([1, 1, 1], gap="large")
    with card_col1:
        st.markdown(
            _tw_dedent_helper.dedent('''
            <div class="tm-start-card">
            <div class="tm-start-card-head">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <h3 style="margin:0;padding:0;font-size:22px;font-weight:800;">Image Project</h3>
            <span style="background:#E8F5E9;color:#2E7D32;padding:2px 6px;border-radius:8px;font-size:9px;font-weight:800;letter-spacing:0.5px;">READY NOW</span>
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
            p = _pick_tmproj_file_dialog()
            if not p:
                st.warning("No file selected.")
                st.stop()
            manifest = _tmproj_read_manifest(p)
            ptype = _tmproj_detect_project_type(p, manifest)
            if ptype not in ("image", PROJECT_TYPE_OCR):
                st.error(f"Unsupported project type: {ptype}")
                st.stop()
            # #region debug-point A:home-open-project-click
            _dbg_open_project_layout("A", "pre-fix", "app.py:_render_new_project", "[DEBUG] home open project selected file", {"picked_path": str(p), "project_type": str(ptype), "session_before_fresh": str(st.session_state.get("session_id", ""))})
            # #endregion
            _tm_open_project_into_session(p, ptype)
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
            <h3 style="margin:0;padding:0;font-size:22px;font-weight:800;">🔠 OCR / Character Project</h3>
            <span style="background:#E8F5E9;color:#2E7D32;padding:2px 6px;border-radius:8px;font-size:9px;font-weight:800;letter-spacing:0.5px;">READY NOW</span>
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
            p = _pick_tmproj_file_dialog()
            if not p:
                st.warning("No file selected.")
                st.stop()
            manifest = _tmproj_read_manifest(p)
            ptype = _tmproj_detect_project_type(p, manifest)
            if ptype not in ("image", PROJECT_TYPE_OCR):
                st.error(f"Unsupported project type: {ptype}")
                st.stop()
            _tm_open_project_into_session(p, ptype)


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
            if isinstance(saved_raw, dict):
                updates = {}
                for k in (
                    "batch_size", "epochs", "validation_split", "learning_rate", "img_size",
                    # OCR mixing / transfer-learning knobs round-trip through
                    # the same file, so an OCR run's mix is reproducible from
                    # the workspace without re-deriving it from the UI.
                    "ocr_emnist_enabled", "ocr_emnist_split", "ocr_emnist_per_class",
                    "ocr_background_enabled", "ocr_background_samples",
                    "ocr_transfer_learning", "ocr_pretrain_per_class",
                    "ocr_pretrain_epochs", "ocr_head_epochs", "ocr_finetune_epochs",
                    "ocr_head_lr", "ocr_finetune_lr", "ocr_unfreeze_blocks",
                ):
                    if k in saved_raw and saved_raw[k] is not None:
                        updates[k] = saved_raw[k]
                if updates:
                    train_cfg = TrainConfig(**{**train_cfg.__dict__, **updates})
    except Exception:
        pass

    meta = _tm_load_classes_meta_full() or {}
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
        "train_cfg": {
            "img_size": int(train_cfg.img_size),
            "batch_size": int(train_cfg.batch_size),
            "epochs": int(train_cfg.epochs),
            "validation_split": float(train_cfg.validation_split),
            "learning_rate": float(train_cfg.learning_rate),
            "conv1_filters": int(train_cfg.conv1_filters),
            "conv2_filters": int(train_cfg.conv2_filters),
            "dense_units": int(train_cfg.dense_units),
            "preprocess_mode": str(getattr(train_cfg, "preprocess_mode", "auto_by_label")),
            "manual_roi": list(getattr(train_cfg, "manual_roi", []) or []) or None,
            # OCR dataset mixing + transfer learning.  Sent for every project
            # so the field set is stable; the panel only renders them for an
            # OCR workspace, and the image strategy never reads them.
            "ocr_emnist_enabled": bool(getattr(train_cfg, "ocr_emnist_enabled", True)),
            "ocr_emnist_split": str(getattr(train_cfg, "ocr_emnist_split", DEFAULT_EMNIST_SPLIT)),
            "ocr_emnist_per_class": int(getattr(train_cfg, "ocr_emnist_per_class", 200)),
            "ocr_background_enabled": bool(getattr(train_cfg, "ocr_background_enabled", True)),
            "ocr_background_samples": int(getattr(train_cfg, "ocr_background_samples", 0)),
            "ocr_transfer_learning": bool(getattr(train_cfg, "ocr_transfer_learning", True)),
            "ocr_pretrain_per_class": int(getattr(train_cfg, "ocr_pretrain_per_class", 1200)),
            "ocr_pretrain_epochs": int(getattr(train_cfg, "ocr_pretrain_epochs", 4)),
            "ocr_head_epochs": int(getattr(train_cfg, "ocr_head_epochs", 12)),
            "ocr_finetune_epochs": int(getattr(train_cfg, "ocr_finetune_epochs", 25)),
            "ocr_head_lr": float(getattr(train_cfg, "ocr_head_lr", 0.0015)),
            "ocr_finetune_lr": float(getattr(train_cfg, "ocr_finetune_lr", 0.0002)),
            "ocr_unfreeze_blocks": int(getattr(train_cfg, "ocr_unfreeze_blocks", 1)),
        },
        # The OCR blocks in the Advanced panel are shown only for an OCR
        # workspace — its knobs are inert for a sign project and vice versa.
        "project_type": str(
            st.session_state.get("project_type") or PROJECT_TYPE_IMAGE
        ),
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
      height: 100%;
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      color: var(--text);
      background: transparent;
    }}
    .wrap {{
      position: relative;
      background: var(--bg);
      border-radius: 12px;
      padding: 72px 16px 92px 16px;
      min-height: 620px;
      overflow: visible;
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
      justify-content: space-between;
      align-items: flex-start;
      gap: 28px;
      z-index: 2;
    }}
    .col-classes {{ width: 500px; min-width: 0; display: flex; flex-direction: column; gap: 24px; }}
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
      width: 320px;
      min-width: 0;
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
        gap: 16px;
      }}
      .col-classes {{
        width: 440px;
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
        width: 260px;
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
      padding: 12px 16px;
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
      display: none;
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
    }}
    .train-adv-row span {{
      line-height: 1.2;
      word-break: normal;
      overflow-wrap: normal;
      white-space: normal;
    }}
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
      align-items: stretch;
      justify-content: stretch;
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
      grid-template-columns: minmax(0, 1.18fr) minmax(260px, 0.82fr);
      gap: 14px;
      align-items: stretch;
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
      flex: 1;
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
      height: 100%;
      overflow: hidden;
    }}
    .preview-stage img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
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
      border-radius: 14px;
      background: var(--surface-soft);
      min-height: 0;
      flex: 1;
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
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-width: 0;
    }}
    .source-right {{
      background: var(--surface-elevated);
      padding: 22px 24px;
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
      padding: 10px 16px;
      font-size: 13px;
      font-weight: 700;
      color: rgba(0,0,0,0.7);
    }}
    .summary-row {{
      padding: 10px 16px 14px 16px;
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
      .addclass {{ border-color: #CBD5E1; color: #475569; }}
      .addclass:hover {{ border-color: #3169FF; color: #3169FF; background: #F0F4FF; }}
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
      <button type="button" class="danger" id="navResetProject"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg></span>Reset project</button>
    </div>
  </div>
  <button type="button" class="tm-toolbar-btn" id="navSettings" title="Global input settings (camera, serial, FPS, frame)"><span class="tm-toolbar-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.19.42.58.74 1.06.91.48.17.99.22 1.51.09"></path></svg></span>Settings</button>
  <button type="button" class="tm-toolbar-btn" id="navPreferences" title="Preferences (display limits, caching)"><span class="tm-toolbar-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg></span>Preferences</button>
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
      <div class="tm-settings-group">
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
      <div class="tm-settings-group">
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
        <div class="tm-settings-field">
          <label>Frame sync word (hex, space separated)</label>
          <input id="settingsSerialSync" type="text" placeholder="AA 55 AA"/>
        </div>
      </div>
    </div>
    <div class="tm-settings-foot">
      <button type="button" class="tm-btn" id="settingsCancel">Cancel</button>
      <button type="button" class="tm-btn primary" id="settingsSave">Save changes</button>
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
          <div class="advanced" id="advBtn">Advanced <span id="advChevron">▾</span></div>
          <button class="adv-reset" id="advReset" type="button">Reset</button>
          <button class="adv-use-rec" id="advUseRec" type="button" title="Apply recommended hyperparameters">💡 Use recommended</button>
        </div>
        <div class="train-adv-panel" id="trainAdvPanel">
          <div class="train-adv-grid">
            <div class="train-adv-row"><span>Image Size</span><input id="advImgSizeInline" type="number" min="8" max="512" step="4"/></div>
            <div class="train-adv-row"><span>Batch Size</span><input id="advBatchInline" type="number" min="1" step="1"/><div class="adv-rec-hint" id="recBatchHint"></div></div>
            <div class="train-adv-row"><span>Epochs</span><input id="advEpochsInline" type="number" min="1" step="1"/><div class="adv-rec-hint" id="recEpochsHint"></div></div>
            <div class="train-adv-row"><span>Validation Split</span><input id="advValInline" type="number" min="0" max="0.5" step="0.05"/><div class="adv-rec-hint" id="recValHint"></div></div>
            <div class="train-adv-row"><span>Learning Rate</span><input id="advLrInline" type="number" min="0" step="0.0001"/><div class="adv-rec-hint" id="recLrHint"></div></div>
            <div id="advArchRows" style="display:contents">
              <div class="train-adv-row"><span>Conv1 Filters</span><input id="advConv1Inline" type="number" min="1" step="1"/></div>
              <div class="train-adv-row"><span>Conv2 Filters</span><input id="advConv2Inline" type="number" min="1" step="1"/></div>
              <div class="train-adv-row"><span>Dense Units</span><input id="advDenseInline" type="number" min="1" step="1"/></div>
            </div>
          </div>
          <div class="train-adv-grid" id="advOcrPanel" style="display:none">
            <div class="train-adv-sub">Dataset mixing</div>
            <div class="train-adv-row"><span>Mix EMNIST into every class</span><input id="advOcrEmnist" type="checkbox"/></div>
            <div class="adv-help" id="advOcrMixHelp">EMNIST glyphs are already isolated and centred, so they bypass segmentation and are polarity-aligned to the capture convention before scaling.</div>
            <div class="train-adv-row" id="advOcrSplitRow"><span>EMNIST split</span><select id="advOcrSplit">{emnist_split_options}</select></div>
            <div class="train-adv-row" id="advOcrPerClassRow"><span>Synthetic per class</span><input id="advOcrPerClass" type="number" min="0" max="5000" step="10"/></div>
            <div class="train-adv-row"><span>Background / noise class</span><input id="advOcrBackground" type="checkbox"/></div>
            <div class="train-adv-row" id="advOcrBackgroundNRow"><span>Background crops</span><input id="advOcrBackgroundN" type="number" min="0" max="5000" step="10"/></div>
            <div class="adv-help">0 mirrors the per-class count. The extra N+1 class is named <b>{BACKGROUND_CLASS_NAME}</b> and teaches the model to reject OOD crops.</div>
            <div class="train-adv-sub">Transfer learning</div>
            <div class="train-adv-row"><span>Fine-tune pre-trained backbone</span><input id="advOcrTransfer" type="checkbox"/></div>
            <div class="train-adv-row"><span>Pre-train samples / class</span><input id="advOcrPretrainN" type="number" min="100" max="6000" step="100"/></div>
            <div class="train-adv-row"><span>Pre-train epochs</span><input id="advOcrPretrainEpochs" type="number" min="1" max="100" step="1"/></div>
            <div class="train-adv-row"><span>Head epochs</span><input id="advOcrHeadEpochs" type="number" min="1" max="200" step="1"/></div>
            <div class="train-adv-row"><span>Head learning rate</span><input id="advOcrHeadLr" type="number" min="0" step="0.0001"/></div>
            <div class="train-adv-row"><span>Fine-tune epochs</span><input id="advOcrFinetuneEpochs" type="number" min="0" max="200" step="1"/></div>
            <div class="train-adv-row"><span>Fine-tune learning rate</span><input id="advOcrFinetuneLr" type="number" min="0" step="0.0001"/></div>
            <div class="train-adv-row"><span>Conv blocks to unfreeze</span><input id="advOcrUnfreeze" type="number" min="0" max="16" step="1"/></div>
            <div class="adv-help">Phase 1 trains only the new N+1 head with the backbone frozen; phase 2 unfreezes the last conv block(s) at the lower fine-tune rate.</div>
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
let previewOodMaxProbMin = 0.60;
let previewOodEntropyRatioMax = 0.70;
let previewUploadImageSrc = '';
let previewUploadImageB64 = '';
let previewUploadFilename = '';
let previewPredictInFlight = false;
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
  return {{ mode, manual_roi: manual, bg_dark_thresh: bg_dark, bg_lum_thresh: bg_lum }};
}}
function getClassPreprocessConfig(className) {{
  const all = STATE.class_preprocess && typeof STATE.class_preprocess === 'object' ? STATE.class_preprocess : {{}};
  return normalizeClassPreprocessConfig(all[className]);
}}
function setClassPreprocessConfig(className, cfg) {{
  const next = Object.assign({{}}, STATE.class_preprocess || {{}});
  next[className] = normalizeClassPreprocessConfig(cfg);
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
async function refreshClassProcessedPreview() {{
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
  classPreprocessProcessedSrc = '';
  classPreprocessAutoCrop = null;
  refreshClassPreprocessResultPane();
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
        sample_config: targetSampleConfig
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
    classPreprocessProcessedSrc = procB64 ? `data:image/png;base64,${{procB64}}` : '';
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
    // Update ROI overlay with auto-detected crop box
    updateClassPreprocessOverlay(null);
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
    if (!next.manual_roi) next.manual_roi = [0, 0, 1, 1];
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
    // Green border = auto-detected, blue border = manual ROI
    overlay.style.borderColor = isManual ? 'var(--accent)' : '#4caf50';
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
  const classThresh = getClassPreprocessConfig(classPreprocessClass);
  const classPreprocessThresholds = {{
    bg_lum_thresh: classThresh.bg_lum_thresh != null ? classThresh.bg_lum_thresh : 150,
    bg_dark_thresh: classThresh.bg_dark_thresh != null ? classThresh.bg_dark_thresh : 20,
}};
  const cfg = normalizeClassPreprocessConfig(
    classPreprocessDraft || getSampleEffectivePreprocessConfig(classPreprocessClass, sampleFilename)
  );
  classPreprocessDraft = cfg;
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
  const lumSlider   = document.getElementById('classPreprocessLum');
  const lumNum       = document.getElementById('classPreprocessLumNum');
  const darkNum      = document.getElementById('classPreprocessDarkNum');
  const darkSlider   = document.getElementById('classPreprocessDark');
  const applyThresholds = () => {{
    if (classPreprocessThresholdTimer) {{
      clearTimeout(classPreprocessThresholdTimer);
      classPreprocessThresholdTimer = null;
    }}
    const existing = getClassPreprocessConfig(classPreprocessClass);
    existing.bg_lum_thresh = Number(lumSlider ? lumSlider.value : 100);
    existing.bg_dark_thresh = Number(darkSlider ? darkSlider.value : 0);
    setClassPreprocessConfig(classPreprocessClass, existing);
    refreshClassProcessedPreview();
  }};
  const debouncedApply = () => {{
    if (classPreprocessThresholdTimer) clearTimeout(classPreprocessThresholdTimer);
    classPreprocessThresholdTimer = setTimeout(() => {{
      classPreprocessThresholdTimer = null;
      applyThresholds();
    }}, 250);
  }};

  // Lum/B-G slider + number sync
  if (lumSlider) lumSlider.oninput = () => {{ if (lumNum) lumNum.value = lumSlider.value; debouncedApply(); }};
  if (darkSlider) darkSlider.oninput = () => {{ if (darkNum) darkNum.value = darkSlider.value; debouncedApply(); }};
  if (darkNum) darkNum.onchange = () => {{
    let v = Number(darkNum.value); if (!isFinite(v) || v < 0) v = 0; else if (v > 100) v = 100;
    darkSlider.value = v; applyThresholds();
  }};
  if (lumNum) lumNum.onchange = () => {{
    let v = Number(lumNum.value); if (!isFinite(v) || v < 50) v = 100; else if (v > 255) v = 255;
    lumSlider.value = v; applyThresholds();
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
    if (data.source === 'webcam' || data.source === 'device' || data.source === 'upload') previewSource = String(data.source);
    if (['auto_by_label', 'sign', 'junction', 'manual_roi', 'none'].includes(String(data.preprocess || ''))) {{
      previewPreprocessMode = String(data.preprocess);
    }}
    previewShowRoi = !!data.showRoi;
    previewShowRaw = !!data.showRaw;
    previewShowScore = !!data.showScore;
    previewThreshUserModified = !!data.threshModified;
    previewOodSignPctMin = Number(data.oodSignPctMin ?? previewOodSignPctMin ?? 0.3);
    previewOodSignPctMax = Number(data.oodSignPctMax ?? previewOodSignPctMax ?? 70.0);
    previewOodMaxProbMin = Number(data.oodMaxProbMin ?? previewOodMaxProbMin ?? 0.60);
    previewOodEntropyRatioMax = Number(data.oodEntropyRatioMax ?? previewOodEntropyRatioMax ?? 0.70);
    if (previewThreshUserModified) {{
      previewDarkThresh = Number(data.darkThresh ?? STATE.preview_dark_thresh ?? 0);
      previewLumThresh = Number(data.lumThresh ?? STATE.preview_lum_thresh ?? 100);
    }}
  }} catch (e) {{}}
}}
function persistPreviewState() {{
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
      oodMaxProbMin: Number((previewOodMaxProbMin ?? 0.60)),
      oodEntropyRatioMax: Number((previewOodEntropyRatioMax ?? 0.70)),
      threshModified: !!previewThreshUserModified
    }}));
  }} catch (e) {{}}
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
          note.textContent = '';
          return;
        }}
        note.textContent = msg || 'Preview starting...';
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
    note.textContent = '';
  }}).catch((err) => {{
    if (ourController !== classSourceAbortController) return;
    if (err && (err.name === 'AbortError' || String(err).includes('aborted'))) return;
    if (classSourceBlobUrl) {{
      try {{ URL.revokeObjectURL(classSourceBlobUrl); }} catch (e) {{}}
      classSourceBlobUrl = '';
    }}
    img.removeAttribute('src');
    note.textContent = String(err && err.message ? err.message : err);
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
  if (openSourceClass === className && openSourceKind === kind) return;
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
      STATE.serial_ports.forEach((opt) => {{
        const o = document.createElement('option');
        o.value = String(opt.path || opt.device || '');
        const lbl = String(opt.path || opt.device || opt.label || '');
        const desc = String(opt.description || opt.label || '').trim();
        o.textContent = desc ? `${{lbl}} (${{desc}})` : lbl || '(unknown)';
        if (o.value === cur) o.selected = true;
        pSel.appendChild(o);
      }});
      if (!Array.from(pSel.options).some((x) => x.value === '')) {{
        const o = document.createElement('option');
        o.value = ''; o.textContent = '— Not connected —';
        pSel.insertBefore(o, pSel.firstChild);
        if (!cur) o.selected = true;
      }}
    }}
    const fps = document.getElementById('settingsFps');
    if (fps) fps.value = Number(STATE.fps ?? 8.0);
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
    if (fps) payload.fps = Math.max(1, Math.min(60, Number(fps.value || 0)));
    const bSel = document.getElementById('settingsSerialBaud');
    if (bSel) payload.serial_baud = Number(bSel.value);
    const fs = document.getElementById('settingsFrameSide');
    if (fs) payload.serial_frame_side = Math.max(32, Math.min(512, Number(fs.value || 96)));
    const ch = document.getElementById('settingsChannels');
    if (ch) payload.serial_channels = Number(ch.value);
    const sw = document.getElementById('settingsSerialSync');
    if (sw) payload.serial_sync = String(sw.value || 'AA 55 AA').trim().toUpperCase() || 'AA 55 AA';

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
  previewOodMaxProbMin = 0.60;
  previewOodEntropyRatioMax = 0.70;
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
  if (s.train_cfg && typeof s.train_cfg === 'object') {{
    STATE.train_cfg = Object.assign({{}}, STATE.train_cfg || {{}}, s.train_cfg);
    persistTrainCfgStorage();
  }}
  if (typeof s.bg_dark_thresh === 'number') previewDarkThresh = s.bg_dark_thresh;
  if (typeof s.bg_lum_thresh === 'number') previewLumThresh = s.bg_lum_thresh;
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
        project_state: {{
          train_cfg: Object.assign({{}}, STATE.train_cfg || {{}}, {{
            class_preprocess: Object.assign({{}}, STATE.class_preprocess || {{}})
          }}),
          class_preprocess: Object.assign({{}}, STATE.class_preprocess || {{}}),
          sample_preprocess: normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}}),
          bg_dark_thresh: Number((previewDarkThresh ?? 0)),
          bg_lum_thresh: Number((previewLumThresh ?? 100))
        }}
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
          project_state: {{
            train_cfg: Object.assign({{}}, STATE.train_cfg || {{}}, {{
              class_preprocess: Object.assign({{}}, STATE.class_preprocess || {{}})
            }}),
            class_preprocess: Object.assign({{}}, STATE.class_preprocess || {{}}),
            sample_preprocess: normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}}),
            bg_dark_thresh: Number((previewDarkThresh ?? 0)),
            bg_lum_thresh: Number((previewLumThresh ?? 100))
          }}
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
async function exportRunWithOverwriteConfirm(exportDirValue, modelNameValue, arrayNameValue) {{
  const payload = {{
    session: STATE.session,
    export_dir: String(exportDirValue || '').trim(),
    model_name: String(modelNameValue || 'tm'),
    array_name: String(arrayNameValue || '')
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
      overwrite: true
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
    const res = await fetch(`${{baseUrl}}/train/start`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        cfg: Object.assign({{}}, STATE.train_cfg || {{}}, {{
          class_preprocess: Object.assign({{}}, STATE.class_preprocess || {{}}),
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
          // Force the preview pane to re-render (fresh static image / fresh
          // predict loop) so the NEW model is visible immediately.
          const trainedPane = document.getElementById('previewPane');
          if (trainedPane) {{ trainedPane.removeAttribute('data-ready'); }}
          renderPreviewCard();
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
    const camSel = document.getElementById('previewCamera');
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
    const res = await fetch(`${{baseUrl}}/preview/predict?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(previewSource)}}&preprocess=${{encodeURIComponent(previewPreprocessMode)}}&bg_dark=${{encodeURIComponent(previewThreshUserModified ? String(previewDarkThresh ?? '') : '')}}&bg_lum=${{encodeURIComponent(previewThreshUserModified ? String(previewLumThresh ?? '') : '')}}&ood_sign_min=${{encodeURIComponent(String(previewOodSignPctMin ?? 0.3))}}&ood_sign_max=${{encodeURIComponent(String(previewOodSignPctMax ?? 70.0))}}&ood_max_prob=${{encodeURIComponent(String(previewOodMaxProbMin ?? 0.60))}}&ood_entropy_max=${{encodeURIComponent(String(previewOodEntropyRatioMax ?? 0.70))}}&_ts=${{Date.now()}}`, {{ signal: previewPredictController.signal }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Preview failed.');
    previewLastPredictData = data;
    applyPreviewSrcToPane(data);
    renderOutputBars(data.labels || [], data.probs || []);
    if (note) {{
      const top = data.top_label ? `${{String(data.top_label)}}` : '';
      const p = Math.round(Number(data.top_prob || 0) * 1000) / 10;
      const roiMsg = data.processed_variant ? (previewPreprocessMode === 'manual_roi' ? ' · ROI/manual crop input' : ' · sign search + crop input') : '';
      const debugMsg = data.ood
        ? ` · OOD: ${{String(data.ood_reason || 'suppressed')}} · sign=${{Number(data.sign_pct || 0).toFixed(1)}}% · max=${{Number(data.max_prob || 0).toFixed(3)}} · ent=${{Number(data.entropy_ratio || 0).toFixed(3)}}`
        : ` · sign=${{Number(data.sign_pct || 0).toFixed(1)}}% · max=${{Number(data.max_prob || 0).toFixed(3)}} · ent=${{Number(data.entropy_ratio || 0).toFixed(3)}}`;
      note.textContent = top ? `Top: ${{top}} (${{p}}%)${{roiMsg}}${{debugMsg}}` : `${{roiMsg.trim()}}${{debugMsg}}`.trim();
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
function updatePreviewRoiOverlay(crop, isManual) {{
  const overlay = document.getElementById('previewRoiOverlay');
  if (!overlay) return;
  const valid = Array.isArray(crop) && crop.length === 4;
  // Green/blue box shows whenever the ROI toggle is OFF — on the default
  // frame AND on the Orig view (the Orig image keeps the full frame's
  // aspect ratio, so the normalized box maps onto it exactly).  ROI ON
  // alone displays the jumping CROP image itself (the search box effect)
  // and the overlay is hidden there.
  const show = valid && !previewShowRoi;
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
  // Green border = auto-detected, blue border = manual ROI
  overlay.style.borderColor = isManual ? 'var(--accent)' : '#4caf50';
}}
async function runPreviewUploadPrediction() {{
  if (!previewInputOn || !STATE.export_enabled) return;
  if (!previewUploadImageB64) {{
    const note = document.getElementById('previewNote');
    if (note) note.textContent = 'Choose an image in settings to test upload preview.';
    return;
  }}
  if (previewPredictInFlight) return;
  previewPredictInFlight = true;
  const pane = document.getElementById('previewPane');
  const note = document.getElementById('previewNote');
  try {{
    const res = await fetch(`${{baseUrl}}/preview/predict_upload`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        image_b64: previewUploadImageB64,
        preprocess: previewPreprocessMode,
        bg_dark: previewThreshUserModified ? Number(previewDarkThresh ?? 0) : '',
        bg_lum: previewThreshUserModified ? Number(previewLumThresh ?? 0) : '', 
        ood_sign_min: Number(previewOodSignPctMin ?? 0.3),
        ood_sign_max: Number(previewOodSignPctMax ?? 70.0),
        ood_max_prob: Number(previewOodMaxProbMin ?? 0.60),
        ood_entropy_max: Number(previewOodEntropyRatioMax ?? 0.70),
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Preview failed.');
    previewLastPredictData = data;
    applyPreviewSrcToPane(data);
    if (pane && !pane.dataset.ready) {{
      pane.innerHTML = '<div class="preview-empty">Choose an upload image in settings.</div>';
    }}
    renderOutputBars(data.labels || [], data.probs || []);
    if (note) {{
      const top = data.top_label ? `${{String(data.top_label)}}` : '';
      const p = Math.round(Number(data.top_prob || 0) * 1000) / 10;
      const roiMsg = data.processed_variant ? (previewPreprocessMode === 'manual_roi' ? ' · ROI/manual crop input' : ' · sign search + crop input') : '';
      const debugMsg = data.ood
        ? ` · OOD: ${{String(data.ood_reason || 'suppressed')}} · sign=${{Number(data.sign_pct || 0).toFixed(1)}}% · max=${{Number(data.max_prob || 0).toFixed(3)}} · ent=${{Number(data.entropy_ratio || 0).toFixed(3)}}`
        : ` · sign=${{Number(data.sign_pct || 0).toFixed(1)}}% · max=${{Number(data.max_prob || 0).toFixed(3)}} · ent=${{Number(data.entropy_ratio || 0).toFixed(3)}}`;
      note.textContent = top
        ? `Upload: ${{String(previewUploadFilename || 'image')}} · Top: ${{top}} (${{p}}%)${{roiMsg}}${{debugMsg}}`
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
  const exportFields = `
    <label>
      Export Name
      <input id="previewExportName" value="${{String(exportModelName || 'tm')}}" placeholder="person_detect"/>
    </label>
    <label>
      Array Name (optional)
      <input id="previewExportArray" value="${{String(exportArrayName || '')}}" placeholder="g_person_detect_model_data"/>
    </label>
    <label>
      Input Source
      <select id="previewInputSource">
        <option value="webcam"${{previewSource === 'webcam' ? ' selected' : ''}}>Webcam</option>
        <option value="device"${{previewSource === 'device' ? ' selected' : ''}}>Device</option>
        <option value="upload"${{previewSource === 'upload' ? ' selected' : ''}}>Upload File</option>
      </select>
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
      <input id="previewOodMaxProbMin" type="number" min="0" max="1" step="0.01" value="${{Number(previewOodMaxProbMin ?? 0.60)}}"/>
    </label>
    <label title="Maximum entropy ratio allowed before Preview suppresses to No Sign.">
      OOD entropy max
      <input id="previewOodEntropyRatioMax" type="number" min="0" max="1" step="0.01" value="${{Number(previewOodEntropyRatioMax ?? 0.70)}}"/>
    </label>
  `;
  if (previewSource === 'upload') {{
    return `
      <div class="source-settings-panel" id="previewSettingsPanel">
        <div class="source-settings-grid">
          ${{exportFields}}
          ${{oodFields}}
          <label>
            Test Image
            <button class="upload-pick" type="button" id="previewUploadPick">Choose image</button>
          </label>
          <label>
            Selected File
            <input id="previewUploadName" value="${{escapeHtml(String(previewUploadFilename || 'No file selected'))}}" readonly/>
          </label>
        </div>
        <div class="source-settings-actions">
          <button class="source-settings-cancel" type="button" id="previewSettingsCancel">Close</button>
          <button class="source-settings-save" type="button" id="previewSettingsSave">Apply</button>
        </div>
      </div>
    `;
  }}
  if (previewSource === 'device') {{
    return `
      <div class="source-settings-panel" id="previewSettingsPanel">
        <div class="source-settings-grid">
          ${{exportFields}}
          ${{oodFields}}
          <label>
            Device Port
            <select id="previewDevicePort">${{buildDeviceOptions(currentSerialPort)}}</select>
          </label>
          <label>
            Image Size
            <select id="previewDeviceFrameSide">${{buildSerialFrameSideOptions(currentSerialFrameSide)}}</select>
          </label>
          <label>
            Color Mode
            <select id="previewDeviceColorMode">${{buildSerialColorModeOptions(currentSerialChannels)}}</select>
          </label>
          <label>
            Baud Rate
            <select id="previewDeviceBaud">${{buildSerialBaudOptions(currentSerialBaud)}}</select>
          </label>
          <label>
            Sync Header
            <input id="previewDeviceSync" value="${{escapeHtml(String(currentSerialSync || 'AA 55 AA'))}}" placeholder="AA 55 AA"/>
          </label>
        </div>
        <div class="source-settings-actions">
          <button class="source-settings-cancel" type="button" id="previewSettingsCancel">Close</button>
          <button class="source-settings-save" type="button" id="previewSettingsSave">Apply</button>
        </div>
      </div>
    `;
  }}
  return `
    <div class="source-settings-panel" id="previewSettingsPanel">
      <div class="source-settings-grid">
        ${{exportFields}}
        ${{oodFields}}
        <label>
          Camera
          <select id="previewCamera">${{buildWebcamOptions(currentWebcamIndex, currentWebcamUid)}}</select>
        </label>
        <label>
          Preview Rate
          <select id="previewPreviewRate">
            <option value="60"${{Number(previewIntervalMs) === 60 ? ' selected' : ''}}>Fast</option>
            <option value="80"${{Number(previewIntervalMs) === 80 ? ' selected' : ''}}>Balanced</option>
            <option value="120"${{Number(previewIntervalMs) === 120 ? ' selected' : ''}}>Stable</option>
            <option value="180"${{Number(previewIntervalMs) === 180 ? ' selected' : ''}}>Low CPU</option>
          </select>
        </label>
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
  if (previewSource === 'device') {{
    // Re-query the serial port list EVERY time the settings panel opens —
    // a port plugged in after page load must appear (same query path as
    // the class-area device dropdown).  The old guard only refreshed when
    // the list was empty, so a newly connected port never showed up.
    refreshSerialPorts(false, 'previewDevicePort').catch(() => {{}});
  }}
  const cancel = document.getElementById('previewSettingsCancel');
  const save = document.getElementById('previewSettingsSave');
  const portSel = document.getElementById('previewDevicePort');
  const sourceSel = document.getElementById('previewInputSource');
  const uploadPick = document.getElementById('previewUploadPick');
  if (portSel) {{
    // Refresh ports once when the settings panel opens — NOT on every
    // dropdown click (refilling mid-interaction resets the selection).
    portSel.onchange = () => {{
      // selection is applied on Save — just keep the UI value
    }};
  }}
  if (sourceSel) sourceSel.onchange = () => {{
    previewSource = String(sourceSel.value || 'webcam');
    renderPreviewSettings();
  }};
  if (uploadPick) uploadPick.onclick = async () => {{
    await pickPreviewUploadFile();
    renderPreviewSettings();
  }};
  if (cancel) cancel.onclick = () => {{
    previewSettingsOpen = false;
    renderPreviewSettings();
  }};
  if (save) save.onclick = async () => {{
    try {{
      const nameEl = document.getElementById('previewExportName');
      const arrayEl = document.getElementById('previewExportArray');
      const sourceEl = document.getElementById('previewInputSource');
      exportModelName = String(nameEl ? nameEl.value : exportModelName || 'tm').trim() || 'tm';
      exportArrayName = String(arrayEl ? arrayEl.value : exportArrayName || '').trim();
      const nextSource = String(sourceEl ? sourceEl.value : previewSource || 'webcam');
      previewSource = nextSource === 'device' ? 'device' : (nextSource === 'upload' ? 'upload' : 'webcam');
      persistPreviewState();
      persistExportSettings();
      if (previewSource === 'device') {{
        const portEl = document.getElementById('previewDevicePort');
        const sideEl = document.getElementById('previewDeviceFrameSide');
        const modeEl = document.getElementById('previewDeviceColorMode');
        const baudEl = document.getElementById('previewDeviceBaud');
        const syncEl = document.getElementById('previewDeviceSync');
        await setDevicePortGlobal(portEl ? portEl.value : currentSerialPort);
        await setSerialFrameSideGlobal(sideEl ? sideEl.value : String(currentSerialFrameSide));
        await setSerialChannelsGlobal(modeEl ? modeEl.value : String(currentSerialChannels));
        await setSerialBaudGlobal(baudEl ? baudEl.value : String(currentSerialBaud));
        await setSerialSyncGlobal(syncEl ? syncEl.value : String(currentSerialSync));
      }} else if (previewSource === 'webcam') {{
        const camEl = document.getElementById('previewCamera');
        const rateEl = document.getElementById('previewPreviewRate');
        const nextCam = camEl ? camEl.value : String(currentWebcamIndex);
        const nextUid = camEl && camEl.selectedOptions && camEl.selectedOptions[0] ? String(camEl.selectedOptions[0].dataset.uid || '') : '';
        const indexChanged = String(nextCam) !== String(currentWebcamIndex);
        const uidChanged = String(nextUid) !== String(currentWebcamUid || '');
        if (indexChanged || uidChanged) {{
          await setWebcamIndexGlobal(nextCam, nextUid);
        }}
        if (rateEl) previewIntervalMs = Number(rateEl.value || previewIntervalMs || 80);
      }}
      const oodSignMinEl = document.getElementById('previewOodSignPctMin');
      const oodSignMaxEl = document.getElementById('previewOodSignPctMax');
      const oodMaxProbEl = document.getElementById('previewOodMaxProbMin');
      const oodEntropyEl = document.getElementById('previewOodEntropyRatioMax');
      previewOodSignPctMin = Math.max(0, Math.min(100, Number(oodSignMinEl ? oodSignMinEl.value : (previewOodSignPctMin ?? 0.3))));
      previewOodSignPctMax = Math.max(0, Math.min(100, Number(oodSignMaxEl ? oodSignMaxEl.value : (previewOodSignPctMax ?? 70.0))));
      previewOodMaxProbMin = Math.max(0, Math.min(1, Number(oodMaxProbEl ? oodMaxProbEl.value : (previewOodMaxProbMin ?? 0.60))));
      previewOodEntropyRatioMax = Math.max(0, Math.min(1, Number(oodEntropyEl ? oodEntropyEl.value : (previewOodEntropyRatioMax ?? 0.70))));
      const darkEl = document.getElementById('previewDarkThresh');
      const lumEl = document.getElementById('previewLumThresh');
      previewDarkThresh = Number(darkEl ? darkEl.value : (previewDarkThresh ?? 0));
      previewLumThresh = Number(lumEl ? lumEl.value : (previewLumThresh ?? 100));
      persistPreviewState();
      const needsAutoPick = previewSource === 'upload' && !previewUploadFilename;
      if (previewSource === 'upload' && previewUploadFilename) {{
        previewInputOn = true;
      }}
      previewSettingsOpen = false;
      renderPreviewSettings();
      renderPreviewCard();
      if (previewSource === 'upload') {{
        if (needsAutoPick) {{
          try {{
            const picked = await pickPreviewUploadFile();
            if (picked) {{
              previewInputOn = true;
              const inputToggle = document.getElementById('previewInputToggle');
              if (inputToggle) inputToggle.checked = true;
              renderPreviewCard();
              persistPreviewState();
            }} else {{
              previewInputOn = false;
              const inputToggle = document.getElementById('previewInputToggle');
              if (inputToggle) inputToggle.checked = false;
              const note = document.getElementById('previewNote');
              if (note) note.textContent = 'Pick a test image in Preview settings (⚙️) → Upload File.';
              renderPreviewCard();
            }}
          }} catch (e) {{
            toast(String(e && e.message ? e.message : e), 5000);
          }}
        }} else {{
          const inputToggle = document.getElementById('previewInputToggle');
          if (inputToggle) inputToggle.checked = true;
          runPreviewUploadPrediction();
        }}
      }} else {{
        if (previewInputOn) {{
          if (previewPredictTimer) stopPreviewPredictLoop();
          startPreviewPredictLoop();
        }} else {{
          // Preview settings Apply with Input OFF + webcam/device source:
          // no live Preview pane — the visual loop must not run (it opened
          // the camera without the user toggling Input).  The class-panel
          // source loop is independent and only ticks when a class source
          // is actually open (guarded inside refreshPreviewImage).
          if (previewVisualTimer) stopPreviewVisualLoop();
          startPreviewLoop();
        }}
      }}
    }} catch (err) {{
      toast(String(err && err.message ? err.message : err));
    }}
  }};
  scheduleLayoutResync();
}}
function showTrainWarningModal(text) {{
  let m = document.getElementById('trainWarningModal');
  if (!m) {{
    m = document.createElement('div');
    m.id = 'trainWarningModal';
    m.style.cssText = 'position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);background:#fff;border:2px solid #c62828;border-radius:12px;box-shadow:0 8px 40px rgba(0,0,0,0.35);padding:20px 24px;max-width:560px;width:90%;z-index:10000;font-size:14px;line-height:1.7;color:#333;';
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
    updatePreviewRoiOverlay(null, false);
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
  const roiSrc = roiFromData || fullFromData || rawSrc;
  const cropSrc = cropFromData || rawSrc;
  const fullSrc = fullFromData || rawSrc;
  let src = rawSrc;
  if (previewShowRaw && previewShowRoi) src = roiSrc;
  else if (previewShowRaw) src = fullSrc;
  else if (previewShowRoi) src = cropSrc;
  if (!pane.dataset.ready) {{
    pane.innerHTML = `<div class="preview-stage"><img id="previewImage" alt="Preview"/><div class="roi-overlay" id="previewRoiOverlay" style="display:none"></div></div>`;
    pane.dataset.ready = '1';
  }}
  const img = document.getElementById('previewImage');
  if (img && src) {{
    img.src = src;
    previewLastStaticImageSrc = src;
  }}
  const cropRect = hasData && Array.isArray(data.crop) && data.crop.length === 4 ? data.crop : null;
  updatePreviewRoiOverlay(cropRect, previewPreprocessMode === 'manual_roi');
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
  setBool('advOcrBackground', nullish(cfg.ocr_background_enabled, true));
  setBool('advOcrTransfer', nullish(cfg.ocr_transfer_learning, true));
  syncOcrPanelVisibility();
  refreshRecHints();
}}
const OCR_BOOL_KEYS = ['ocr_emnist_enabled', 'ocr_background_enabled', 'ocr_transfer_learning'];
const OCR_STRING_KEYS = ['ocr_emnist_split'];
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
function isOcrProject() {{
  return String((STATE && STATE.project_type) || '').toLowerCase() === 'ocr';
}}
function syncOcrPanelVisibility() {{
  const cfg = STATE.train_cfg || {{}};
  const ocr = isOcrProject();
  const ocrPanel = document.getElementById('advOcrPanel');
  if (ocrPanel) ocrPanel.style.display = ocr ? 'grid' : 'none';
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
  const mix = !!nullish(cfg.ocr_emnist_enabled, true);
  show('advOcrSplitRow', ocr && mix);
  show('advOcrPerClassRow', ocr && mix);
  show('advOcrMixHelp', ocr && mix);
  show('advOcrBackgroundNRow', ocr && !!nullish(cfg.ocr_background_enabled, true));
}}
function renderAdvancedPanel() {{
  const panel = document.getElementById('trainAdvPanel');
  const chev = document.getElementById('advChevron');
  if (!panel) return;
  panel.style.display = trainAdvancedOpen ? 'block' : 'none';
  if (chev) chev.textContent = trainAdvancedOpen ? '▴' : '▾';
  syncOcrPanelVisibility();
  if (trainAdvancedOpen) syncAdvancedInlineInputs();
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
  bindNum('advOcrPerClass', 'ocr_emnist_per_class');
  bindCheck('advOcrBackground', 'ocr_background_enabled');
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
  STATE.train_cfg = {{
    img_size: 96,
    batch_size: 32,
    epochs: 20,
    validation_split: 0.25,
    learning_rate: 0.0016,
    conv1_filters: 8,
    conv2_filters: 16,
    dense_units: 32,
    // OCR defaults, kept in step with trainer.TrainConfig.
    ocr_emnist_enabled: true,
    ocr_emnist_split: 'balanced',
    ocr_emnist_per_class: 200,
    ocr_background_enabled: true,
    ocr_background_samples: 0,
    ocr_transfer_learning: true,
    ocr_pretrain_per_class: 1200,
    ocr_pretrain_epochs: 4,
    ocr_head_epochs: 12,
    ocr_head_lr: 0.0015,
    ocr_finetune_epochs: 25,
    ocr_finetune_lr: 0.0002,
    ocr_unfreeze_blocks: 1
  }};
  persistTrainCfgStorage();
  syncAdvancedInlineInputs();
  renderTrainStatus();
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
  const root = document.getElementById('classes');
  root.innerHTML = '';
  for (let i=0; i<STATE.classes.length; i++) {{
    const name = STATE.classes[i];
    try {{
      const card = document.createElement('div');
      card.className = 'card class-card';
      const head = document.createElement('div');
      head.className = 'class-head';
      const dragHandle = document.createElement('span');
      dragHandle.className = 'class-drag-handle';
      dragHandle.textContent = '⠿';
      dragHandle.title = 'Drag to reorder classes';
      bindClassDragHandle(dragHandle, i);
      head.appendChild(dragHandle);
      const title = document.createElement('div');
      title.className = 'class-title';
      title.textContent = name;
      const preprocessBtn = document.createElement('button');
      preprocessBtn.className = 'preprocess-chip';
      preprocessBtn.type = 'button';
      preprocessBtn.textContent = 'Edit';
      preprocessBtn.onclick = (e) => {{
        if (e) {{
          e.preventDefault();
          e.stopPropagation();
        }}
        openClassPreprocessEditor(name);
      }};
      const edit = document.createElement('button');
      edit.className = 'iconbtn tm-rename-btn';
      edit.title = 'Rename this class';
      edit.innerHTML = createIcon('rename');
      edit.onclick = () => renameClass(name);
      title.appendChild(preprocessBtn);
      title.appendChild(edit);
      head.appendChild(title);
      const more = document.createElement('button');
      more.className = 'iconbtn more tm-delete-class';
      more.innerHTML = createIcon('delete');
      more.title = 'Delete this class';
      more.onclick = async (e) => {{
        if (e) {{
          e.preventDefault();
          e.stopPropagation();
        }}
        await deleteClass(name);
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
      head.appendChild(clearBtn);
      head.appendChild(more);  // delete ⋮ sits at the head's right end
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
                  <button class="iconbtn" type="button" title="Close" id="sourceClose-${{cssSafe(name)}}">✕</button>
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
    const add = document.createElement('div');
    add.className = 'addclass';
    add.innerHTML = `<span class="addclass-icon">${{createIcon('plus')}}</span><span>Add a class</span>`;
    add.onclick = () => addClass();
    root.appendChild(add);

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
    // Settings button (Problem 3.2 SSOT modal) + settings close/save bindings
    const navSettings = document.getElementById('navSettings');
    if (navSettings) navSettings.onclick = (e) => {{
      if (e) e.stopPropagation();
      closeToolbarDropdowns();
      openGlobalSettings();
    }};
    const setCancel = document.getElementById('settingsCancel');
    if (setCancel) setCancel.onclick = (e) => {{ if (e) e.stopPropagation(); closeGlobalSettings(); }};
    const setSave = document.getElementById('settingsSave');
    if (setSave) setSave.onclick = (e) => {{ if (e) e.stopPropagation(); saveGlobalSettings(); }};
    const setModal = document.getElementById('globalSettingsModal');
    if (setModal) setModal.onclick = (e) => {{
      if (e && e.target === setModal) closeGlobalSettings();
    }};
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
  window.addEventListener('keydown', (e) => {{
    const mod = isMac ? (e.metaKey && !e.ctrlKey && !e.altKey) : (e.ctrlKey && !e.metaKey && !e.altKey);
    if (!mod) return;
    const k = String(e.key || '').toLowerCase();
    if (k === 's' && !e.shiftKey) {{
      e.preventDefault(); e.stopImmediatePropagation();
      saveProjectQuick();
    }} else if (k === 's' && e.shiftKey) {{
      e.preventDefault(); e.stopImmediatePropagation();
      saveProjectAs();
    }} else if (k === 'o') {{
      e.preventDefault(); e.stopImmediatePropagation();
      openProject();
    }} else if (k === 'n') {{
      e.preventDefault(); e.stopImmediatePropagation();
      newProject();
    }} else if (k === 'e') {{
      e.preventDefault(); e.stopImmediatePropagation();
      exportModel();
    }}
  }}, true);
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


def _tm_write_class_preprocess(class_preprocess: Dict[str, Dict[str, Any]]) -> None:
    """Persist per-class preprocess through the SAME store the SPA uses.

    The SPA writes this by AJAX to /classes/save_config; Streamlit-side
    controls must land in the identical place (tm_classes.json, merged) or
    the two writers would fight over the class config.
    """
    p = _tm_classes_meta_path()
    raw: Dict[str, Any] = {}
    if p.exists():
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}
    merged = dict(raw.get("class_preprocess") or {})
    for name, cfg in class_preprocess.items():
        merged[name] = {**dict(merged.get(name) or {}), **cfg}
    raw["class_preprocess"] = merged
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    controller = _ensure_fresh_record_controller()
    try:
        controller._rebuild_processed_cache(
            _tm_dataset_dir(),
            class_preprocess=normalize_class_preprocess_map(merged),
            out_size=int(getattr(st.session_state.get("train_cfg"), "img_size", GLYPH_SIZE) or GLYPH_SIZE),
            fast_mode=False,
            project_type=PROJECT_TYPE_OCR,
        )
    except Exception:
        pass


def _render_ocr_glyph_settings() -> None:
    """Per-class ink polarity / segmentation / normalisation.

    These are the only OCR-specific knobs, and they live in the existing
    per-class preprocess record (see normalize_class_preprocess) rather than
    a parallel OCR config — so .tmproj saving, sample overrides and the
    preview calls all carry them for free.
    """
    from ocr_preprocess import normalize_ink_polarity, normalize_norm_convention

    classes = list(st.session_state.tm_classes or ["Class 1", "Class 2"])
    meta = _tm_load_classes_meta_full() or {}
    stored = normalize_class_preprocess_map(meta.get("class_preprocess"))
    side = int(getattr(st.session_state.get("train_cfg"), "img_size", GLYPH_SIZE) or GLYPH_SIZE)

    with st.expander(f"🔠 Glyph preprocessing — {side}×{side} model input", expanded=False):
        st.caption(
            "Every step is integer-exact and mirrorable on the device: BT.601 luminance → exact "
            "integer Otsu (cross-multiplied, no floats) → vertical-projection segmentation → "
            "nearest-neighbour resize by the exact rational map floor((2o+1)·in/(2·out)) → "
            "gray − 128.  Verified byte-identical to the C reference on 386/386 corpus frames."
        )
        updated: Dict[str, Dict[str, Any]] = {}
        for idx, name in enumerate(classes):
            cfg = stored.get(name) or {}
            polarity_label = st.selectbox(
                f"{name} — ink",
                options=["Dark ink on light (camera captures)", "Bright ink on dark (EMNIST/MNIST)"],
                index=1 if int(normalize_ink_polarity(cfg.get("ink_polarity"))) else 0,
                key=f"tm_ocr_polarity_{idx}",
            )
            segment = st.checkbox(
                f"{name} — segment glyph by vertical projection",
                value=bool(cfg.get("glyph_segment", True)),
                key=f"tm_ocr_segment_{idx}",
                help="On for camera captures (the glyph sits somewhere in a larger frame). "
                     "Off for already-isolated imported datasets (EMNIST).",
            )
            convention_label = st.selectbox(
                f"{name} — normalisation",
                options=["Stretch bbox to the full grid", "MNIST/EMNIST canonical (fit 20 px, centre of mass)"],
                index=1 if str(normalize_norm_convention(cfg.get("glyph_convention"))) == "canon" else 0,
                key=f"tm_ocr_convention_{idx}",
            )
            updated[name] = {
                "ink_polarity": 1 if polarity_label.startswith("Bright") else 0,
                "glyph_segment": bool(segment),
                "glyph_convention": "canon" if convention_label.startswith("MNIST") else "stretch",
            }
        if st.button("Apply glyph settings", key="tm_ocr_apply", type="primary"):
            _tm_write_class_preprocess(updated)
            st.success("Glyph settings applied — the processed cache was rebuilt.")
            st.rerun()


def _render_ocr_project_page() -> None:
    """OCR / Character workspace.

    Deliberately NOT a second frontend.  Capture, class editing, the sample
    strip, training, quantisation, export and the OOD preview are all
    project-agnostic: what differs is the sample transform (dispatched
    server-side on `project_type`) and the model-input size (28).  So this
    page renders the shared workspace and only adds the glyph-specific
    explanation + per-class polarity/segmentation controls.
    """
    inject_teachable_style()
    st.markdown(
        '''
        <div style="background-color:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:24px 32px;margin-bottom:1.5rem;margin-top:1rem;">
            <h1 style="margin:0 0 10px 0;font-size:2.2rem;font-weight:800;color:#0F172A;">🔠 OCR / Character Project</h1>
            <p style="margin:0;max-width:860px;line-height:1.6;color:#475569;font-size:15px;">
                Character recognition for the ESP32-P4. Samples are captured exactly like the image project,
                then each one is reduced to a glyph tensor by an integer-exact pipeline whose Python and C
                implementations agree bit-for-bit. Training, INT8 quantisation, export and the OOD gate are
                the same code the image project uses.
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    _render_ocr_glyph_settings()
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
        <div style="background-color:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:32px 40px;margin-bottom:2.5rem;margin-top:1rem;">
            <h1 style="margin:0 0 12px 0;font-size:2.2rem;font-weight:800;color:#0F172A;">Import a classified image folder</h1>
            <p style="margin:0;max-width:800px;line-height:1.6;color:#475569;font-size:15px;">
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


def main() -> None:
    _persist_session_id()
    st.set_page_config(page_title="TF Lite Training", layout="wide", page_icon="AIoScouts.png")
    _init_session()

    if st.session_state.project_type is None:
        _render_new_project()
        return

    if st.session_state.project_type == "image":
        _render_image_project()
    elif st.session_state.project_type == "image_classified_import":
        _render_classified_import_page()
    elif st.session_state.project_type == "ocr":
        _render_ocr_project_page()
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
