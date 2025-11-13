import os
import sys
import uuid
from datetime import datetime
import logging
import importlib.util

from flask import Flask, request, jsonify, send_from_directory
import io
import requests

PREDICT_IMPORT_ERROR = None  # populated if import fails


def _safe_exec_module(spec):
    global PREDICT_IMPORT_ERROR
    try:
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    except Exception as e:
        PREDICT_IMPORT_ERROR = f"{type(e).__name__}: {e}"
        logging.exception("Failed to import prediction utilities")
    return None


def _load_predict_utils(app_root: str):
    """Try importing predict utilities from several common locations.

    Order:
    1) package import: from utils import predict
    2) file at app_root/utils/predict.py
    3) file at app_root/predict.py
    4) package import: from brain_tumor_api import predict
    5) file at app_root/brain_tumor_api/predict.py
    6) package import: from brain_tumor_api.utils import predict
    7) file at app_root/brain_tumor_api/utils/predict.py
    """
    # 1) regular package import
    try:
        from utils import predict as predict_mod  # type: ignore
        logging.info("Loaded predict utils via package import 'from utils import predict'")
        return predict_mod
    except Exception as e:
        logging.debug(f"utils package import failed: {e}")

    # 2) direct file import utils/predict.py
    utils_predict_path = os.path.join(app_root, "utils", "predict.py")
    if os.path.exists(utils_predict_path):
        spec = importlib.util.spec_from_file_location("utils.predict", utils_predict_path)
        predict_mod = _safe_exec_module(spec)
        if predict_mod is not None:
            logging.info(f"Loaded predict utils from {utils_predict_path}")
            return predict_mod

    # 3) direct file import predict.py in root
    root_predict_path = os.path.join(app_root, "predict.py")
    if os.path.exists(root_predict_path):
        spec = importlib.util.spec_from_file_location("predict", root_predict_path)
        predict_mod = _safe_exec_module(spec)
        if predict_mod is not None:
            logging.info(f"Loaded predict utils from {root_predict_path}")
            return predict_mod

    # 4) package import: brain_tumor_api.predict
    try:
        from brain_tumor_api import predict as predict_mod  # type: ignore
        logging.info("Loaded predict utils via package import 'from brain_tumor_api import predict'")
        return predict_mod
    except Exception as e:
        logging.debug(f"brain_tumor_api package import failed: {e}")

    # 5) direct file import brain_tumor_api/predict.py
    bta_predict_path = os.path.join(app_root, "brain_tumor_api", "predict.py")
    if os.path.exists(bta_predict_path):
        spec = importlib.util.spec_from_file_location("brain_tumor_api.predict", bta_predict_path)
        predict_mod = _safe_exec_module(spec)
        if predict_mod is not None:
            logging.info(f"Loaded predict utils from {bta_predict_path}")
            return predict_mod

    # 6) package import: brain_tumor_api.utils.predict
    try:
        from brain_tumor_api.utils import predict as predict_mod  # type: ignore
        logging.info("Loaded predict utils via package import 'from brain_tumor_api.utils import predict'")
        return predict_mod
    except Exception as e:
        logging.debug(f"brain_tumor_api.utils package import failed: {e}")

    # 7) direct file import brain_tumor_api/utils/predict.py
    bta_utils_predict_path = os.path.join(app_root, "brain_tumor_api", "utils", "predict.py")
    if os.path.exists(bta_utils_predict_path):
        spec = importlib.util.spec_from_file_location("brain_tumor_api.utils.predict", bta_utils_predict_path)
        predict_mod = _safe_exec_module(spec)
        if predict_mod is not None:
            logging.info(f"Loaded predict utils from {bta_utils_predict_path}")
            return predict_mod

    logging.error("Could not locate a predict.py module. Place your inference code under utils/, brain_tumor_api/, or root.")
    return None


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_ROOT, "static")
GRADCAM_DIR = os.path.join(STATIC_DIR, "gradcam")
REPORTS_DIR = os.path.join(STATIC_DIR, "reports")

os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
logger = logging.getLogger(__name__)

# Load predict utils once at startup
predict_utils = _load_predict_utils(APP_ROOT)


def build_treatment_suggestion(label: str) -> str:
    mapping = {
        "glioma": (
            "Biopsy for histologic and molecular profiling; maximal safe surgical resection; "
            "adjuvant radiotherapy and/or temozolomide based on grade and markers (e.g., IDH, 1p/19q); "
            "neuro‑oncology follow‑up with guideline‑based MRI surveillance."
        ),
        "meningioma": (
            "Surgical resection when feasible; consider stereotactic radiosurgery for small, skull‑base, or residual tumors; "
            "postoperative MRI at 3–6 months, then every 6–12 months depending on WHO grade and symptoms."
        ),
        "pituitary": (
            "Transsphenoidal resection for symptomatic adenomas; endocrinology management and targeted medical therapy "
            "(e.g., dopamine agonists for prolactinoma); adjuvant radiotherapy for residual/recurrent disease; "
            "MRI and hormone follow‑up every 3–6 months initially."
        ),
        "no_tumor": (
            "No tumor detected. Correlate with clinical findings; routine follow‑up or repeat imaging only if symptoms persist."
        ),
    }
    return mapping.get(label, "Consult a specialist for a tailored treatment plan")


def human_readable_diagnostic(label: str, confidence: float) -> str:
    if label == "no_tumor":
        return "The model did not detect a brain tumor in this MRI."
    return f"The model suggests a {label} with a confidence of {confidence:.1%}. Consider clinical correlation and further evaluation."


def estimate_lobe_from_gradcam(gradcam_path: str) -> str:
    """Coarse lobe + hemisphere estimate directly from the Grad-CAM overlay image.

    Uses the most activated pixel (argmax) from a color overlay as a proxy.
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(gradcam_path).convert("RGB")
        arr = np.asarray(img).astype("float32") / 255.0
        # Use red channel as proxy for activation (typical hot colormap), fall back to blue if needed
        red = arr[..., 0]
        if red.max() <= 0.0:
            red = arr[..., 2]
        y, x = np.unravel_index(np.argmax(red), red.shape)
        h, w = red.shape

        hemisphere = "left" if x < w / 2 else "right"

        y_rel = y / max(h, 1)
        # Split into thirds vertically to approximate lobes
        if y_rel < 0.33:
            lobe = "frontal or parietal (superior)"
        elif y_rel < 0.66:
            lobe = "parietal or temporal (mid)"
        else:
            lobe = "temporal or occipital (inferior/posterior)"

        return f"{lobe}, {hemisphere} hemisphere"
    except Exception:
        return "undetermined"


def build_highlight_tokens(label: str, localization: str) -> list[dict]:
    tokens: list[dict] = []
    if label:
        tokens.append({"text": label, "type": "label"})
    if localization and localization != "undetermined":
        for word in ["frontal", "parietal", "temporal", "occipital", "left", "right"]:
            if word in localization.lower():
                tokens.append({"text": word, "type": "anat"})
    proc_by_label = {
        "glioma": ["biopsy", "resection", "radiotherapy", "temozolomide"],
        "meningioma": ["resection", "stereotactic radiosurgery"],
        "pituitary": ["transsphenoidal resection", "endocrinology", "radiotherapy"],
    }
    for t in proc_by_label.get(label, []):
        tokens.append({"text": t, "type": "proc"})
    return tokens


def try_predict_with_utils(image_path: str, class_names: list[str]):
    """Attempt to call into utils.predict using common function names.

    Expected returns: (pred_label: str, confidences: dict[str, float], gradcam_np_or_path)
    """
    if predict_utils is None:
        raise RuntimeError("utils/predict.py not importable. Ensure it exists and is importable.")

    # Flexible discovery of available functions
    # Priority 1: a single entrypoint that returns everything
    for fn_name in [
        "predict_and_gradcam",
        "run_inference",
        "predict_full",
    ]:
        if hasattr(predict_utils, fn_name):
            fn = getattr(predict_utils, fn_name)
            logger.info(f"Calling utils.{fn_name}(...) with image_path={os.path.basename(image_path)}")
            # Try with common kwargs; fall back to positional
            try:
                result = fn(image_path=image_path, class_names=class_names, output_dir=GRADCAM_DIR)  # type: ignore
            except TypeError:
                result = fn(image_path, class_names, GRADCAM_DIR)  # type: ignore
            # Expect tuple
            return result

    # Priority 2: separate prediction and gradcam generators
    pred_fn = None
    for name in ["predict_image", "predict", "classify_image"]:
        if hasattr(predict_utils, name):
            pred_fn = getattr(predict_utils, name)
            break

    cam_fn = None
    for name in ["generate_gradcam", "gradcam", "make_gradcam", "compute_gradcam"]:
        if hasattr(predict_utils, name):
            cam_fn = getattr(predict_utils, name)
            break

    if pred_fn is None or cam_fn is None:
        raise RuntimeError(
            "predict module must expose either predict_and_gradcam(...) or both predict/predict_image(...) and generate_gradcam(...)."
        )

    logger.info("Calling classification function from utils …")
    # Try common call signatures
    try:
        pred_label, confidences = pred_fn(image_path=image_path, class_names=class_names)  # type: ignore
    except TypeError:
        # Some modules accept only image_path
        try:
            pred_label, confidences = pred_fn(image_path)  # type: ignore
        except TypeError:
            pred_label, confidences = pred_fn(image_path, class_names)  # type: ignore
    logger.info("Calling Grad-CAM generator from utils …")
    try:
        gradcam_obj = cam_fn(image_path=image_path, output_dir=GRADCAM_DIR)  # type: ignore
    except TypeError:
        try:
            gradcam_obj = cam_fn(image_path, GRADCAM_DIR)  # type: ignore
        except TypeError:
            gradcam_obj = cam_fn(image_path)  # type: ignore
    return pred_label, confidences, gradcam_obj


def ensure_gradcam_path(gradcam_obj) -> str:
    """Accepts a numpy array/PIL.Image/str path and ensures it becomes a saved file path under static/gradcam."""
    try:
        # If it's already a path
        if isinstance(gradcam_obj, str) and os.path.exists(gradcam_obj):
            return gradcam_obj

        # Try PIL Image save
        from PIL import Image  # lazy import
        if isinstance(gradcam_obj, Image.Image):
            filename = f"gradcam_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
            out_path = os.path.join(GRADCAM_DIR, filename)
            gradcam_obj.save(out_path)
            return out_path

        # Try numpy array -> save via PIL
        import numpy as np
        if isinstance(gradcam_obj, np.ndarray):
            from PIL import Image  # type: ignore
            img = Image.fromarray(gradcam_obj)
            filename = f"gradcam_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
            out_path = os.path.join(GRADCAM_DIR, filename)
            img.save(out_path)
            return out_path
    except Exception:
        pass

    # Fallback: write bytes-like object if possible
    filename = f"gradcam_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(GRADCAM_DIR, filename)
    try:
        with open(out_path, "wb") as f:
            f.write(gradcam_obj)
        return out_path
    except Exception:
        # As a last resort, return under static even if not existent; frontend will handle error
        return out_path


def normalize_confidences(confidences, class_names: list[str]):
    """Convert confidences to a JSON-serializable dict[label -> float].

    Accepts: dict/Mapping label->score, list/tuple aligned with class_names,
    or numpy array. Returns dict with native Python floats.
    """
    try:
        import numpy as np  # local import
    except Exception:  # pragma: no cover
        np = None

    # If dict-like already
    if isinstance(confidences, dict):
        normalized = {}
        for k, v in confidences.items():
            try:
                normalized[str(k)] = float(v)
            except Exception:
                try:
                    normalized[str(k)] = float(v.item())  # type: ignore[attr-defined]
                except Exception:
                    normalized[str(k)] = 0.0
        return normalized

    # If it's a numpy array or list-like aligned to class_names
    values = confidences
    if np is not None and hasattr(np, "ndarray") and isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        out = {}
        for idx, label in enumerate(class_names):
            try:
                out[label] = float(values[idx])
            except Exception:
                out[label] = 0.0
        return out

    # Fallback: try to coerce to float and attach to predicted label later
    return {label: 0.0 for label in class_names}


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    # Class names as provided
    class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

    # Optional: preload model to reduce first-request latency, if utils exposes it
    model_handle = None
    if predict_utils is not None:
        for name in ["load_model", "get_model", "init_model"]:
            if hasattr(predict_utils, name):
                try:
                    loader = getattr(predict_utils, name)
                    logger.info(f"Preloading model via utils.{name}() …")
                    model_handle = loader()
                    logger.info("Model preloaded successfully")
                except Exception:
                    logger.exception("Model preload failed; proceeding without preloaded model")
                    model_handle = None
                break

    @app.route("/")
    def serve_index():
        return send_from_directory(APP_ROOT, "index.html")

    @app.route("/analysis")
    def serve_analysis():
        return send_from_directory(APP_ROOT, "analysis.html")

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded under form field 'image'"}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Save upload to a temp file
        filename = f"upload_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}_{file.filename}"
        upload_path = os.path.join(GRADCAM_DIR, filename)  # reuse same dir; it's served and ignored typically
        file.save(upload_path)
        logger.info(f"Upload saved to {upload_path}")

        try:
            # If utils exposes functions that accept a preloaded model, try that first
            if predict_utils is not None and hasattr(predict_utils, "predict_with_model"):
                logger.info("Invoking utils.predict_with_model …")
                pred_label, confidences = predict_utils.predict_with_model(  # type: ignore
                    image_path=upload_path, class_names=class_names, model=model_handle
                )
                if hasattr(predict_utils, "generate_gradcam_with_model"):
                    logger.info("Invoking utils.generate_gradcam_with_model …")
                    gradcam_obj = predict_utils.generate_gradcam_with_model(  # type: ignore
                        image_path=upload_path, model=model_handle, output_dir=GRADCAM_DIR
                    )
                else:
                    gradcam_obj = None
            else:
                logger.info("Invoking try_predict_with_utils …")
                pred_label, confidences, gradcam_obj = try_predict_with_utils(upload_path, class_names)
        except Exception as e:
            logger.exception("Prediction failed")
            return jsonify({"error": f"Prediction failed: {e}"}), 500

        # gradcam_obj may now be a dict with overlay/mask/bbox
        gradcam_overlay_path = None
        gradcam_mask_path = None
        gradcam_bbox = None
        if isinstance(gradcam_obj, dict):
            gradcam_overlay_path = gradcam_obj.get("overlay_path")
            gradcam_mask_path = gradcam_obj.get("mask_path")
            gradcam_bbox = gradcam_obj.get("bbox")
            gradcam_center = gradcam_obj.get("center")
            gradcam_path = gradcam_overlay_path or gradcam_mask_path
        else:
            gradcam_path = ensure_gradcam_path(gradcam_obj) if gradcam_obj is not None else None
            gradcam_center = None
        # Convert absolute path to URL under /static, only if the path exists
        gradcam_url = None
        gradcam_mask_url = None
        try:
            if gradcam_path and os.path.exists(gradcam_path):
                rel_gradcam = os.path.relpath(gradcam_path, STATIC_DIR).replace("\\", "/")
                gradcam_url = f"/static/{rel_gradcam}"
        except Exception:
            gradcam_url = None
        if gradcam_mask_path:
            try:
                if os.path.exists(gradcam_mask_path):
                    rel_mask = os.path.relpath(gradcam_mask_path, STATIC_DIR).replace("\\", "/")
                    gradcam_mask_url = f"/static/{rel_mask}"
            except Exception:
                gradcam_mask_url = None
        if gradcam_path:
            logger.info(f"Grad-CAM saved to {gradcam_path}")

        # Original uploaded image URL (we saved it under GRADCAM_DIR)
        rel_original = os.path.relpath(upload_path, STATIC_DIR).replace("\\", "/")
        original_url = f"/static/{rel_original}"

        # Normalize confidences to JSON-safe dict
        safe_confidences = normalize_confidences(confidences, class_names)
        # Debug: log per-class confidences sorted desc for troubleshooting
        try:
            sorted_conf = sorted(safe_confidences.items(), key=lambda kv: kv[1], reverse=True)
            logger.info("Confidences: " + ", ".join([f"{k}={v:.3f}" for k, v in sorted_conf]))
        except Exception:
            pass
        primary_conf = float(safe_confidences.get(pred_label, 0.0))

        suggestion = build_treatment_suggestion(pred_label)
        diagnostic = human_readable_diagnostic(pred_label, primary_conf)
        # If predicted no_tumor: do not show localization or Grad-CAM
        if pred_label == "no_tumor":
            localization = "NIL"
            gradcam_url = None
            gradcam_mask_url = None
            gradcam_bbox = None
        else:
            # Prefer precise localization from hotspot center if provided
            localization = "undetermined"
            try:
                if gradcam_center and gradcam_path and os.path.exists(gradcam_path):
                    from PIL import Image  # lazy import
                    cx, cy = gradcam_center if isinstance(gradcam_center, (list, tuple)) else (None, None)
                    img = Image.open(gradcam_path)
                    w, h = img.size
                    if isinstance(cx, int) and isinstance(cy, int) and w > 0 and h > 0:
                        hemisphere = "left" if cx < w/2 else "right"
                        y_rel = cy / float(h)
                        if y_rel < 0.33:
                            lobe = "frontal or parietal (superior)"
                        elif y_rel < 0.66:
                            lobe = "parietal or temporal (mid)"
                        else:
                            lobe = "temporal or occipital (inferior/posterior)"
                        localization = f"{lobe}, {hemisphere} hemisphere"
                    else:
                        localization = estimate_lobe_from_gradcam(gradcam_path)
                else:
                    localization = estimate_lobe_from_gradcam(gradcam_path)
            except Exception:
                localization = estimate_lobe_from_gradcam(gradcam_path)

        # If an LLM API key is configured, refine treatment suggestion via LLM (Groq preferred, then Grok)
        try:
            loc_text = localization if pred_label != "no_tumor" and localization else ""
            prompt_ts = (
                "Provide a concise, evidence-aligned treatment plan for the suspected brain tumor. "
                "Return 2-3 sentences, no bullets, no prognosis. Keep under 60 words.\n\n"
                f"Tumor type: {pred_label}\n"
                f"Confidence: {primary_conf:.3f}\n"
                f"Estimated location: {loc_text}\n"
                f"Baseline plan: {suggestion}"
            )
            llm_ts = ""
            if os.environ.get("GROQ_API_KEY", "").strip():
                llm_ts = _call_groq_api(prompt_ts).strip()
            elif os.environ.get("GROK_API_KEY", "").strip():
                llm_ts = _call_grok_api(prompt_ts).strip()
            if llm_ts:
                suggestion = llm_ts
        except Exception:
            pass

        # Probe overlay image size for frontend auto-view logic
        gradcam_size = None
        try:
            if gradcam_path and os.path.exists(gradcam_path):
                from PIL import Image as _Image
                _im = _Image.open(gradcam_path)
                gradcam_size = (int(_im.size[0]), int(_im.size[1]))
        except Exception:
            gradcam_size = None

        response = {
            "prediction": pred_label,
            "confidence": primary_conf,
            "confidences": safe_confidences,
            "gradcam_url": gradcam_url,
            "treatment_suggestion": suggestion,
            "diagnostic_text": diagnostic,
            "localization": localization,
            "highlights": build_highlight_tokens(pred_label, localization),
            "bbox": gradcam_bbox,
            "gradcam_mask_url": gradcam_mask_url,
            "original_url": original_url,
            "hotspot_center": gradcam_center,
            "gradcam_size": gradcam_size,
        }
        logger.info(f"Responding with prediction={pred_label}, confidence={primary_conf:.3f}")
        return jsonify(response)

    def _call_grok_api(prompt: str) -> str:
        api_key = os.environ.get("GROK_API_KEY", "").strip()
        if not api_key:
            return ""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "grok-beta",
                "messages": [
                    {"role": "system", "content": "You are a clinical report assistant. Keep language clear and concise."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            resp = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text or ""
        except Exception as e:
            logger.warning(f"Grok API call failed: {e}")
            return ""

    def _call_groq_api(prompt: str) -> str:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            return ""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                # Common Groq chat completions endpoint (OpenAI-compatible)
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": "You are a clinical report assistant. Keep language clear and concise."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text or ""
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}")
            return ""

    def _build_pdf(report_data: dict) -> str:
        # Create a formatted PDF report using Platypus with proper wrapping
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_LEFT
            from reportlab.lib import colors
        except Exception as e:
            raise RuntimeError("Report generation requires the 'reportlab' package. Install with: pip install reportlab") from e

        import unicodedata

        def norm(text: str) -> str:
            if not isinstance(text, str):
                return ""
            # Replace special dashes/bullets and normalize to ASCII to avoid font glyph issues
            text = text.replace("–", "-").replace("—", "-").replace("•", "-")
            return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

        filename = f"report_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}.pdf"
        out_path = os.path.join(REPORTS_DIR, filename)

        doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        title = ParagraphStyle('title', parent=styles['Heading1'], alignment=TA_LEFT, spaceAfter=12)
        h2 = ParagraphStyle('h2', parent=styles['Heading2'], spaceBefore=12, spaceAfter=6)
        body = ParagraphStyle('body', parent=styles['BodyText'], leading=14, spaceAfter=6)

        elems = []
        elems.append(Paragraph("Brain Tumor Analysis Report", title))
        patient_name = report_data.get('patient_name') or ''
        patient_phone = report_data.get('patient_phone') or ''
        if patient_name:
            elems.append(Paragraph(norm(f"Patient: {patient_name}"), styles['BodyText']))
        if patient_phone:
            elems.append(Paragraph(norm(f"Phone: {patient_phone}"), styles['BodyText']))
        elems.append(Paragraph(norm(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"), styles['BodyText']))
        elems.append(Spacer(1, 6))

        # Summary
        elems.append(Paragraph("SUMMARY", h2))
        elems.append(Paragraph(norm(f"Tumor type: {report_data.get('prediction','-')}"), body))
        elems.append(Paragraph(norm(f"Confidence: {(report_data.get('confidence',0.0)*100):.1f}%"), body))
        loc = report_data.get("localization") or "NIL"
        elems.append(Paragraph(norm(f"Estimated location: {loc}"), body))
        elems.append(Spacer(1, 6))

        narrative = report_data.get("narrative") or report_data.get("diagnostic_text") or ""
        if narrative:
            elems.append(Paragraph("DIAGNOSTIC NOTE", h2))
            elems.append(Paragraph(norm(narrative), body))
            elems.append(Spacer(1, 6))

        treatment = report_data.get("treatment_suggestion") or ""
        if treatment:
            elems.append(Paragraph("TREATMENT SUGGESTION", h2))
            elems.append(Paragraph(norm(treatment), body))
            elems.append(Spacer(1, 6))

        # Images side-by-side if available
        from urllib.parse import urlparse
        def to_path(url: str) -> str:
            if not url:
                return ""
            u = url
            # Strip origin if absolute URL
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    u = parsed.path or url
            except Exception:
                u = url
            if u.startswith("/static/"):
                rel = u.replace("/static/", "").lstrip("/")
                return os.path.join(STATIC_DIR, rel.replace("/", os.sep))
            # Fallback: treat as path relative to app root
            return os.path.join(APP_ROOT, u.lstrip("/"))

        gradcam_url = report_data.get("gradcam_url")
        orig_url = report_data.get("original_url")
        left_path = to_path(orig_url)
        right_path = to_path(gradcam_url)
        row = []
        img_w = 240
        img_h = 180
        try:
            if left_path and os.path.exists(left_path):
                row.append(RLImage(left_path, width=img_w, height=img_h))
            if right_path and os.path.exists(right_path):
                row.append(RLImage(right_path, width=img_w, height=img_h))
            if row:
                tbl = Table([row], colWidths=[img_w] * len(row))
                tbl.setStyle(TableStyle([
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('BOX', (0,0), (-1,-1), 0.25, colors.grey),
                ]))
                elems.append(Spacer(1, 6))
                elems.append(tbl)
        except Exception as e:
            elems.append(Paragraph(norm(f"[Image unavailable: {e}]"), styles['BodyText']))

        doc.build(elems)
        return out_path

    @app.route("/report", methods=["POST"])
    def generate_report():
        data = request.get_json(silent=True) or {}
        # Optionally call Grok to craft a human-readable narrative
        narrative = ""
        prediction = data.get("prediction", "")
        confidence = data.get("confidence", 0.0)
        localization = data.get("localization", "")
        diagnostic = data.get("diagnostic_text", "")
        suggestion = data.get("treatment_suggestion", "")

        prompt = (
            "Create a short, patient-friendly MRI brain tumor report. Include tumor type, confidence as %," \
            " location (if provided), one-paragraph diagnostic explanation, and concise treatment plan. " \
            "Avoid prognosis. Keep under 120 words.\n\n" \
            f"Tumor type: {prediction}\nConfidence: {confidence:.3f}\nLocation: {localization}\n" \
            f"Diagnostic: {diagnostic}\nTreatment: {suggestion}"
        )
        # Prefer Groq for narrative, then Grok; fall back to diagnostic text
        narrative = ""
        try:
            if os.environ.get("GROQ_API_KEY", "").strip():
                narrative = _call_groq_api(prompt)
            elif os.environ.get("GROK_API_KEY", "").strip():
                narrative = _call_grok_api(prompt)
        except Exception:
            narrative = ""
        if not narrative:
            narrative = diagnostic

        payload = {
            "prediction": prediction,
            "confidence": confidence,
            "localization": localization or "NIL",
            "diagnostic_text": diagnostic,
            "treatment_suggestion": suggestion,
            "gradcam_url": data.get("gradcam_url"),
            "original_url": data.get("original_url"),
            "narrative": narrative,
            "patient_name": data.get("patient_name", ""),
            "patient_phone": data.get("patient_phone", ""),
        }
        try:
            pdf_path = _build_pdf(payload)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        rel = os.path.relpath(pdf_path, STATIC_DIR).replace("\\", "/")
        return jsonify({"report_url": f"/static/{rel}"})

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


