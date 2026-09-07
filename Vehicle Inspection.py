import os
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field


# 1. Define the exact JSON structure we want Gemini to return
class ValidationResult(BaseModel):
    is_valid: bool = Field(description="True if the image matches the instruction perfectly, False otherwise.")
    confidence: float = Field(
        description="A confidence score from 0 to 100 representing how strongly the image matches "
                     "the instruction (e.g. 98.5 means 98.5% confident it matches)."
    )
    reason: str = Field(description="A brief explanation of why it passed or failed.")


# 2. Look for the key in the server's hidden environment variables or secrets file
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

# 3. Safety check: Stop the app if the server doesn't have the key configured yet
if not api_key:
    st.error("Missing Gemini API Key. Please configure it in your Streamlit Secrets.")
    st.stop()

# 4. Securely initialize the Gemini client
client = genai.Client(api_key=api_key)


# 5. Central place for all instruction prompts.
#    To add a new angle in the future, just add one more key/value pair here
#    and it will automatically show up in the dropdown below.
INSTRUCTION_PROMPTS = {
    "Engine Photo": (
        "Analyze this image carefully. Is it a clear photo looking directly "
        "into a car's engine bay/compartment? Check if mechanical engine parts "
        "are explicitly visible. If it's a photo of the outside of the car, the interior, "
        "or an unrelated object, mark it as invalid."
    ),
    "Engine Bay Photo": (
        "Analyze this image carefully. Is it a clear photo taken with the hood/bonnet "
        "open, showing the full engine bay from above? The engine block, surrounding "
        "compartment walls, firewall, and under-hood components must be visible in one "
        "wide shot (not a tight close-up of a single part). If the hood is closed, if it's "
        "a close-up of a single component only, or if it's an unrelated part of the car, "
        "mark it as invalid."
    ),
    "Front View Car Photo": (
        "Analyze this image carefully. Is it a clear, uncropped photo of a vehicle "
        "taken directly from the front view? It must clearly display the front grille "
        "and both headlights symmetrically. Side profiles or rear angles are strictly invalid."
    ),
    "Rear View Car Photo": (
        "Analyze this image carefully. Is it a clear, uncropped photo of a vehicle "
        "taken directly from the rear/back view? It must clearly display the rear "
        "bumper, tail lights, and (if visible) the license plate area symmetrically. "
        "Front views, side profiles, or angled 3/4 shots are strictly invalid."
    ),
    "Side View Car Photo": (
        "Analyze this image carefully. Is it a clear, uncropped photo of a vehicle "
        "taken directly from the side (a full profile shot)? The entire length of the "
        "car from front bumper to rear bumper must be visible in one straight line, "
        "with both wheels on that side visible. Front-on, rear-on, or angled 3/4 shots "
        "are strictly invalid."
    ),
    "Exhaust Photo": (
        "Analyze this image carefully. Is it a clear, close-up photo looking directly "
        "at a car's exhaust pipe/tailpipe area underneath the rear bumper? The exhaust "
        "outlet(s) must be explicitly visible and in focus. If it's a photo of the "
        "engine bay, the full car, the interior, or an unrelated object, mark it as invalid."
    ),
    "Top View Car Photo": (
        "Analyze this image carefully. Is it a clear photo of a vehicle taken directly "
        "from above (a bird's-eye/top-down view), showing the roof, hood, and trunk "
        "outline from overhead? Photos taken from ground level or at an angle (front, "
        "side, or rear views) are strictly invalid."
    ),
}


import time


class CalibratedResult(BaseModel):
    is_valid: bool
    confidence: float          # agreement rate across samples (0-100) — the calibrated number
    avg_self_reported: float   # average of the model's own stated confidence, for comparison
    reason: str                # reason taken from a run that matches the majority verdict
    votes_valid: int
    votes_total: int


def _single_call(img, prompt, max_retries: int = 3) -> ValidationResult:
    """
    One sampled call to Gemini. Temperature > 0 so runs can actually disagree.
    Retries with backoff on rate-limit (429) errors, which are common on the free tier.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[img, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ValidationResult,
                    temperature=0.7,   # higher temp = meaningful variation between samples
                ),
            )
            return response.parsed
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(15 * (attempt + 1))  # back off: 15s, 30s, ...
                continue
            raise


def verify_app_image(uploaded_file, instruction_type: str, n_samples: int = 3,
                      progress_callback=None) -> CalibratedResult:
    """
    Runs Gemini n_samples times SEQUENTIALLY (free-tier friendly — avoids bursting
    the requests-per-minute limit that parallel calls would trigger) and derives a
    calibrated confidence score from how often the runs agree.
    """
    img = Image.open(uploaded_file)

    base_prompt = INSTRUCTION_PROMPTS[instruction_type]
    prompt = (
        f"{base_prompt}\n\n"
        "Additionally, provide a confidence score from 0 to 100 (as a decimal, e.g. 97.5) "
        "reflecting how certain you are that this image matches the criteria above."
    )

    results = []
    for i in range(n_samples):
        try:
            results.append(_single_call(img, prompt))
        except Exception:
            pass  # one failed sample shouldn't kill the whole batch
        if progress_callback:
            progress_callback(i + 1, n_samples)
        if i < n_samples - 1:
            time.sleep(4)  # small gap between calls to stay under free-tier RPM limits

    if not results:
        raise RuntimeError("All Gemini calls failed — no samples to aggregate. "
                            "You may have hit the free-tier rate limit; wait a minute and retry.")

    votes_valid = sum(1 for r in results if r.is_valid)
    votes_total = len(results)
    majority_valid = votes_valid > votes_total / 2

    # Agreement rate WITH the majority verdict = the calibrated confidence.
    agreeing = votes_valid if majority_valid else (votes_total - votes_valid)
    calibrated_confidence = (agreeing / votes_total) * 100

    avg_self_reported = sum(r.confidence for r in results) / votes_total

    # Use the reason from a sample that matches the majority verdict
    matching = [r for r in results if r.is_valid == majority_valid]
    reason = matching[0].reason if matching else results[0].reason

    return CalibratedResult(
        is_valid=majority_valid,
        confidence=calibrated_confidence,
        avg_self_reported=avg_self_reported,
        reason=reason,
        votes_valid=votes_valid,
        votes_total=votes_total,
    )


# --- STREAMLIT WEB INTERFACE ---
st.set_page_config(page_title="Car Photo Validator", page_icon="🚗")
st.title("🚗 Car Photo Verification Dashboard")
st.write("Upload an image to test how the Gemini API validates your app's rules.")

# Step 1: Dropdown selection for instruction type (built from the dict above)
instruction = st.selectbox(
    "Select the active instruction rule:",
    list(INSTRUCTION_PROMPTS.keys())
)

# Step 1b: How many samples to run for calibration.
# Kept low by default since the free tier has tight per-minute/per-day request limits —
# each validation uses n_samples API calls.
n_samples = st.slider(
    "Calibration samples (more = more reliable, but uses more free-tier quota)",
    min_value=2, max_value=6, value=3
)

# Step 2: File uploader widget
uploaded_file = st.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image side-by-side or centered
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Process button
    if st.button("Validate Image ✨"):
        progress_bar = st.progress(0, text="Starting...")

        def _update_progress(done, total):
            progress_bar.progress(done / total, text=f"Sample {done}/{total} complete...")

        try:
            # Call our validation function
            result = verify_app_image(uploaded_file, instruction, n_samples=n_samples,
                                       progress_callback=_update_progress)
            progress_bar.empty()

            # Step 3: Display results beautifully on the webpage
            st.subheader("API Return Value")

            # Show an alert banner depending on the boolean pass/fail
            if result.is_valid:
                st.success(f"✅ Image Validated Successfully! ({result.confidence:.1f}% calibrated confidence)")
            else:
                st.error(f"❌ Image Rejected! ({result.confidence:.1f}% calibrated confidence)")

            # Visual confidence meter
            st.progress(min(max(result.confidence / 100, 0.0), 1.0))

            st.caption(
                f"{result.votes_valid} of {result.votes_total} samples voted 'valid' "
                f"· model's own average self-reported confidence: {result.avg_self_reported:.1f}%"
            )

            # Show the raw structured data/JSON
            st.json({
                "is_valid": result.is_valid,
                "calibrated_confidence": result.confidence,
                "avg_self_reported_confidence": result.avg_self_reported,
                "votes_valid": result.votes_valid,
                "votes_total": result.votes_total,
                "reason": result.reason
            })

        except Exception as e:
            progress_bar.empty()
            st.error(f"An error occurred: {e}")