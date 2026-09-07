import os
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field


# 1. Define the exact JSON structure we want Gemini to return
class ValidationResult(BaseModel):
    is_valid: bool = Field(description="True if the image matches the instruction perfectly, False otherwise.")
    reason: str = Field(description="A brief explanation of why it passed or failed.")


# Initialize the Gemini Client
# Best practice is setting GEMINI_API_KEY environment variable, or paste it directly below:
import os
import streamlit as st

# 1. Look for the key in the server's hidden environment variables or secrets file
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

# 2. Safety check: Stop the app if the server doesn't have the key configured yet
if not api_key:
    st.error("Missing Gemini API Key. Please configure it in your Streamlit Secrets.")
    st.stop()

# 3. Securely initialize the Gemini client
client = genai.Client(api_key=api_key)


def verify_app_image(uploaded_file, instruction_type: str) -> ValidationResult:
    """Sends the uploaded file directly to Gemini for validation."""
    # Convert the uploaded file bytes into a PIL Image object
    img = Image.open(uploaded_file)

    # Set up prompts based on selection
    if instruction_type == "Engine Photo":
        prompt = (
            "Analyze this image carefully. Is it a clear photo looking directly "
            "into a car's engine bay/compartment? Check if mechanical engine parts "
            "are explicitly visible. If it's a photo of the outside of the car, the interior, "
            "or an unrelated object, mark it as invalid."
        )
    else:
        prompt = (
            "Analyze this image carefully. Is it a clear, uncropped photo of a vehicle "
            "taken directly from the front view? It must clearly display the front grille "
            "and both headlights symmetrically. Side profiles or rear angles are strictly invalid."
        )

    # Call Gemini 2.5 Flash
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ValidationResult,
            temperature=0.1,
        ),
    )
    return response.parsed


# --- STREAMLIT WEB INTERFACE ---
st.set_page_config(page_title="Car Photo Validator", page_icon="🚗")
st.title("🚗 Car Photo Verification Dashboard")
st.write("Upload an image to test how the Gemini API validates your app's rules.")

# Step 1: Dropdown selection for instruction type
instruction = st.selectbox(
    "Select the active instruction rule:",
    ["Engine Photo", "Front View Car Photo"]
)

# Step 2: File uploader widget
uploaded_file = st.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image side-by-side or centered
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Process button
    if st.button("Validate Image ✨"):
        with st.spinner("Gemini is analyzing the image against instructions..."):
            try:
                # Call our validation function
                result = verify_app_image(uploaded_file, instruction)

                # Step 3: Display results beautifully on the webpage
                st.subheader("API Return Value")

                # Show an alert banner depending on the boolean pass/fail
                if result.is_valid:
                    st.success("✅ Image Validated Successfully!")
                else:
                    st.error("❌ Image Rejected!")

                # Show the raw structured data/JSON
                st.json({
                    "is_valid": result.is_valid,
                    "reason": result.reason
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")
