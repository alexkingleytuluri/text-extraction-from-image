import os
import tempfile

import streamlit as st

from services.traditional_ocr import predict_character
from services.printed_ocr import extract_text as printed_ocr
from services.deep_learning_ocr import extract_text as deep_learning_ocr
from services.gemini_ocr import extract_text as gemini_ocr
from services.mistral_ocr import extract_text as mistral_ocr


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Multi-Model OCR",
    page_icon="📝",
    layout="centered"
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📝 Multi-Model OCR System")

st.write(
    "Upload an image or PDF and choose an OCR method "
    "to extract text."
)


# ---------------------------------------------------------
# OCR method descriptions
# ---------------------------------------------------------

ocr_methods = {
    "Traditional ML — Character Recognition":
        "HOG + RBF-SVM for individual handwritten characters.",

    "Printed Document OCR":
        "Tesseract OCR for printed or typed images and PDFs.",

    "Deep Learning — Handwriting OCR":
        "Fine-tuned TrOCR for handwritten words and lines.",

    "Gemini AI":
        "Gemini multimodal API for handwriting and document OCR.",

    "Mistral OCR":
        "Mistral OCR API for images and PDF documents."
}


# ---------------------------------------------------------
# File upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image or PDF",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "pdf"
    ]
)


# ---------------------------------------------------------
# OCR selection
# ---------------------------------------------------------

ocr_method = st.selectbox(
    "Choose OCR Method",
    list(ocr_methods.keys())
)

st.caption(
    ocr_methods[ocr_method]
)


# ---------------------------------------------------------
# Deep Learning options
# ---------------------------------------------------------

use_dictionary = False

if ocr_method == "Deep Learning — Handwriting OCR":

    use_dictionary = st.checkbox(
        "Enable dictionary correction",
        value=True,
        help=(
            "Corrects recognized words using the "
            "project's handwriting vocabulary."
        )
    )


# ---------------------------------------------------------
# File preview
# ---------------------------------------------------------

if uploaded_file is not None:

    file_extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    if file_extension in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]:

        st.image(
            uploaded_file,
            caption="Uploaded image",
            use_container_width=True
        )

    else:

        st.info(
            f"PDF uploaded: {uploaded_file.name}"
        )


# ---------------------------------------------------------
# Run OCR
# ---------------------------------------------------------

if st.button(
    "🔍 Run OCR",
    type="primary",
    use_container_width=True
):

    if uploaded_file is None:

        st.warning(
            "Please upload an image or PDF first."
        )

    else:

        temp_path = None

        try:

            # -------------------------------------------------
            # Save uploaded file temporarily
            # -------------------------------------------------

            file_extension = os.path.splitext(
                uploaded_file.name
            )[1].lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name


            # -------------------------------------------------
            # Check file compatibility
            # -------------------------------------------------

            is_pdf = file_extension == ".pdf"

            if is_pdf and ocr_method in [
                "Traditional ML — Character Recognition",
                "Deep Learning — Handwriting OCR"
            ]:

                st.warning(
                    f"{ocr_method} currently works with "
                    "image files rather than PDF files. "
                    "Please upload JPG, JPEG, PNG, or WEBP."
                )

                st.stop()


            # -------------------------------------------------
            # Traditional ML
            # -------------------------------------------------

            if (
                ocr_method
                == "Traditional ML — Character Recognition"
            ):

                with st.spinner(
                    "Running Traditional ML OCR..."
                ):

                    result = predict_character(
                        temp_path
                    )


            # -------------------------------------------------
            # Printed OCR
            # -------------------------------------------------

            elif ocr_method == "Printed Document OCR":

                with st.spinner(
                    "Running Printed Document OCR..."
                ):

                    result = printed_ocr(
                        temp_path
                    )


            # -------------------------------------------------
            # Deep Learning
            # -------------------------------------------------

            elif (
                ocr_method
                == "Deep Learning — Handwriting OCR"
            ):

                with st.spinner(
                    "Running Deep Learning OCR..."
                ):

                    result = deep_learning_ocr(
                        temp_path,
                        use_dictionary=use_dictionary
                    )


            # -------------------------------------------------
            # Gemini
            # -------------------------------------------------

            elif ocr_method == "Gemini AI":

                with st.spinner(
                    "Sending document to Gemini AI..."
                ):

                    result = gemini_ocr(
                        temp_path
                    )


            # -------------------------------------------------
            # Mistral
            # -------------------------------------------------

            elif ocr_method == "Mistral OCR":

                with st.spinner(
                    "Sending document to Mistral OCR..."
                ):

                    result = mistral_ocr(
                        temp_path
                    )


            # -------------------------------------------------
            # Display result
            # -------------------------------------------------

            st.subheader("📄 Extracted Text")

            if result:

                st.text_area(
                    "OCR Result",
                    value=result,
                    height=350
                )

                # ---------------------------------------------
                # Download result
                # ---------------------------------------------

                st.download_button(
                    label="⬇️ Download Text",
                    data=result,
                    file_name="ocr_result.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            else:

                st.warning(
                    "The selected OCR method returned no text."
                )


        except Exception as error:

            st.error(
                f"OCR failed: {error}"
            )


        finally:

            # -------------------------------------------------
            # Remove temporary file
            # -------------------------------------------------

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                os.remove(temp_path)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Multi-Model OCR System • "
    "Traditional ML • Printed OCR • "
    "Deep Learning • Gemini • Mistral"
)