"""
Medical Report Extraction and Processing System

This module provides functionality for extracting and processing medical reports using OCR and AI.
It handles various types of medical reports (e.g., pathology, radiology) and extracts structured
information using multiple AI models (Azure OpenAI, Gemini) with fallback mechanisms.

Key Features:
- PDF to image conversion with multiple fallback methods
- OCR text extraction with preprocessing for better accuracy
- Image-based extraction using Gemini 1.5 Flash
- Report classification using AI models
- Text enhancement and verification using multiple AI providers
- Structured data extraction in JSON format
- Image-aware processing for better accuracy

Dependencies:
- OCR: Tesseract
- Image Processing: PIL, OpenCV
- AI Models: Azure OpenAI, Google Gemini
- Database: PostgreSQL
"""

# Standard library imports
import base64
import json
import os
import re
import time
from datetime import datetime

# Third-party imports
import cv2
import google.generativeai as genai
import numpy as np
import psycopg2
import pypdfium2 as pdfium
import pytesseract
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from psycopg2.extras import Json
from werkzeug.utils import secure_filename

pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)

# Remove PIL image size limit for large medical images
Image.MAX_IMAGE_PIXELS = 500000000

# # Azure OpenAI configuration
# API_BASE = "https://azure-isv-success-in.openai.azure.com/"
# API_KEY = "7c90d344cb524b9885202a7603641589"
DEPLOYMENT_NAME = "gpt-4o-mini"
# API_VERSION = "2024-06-01"

# # Initialize AI clients
# client = AzureOpenAI(
# Load environment variables
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Configure and initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GMODEL = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("Warning: GEMINI_API_KEY not found in .env file")
    GMODEL = None

def load_system_prompt(report_type):
    """
    Load the appropriate system prompt based on the type of medical report.

    This function selects a system prompt file depending on the report type.
    For certain categories of histopathology reports, common prompt files are used.
    If a specific prompt file for the report type exists, it is used instead.
    If no file is found, a default prompt message is returned.

    Args:
        report_type (str): Type of medical report to process (e.g., 'histopathology_biopsy', 'histopathology_cytology')

    Returns:
        str: The content of the system prompt from the corresponding file, or a default prompt message if the file is not found.
    """
    # Define lists of histopathology report types for classification
    histopathology_surgical_types = [
        "histopathology_biopsy",
        "histopathology_lumpectomy",
        "histopathology_resection"
    ]

    histopathology_diagnostic_types = [
        "histopathology_cytology",
        "histopathology_fnac"
    ]

    # Determine the appropriate prompt file path based on the report type
    if any(
        report_type.lower().startswith(surgical_type)
        for surgical_type in histopathology_surgical_types
    ):
        # Use a common prompt for surgical histopathology reports
        prompt_file_path = os.path.join(
            "System_prompts", "histopathology_surgical_prompt.txt"
        )
        print(f"Using common surgical histopathology prompt for {report_type}")
    elif any(
        report_type.lower().startswith(diagnostic_type)
        for diagnostic_type in histopathology_diagnostic_types
    ):
        # Use a common prompt for diagnostic histopathology reports
        prompt_file_path = os.path.join(
            "System_prompts", "histopathology_diagnostic_prompt.txt"
        )
        print(f"Using common diagnostic histopathology prompt for {report_type}")
    else:
        # Use a specific prompt file for other report types
        prompt_file_path = os.path.join("System_prompts", f"{report_type}_prompt.txt")

    # Load and return the content of the prompt file, or a default message if file not found
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, "r") as file:
            return file.read()
    else:
        return ("Assistant is a large language model trained by OpenAI. "
                "Enhance the extracted medical text and provide it in JSON "
                "format.")

def generate_prompt(extracted_text, report_type):
    """
    Generate system and user prompts for text processing.
    
    Args:
        extracted_text (str): The text extracted from the medical report
        report_type (str): Type of medical report
    
    Returns:
        tuple: A pair of (system_prompt, user_prompt) strings
    """
    system_prompt = load_system_prompt(report_type)  # Load prompt dynamically
    user_prompt = extracted_text
    return system_prompt, user_prompt

def encode_image(image_path):
    """
    Encode an image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def preprocess_image(image_path):
    """
    Preprocess image to improve OCR quality.
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        PIL.Image: Preprocessed image with enhanced contrast and sharpness
    """
    image = Image.open(image_path).convert("L")
    image = image.filter(ImageFilter.MedianFilter(size=3))  # Median filter
    image = image.filter(ImageFilter.SHARPEN)  # Sharpen image
    enhancer = ImageEnhance.Contrast(image)  # Enhance contrast
    image = enhancer.enhance(2)
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Binary image
    return image

def resize_if_necessary(image_path):
    """
    Resize image if it exceeds the maximum pixel limit.
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        str: Path to the resized image if resizing was necessary, otherwise original path
    """
    img = Image.open(image_path)
    width, height = img.size
    current_pixels = width * height
    max_pixels = 33177600
    if current_pixels > max_pixels:
        scale_factor = (max_pixels / current_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img_resized = img.resize((new_width, new_height))
        resized_path = "resized_image.png"
        img_resized.save(resized_path)
        return resized_path
    return image_path

def convert_pdf_to_images_option1(pdf_path):
    """
    Convert PDF to images using pdf2image library.
    
    Args:
        pdf_path (str): Path to the input PDF file
    
    Returns:
        list: List of paths to the converted image files, or None if conversion fails
    """
    try:
        images = convert_from_path(pdf_path)  # Convert all pages
        image_paths = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]

        for i, image in enumerate(images):  # Save each page as image
            image_path = f"{base_name}_page_{i + 1}_converted.png"
            image.save(image_path, "PNG")
            image_paths.append(image_path)

        return image_paths
    except Exception as e:
        print(f"Option 1 failed with error: {e}")
        return None

def convert_pdf_to_images_option2(pdf_path, scale=300 / 72):
    """
    Convert PDF to images using pypdfium2 library.
    
    Args:
        pdf_path (str): Path to the input PDF file
        scale (float, optional): Scale factor for rendering. Defaults to 300/72
    
    Returns:
        list: List of paths to the converted image files, or None if conversion fails
    """
    try:
        pdf_file = pdfium.PdfDocument(pdf_path)
        page_indices = [i for i in range(len(pdf_file))]  # Get all pages
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]

        renderer = pdf_file.render(  # Render pages
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices,
            scale=scale,
        )

        image_paths = []
        images = list(renderer)  # Convert generator to list
        for i, image in enumerate(images):  # Save all images
            image_path = f"{base_name}_page_{i + 1}_converted.png"
            image.save(image_path, format='PNG')
            image_paths.append(image_path)

        return image_paths

    except Exception as e:
        print(f"Option 2 failed with error: {e}")
        return None

def convert_pdf_to_images(pdf_path):
    """
    Convert PDF to images using fallback logic between different conversion methods.
    
    Args:
        pdf_path (str): Path to the input PDF file
    
    Returns:
        list: List of paths to the successfully converted image files
    """
    output_dir = "converted_images"  # Output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    final_image_paths = []

    try:  # Attempt Option 1
        image_paths = convert_pdf_to_images_option1(pdf_path)
        if image_paths and len(image_paths) > 0:
            insufficient_text = False
            temp_image_paths = []

            for i, img_path in enumerate(image_paths):  # Process each image
                text = extract_text_from_image(img_path)
                word_count = len(text.split())
                print(f"Option 1 - Page {i + 1} word count: {word_count}")

                if word_count < 14:  # Check for insufficient text
                    insufficient_text = True
                    print(f"Page {i + 1} has insufficient text ({word_count} words)")
                    break

                final_path = os.path.join(
                    output_dir, f"{base_name}_page_{i + 1}_converted.png"
                )
                if os.path.exists(img_path):
                    os.replace(img_path, final_path)
                    temp_image_paths.append(final_path)

            for img_path in image_paths:  # Clean up original images
                if os.path.exists(img_path):
                    os.remove(img_path)

            if not insufficient_text:
                return temp_image_paths
            else:
                print("Switching to Option 2 due to insufficient text")
                for path in temp_image_paths:  # Clean up Option 1 images
                    if os.path.exists(path):
                        os.remove(path)

    except Exception as e:
        print(f"Error in Option 1: {e}")

    try:  # Use Option 2
        print("Attempting Option 2 conversion")
        image_paths = convert_pdf_to_images_option2(pdf_path)
        if image_paths and len(image_paths) > 0:
            for i, img_path in enumerate(image_paths):  # Move each image
                final_path = os.path.join(
                    output_dir, f"{base_name}_page_{i + 1}_converted.png"
                )
                if os.path.exists(img_path):
                    os.replace(img_path, final_path)
                    final_image_paths.append(final_path)
            return final_image_paths
    except Exception as e:
        print(f"Error in Option 2: {e}")

    return None

def compress_image_for_api(image_path, max_size_mb=19):
    """
    Compress and resize image to ensure it's under the API size limit.
    
    Args:
        image_path (str): Path to the input image
        max_size_mb (int, optional): Maximum file size in MB. Defaults to 19
    
    Returns:
        str: Path to the compressed image
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            # Initial quality
            quality = 95
            output_path = f"compressed_{os.path.basename(image_path)}"

            while True:
                # Save with current quality
                img.save(output_path, 'JPEG', quality=quality)
                # Check file size
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                if size_mb <= max_size_mb:
                    return output_path
                # Reduce quality if file is still too large
                quality -= 10

                # If quality is too low, try resizing
                if quality < 30:
                    # Resize image
                    width, height = img.size
                    new_width = int(width * 0.8)
                    new_height = int(height * 0.8)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    quality = 95  # Reset quality

                # Prevent infinite loop
                if quality < 20 and size_mb > max_size_mb:
                    raise ValueError("Unable to compress image sufficiently")
    except Exception as e:
        print(f"Error compressing image: {e}")
        return None

def extract_text_from_image(image_path):
    """
    Extract text from a single image using Tesseract or another OCR engine.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text
    """
    # Example using Tesseract:
    try:
        # Preprocess image if needed
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {e}")
        return ""


def extract_text_using_gemini(image_paths):
    """
    Extract text directly from images using Gemini 1.5 Flash model.
    
    Args:
        image_paths (list): List of paths to the images
    
    Returns:
        str: Extracted text from the images
        bool: True if extraction was successful, False otherwise
    """
    try:
        content_parts = []
        
        # Add system instructions for text extraction
        instruction = """
        Extract all text from these medical report images. 
        Pay special attention to:
        1. Patient information
        2. Test results and values
        3. Medical terminology
        4. Doctor's notes and diagnoses
        5. Reference ranges and indicators

        Provide the extracted text in a clear, structured format preserving the overall layout.
        """
        content_parts.append(instruction)
        
        # Add images for analysis using base64 encoding
        for i, image_path in enumerate(image_paths):
            try:
                # Convert image to base64
                with open(image_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
                
                # Create Part object with MIME type
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": base64_encoded
                }
                
                # Add to content parts
                content_parts.append(image_part)
                print(f"Successfully added image {i+1} to content parts")
                
            except Exception as img_error:
                print(f"Error processing image {image_path}: {img_error}")
                continue
        
        if len(content_parts) <= 1:  # Only instruction, no images
            return "No valid images for extraction", False
            
        # Configure the model for text extraction
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        # Generate response with the configured model using multipart content
        print(f"Sending {len(content_parts)-1} images to Gemini")
        
        # Process images one at a time if there are multiple
        if len(content_parts) > 2:  # More than instruction + 1 image
            extracted_texts = []
            
            # Process each image separately to avoid potential model limitations
            for i in range(1, len(content_parts)):
                try:
                    single_image_parts = [content_parts[0], content_parts[i]]
                    print(f"Processing image {i} with Gemini")
                    
                    single_response = GMODEL.generate_content(
                        single_image_parts,
                        generation_config=generation_config
                    )
                    
                    if single_response and single_response.text:
                        extracted_texts.append(f"===== PAGE {i} =====\n{single_response.text.strip()}")
                except Exception as e:
                    print(f"Error processing image {i}: {e}")
                    continue
            
            if not extracted_texts:
                return "Failed to extract text from any images", False
                
            combined_text = "\n\n".join(extracted_texts)
            print("Successfully extracted text from multiple images using Gemini")
            return combined_text, True
            
        else:
            # Single image, process directly
            response = GMODEL.generate_content(
                content_parts,
                generation_config=generation_config
            )
            
            if not response or not response.text:
                return "No text extracted by Gemini", False
                
            extracted_text = response.text.strip()
            print("Successfully extracted text using Gemini")
            
            # Validate the extracted text
            if len(extracted_text.split()) < 20:  # Arbitrary threshold
                print("Extracted text too short, possible extraction failure")
                return extracted_text, False
                
            return extracted_text, True
        
    except Exception as e:
        print(f"Error in Gemini text extraction: {e}")
        return f"Extraction failed: {str(e)}", False

def extract_text_using_openai(image_paths):
    """
    Extract text directly from images using OpenAI's API with image URL input.
    
    Args:
        image_paths (list): List of paths to the images
    
    Returns:
        str: Extracted text from the images
        bool: True if extraction was successful, False otherwise
    """
    try:
        combined_text = ""
        for i, image_path in enumerate(image_paths):
            # Compress and prepare the image for API submission
            compressed_path = compress_image_for_api(image_path)
            if not compressed_path:
                print(f"Failed to process image {image_path}, skipping...")
                continue
            
            # Encode image to base64 URL format
            base64_encoded = encode_image(compressed_path)
            image_url = f"data:image/jpeg;base64,{base64_encoded}"
            
            prompt = (
                "Extract all text from this medical report image. "
                "Focus on patient details, test results, and diagnoses."
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # specify the OpenAI model
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}]}
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            
            if response and response.choices:
                page_text = response.choices[0].message.content.strip()
                combined_text += f"\n===== PAGE {i+1} =====\n{page_text}"
            else:
                print(f"OpenAI extraction failed for image {i+1}")
                return combined_text, False
            
            # Clean up compressed image after use
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
        
        return combined_text, True
    except Exception as e:
        print(f"Error in OpenAI text extraction: {e}")
        return "", False

def classify_report(corrected_text, system_prompt_content):
    """
    Classify the report using Gemini first, falls back to Azure if error occurs.
    
    Args:
        corrected_text (str): The text to classify
        system_prompt_content (str): System prompt for the classification
    
    Returns:
        dict: Classification results including report type and confidence score
    """
    try:  # First attempt with Gemini
        result = classify_report_with_gemini(
            corrected_text, system_prompt_content
        )

        if (  # Check for Gemini failure
            result["report_type"] == "Unknown" and
            result["confidence_score"] == 0 and
            not result["keywords_identified"]
        ):
            print("Gemini classification failed or unknown, trying Open AI...")
            result = classify_report_with_azure(  # Fall back to Azure
                corrected_text, system_prompt_content
            )

        return result

    except Exception as e:
        print(f"Error in Gemini classification: {str(e)}, falling back to Azure...")
        return classify_report_with_azure(  # Fallback to Azure on Gemini error
            corrected_text, system_prompt_content
        )

def classify_report_with_gemini(corrected_text, system_prompt_content):
    """
    Classify report using Gemini 1.5 Flash model.
    
    Args:
        corrected_text (str): The text to classify
        system_prompt_content (str): System prompt for the classification
    
    Returns:
        dict: Classification results from Gemini model
    """
    print("This is corrected text",corrected_text)
    if not system_prompt_content:
        print("Error: System prompt is empty")
        return {"report_type": "Unknown", "confidence_score": 0,
                "keywords_identified": []}

    try:
        print(f"Using system prompt (first 100 chars): "
              f"{system_prompt_content[:100]}...")

        chat = GMODEL.start_chat(history=[])
        response = chat.send_message(
            f"{system_prompt_content}\n\n{corrected_text}"
        )

        classification_result = response.text
        print(f"Raw Classification Result by gemini: {classification_result}")

        cleaned_result = classification_result.strip()  # Clean up response

        if cleaned_result.startswith("```json"):  # Handle JSON formatting
            cleaned_result = cleaned_result[7:]
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3]
        elif cleaned_result.startswith("```"):
            cleaned_result = cleaned_result[3:]
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3]

        cleaned_result = cleaned_result.strip()

        try:  # Parse JSON
            result_json = json.loads(cleaned_result)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Cleaned result: {cleaned_result}")
            json_match = re.search(r'\{.*\}', cleaned_result, re.DOTALL)
            if json_match:
                try:
                    result_json = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {
                        "report_type": "Unknown",
                        "confidence_score": 0,
                        "keywords_identified": []
                    }
            else:
                return {
                    "report_type": "Unknown",
                    "confidence_score": 0,
                    "keywords_identified": []
                }

        report_type = None  # Standardize report type

        if "category" in result_json:
            report_type = result_json["category"]
        elif "type" in result_json:
            report_type = result_json["type"]
        elif "report_type" in result_json:
            report_type = result_json["report_type"]

        if report_type:
            report_type_mapping = {  # Map report types
                "SERUM": "Serum_Analysis",
                "CBC": "CBC",
                "ENDOSCOPY": "Endoscopy",
                "CLINICAL BIOCHEMISTRY": "Clinical_Biochemistry",
                "BIOCHEMISTRY": "Clinical_Biochemistry",
                "IMMUNOHISTOCHEMISTRY": "IMMUNOHISTOCHEMISTRY",
                "IHC_HCG": "IHC_HCG",
                "HISTOPATHOLOGY": "Histopathology",
                "HISTOLOGY": "Histopathology",
                "Molecular_Biochemistry": "Clinical_Biochemistry"
            }

            report_type = report_type.upper()
            standardized_type = report_type_mapping.get(report_type, report_type)

            standardized_response = {  # Create standardized response
                "report_type": standardized_type,
                "confidence_score": (
                    result_json.get("confidence",
                                    result_json.get("match_percentage", 0))
                    if isinstance(result_json.get("confidence"), (int, float))
                    else 0.5
                ),
                "keywords_identified": result_json.get("matched_keywords", []),
                "reason": result_json.get("reason", "No reason provided")
            }
            if report_type.upper() == "HISTOPATHOLOGY":
                subcategory = result_json.get("subcategory", "Unknown")
                standardized_response["subcategory"] = subcategory
                standardized_response["report_type"] = (
                    f"histopathology_{subcategory.lower()}"
                )
            return standardized_response
            print("Final standardized response:", standardized_response)  

        return {"report_type": "Unknown", "confidence_score": 0,
                "keywords_identified": [], "reason": "Classification failed"}

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        print(f"Full error details: {str(e.__class__.__name__)}")
        return {"report_type": "Unknown", "confidence_score": 0,
                "keywords_identified": [],"reason": f"Error occurred: {str(e)}"}

def classify_report_with_azure(corrected_text, system_prompt_content):
    """
    Classify report using Azure OpenAI LLM.
    
    Args:
        corrected_text (str): The text to classify
        system_prompt_content (str): System prompt for the classification
    
    Returns:
        dict: Classification results from Azure OpenAI
    """
    try:
        classification_response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_content,
                },
                {
                    "role": "user",
                    "content": corrected_text,
                }
            ],
            temperature=0.7,
            max_tokens=2024,
            top_p=1,
        )

        classification_result = classification_response.choices[0].message.content
        # print(f"Raw Classification Result: {classification_result}")

        cleaned_result = classification_result.strip()  # Clean up response
        if cleaned_result.startswith("```json"):
            cleaned_result = cleaned_result[7:-3]

        result_json = json.loads(cleaned_result)  # Parse JSON

        report_type = None  # Standardize report type

        if "category" in result_json:
            report_type = result_json["category"]
        elif "type" in result_json:
            report_type = result_json["type"]
        elif "report_type" in result_json:
            report_type = result_json["report_type"]

        if report_type:
            report_type_mapping = {  # Map report types
                "SERUM": "Serum_Analysis",
                "CBC": "CBC",
                "ENDOSCOPY": "Endoscopy",
                "CLINICAL BIOCHEMISTRY": "Clinical_Biochemistry",
                "BIOCHEMISTRY": "Clinical_Biochemistry",
                "IMMUNOHISTOCHEMISTRY": "IMMUNOHISTOCHEMISTRY",
                "IHC_HCG": "IHC_HCG",
                "HISTOPATHOLOGY": "Histopathology",
                "HISTOLOGY": "Histopathology",
                "Molecular_Biochemistry": "Clinical_Biochemistry"
            }

            report_type = report_type.upper()
            standardized_type = report_type_mapping.get(report_type, report_type)

            standardized_response = {  # Create standardized response
                "report_type": standardized_type,
                "confidence_score": (
                    result_json.get("confidence",
                                    result_json.get("match_percentage", 0))
                    if isinstance(result_json.get("confidence"), (int, float))
                    else 0.5
                ),
                "keywords_identified": result_json.get("matched_keywords", [])
            }
            if report_type.upper() == "HISTOPATHOLOGY":
                subcategory = result_json.get("subcategory", "Unknown")
                standardized_response["subcategory"] = subcategory
                standardized_response["report_type"] = (
                    f"histopathology_{subcategory.lower()}"
                )

            return standardized_response

        return {"report_type": "Unknown", "confidence_score": 0,
                "keywords_identified": []}

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return {"report_type": "Unknown", "confidence_score": 0,
                "keywords_identified": []}

def read_system_prompt(prompt_file):
    """
    Read system prompt from file.
    
    Args:
        prompt_file (str): Path to the prompt file
    
    Returns:
        str: Content of the prompt file, or default prompt if file not found
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return None  # Default system prompt if file not found

def enhance_text_with_azure(extracted_text, report_type, image_paths=None):
    """
    Enhanced version that uses Azure OpenAI for text enhancement with image support.
    
    Args:
        extracted_text (str): The text to enhance
        report_type (str): Type of the report
        image_paths (list, optional): List of paths to images. Defaults to None
    
    Returns:
        str: Enhanced text in JSON format
    """
    system_prompt, user_prompt = generate_prompt(extracted_text, report_type)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
            top_p=0.95,
            response_format={"type": "json_object"}
        )

        enhanced_text = response.choices[0].message.content
        print("\nReceived response from Azure OpenAI after enhancement")
        # print("THIS IS ENHANCED TEXT", enhanced_text)
        print(enhanced_text)
        return enhanced_text

    except Exception as e:
        print(f"Azure OpenAI failed with error: {e}")
        print("Falling back to Gemini...")
        try:
            return enhance_text_with_gemini(  # Fall back to Gemini
                extracted_text, report_type, image_paths
            )

        except Exception as gemini_error:
            print(f"Gemini also failed with error: {gemini_error}")
            return None

def enhance_text_with_gemini(extracted_text, report_type, image_paths=None):
    """
    Enhanced version using Gemini 1.5 Flash model with structured configuration.
    
    Args:
        extracted_text (str): The text to enhance
        report_type (str): Type of the report
        image_paths (list, optional): List of paths to images. Defaults to None
    
    Returns:
        str: Enhanced text in JSON format
    """
    try:
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
            "response_mime_type": "text/plain",
        }

        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )

        system_prompt, user_prompt = generate_prompt(extracted_text, report_type)
            # print("This is user prompt", user_prompt)

        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        chat = gemini_model.start_chat(history=[])
        response = chat.send_message(combined_prompt)

        enhanced_text = response.text
        print("\nReceived response from Gemini")

        try:
            # Clean up JSON response
            temp = re.sub(r'```json\n|```', '', enhanced_text).strip()
            enhanced_text = re.sub(r'```json\n|```', '', enhanced_text).strip()

            # Replace newlines with spaces
            temp = re.sub(r'\n', ' ', temp)

            try:
                enhanced_text = json.loads(temp)
                if isinstance(enhanced_text, list) and len(enhanced_text) > 0:
                        enhanced_text = enhanced_text[0]
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                enhanced_text = None

            if enhanced_text is not None:
                enhanced_text = json.dumps(enhanced_text)
            json_response = json.loads(enhanced_text)
            return json.dumps(json_response, indent=2)
        except json.JSONDecodeError:
            print(
                    "Warning: Response was not in valid JSON format. "
                    "Returning raw response."
                )
            return enhanced_text

    except Exception as e:
        print(f"Error during enhancement with Gemini: {e}")
        return None

def verify_results_with_azure(enhanced_text, image_paths, report_type):
    """
    Verify and validate extracted medical data using Azure OpenAI.

    This function performs a second pass over the enhanced text to:
    1. Validate numerical values against reference ranges
    2. Cross-reference data with visual elements in images
    3. Pay special attention to circled values in measurement bars
    4. Ensure consistent JSON structure
    5. Check patient information accuracy
    
    Args:
        enhanced_text (str): Previously enhanced text to verify
        image_paths (list): Paths to report images for visual validation
        report_type (str): Type of medical report for context
    
    Returns:
        str: JSON-formatted verified data with corrected values
    """
    system_prompt = load_system_prompt(report_type)

    verification_prompt = (
        "Verify the accuracy of the following medical report data by comparing "
        "it with the provided images and also check for patient name which is "
        "important.\nUse the same structure and format as defined in the system "
        "prompt and if the result is not in valid json format then correct it. "
        "Analyse the image and if there is any values inside circles then give "
        "more attention to it to give the correct results if it's incorrect.\n\n"
        f"Enhanced Data:\n{enhanced_text}\n\n"
        "####PRIORITY\n"
        "Always extract the numerical values shown inside circles on the "
        "measurement bars - these represent the actual test results.\n"
        "Pay special attention to distinguishing between:\n"
        "- Actual results (typically shown in circles)\n"
        "- Reference ranges (typically shown as ranges like 0-7.22)\n\n"
        "Instructions:\n"
        "1. Compare all values with the images\n"
        "2. Maintain the exact same JSON structure as the enhanced data\n"
        "3. Update any incorrect values\n"
        "4. Add any missing values visible in the images\n"
        "5. Keep all field names and structure identical to the system prompt\n"
        "6. Return the complete verified data in the same format\n"
        "7. If the values are in circles then, Focus on extracting values from "
        "the circles in the image. Do not use the reference range values.\n\n"
        "Important note:\n"
        "I want the same structure as in the system prompt. Don't change the "
        "structure.\nFor each test parameter, extract the actual measured value "
        "(shown in circles), NOT the reference range values."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Prepare content with text and images
    content = [
        {
            "type": "text",
            "text": verification_prompt
        }
    ]

    # Add images for visual verification
    if image_paths:
        for image_path in image_paths:
            compressed_path = compress_image_for_api(image_path)
            if compressed_path:
                try:
                    base64_image = encode_image(compressed_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    })
                finally:
                    if os.path.exists(compressed_path):
                        os.remove(compressed_path)

    messages.append({
        "role": "user",
        "content": content
    })

    try:
        # Get verification from Azure OpenAI
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
            top_p=0.95,
            response_format={"type": "json_object"}
        )

        enhanced_text = response.choices[0].message.content
        print("Received verification response from Azure OpenAI")

        try:
            json_response = json.loads(enhanced_text)

            def clean_text_values(obj):
                """Remove unwanted characters and normalize spacing in text values."""
                if isinstance(obj, dict):
                    return {k: clean_text_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_text_values(item) for item in obj]
                elif isinstance(obj, str):
                    cleaned = obj.replace('\n', ' ')
                    return ' '.join(cleaned.split())
                return obj

            cleaned_json = clean_text_values(json_response)
            return json.dumps(cleaned_json, indent=2)

        except json.JSONDecodeError:
            print("Warning: Non-JSON response received, returning cleaned text")
            cleaned_text = enhanced_text.replace('\n', ' ')
            print(cleaned_text)
            return ' '.join(cleaned_text.split())

    except Exception as e:
        print(f"Azure OpenAI verification failed: {e}")
        print("Falling back to Gemini...")

        try:
            return verify_results_with_gemini(
                enhanced_text,
                image_paths,
                report_type
            )
        except Exception as gemini_error:
            print(f"Gemini verification also failed: {gemini_error}")
            return None

def verify_results_with_gemini(enhanced_text, image_paths, report_type):
    """
    Verify medical report data using Google's Gemini model.
    """
    try:
        # Parse input text
        if isinstance(enhanced_text, str):
            try:
                enhanced_json = json.loads(enhanced_text)
            except json.JSONDecodeError:
                enhanced_json = {"raw_text": enhanced_text}
        else:
            enhanced_json = enhanced_text

        system_prompt = load_system_prompt(report_type)

        verification_prompt = (
        "Verify the accuracy of the following medical report data by comparing "
        "it with the provided images and also check for patient name which is "
        "important.\nUse the same structure and format as defined in the system "
        "prompt and if the result is not in valid json format then correct it. "
        "Analyse the image and if there is any values inside circles then give "
        "more attention to it to give the correct results if it's incorrect.\n\n"
        f"Enhanced Data:\n{enhanced_text}\n\n"
        "####PRIORITY\n"
        "Always extract the numerical values shown inside circles on the "
        "measurement bars - these represent the actual test results.\n"
        "Pay special attention to distinguishing between:\n"
        "- Actual results (typically shown in circles)\n"
        "- Reference ranges (typically shown as ranges like 0-7.22)\n\n"
        "Instructions:\n"
        "1. Compare all values with the images\n"
        "2. Maintain the exact same JSON structure as the enhanced data\n"
        "3. Update any incorrect values\n"
        "4. Add any missing values visible in the images\n"
        "5. Keep all field names and structure identical to the system prompt\n"
        "6. Return the complete verified data in the same format\n"
        "7. If the values are in circles then, Focus on extracting values from "
        "the circles in the image. Do not use the reference range values.\n\n"
        "Important note:\n"
        "I want the same structure as in the system prompt. Don't change the "
        "structure.\nFor each test parameter, extract the actual measured value "
        "(shown in circles), NOT the reference range values."
        )

        # Prepare content for Gemini
        content_parts = [verification_prompt]

        # Add images for visual verification
        if image_paths:
            for image_path in image_paths:
                try:
                    # Load image directly
                    with Image.open(image_path) as img:
                        # Convert to RGB if necessary
                        if img.mode in ('RGBA', 'P'):
                            img = img.convert('RGB')
                        
                        # Resize if needed
                        width, height = img.size
                        current_pixels = width * height
                        max_pixels = 25000000
                        
                        if current_pixels > max_pixels:
                            scale_factor = (max_pixels / current_pixels) ** 0.5
                            new_width = int(width * scale_factor)
                            new_height = int(height * scale_factor)
                            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Add image to content parts
                        content_parts.append(img.copy())
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue

        # Generate verification response
        response = GMODEL.generate_content(
            content_parts,
            generation_config={
                'temperature': 0.3,
                'top_p': 0.95,
                'max_output_tokens': 2048,
            }
        )

        if not response or not response.text:
            raise Exception("Empty response from Gemini")

        # Clean up response text
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        try:
            # Parse and clean the verified data
            verified_json = json.loads(response_text.strip())

            def clean_text_values(obj):
                """Normalize text values by removing unwanted characters."""
                if isinstance(obj, dict):
                    return {k: clean_text_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_text_values(item) for item in obj]
                elif isinstance(obj, str):
                    cleaned = obj.replace('\n', ' ').replace('/', ' ')
                    return ' '.join(cleaned.split())
                return obj

            cleaned_json = clean_text_values(verified_json)
            return json.dumps(cleaned_json, indent=2)

        except json.JSONDecodeError as json_error:
            print(f"Error parsing Gemini response: {json_error}")
            return json.dumps(enhanced_json, indent=2)

    except Exception as e:
        print(f"Error in Gemini verification: {e}")
        if isinstance(enhanced_text, str):
            return enhanced_text
        return json.dumps(enhanced_text, indent=2)

def extract_text_from_pdf_with_image_based(image_paths):
    """
    Extract text directly from PDF images using Gemini model.
    If Gemini extraction fails, fall back to OpenAI's 03-mini model.
    If OpenAI extraction also fails, fall back to OCR.
    
    Args:
        image_paths (list): List of paths to the converted PDF images
        
    Returns:
        str: Combined extracted text
        bool: Success indicator
    """
    print("Attempting image-based extraction using Gemini 1.5 Flash...")
    
    # First try image-based extraction using Gemini
    extracted_text, success = extract_text_using_gemini(image_paths)
    
    if success:
        print("Successfully extracted text using Gemini")
        return extracted_text, True
    else:
        print("Gemini extraction failed, falling back to OpenAI 03-mini extraction...")
        extracted_text, openai_success = extract_text_using_openai(image_paths)
        
        if openai_success:
            print("Successfully extracted text using OpenAI")
            return extracted_text, True
        else:
            print("OpenAI extraction failed, falling back to OCR extraction...")
            extracted_text = extract_text_from_pdf_with_ocr(image_paths)
            return extracted_text, False

def extract_text_from_pdf_with_ocr(image_paths):
    """
    Extract text from PDF images using OCR as a fallback method.
    
    Args:
        image_paths (list): List of paths to the converted PDF images
        
    Returns:
        str: Combined extracted text
    """
    print("Falling back to OCR-based extraction...")
    
    combined_text = ""
    for i, image_path in enumerate(image_paths):
        print(f"Processing page {i+1} with OCR...")
        page_text = extract_text_from_image(image_path)
        combined_text += f"\nPage {i+1}:\n{page_text}"
    
    print(f"OCR extraction complete. Total text length: {len(combined_text.split())} words")
    return combined_text

def process_pdf(pdf_path):
    """
    Process a PDF file by first attempting image-based extraction, then falling back to OCR.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: JSON-formatted extraction result
    """
    start_time = time.time()
    
    original_filename = os.path.basename(pdf_path)
    print(f"Processing PDF: {original_filename}")
    
    # Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path)
    
    if not image_paths or len(image_paths) == 0:
        print("Failed to convert PDF to images")
        return "Error: Failed to convert PDF to images"
    
    print(f"Successfully converted PDF to {len(image_paths)} images")
    
    # First try image-based extraction
    extracted_text, image_extraction_success = extract_text_from_pdf_with_image_based(image_paths)
    
    # Fall back to OCR if image-based extraction fails
    if not image_extraction_success:
        print("Image-based extraction failed, falling back to OCR...")
        extracted_text = extract_text_from_pdf_with_ocr(image_paths)
    
    # Read classification system prompt
    classification_system_prompt = read_system_prompt("classify_sys_prompt.txt")

    # Classify the report type
    classification_result = classify_report(extracted_text, classification_system_prompt)
    
    # Process classification result and get report type
    if isinstance(classification_result, dict):
        report_type = classification_result.get("report_type", "Unknown")
        reason = classification_result.get("reason", "Unknown")
    else:
        report_type = "Unknown"
        print("Classification failed, using Unknown as report type")
    
    # Standardize the report type name for file naming
    report_type_standardized = report_type.lower().replace(" ", "_")
    
    # Load the appropriate system prompt based on the classification
    system_prompt = load_system_prompt(report_type_standardized)
    
    # Enhance the extracted text using Azure OpenAI
    enhanced_text = enhance_text_with_azure(
        extracted_text=extracted_text,
        report_type=report_type,
        image_paths=image_paths
    )
    
    # Verify the enhanced results
    verification_result = verify_results_with_azure(
        enhanced_text=enhanced_text,
        image_paths=image_paths,
        report_type=report_type
    )
    
    # Add classification result and extraction method to the verification result
    if isinstance(verification_result, str):
        try:
            verification_json = json.loads(verification_result)
            verification_json["classification_result"] = classification_result
            verification_json["extraction_method"] = "image_based" if image_extraction_success else "ocr_based"
            verification_result = json.dumps(verification_json, indent=2)
        except json.JSONDecodeError:
            pass
    
    processing_time = time.time() - start_time
    print(f"PDF processing completed in {processing_time:.2f} seconds")
    
    return verification_result

# Run the processing
if __name__ == "__main__":
    main() # type: ignore