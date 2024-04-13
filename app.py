# # import streamlit as st 
# # from PIL import Image 
# # from pytesseract import pytesseract 
# # from transformers import BlipProcessor, BlipForConditionalGeneration,pipeline
# # import os
# # from pypdf import PdfReader
# # import fitz

# # file_path = st.file_uploader("Upload PDF file", type=('pdf'))
# # # file_path = 'sample_file.pdf'

# # images_path = 'tp/'
# # pdf_file = fitz.open(file_path)
# # page_nums = len(pdf_file)
# # images_list = []
# # for page_num in range(page_nums):
# #     page_content = pdf_file[page_num]
# #     images_list.extend(page_content.get_images())
# # for i, img in enumerate(images_list, start=1):
# #     xref = img[0]
# #     base_image = pdf_file.extract_image(xref)
# #     image_bytes = base_image['image']
# #     image_ext = base_image['ext']
# #     image_name = str(i) + '.' + image_ext

# # reader = PdfReader(file_path)
# # num_pages = len(reader.pages)
# # final_text = ""
# # for page_num in range(num_pages):
# #     page = reader.pages[page_num]
# #     text = page.extract_text()
# #     final_text += text
# # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# # image_folder = "tp"
# # for filename in os.listdir(image_folder):
# #     if filename.endswith(".jpeg") or filename.endswith(".png"):
# #         image_path = os.path.join(image_folder, filename)

# #         try:
# #             raw_image = Image.open(image_path)
# #             text = pytesseract.image_to_string(raw_image)

# #             if(len(text)<10):
# #                 text1 = "a photography of"
# #                 inputs = processor(raw_image, text1, return_tensors="pt")

# #                 out = model.generate(**inputs)
# #                 conditional_caption = processor.decode(out[0], skip_special_tokens=True)

# #                 inputs = processor(raw_image, return_tensors="pt")

# #                 out = model.generate(**inputs)
# #                 unconditional_caption = processor.decode(out[0], skip_special_tokens=True)

# #                 final_text += "There is an image with text: '" + conditional_caption + "' and '" + unconditional_caption + "'"
            
# #             else:
# #                 final_text += "There is an image with text: '" + text + "'"

# #         except (FileNotFoundError, IOError) as e:
# #             print(f"Error processing {filename}: {e}")


# # summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
# # tt = summarizer(final_text, max_length=330, min_length=130, do_sample=False)
# # summary_text = tt[0]['summary_text']
# # st.write(summary_text)


# import streamlit as st
# from PIL import Image
# from pytesseract import pytesseract
# from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
# import fitz
# import io
# import os
# from typing import Generator, Tuple

# @st.cache_resource
# def load_blip_model():
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
#     model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
#     return processor, model

# @st.cache_resource
# def load_summarizer():
#     return pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

# def extract_images_and_text(file_path: str) -> Generator[Tuple[dict, str], None, None]:
#     """
#     Extract images and text from a PDF file, yielding a dictionary of images and the corresponding text for each page.
#     """
#     pdf_file = fitz.open(file_path)
#     for page_num in range(len(pdf_file)):
#         page = pdf_file[page_num]
#         text = page.get_text("text")
#         images = page.get_image_info()
#         yield images, text

# def process_images(images: dict, processor: BlipProcessor, model: BlipForConditionalGeneration, pdf_file: fitz.Document) -> str:
#     """
#     Process images using the BLIP model and generate captions.
#     """
#     image_captions = []
#     for image_info in images.values():
#         xref = image_info["xref"]
#         base_image = pdf_file.extract_image(xref)
#         image_bytes = base_image["image"]
#         image = Image.open(io.BytesIO(image_bytes))
#         text = pytesseract.image_to_string(image)

#         if len(text) < 10:
#             text1 = "a photography of"
#             inputs = processor(image, text1, return_tensors="pt")
#             out = model.generate(**inputs)
#             conditional_caption = processor.decode(out[0], skip_special_tokens=True)
#             inputs = processor(image, return_tensors="pt")
#             out = model.generate(**inputs)
#             unconditional_caption = processor.decode(out[0], skip_special_tokens=True)
#             caption = f"There is an image with text: '{conditional_caption}' and '{unconditional_caption}'"
#         else:
#             caption = f"There is an image with text: '{text}'"
#         image_captions.append(caption)

#     return " ".join(image_captions)

# st.set_page_config("PDF Summarizor", ":turkey:", layout="wide")

# file_path = st.file_uploader("Upload PDF file", type=('pdf'))

# if file_path is not None:
#     final_text = ""
#     processor, model = load_blip_model()
#     summarizer = load_summarizer()
#     pdf_file = fitz.open(file_path.name)

#     for images, text in extract_images_and_text(file_path.name):
#         final_text += text
#         image_captions = process_images(images, processor, model, pdf_file)
#         final_text += " " + image_captions

#     tt = summarizer(final_text, max_length=330, min_length=130, do_sample=False)
#     summary_text = tt[0]['summary_text']
#     st.write(summary_text)

import streamlit as st
from PIL import Image
from pytesseract import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import fitz
import io
import os
from typing import Generator, Tuple

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

def extract_images_and_text(file_path: str) -> Generator[Tuple[list, str], None, None]:
    """
    Extract images and text from a PDF file, yielding a list of images and the corresponding text for each page.
    """
    pdf_file = fitz.open(file_path)
    for page_num in range(len(pdf_file)):
        page = pdf_file[page_num]
        text = page.get_text("text")
        images = [pdf_file.extract_image(xref)["image"] for xref, *_ in page.get_images()]
        yield images, text

def process_images(images: list, processor: BlipProcessor, model: BlipForConditionalGeneration) -> str:
    """
    Process images using the BLIP model and generate captions.
    """
    image_captions = []
    for image_bytes in images:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)

        if len(text) < 10:
            text1 = "a photography of"
            inputs = processor(image, text1, return_tensors="pt")
            out = model.generate(**inputs)
            conditional_caption = processor.decode(out[0], skip_special_tokens=True)
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            unconditional_caption = processor.decode(out[0], skip_special_tokens=True)
            caption = f"There is an image of: '{conditional_caption}' and '{unconditional_caption}'"
        else:
            caption = f"There is an image of:  '{text}'"
        image_captions.append(caption)

    return " ".join(image_captions)

st.set_page_config("PDF Summarizor", ":turkey:", layout="wide")

file_path = st.file_uploader("Upload PDF file", type=('pdf'))

if file_path is not None:
    final_text = ""
    processor, model = load_blip_model()
    summarizer = load_summarizer()

    for images, text in extract_images_and_text(file_path.name):
        final_text += text
        image_captions = process_images(images, processor, model)
        final_text += " " + image_captions

    tt = summarizer(final_text, max_length=330, min_length=130, do_sample=False)
    summary_text = tt[0]['summary_text']
    st.write(summary_text)
