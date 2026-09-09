# import torch
# from transformers import AutoModelForCausalLM
# from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
# from deepseek_vl.utils.io import load_pil_images
from pdf2image import convert_from_path
import os
from natsort import natsorted
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
# from mlx_vlm.utils import load_config



# from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
# from docling.datamodel.base_models import InputFormat
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, RapidOcrOptions, EasyOcrOptions
# # from docling.document_converter import DocumentConverter
# from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
# from docling.models.tesseract_ocr_cli_model import TesseractCliOcrOptions
# from docling.models.tesseract_ocr_model import TesseractOcrOptions

from gmft.detectors.tatr import TATRDetector
from gmft.auto import CroppedTable, TableDetector, AutoTableFormatter, AutoTableDetector, AutoFormatConfig
from gmft.pdf_bindings import PyPDFium2Document
from gmft_pymupdf import PyMuPDFDocument
from gmft.formatters.ditr import DITRFormatConfig, DITRFormatter
import re
import os
import pymupdf
# from IPython.display import display
from tabulate import tabulate
from gmft.formatters.histogram import HistogramFormatter, HistogramConfig
import shutil

def remove_b_from_a(A, B):
    # Split B into tokens (non-whitespace parts)
    tokens = B.split()
    if not tokens:
        # If B doesn't contain any non-whitespace characters, return A unchanged
        return A

    # Build a regex pattern that matches the tokens in sequence, allowing any whitespace in between.
    # For example, if B is "Test Data, More Content", tokens will be ["Test", "Data,", "More", "Content"],
    # and the pattern will be "Test\s*Data,\s*More\s*Content".
    pattern = r'\s*'.join(map(re.escape, tokens))

    # Compile the regex. Adding re.DOTALL will allow '.' to match newlines if needed.
    regex = re.compile(pattern, re.DOTALL)

    # Replace all occurrences of the pattern in A with an empty string
    result = regex.sub('', A)
    return result

def split_pdf(input_pdf, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    doc = pymupdf.open(input_pdf)

    for page_num in range(len(doc)):
        new_doc = pymupdf.open()  # Create a new PDF
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)  # Extract one page
        output_path = os.path.join(output_folder, f"out{page_num}.pdf")
        new_doc.save(output_path)  # Save as new PDF
        new_doc.close()

    doc.close()
    print(f"PDF split successfully. Pages saved in: {output_folder}")


def clean_text(text):
    # 1. Remove extra spaces around decimal points
    #    Example: "0 . 7" -> "0.7"
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)

    # 2. Remove extra spaces after a minus sign when attached to digits
    #    Example: "yr -1" -> "yr-1"
    text = re.sub(r'-\s+(\d)', r'-\1', text)

    # 3. Remove spaces immediately after an opening parenthesis and before a closing parenthesis
    #    Example: "( 0.018 Gt C yr-1 )" -> "(0.018 Gt C yr-1)"
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)

    # 4. (Optional) Remove extra spaces if there are any multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text


# source = 'global_carbon_budge_2023.pdf'


def convert_to_markdown_and_save(source, output_folder, sanitized_indexed_file_name, model, processor, config):
    markdown_content = []

    markdown_filename = f"{sanitized_indexed_file_name}.md"

    markdown_filepath = os.path.join(output_folder, markdown_filename)

    ext = os.path.splitext(source)[1].lower()

    if ext == ".pdf":
        # model_path = "deepseek-ai/deepseek-vl-7b-chat"
        # vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        # tokenizer = vl_chat_processor.tokenizer

        # vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        # vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

        # Converting pdf into images
        images_parent_path = os.path.dirname(source)
        pdf_dpi = int(os.environ.get("MILONET_PDF_DPI", "500"))
        pages = convert_from_path(source, pdf_dpi)
        split_pages_folder = os.path.join(images_parent_path, "split_pages")
        os.makedirs(split_pages_folder, exist_ok=True)
        for count, page in enumerate(pages):
            img_path = os.path.join(split_pages_folder, f"out{count}.jpg")
            page.save(img_path, 'JPEG')


        # Splitting pdf into a lit of pages
        split_pdf(source, split_pages_folder)

        sorted_filenames = natsorted(os.listdir(split_pages_folder))

        for filename in sorted_filenames:
            img_path = os.path.join(split_pages_folder, filename)
            if ".jpg" in img_path:
                prompt = ("<image>This is a paper page. "
                          "Determine if the page contains any data tables and answer only with Yes or No: "
                          "A data table is defined as structured information organized into rows and columns, with a clear title or caption. "
                          "The table must have a clear grid-like structure with rows or columns. It may include boundaries, or it can be presented without vertical or horizontal boundaries. "
                          "If the page contains a table, return `Yes` and provide the table's title or caption. "
                          "If the page does not contain a table, return `No`. "
                          "Please carefully check the content the table must have a clear grid-like structure with rows and columns. It may include boundaries, or it can be presented without vertical or horizontal boundaries, or avoid mistaking the following for tables: "
                          "- **Text descriptions (e.g., 'The annual CO2 emissions were 0.02 Gt C yr^-1')**. "
                          "- **Figures or charts (e.g., bar charts, line graphs, scatter plots)**. "
                          "- **Lists or bullet points**. "
                          "- **Numbers or data points embedded in the text.** "
                          "Example 1: "
                          "If the page contains 'Table X. ' or 'Table X: ', and the table has a clear grid structure with rows or columns, return `Yes` and provide 'Table X: ...'. "
                          "Example 2: "
                          "If the page only contains text descriptions like 'The annual CO2 emissions were 0.02 Gt C yr^-1', or figures and (or) figure captions, return `No`. ")
                formatted_prompt = apply_chat_template(
                    processor, config, prompt, num_images=1
                )
                answer = generate(model, processor, formatted_prompt, [img_path], verbose=False)

                print(img_path)
                # if "No" in answer:
                print(answer)
                pdf_path = img_path.replace(".jpg", ".pdf")

                if "Yes" in answer:
                    md_output_path = pdf_path.replace(".pdf", "_pymupdf_gmft.md")


                    formatter = AutoTableFormatter()
                    detector = TATRDetector()
                    # detector = AutoTableDetector()
                    # formatter = HistogramFormatter(HistogramConfig(col_sep_threshold=3))
                    # formatter = DITRFormatter(DITRFormatConfig(formatter_path="conjuncts/ditr-e15", formatter_base_threshold=0.01))


                    # detector = Img2TableDetector()
                    # Extract tables from PDF
                    def ingest_pdf(pdf_path): # produces list[CroppedTable]
                        doc = PyMuPDFDocument(pdf_path)
                        tables = []
                        removed_page = ""
                        for page in doc:
                            removed_page = page._get_text()
                            tables_page = detector.extract(page)
                            if len(tables_page) > 0:
                                for table in tables_page:
                                    removed_page = remove_b_from_a(removed_page, table.text())
                                    captions = table.captions()
                                    for cap in captions:
                                        removed_page = remove_b_from_a(removed_page, cap)
                                tables += tables_page

                            # print(tables[0].text())
                        return tables, doc, removed_page
                    tables, doc, removed_page = ingest_pdf(pdf_path)
                    # doc = PyMuPDFDocument(source)
                    # tables = detector.extract(doc)
                    # doc = PyMuPDFDocument(source)
                    # with open(md_output_path, "w", encoding="utf-8") as f:
                    markdown_content.append("\n")
                        # f.write("\n")
                    markdown_content.append(removed_page+'\n')
                        # f.write(removed_page+'\n')
                    markdown_content.append('\n')
                        # f.write("\n")
                    # print(f"Markdown text file created: {md_output_path}")
                    for table in tables:
                        # print(table.text())
                        ft = formatter.format(table)
                        caption = table.captions()
                        # caption = caption[0]
                        # print(caption)
                        # df = ft.df()

                        # with open(md_output_path, "w", encoding="utf-8") as f:
                        #     f.write(markdown_table)
                        # print(f"Markdown file created: {md_output_path}")  # output: "## Docling Technical Report[...]"
                        config_overrides = AutoFormatConfig(semantic_spanning_cells=True, enable_multi_header=False )
                        df = ft.df(config_overrides=config_overrides)  # if provided, config_overrides replaces config, so verbosity is reverted
                        markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
                        # with open(md_output_path, "a", encoding="utf-8") as f:
                        #     if len(caption)>0:
                        #         f.write("\n")
                        #         f.write(caption[0]+'\n')
                        #         f.write("\n")
                        #
                        #     f.write("\n")
                        #     f.write(markdown_table+'\n')
                        #     f.write("\n")
                        #
                        #     if len(caption) > 1:
                        #         f.write("\n")
                        #         f.write(caption[1]+'\n')
                        #         f.write("\n")
                        # print(f"Markdown file table created: {md_output_path}")  # output: "## Docling Technical Report[...]"

                        # with open(md_output_path, "a", encoding="utf-8") as f:
                        if len(caption)>0:
                            # f.write("\n")
                            # f.write(caption[0] + '\n')
                            # f.write("\n")
                            markdown_content.append('\n')
                            markdown_content.append(caption[0]+'\n')
                            markdown_content.append('\n')

                        markdown_content.append('\n')
                        markdown_content.append(markdown_table+'\n')
                        markdown_content.append('\n')

                        # f.write("\n")
                        # f.write(markdown_table+'\n')
                        # f.write("\n")

                        if len(caption) > 1:
                            # f.write("\n")
                            # f.write(caption[1]+'\n')
                            # f.write("\n")
                            markdown_content.append('\n')
                            markdown_content.append(caption[1] + '\n')
                            markdown_content.append('\n')
                        # print(f"Markdown file table created: {md_output_path}")  # output: "## Docling Technical Report[...]"

                    doc.close() # once you're done with the document

                else:
                    # md_output_path = pdf_path.replace(".pdf", "_pymupdf.md")
                    doc = pymupdf.open(pdf_path)
                    # with open(md_output_path, "w", encoding="utf-8") as f:
                    #     for page in doc:
                    #         text = page.get_text()
                    #         f.write(text)

                    # with open(md_output_path, "w", encoding="utf-8") as f:
                    for page in doc:
                        text = page.get_text()
                        markdown_content.append('\n')
                        markdown_content.append(text)
                        markdown_content.append('\n')
                    doc.close()
                    # print(f"Markdown file created: {md_output_path}")
        shutil.rmtree(split_pages_folder)
    elif ext == ".txt":
        # source = source.replace("/", "//")
        # source = source.split("/Original_documents/")[1]
        # source = "Original_documents//"+source
        # shorter_source = filename_map[source]
        with open(source, "r", encoding="utf-8") as file:
            text_data = file.read()
        markdown_content.append(text_data)

    all_texts = " ".join([markdown_content[i] for i in range(len(markdown_content))])

    with open(markdown_filepath, 'w', encoding='utf-8') as f:  # ,  encoding='utf-8'
        f.write(all_texts)
    return markdown_filepath
