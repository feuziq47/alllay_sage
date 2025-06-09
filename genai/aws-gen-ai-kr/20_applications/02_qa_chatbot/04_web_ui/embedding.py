# 표준 라이브러리
import base64
import json
import math
import os
import shutil
import sys
import tempfile
import time
from glob import glob
from io import BytesIO
from itertools import chain

# 서드파티 라이브러리
import boto3
import botocore
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdfplumber
from PIL import Image
from pprint import pprint
from PyPDF2 import PdfWriter

# PDF/문서처리 라이브러리
from pdf2image import convert_from_path
from unstructured.cleaners.core import clean_bullets, clean_extra_whitespace

# LangChain 관련
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.embeddings import BedrockEmbeddings
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import OpenSearchVectorSearch
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    UnstructuredAPIFileLoader,
)

# 프로젝트 내부 유틸
from bedrock import get_embedding_model, get_llm
# py파일
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
# ipynb 파일
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
from utils import bedrock, print_ww
from utils.bedrock import bedrock_info
from utils.chunk import parant_documents
from utils.common_utils import (
    to_pickle,
    load_pickle,
    retry,
    print_html,
)
from utils.opensearch import opensearch_utils
from utils.rag import (
    qa_chain,
    prompt_repo,
    show_context_used,
    retriever_utils,
    OpenSearchHybridSearchRetriever,
)
from utils.ssm import parameter_store


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('utf-8')

def add_python_path(module_path):
    if os.path.abspath(module_path) not in sys.path:
        sys.path.append(os.path.abspath(module_path))
        print(f"python path: {os.path.abspath(module_path)} is added")
    else:
        print(f"python path: {os.path.abspath(module_path)} already exists")
    print("sys.path: ", sys.path)
    
@retry(total_try_cnt=5, sleep_in_sec=10, retryable_exceptions=(botocore.exceptions.EventStreamError))
def summary_img(summarize_chain, img_base64):

    img = Image.open(BytesIO(base64.b64decode(img_base64)))
    plt.imshow(img)
    plt.show()

    summary = summarize_chain.invoke(
        {
            "image_base64": img_base64
        }
    )

    return summary
def create_index(os_client, index_name):
    region=boto3.Session().region_name
    pm = parameter_store(region)
    dimension = 1024
    pm.put_params(
        key="opensearch_customer_index_name",
        value=f'{index_name}',
        overwrite=True,
        enc=False
    )
    index_body = {
        'settings': {
            'analysis': {
                'analyzer': {
                    'my_analyzer': {
                             'char_filter':['html_strip'],
                        'tokenizer': 'nori',
                        'filter': [
                            #'nori_number',
                            #'lowercase',
                            #'trim',
                            'my_nori_part_of_speech'
                        ],
                        'type': 'custom'
                    }
                },
                'tokenizer': {
                    'nori': {
                        'decompound_mode': 'mixed',
                        'discard_punctuation': 'true',
                        'type': 'nori_tokenizer'
                    }
                },
                "filter": {
                    "my_nori_part_of_speech": {
                        "type": "nori_part_of_speech",
                        "stoptags": [
                            "J", "XSV", "E", "IC","MAJ","NNB",
                            "SP", "SSC", "SSO",
                            "SC","SE","XSN","XSV",
                            "UNA","NA","VCP","VSV",
                            "VX"
                        ]
                    }
                }
            },
            'index': {
                'knn': True,
                'knn.space_type': 'cosinesimil'  # Example space type
            }
        },
        'mappings': {
            'properties': {
                'metadata': {
                    'properties': {
                        'source': {'type': 'keyword'},
                        'page_number': {'type':'long'},
                        'category': {'type':'text'},
                        'file_directory': {'type':'text'},
                        'last_modified': {'type': 'text'},
                        'type': {'type': 'keyword'},
                        'image_base64': {'type':'text'},
                        'origin_image': {'type':'text'},
                        'origin_table': {'type':'text'},
                    }
                },
                'text': {
                    'analyzer': 'my_analyzer',
                    'search_analyzer': 'my_analyzer',
                    'type': 'text'
                },
                'vector_field': {
                    'type': 'knn_vector',
                    'dimension': f"{dimension}" # Replace with your vector dimension
                }
            }
        }
    }
    index_exists = opensearch_utils.check_if_index_exists(
        os_client,
        index_name
    )

    if not index_exists:
        opensearch_utils.create_index(os_client, index_name, index_body)
        index_info = os_client.indices.get(index=index_name)
        pprint(index_info)
    
def get_bedrock():

    
# def embedding():
def embedding(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = f"./data/complex_pdf/{tmp_file.name}.pdf"
    # bedrock 클라이언트
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None),
    )
    # llm 모델
    llm_text = ChatBedrock(
        model_id=bedrock_info.get_model_id(model_name="Claude-V3-Sonnet"),
        client=boto3_bedrock,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs={
            "max_tokens": 2048,
            "stop_sequences": ["\n\nHuman"],
            # "temperature": 0,
            # "top_k": 350,
            # "top_p": 0.999
        }
    )

    # Embedding 모델
    llm_emb = BedrockEmbeddings(
        client=boto3_bedrock,
        model_id=bedrock_info.get_model_id(model_name="Titan-Text-Embeddings-V2")
    )
    dimension = 1024

    # module_path = "../../.."
    # add_python_path(module_path)

    # input파일 테스트를 위한 임시 세팅
    file_name = "facility_manual"
    tmp_path = f"./data/origin/{file_name}.pdf"
    image_path = f"./data/fig/{file_name}"

    # 꼬리말 자르기
    file_path = f"./data/processed/{file_name}_no_footer.pdf"  # 꼬리말 잘라낸 새 PDF 경로
    cut_height = 40            # 하단에서 잘라낼 높이(pt, 필요시 조정)

    with pdfplumber.open(tmp_path) as pdf:
        writer = PdfWriter()

        for page in pdf.pages:
            # 1. 하단 꼬리말 빼고 이미지 렌더링
            cropped = page.within_bbox((0, 0, page.width, page.height - cut_height))
            img = cropped.to_image(resolution=300).original

            # 2. 이미지 → PDF 한페이지로 변환
            buf = BytesIO()
            img.save(buf, format="PDF")
            buf.seek(0)
            writer.append(buf)

        # 3. 새 PDF로 저장
        with open(file_path, "wb") as f:
            writer.write(f)

    if os.path.isdir(image_path):
        shutil.rmtree(image_path)
    os.mkdir(image_path)

    loader = UnstructuredFileLoader(
        file_path=file_path,

        chunking_strategy="by_title",
        mode="elements",

        strategy="hi_res",
        hi_res_model_name="yolox_quantized", #"detectron2_onnx", "yolox", "yolox_quantized"

        extract_images_in_pdf=True,
        #skip_infer_table_types='[]', # ['pdf', 'jpg', 'png', 'xls', 'xlsx', 'heic']
        pdf_infer_table_structure=True, ## enable to get table as html using tabletrasformer

        extract_image_block_output_dir=image_path,
        extract_image_block_to_payload=False, ## False: to save image

        max_characters=4096,
        new_after_n_chars=4000,
        combine_text_under_n_chars=2000,

        languages= ["kor+eng"],

        post_processors=[clean_bullets, clean_extra_whitespace]
    )

    # file로드
    docs = loader.load()
    to_pickle(docs, "./data/pickle/parsed_unstructured.pkl")
    docs = load_pickle("./data/pickle/parsed_unstructured.pkl")

    tables, texts = [], []
    images = glob(os.path.join(image_path, "*"))

    tables, texts = [], []

    for doc in docs:

        category = doc.metadata["category"]

        if category == "Table":
            tables.append(doc)
        elif category == "Image":
            images.append(doc)
        else:
            texts.append(doc)

        images = glob(os.path.join(image_path, "*"))

    print (f' # texts: {len(texts)} \n # tables: {len(tables)} \n # images: {len(images)}')
    # 이미지 resizing
    table_as_image = True
    if table_as_image:
        image_tmp_path = os.path.join(image_path, "tmp")
        if os.path.isdir(image_tmp_path):
            shutil.rmtree(image_tmp_path)
        os.mkdir(image_tmp_path)

        # from pdf to image
        pages = convert_from_path(file_path)
        for i, page in enumerate(pages):
            page.save(f'{image_tmp_path}/{str(i+1)}.jpg', "JPEG")



        #table_images = []
        for idx, table in enumerate(tables):
            points = table.metadata["coordinates"]["points"]
            page_number = table.metadata["page_number"]
            layout_width, layout_height = table.metadata["coordinates"]["layout_width"], table.metadata["coordinates"]["layout_height"]

            img = cv2.imread(f'{image_tmp_path}/{page_number}.jpg')
            crop_img = img[math.ceil(points[0][1]):math.ceil(points[1][1]), \
                           math.ceil(points[0][0]):math.ceil(points[3][0])]
            table_image_path = f'{image_path}/table-{idx}.jpg'
            cv2.imwrite(table_image_path, crop_img)
            #table_images.append(table_image_path)

            # print (f'unstructured width: {layout_width}, height: {layout_height}')
            # print (f'page_number: {page_number}')
            # print ("==")

            width, height, _ = crop_img.shape
            image_token = width*height/750
            # print (f'image: {table_image_path}, shape: {img.shape}, image_token_for_claude3: {image_token}' )

            ## Resize image
            if image_token > 1500:
                resize_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                print("   - resize_img.shape = {0}".format(resize_img.shape))
                table_image_resize_path = table_image_path.replace(".jpg", "-resize.jpg")
                cv2.imwrite(table_image_resize_path, resize_img)
                os.remove(table_image_path)
                table_image_path = table_image_resize_path

            img_base64 = image_to_base64(table_image_path)
            table.metadata["image_base64"] = img_base64

        if os.path.isdir(image_tmp_path):
            shutil.rmtree(image_tmp_path)
        images = glob(os.path.join(image_path, "*"))
    
    # 이미지 요약
    system_prompt = "You are an assistant tasked with describing table and image."
    system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
    human_prompt = [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + "{image_base64}",
            },
        },
        {
            "type": "text",
            "text": '''
                     Given image, give a concise summary in 50 characters or less.
                     Don't insert any XML tag such as <text> and </text> when answering.
                     Write in Korean.
            '''
        },
    ]
    human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_message_template,
            human_message_template
        ]
    )

    summarize_chain = prompt | llm_text | StrOutputParser()
    #summarize_chain = {"image_base64": lambda x:x} | prompt | llm_text | StrOutputParser()
    img_info = [image_to_base64(img_path) for img_path in images if os.path.basename(img_path).startswith("figure")]

    image_summaries = []
    for idx, img_base64 in enumerate(img_info):
        summary = summary_img(summarize_chain, img_base64)
        image_summaries.append(summary)

    #image_summaries = summarize_chain.batch(img_info, config={"max_concurrency": 1})


    # Document 생성
    # `요약`된 내용을 Document의 `page_content`로, `OCR`결과는 metadata의 `origin_image`로 사용
    images_preprocessed = []
    for img_path, image_base64, summary in zip(images, img_info, image_summaries):

        metadata = {}
        metadata["img_path"] = img_path
        metadata["category"] = "Image"
        metadata["image_base64"] = image_base64

        doc = Document(
            page_content=summary,
            metadata=metadata
        )
        images_preprocessed.append(doc)

    # 테이블 처리
    human_prompt = [
        {
            "type": "text",
            "text": '''
                     Here is the table: <table>{table}</table>
                     Given table, give a concise summary in 50 characters or less.
                     Don't insert any XML tag such as <table> and </table> when answering.
                     Write in Korean.
            '''
        },
    ]
    human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_message_template,
            human_message_template
        ]
    )

    #summarize_chain = prompt | llm_text | StrOutputParser()
    summarize_chain = {"table": lambda x:x} | prompt | llm_text | StrOutputParser()

    table_info = [(t.page_content, t.metadata["text_as_html"]) for t in tables]
    table_summaries = summarize_chain.batch(table_info, config={"max_concurrency": 1})
    if table_as_image: table_info = [(t.page_content, t.metadata["text_as_html"], t.metadata["image_base64"]) if "image_base64" in t.metadata else (t.page_content, t.metadata["text_as_html"], None) for t in tables]

    tables_preprocessed = []
    for origin, summary in zip(tables, table_summaries):
        metadata = origin.metadata
        metadata["origin_table"] = origin.page_content
        doc = Document(
            page_content=summary,
            metadata=metadata
        )
        tables_preprocessed.append(doc)


    # Opensearch client 생성
    region = boto3.Session().region_name
    pm = parameter_store(region)
    opensearch_domain_endpoint = pm.get_params(
        key="opensearch_domain_endpoint",
        enc=False
    )

    secrets_manager = boto3.client('secretsmanager')

    response = secrets_manager.get_secret_value(
        SecretId='opensearch_user_password'
    )

    secrets_string = response.get('SecretString')
    secrets_dict = eval(secrets_string)

    opensearch_user_id = secrets_dict['es.net.http.auth.user']
    opensearch_user_password = secrets_dict['pwkey']

    opensearch_domain_endpoint = opensearch_domain_endpoint
    rag_user_name = opensearch_user_id
    rag_user_password = opensearch_user_password

    http_auth = (rag_user_name, rag_user_password) # Master username, Master password

    aws_region = os.environ.get("AWS_DEFAULT_REGION", None)

    os_client = opensearch_utils.create_aws_opensearch_client(
        aws_region,
        opensearch_domain_endpoint,
        http_auth
    )

    # index 생성
    index_name = "alllay_index"
    create_index(os_client, index_name)

    # Vector 생성
    vector_db = OpenSearchVectorSearch(
        index_name=index_name,
        opensearch_url=f"https://{opensearch_domain_endpoint}",
        embedding_function=llm_emb,
        http_auth=http_auth, # http_auth
        is_aoss=False,
        engine="faiss",
        space_type="l2",
        bulk_size=100000,
        timeout=60
    )

    # parent chunk 생성
    parent_chunk_size = 4096
    parent_chunk_overlap = 0

    child_chunk_size = 1024
    child_chunk_overlap = 256

    opensearch_parent_key_name = "parent_id"
    opensearch_family_tree_key_name = "family_tree"

    parent_chunk_docs = parant_documents.create_parent_chunk(
        docs=texts,
        parent_id_key=opensearch_parent_key_name,
        family_tree_id_key=opensearch_family_tree_key_name,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap
    )

    parent_ids = vector_db.add_documents(
        documents = parent_chunk_docs, 
        vector_field = "vector_field",
        bulk_size = 1000000
    )
    total_count_docs = opensearch_utils.get_count(os_client, index_name)

    # child chunk 생성
    child_chunk_docs = parant_documents.create_child_chunk(
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        docs=parent_chunk_docs,
        parent_ids_value=parent_ids,
        parent_id_key=opensearch_parent_key_name,
        family_tree_id_key=opensearch_family_tree_key_name
    )

    parent_id = child_chunk_docs[0].metadata["parent_id"]
    response = opensearch_utils.get_document(os_client, doc_id = parent_id, index_name = index_name)

    for table in tables_preprocessed:
        table.metadata["family_tree"], table.metadata["parent_id"] = "parent_table", "NA"
    for image in images_preprocessed:
        image.metadata["family_tree"], image.metadata["parent_id"] = "parent_image", "NA"
    docs_preprocessed = list(chain(child_chunk_docs, tables_preprocessed, images_preprocessed))

    child_ids = vector_db.add_documents(
        documents=docs_preprocessed,
        vector_field = "vector_field",
        bulk_size=1000000
    )
    print("length of child_ids: ", len(child_ids))


def hybrid_search(streaming_callback, query: str):
    index_name = "alllay_index"
    opensearch_domain_endpoint = pm.get_params(
        key="opensearch_domain_endpoint",
        enc=False
    )


    secrets_manager = boto3.client('secretsmanager')

    response = secrets_manager.get_secret_value(
        SecretId='opensearch_user_password'
    )

    secrets_string = response.get('SecretString')
    secrets_dict = eval(secrets_string)

    opensearch_user_id = secrets_dict['es.net.http.auth.user']
    opensearch_user_password = secrets_dict['pwkey']

    opensearch_domain_endpoint = opensearch_domain_endpoint
    rag_user_name = opensearch_user_id
    rag_user_password = opensearch_user_password

    http_auth = (rag_user_name, rag_user_password) # Master username, Master password

    aws_region = os.environ.get("AWS_DEFAULT_REGION", None)

    os_client = opensearch_utils.create_aws_opensearch_client(
        aws_region,
        opensearch_domain_endpoint,
        http_auth
    )
    opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
        os_client=os_client,
        index_name=index_name,
        llm_text=get_llm(streaming_callback), # llm for query augmentation in both rag_fusion and HyDE
        llm_emb=get_embedding_model(), # Used in semantic search based on opensearch

        # hybird-search debugger
        #hybrid_search_debugger = "semantic", #[semantic, lexical, None]

        # option for lexical
        minimum_should_match=0,
        filter=[],

        # option for search
        fusion_algorithm="RRF", # ["RRF", "simple_weighted"], rank fusion 방식 정의
        ensemble_weights=[.51, .49], # [for semantic, for lexical], Semantic, Lexical search 결과에 대한 최종 반영 비율 정의
        reranker=False, # enable reranker with reranker model
        #reranker_endpoint_name=endpoint_name, # endpoint name for reranking model
        parent_document=True, # enable parent document

        # option for complex pdf consisting of text, table and image
        complex_doc=True,

        # option for async search
        async_mode=True,

        # option for output
        k=7, # 최종 Document 수 정의
        verbose=False,
    )
    search_hybrid_result, tables, images = opensearch_hybrid_retriever.get_relevant_documents(query)
    result = ""
    for idx, context in enumerate(search_hybrid_result):
        result += f"Document {idx+1}:"
        result += f"Page Content: {context.page_content}"
        result += f"Metadata: {context.metadata}"
        result += "="*50
    return result
