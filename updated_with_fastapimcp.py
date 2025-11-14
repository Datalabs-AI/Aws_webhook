from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
from botocore.exceptions import ClientError
import io
import fitz  # PyMuPDF
from PIL import Image
import os
import json
import requests
import openai
import base64
from io import BytesIO
from dotenv import load_dotenv
import time
import asyncio
import aiohttp
from datetime import datetime
import sys
import traceback



load_dotenv()

# --- CONFIG ---
# Use exact Coolify variable names (reads from .env via load_dotenv() or environment)
AWS_ACCESS_KEY = os.getenv("aws_access_key_id")
AWS_SECRET_KEY = os.getenv("aws_secret_access_key")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "us-east-1")
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_s3_client():
    """Get S3 client, creating it if needed."""
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise ValueError(f"AWS credentials not configured. AWS_ACCESS_KEY={'set' if AWS_ACCESS_KEY else 'missing'}, AWS_SECRET_KEY={'set' if AWS_SECRET_KEY else 'missing'}")
    
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION_NAME
    )

S3 = None  # Will be initialized on first use

PPLX_URL = "https://api.perplexity.ai/chat/completions"
HEADERS = {"Authorization": f"Bearer {PPLX_API_KEY}", "Content-Type": "application/json"}

def get_openai_client():
    """Get OpenAI client, creating it if needed."""
    if not OPENAI_API_KEY:
        return None
    return openai.OpenAI(api_key=OPENAI_API_KEY)

OPENAI_CLIENT = None  # Will be initialized on first use

app = FastAPI(title="Hybrid Metadata Extractor")


# --- MODELS ---
class S3Input(BaseModel):
    bucket: str
    key: str


# --- HELPERS ---
def _derive_tenant_and_paths_from_key(s3_key: str):
    """Given a key like
    banks/NEXI/tenants/<tenantId>/bank-statements/raw/<documentName>.pdf
    return (tenant_id, metadata_key, document_id)
    where metadata_key is banks/NEXI/tenants/<tenantId>/metadata.json
    and document_id is <documentName> without extension.
    """
    parts = s3_key.split("/")
    try:
        tenants_index = parts.index("tenants")
        tenant_id = parts[tenants_index + 1]
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail="Invalid S3 key format: missing tenants/<tenantId> segment")

    # metadata.json lives directly under the tenant directory
    metadata_key = "/".join(parts[: tenants_index + 2] + ["metadata.json"])
    document_name = os.path.splitext(os.path.basename(s3_key))[0]
    return tenant_id, metadata_key, document_name


async def _append_metadata_to_s3_json(bucket: str, metadata_key: str, document_id: str, document_metadata: dict):
    """Read existing metadata.json, append or upsert this document's metadata, and write back."""
    global S3
    if S3 is None:
        S3 = get_s3_client()
    
    existing = {}
    try:
        # Download existing metadata.json if present
        obj = await asyncio.to_thread(S3.get_object, Bucket=bucket, Key=metadata_key)
        body_bytes = obj["Body"].read()
        if body_bytes:
            try:
                existing = json.loads(body_bytes.decode("utf-8"))
                if isinstance(existing, dict):
                    pass  # good, merge into it
                else:
                    # The old structure is not a dict (could be list/str/int/etc), preserve under special key
                    existing = {"_previous_data": existing}
            except Exception:
                # If corrupted or not JSON, start fresh to avoid breaking the flow
                existing = {}
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code not in ("NoSuchKey", "404", "NotFound"):
            raise
    except Exception:
        # Any other unexpected error while reading: proceed with empty
        existing = {}

    # Upsert the document metadata under its document_id key
    existing[document_id] = document_metadata

    # Write back to S3
    body = json.dumps(existing, indent=2).encode("utf-8")
    await asyncio.to_thread(
        S3.put_object,
        Bucket=bucket,
        Key=metadata_key,
        Body=body,
        ContentType="application/json",
    )
def pdf_to_first_two_page_images(pdf_bytes: bytes) -> list:
    """Convert the first and second pages of PDF to PIL images (blocking)."""
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i in range(min(2, pdf.page_count)):
        page = pdf.load_page(i)
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    pdf.close()
    return images

async def pdf_to_first_two_page_images_async(pdf_bytes: bytes) -> list:
    """Async: Convert first two PDF pages to PIL Images."""
    return await asyncio.to_thread(pdf_to_first_two_page_images, pdf_bytes)

async def call_openai_vision(images: list, system_prompt: str) -> dict:
    """Send images + prompt to OpenAI GPT-4o Vision."""
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        OPENAI_CLIENT = get_openai_client()
        if OPENAI_CLIENT is None:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        image_entries = []
        for image in images:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            image_entries.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })
        image_entries.append({"type": "text", "text": "Analyze this bank statement and extract the metadata."})

        response = await asyncio.to_thread(
            OPENAI_CLIENT.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": image_entries
                }
            ]
        )
        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1].split("```")
            raw_text = raw_text[0].strip() if raw_text else raw_text.strip()
        return json.loads(raw_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Vision API failed: {str(e)}")


async def call_perplexity(prompt: str, system: str = None) -> dict:
    """Call Perplexity AI and return parsed JSON."""
    payload = {
        "model": "sonar-pro",
        "messages": []
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})

    # Use async HTTP client for non-blocking requests
    async with aiohttp.ClientSession() as session:
        async with session.post(PPLX_URL, headers=HEADERS, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise HTTPException(status_code=500, detail=f"Perplexity API error: {error_text}")
            
            data = await resp.json()
            raw = data["choices"][0]["message"]["content"]
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            try:
                return json.loads(cleaned)
            except Exception:
                return {"raw_response": raw}


# --- MAIN ENDPOINT ---
@app.post("/extract-bank-metadata")
async def extract_bank_metadata(data: S3Input):
    start_time = time.time()
    bucket = data.bucket
    key = data.key
    request_id = f"{bucket}_{key.replace('/', '_')}_{int(time.time())}"
    print(f"[{request_id}] Starting processing S3: {bucket}/{key}")

    try:
        # Step 1️⃣: Download PDF
        global S3
        if S3 is None:
            # Debug: Check if credentials are available
            print(f"[{request_id}] AWS_ACCESS_KEY present: {bool(AWS_ACCESS_KEY)}")
            print(f"[{request_id}] AWS_SECRET_KEY present: {bool(AWS_SECRET_KEY)}")
            print(f"[{request_id}] AWS_REGION: {AWS_REGION_NAME}")
            S3 = get_s3_client()
            
        try:
            pdf_stream = io.BytesIO()
            # Run S3 download in thread pool since it's blocking I/O
            download_start = time.time()
            await asyncio.to_thread(S3.download_fileobj, bucket, key, pdf_stream)
            pdf_bytes = pdf_stream.getvalue()
            download_time = time.time() - download_start
            print(f"[{request_id}] Downloaded PDF in {download_time:.2f}s")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")

        # Step 2️⃣: Convert first and second pages → images
        conversion_start = time.time()
        images = await pdf_to_first_two_page_images_async(pdf_bytes)
        conversion_time = time.time() - conversion_start
        print(f"[{request_id}] Converted PDF to images in {conversion_time:.2f}s")

        # Step 3️⃣: OpenAI GPT-4o Vision → Extract core metadata
        system_prompt_1 = (
    "You are a precise document parser. Analyze the uploaded image of a bank statement and extract the following as JSON:\n"
    "{\n"
    '"legal_name": string,\n'
    '"trade_name": string or null,\n'
    '"company_address": string or null,\n'
    '"business_type": LLC | Corporation | Sole Proprietorship | Partnership | etc.,\n'
    '"bank_name": string (must not be null),\n'
    '"bank_type": "National Bank" | "Credit Union", do some research to determine the bank type based on the bank name.\n'
    '"bank_account_number": string or null,\n'
    '"statement_month": string (never null),\n'
    '"statement_year": string (4-digit),\n'
    '"starting_date": string (MM-DD-YYYY) or null,\n'
    '"ending_date": string (MM-DD-YYYY) or null,\n'
    '"starting_balance": string or null,\n'
    '"ending_balance": string or null\n'
    "}\n"
    "Rules:\n"
    "- Do NOT hallucinate missing data; use null, except statement_month must never be null.\n"
    "- bank_name must always be present.\n"
    "- statement_month must always be present.\n"
    "- statement_year must always be present.\n"
    "- business_type must always be present and should not be null.\n"
    "- bank_type must be exactly either 'National Bank' or 'Credit Union'.\n"
    "- trade_name is DBA if available.\n"
    "- Dates must be formatted as MM-DD-YYYY.\n"
    "- starting_balance and ending_balance must not include currency symbols like '$'.\n"
    "- Output strictly valid JSON, nothing else."
)

        openai_start = time.time()
        openai_result = await call_openai_vision(images, system_prompt_1)
        openai_time = time.time() - openai_start
        print(f"[{request_id}] OpenAI Vision completed in {openai_time:.2f}s")
        print(f"[{request_id}] OpenAI Output:", openai_result)

        # Step 4️⃣: Perplexity → Industry & Aggregator Links
        company = openai_result.get("legal_name")
        address = openai_result.get("company_address")

        system_prompt_2 = (
            "You are a business research assistant. "
            "Given a company name and its address, find the most probable industry type and related aggregator links. "
            "Return a JSON strictly as: { 'industry': string, 'aggregator_links': array of URLs }."
        )
        pplx_prompt = f"Company: {company}\nAddress: {address}"

        perplexity_start = time.time()
        pplx_result = await call_perplexity(pplx_prompt, system_prompt_2)
        perplexity_time = time.time() - perplexity_start
        print(f"[{request_id}] Perplexity completed in {perplexity_time:.2f}s")
        print(f"[{request_id}] Perplexity Output:", pplx_result)

        # Step 5️⃣: Merge results
        final_metadata = {**openai_result, **pplx_result}
        
        # Step 5.1️⃣: Append to tenant-level metadata.json in S3
        tenant_id, tenant_metadata_key, document_id = _derive_tenant_and_paths_from_key(key)
        await _append_metadata_to_s3_json(bucket, tenant_metadata_key, document_id, final_metadata)
        
        # Step 6️⃣: Save to JSON file
        output_dir = "extracted_metadata"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{request_id}.json")
        
        output_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "s3_bucket": bucket,
            "s3_key": key,
            "timing": {
                "total_time": time.time() - start_time,
                "download_time": download_time,
                "conversion_time": conversion_time,
                "openai_time": openai_time,
                "perplexity_time": perplexity_time
            },
            "metadata": final_metadata
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"[{request_id}] ✅ Completed in {total_time:.2f}s - Saved to {output_file}")
        print(f"[{request_id}] Timing breakdown: Download={download_time:.2f}s, Conversion={conversion_time:.2f}s, OpenAI={openai_time:.2f}s, Perplexity={perplexity_time:.2f}s")

        return {
            "message": "Metadata extracted successfully",
            "data": final_metadata,
            "tenant_metadata_s3_key": tenant_metadata_key,
            "document_id": document_id,
            "timing": total_time,
        }

    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        error_time = time.time() - start_time
        error_traceback = traceback.format_exc()
        exc_type, exc_value, exc_tb = sys.exc_info()
        
        # Get comprehensive error information
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else repr(e)
        
        # If error message is empty, try to get more info
        if not error_msg or error_msg.strip() == "":
            error_msg = f"{error_type}: {repr(e)}"
            if hasattr(e, 'args') and e.args:
                error_msg += f" (args: {e.args})"
        
        # Print detailed error information
        print(f"[{request_id}] ❌ Failed after {error_time:.2f}s")
        print(f"[{request_id}] Exception Type: {error_type}")
        print(f"[{request_id}] Error Message: {error_msg}")
        print(f"[{request_id}] Full traceback:\n{error_traceback}")
        
        # Also print exception args if available
        if hasattr(e, 'args') and e.args:
            print(f"[{request_id}] Exception args: {e.args}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {error_type} - {error_msg}"
        )

    
