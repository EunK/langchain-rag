from typing import List, Dict, Any
from supabase import Client
from clients import get_supabase_client
from config import Settings


def ensure_bucket_exists(sb: Client, bucket: str, public: bool = True) -> None:
    try:
        buckets = sb.storage.list_buckets()
        exists = any(b.get("name") == bucket for b in buckets)
        if not exists:
            sb.storage.create_bucket(bucket, public=public)
        return
    except Exception:
        pass

    try:
        sb.storage.create_bucket(bucket, public=public)
    except Exception:
        return


def supabase_upload_png(sb: Client, bucket: str, path: str, png_bytes: bytes) -> str:
    ensure_bucket_exists(sb, bucket, public=True)
    try:
        sb.storage.from_(bucket).upload(
            path=path,
            file=png_bytes,
            file_options={"content-type": "image/png", "upsert": "true"},
        )
    except Exception:
        ensure_bucket_exists(sb, bucket, public=True)
        sb.storage.from_(bucket).upload(
            path=path,
            file=png_bytes,
            file_options={"content-type": "image/png", "upsert": "true"},
        )
    return sb.storage.from_(bucket).get_public_url(path)


def _chunks(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def delete_doc_and_assets(settings: Settings, doc_id: int) -> Dict[str, Any]:
    sb = get_supabase_client(settings.supabase_url, settings.supabase_service_key)

    pages_res = (
        sb.table("manual_pages")
        .select("image_path")
        .eq("doc_id", doc_id)
        .execute()
    )
    image_paths = [r["image_path"] for r in (pages_res.data or []) if r.get("image_path")]

    storage_deleted = 0
    storage_failed: List[str] = []
    if image_paths:
        for batch in _chunks(image_paths, 100):
            try:
                sb.storage.from_(settings.storage_bucket).remove(batch)
                storage_deleted += len(batch)
            except Exception:
                storage_failed.extend(batch)

    try:
        sb.table("rag_chunks").delete().eq("doc_id", doc_id).execute()
        sb.table("manual_pages").delete().eq("doc_id", doc_id).execute()
        sb.table("manual_docs").delete().eq("id", doc_id).execute()
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "storage_deleted": storage_deleted,
            "storage_failed": storage_failed,
        }

    return {"ok": True, "storage_deleted": storage_deleted, "storage_failed": storage_failed}
