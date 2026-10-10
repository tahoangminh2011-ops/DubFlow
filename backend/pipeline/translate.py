"""pipeline/translate.py - Groq/xKiro translation, context và strict named-entity preservation."""
import time
import hashlib
import re
import threading
import logging
import json
from typing import List, Dict, Any, Optional, Union
import requests
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import config

logger = logging.getLogger(__name__)

CACHE_DIR = config.TRANSLATE_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = int(getattr(config, "MAX_TRANSLATE_WORKERS", 10))
BATCH_SIZE = int(getattr(config, "TRANS_BATCH_SIZE", 25))
MAX_BATCH_CHARS = int(getattr(config, "TRANS_MAX_BATCH_CHARS", 2200))
TIMEOUT = int(getattr(config, "TRANS_TIMEOUT", 90))
RETRY_COUNT = int(getattr(config, "TRANS_RETRY_COUNT", 2))
BACKOFF_BASE = float(getattr(config, "TRANS_BACKOFF_BASE", 1.5))
MAX_CONCURRENCY_CAP = int(getattr(config, "TRANS_MAX_CONCURRENCY_CAP", 10))
AUTO_RETRY = int(getattr(config, "TRANS_AUTO_RETRY", 2))
GROQ_MAX_CONCURRENT_PER_KEY = int(getattr(config, "GROQ_MAX_CONCURRENT_PER_KEY", 1))
CJK_ACCEPT_MAX_LEN = int(getattr(config, "TRANS_ACCEPT_UNTRANSLATED_CJK_MAX_LEN", 6))

_PRIMARY_MODEL = getattr(config, "GROQ_TRANSLATE_MODEL", "openai/gpt-oss-120b")
_FALLBACK_MODELS = getattr(config, "GROQ_TRANSLATE_FALLBACK_MODELS", [])
if isinstance(_FALLBACK_MODELS, str):
    _FALLBACK_MODELS = [x.strip() for x in _FALLBACK_MODELS.split(",") if x.strip()]

DECOMMISSIONED_GROQ_MODELS = {
    "llama-3.1-70b-versatile", "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
}
_dead_models_lock = threading.Lock()
_dead_models = set(DECOMMISSIONED_GROQ_MODELS)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3131-\uD79D\u3040-\u309F\u30A0-\u30FF]")


def _dedupe_models(models: List[str]) -> List[str]:
    seen = set()
    out = []
    for m in models:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


GROQ_MODELS = _dedupe_models([_PRIMARY_MODEL] + list(_FALLBACK_MODELS))
GROQ_MODELS = [m for m in GROQ_MODELS if m not in DECOMMISSIONED_GROQ_MODELS]
if not GROQ_MODELS:
    GROQ_MODELS = ["openai/gpt-oss-120b"]


def _live_models() -> List[str]:
    with _dead_models_lock:
        live = [m for m in GROQ_MODELS if m not in _dead_models]
    return live or list(GROQ_MODELS)


_tls = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_tls, "session"):
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _tls.session = session
    return _tls.session


def _get_groq_api_keys(api_key: Optional[str] = None) -> List[str]:
    keys = []
    if hasattr(config, "get_groq_api_keys"):
        try:
            keys = config.get_groq_api_keys()
        except Exception:
            keys = []
    if not keys and api_key and str(api_key).strip():
        keys = [str(api_key).strip()]
    if not keys:
        single_key = getattr(config, "GROQ_API_KEY", "")
        if single_key and str(single_key).strip():
            keys = [str(single_key).strip()]
    if not keys:
        raise RuntimeError("Groq: thiếu API key. Set GROQ_API_KEY_1... trong .env")
    keys = [k for k in keys if k and str(k).strip()]
    logger.info(f"Đã tìm thấy {len(keys)} Groq API keys")
    return keys


def parse_response(translated_text: str) -> Dict[int, str]:
    parsed = {}
    last_n = None
    for line in str(translated_text or "").split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^\s*(?:[-*•]\s*)?(\d+)[.:)\-\]]?\s*(.+)$", line)
        if match:
            n = int(match.group(1)) - 1
            content = match.group(2).strip()
            if content:
                parsed[n] = content
                last_n = n
        elif last_n is not None and last_n in parsed:
            parsed[last_n] = f"{parsed[last_n]} {line}".strip()
    return parsed



# ============================================================
# xKiro / Qwen support
# ============================================================
XKIRA_API_BASE = str(getattr(config, "XKIRA_API_BASE", "https://api.xkiro.com/v1")).rstrip("/")
XKIRA_CHAT_URL = XKIRA_API_BASE + "/chat/completions"
XKIRA_MODEL_CACHE_SECONDS = int(getattr(config, "XKIRA_MODEL_CACHE_SECONDS", 300))
XKIRA_TRANSLATE_BATCH_SIZE = max(1, min(50, int(getattr(config, "XKIRA_TRANSLATE_BATCH_SIZE", 50))))
XKIRA_TRANSLATE_MAX_BATCH_CHARS = int(getattr(config, "XKIRA_TRANSLATE_MAX_BATCH_CHARS", 12000))
XKIRA_TRANSLATE_TIMEOUT = int(getattr(config, "XKIRA_TRANSLATE_TIMEOUT", 90))
XKIRA_TRANSLATE_RETRY_COUNT = int(getattr(config, "XKIRA_TRANSLATE_RETRY_COUNT", 2))
XKIRA_TRANSLATE_MAX_CONCURRENCY = int(getattr(config, "XKIRA_TRANSLATE_MAX_CONCURRENCY", 10))
XKIRA_USE_PAID_QWEN = bool(getattr(config, "XKIRA_USE_PAID_QWEN", True))
_xkira_models_cache: List[Dict[str, Any]] = []
_xkira_models_cache_at = 0.0
_xkira_models_lock = threading.Lock()
_xkira_dead_models: Dict[str, float] = {}
_xkira_dead_models_lock = threading.Lock()


def _get_xkira_api_keys(api_key: Optional[Union[str, List[str], tuple]] = None) -> List[str]:
    keys: List[str] = []
    if isinstance(api_key, (list, tuple)):
        keys.extend([str(k).strip() for k in api_key if str(k).strip()])
    elif api_key and str(api_key).strip():
        keys.append(str(api_key).strip())
    if not keys and hasattr(config, "get_xkira_api_keys"):
        try:
            keys = [str(k).strip() for k in (config.get_xkira_api_keys() or []) if str(k).strip()]
        except Exception:
            keys = []
    if not keys:
        single = getattr(config, "XKIRA_API_KEY", "")
        if single and str(single).strip():
            keys = [str(single).strip()]
    if not keys:
        raise RuntimeError("xKiro: thiếu API key. Set XKIRA_API_KEY_1...XKIRA_API_KEY_8 trong .env")
    return list(dict.fromkeys(keys))


def get_xkira_qwen_models(force: bool = False, include_paid: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Live Qwen model catalog from xKiro. The public /v1/models endpoint is the source of truth."""
    global _xkira_models_cache, _xkira_models_cache_at
    now = time.time()
    include_paid = XKIRA_USE_PAID_QWEN if include_paid is None else bool(include_paid)
    with _xkira_models_lock:
        if not force and _xkira_models_cache and (now - _xkira_models_cache_at) < XKIRA_MODEL_CACHE_SECONDS:
            models = list(_xkira_models_cache)
        else:
            try:
                resp = requests.get(XKIRA_API_BASE + "/models", timeout=12)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                models = [m for m in data if str(m.get("id", "")).lower().startswith("qwen/")]
                models.sort(key=lambda m: (0 if str(m.get("access_tier", "")).lower() == "free" or str(m.get("id", "")).endswith(":free") else 1, str(m.get("id", ""))))
                _xkira_models_cache = models
                _xkira_models_cache_at = now
            except Exception as exc:
                logger.warning("xKiro: không tải được catalog Qwen: %s", exc)
                models = list(_xkira_models_cache)
    if not include_paid:
        models = [m for m in models if str(m.get("access_tier", "")).lower() == "free" or str(m.get("id", "")).endswith(":free")]
    with _xkira_dead_models_lock:
        live = [m for m in models if _xkira_dead_models.get(str(m.get("id")), 0) <= now]
    return live


def _mark_xkira_model_dead(model: str, seconds: float = 900.0) -> None:
    with _xkira_dead_models_lock:
        _xkira_dead_models[str(model)] = time.time() + max(30.0, seconds)



_COMMON_LATIN_WORDS = {
    "the","and","but","for","with","from","this","that","you","your","i","we","they",
    "he","she","it","is","are","was","were","not","have","has","had","to","of","in","on",
    "at","by","or","so","if","then","than","a","an","my","our","their","me","him","her",
    "do","does","did","can","will","would","could","should","just","very","really","okay","ok",
    "here","hey","thank","thanks","damn","right","go","up","what","where","when","why","how",
    "please","yeah","yes","no","now","then","one","two","three","a","an","the"
}


def _collect_protected_terms(texts: List[str], glossary=None) -> List[tuple[str, str]]:
    """Collect terms that must survive translation.

    Project/user glossary is authoritative. Automatic Latin protection is conservative
    so ordinary English phrases remain translatable; proper nouns in Chinese are kept
    when the entity-audit/glossary marks them.
    """
    terms = {}
    for item in (glossary or []):
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "") or "").strip()
        if len(src) < 2:
            continue
        dst = str(item.get("target", "") or "").strip() or src
        terms[src] = dst

    if bool(getattr(config, "TRANSLATION_PROTECT_LATIN_TERMS", True)):
        rx = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9._'’\-]*(?:\s+[A-Za-z][A-Za-z0-9._'’\-]*)*(?![A-Za-z0-9_])")
        for text in texts or []:
            for m in rx.finditer(str(text or "")):
                token = m.group(0).strip()
                if len(token) < 2 or token.lower() in _COMMON_LATIN_WORDS:
                    continue
                words = [w for w in re.split(r"\s+", token) if w]
                alpha_chars = [c for c in token if c.isalpha()]
                has_digit = any(c.isdigit() for c in token)
                has_internal_upper = any(any(c.isupper() for c in w[1:] if c.isalpha()) for w in words)
                all_upper = bool(alpha_chars) and all(c.isupper() for c in alpha_chars)
                title_words = sum(1 for w in words if any(c.isalpha() for c in w) and next((c for c in w if c.isalpha()), '').isupper())
                title_phrase = len(words) >= 2 and title_words == len(words)
                dotted_or_branded = bool(re.search(r"[A-Za-z0-9]\.[A-Za-z0-9]", token) or re.search(r"[A-Za-z0-9][_-][A-Za-z0-9]", token))
                if has_digit or has_internal_upper or all_upper or title_phrase or dotted_or_branded:
                    terms.setdefault(token, token)
    return sorted(terms.items(), key=lambda x: len(x[0]), reverse=True)


def _protect_terms_in_texts(texts: List[str], glossary=None):
    terms = _collect_protected_terms(texts, glossary)
    mapping = {}
    protected_texts = [str(t or "") for t in texts]
    expected = [[] for _ in protected_texts]
    for term_i, (src, dst) in enumerate(terms, 1):
        placeholder = f"KEEPTERM{term_i:03d}"
        for i, text in enumerate(protected_texts):
            if src in text:
                protected_texts[i] = text.replace(src, placeholder)
                mapping[placeholder] = dst
                expected[i].append(placeholder)
    return protected_texts, mapping, expected


def _restore_protected_terms(text: str, mapping: Dict[str, str]) -> str:
    out = str(text or "")
    # Replace exact placeholders first; also tolerate a model adding spaces inside the token.
    for placeholder, value in sorted((mapping or {}).items(), key=lambda kv: len(kv[0]), reverse=True):
        out = out.replace(placeholder, value)
        loose = r"[^A-Za-z0-9]*".join(re.escape(ch) for ch in placeholder)
        out = re.sub(loose, lambda _m, v=value: v, out, flags=re.I)
    return out.strip()


def _protected_terms_missing(text: str, expected: List[str]) -> List[str]:
    t = str(text or "")
    normalized = re.sub(r"[^A-Za-z0-9]+", "", t).lower()
    missing = []
    for p in expected:
        p = str(p)
        p_norm = re.sub(r"[^A-Za-z0-9]+", "", p).lower()
        if p in t or (p_norm and p_norm in normalized):
            continue
        loose = r"\W*".join(re.escape(ch) for ch in p)
        if re.search(loose, t, flags=re.I):
            continue
        missing.append(p)
    return missing


def _canonical_terms_for_source(source: str, glossary=None) -> List[tuple[str, str]]:
    src = str(source or "")
    return [(term, target) for term, target in _collect_protected_terms([src], glossary) if term in src]


def _canonical_translation_terms_missing(source: str, translated: str, glossary=None) -> List[str]:
    out = str(translated or "")
    missing = []
    for term, target in _canonical_terms_for_source(source, glossary):
        canonical = str(target or term).strip()
        if len(canonical) >= 2 and canonical not in out:
            missing.append(canonical)
    return missing


def _merge_unique_glossary(base_glossary=None, discovered=None) -> List[Dict[str, str]]:
    out=[]
    seen={}
    for item in list(base_glossary or []) + list(discovered or []):
        if not isinstance(item, dict):
            continue
        src=str(item.get("source", "") or "").strip()
        if len(src)<2:
            continue
        dst=str(item.get("target", "") or "").strip() or src
        key=src.casefold()
        if key in seen:
            # Explicit/project glossary remains authoritative.
            continue
        seen[key]=True
        out.append({"source":src,"target":dst})
    return out


def _parse_entity_audit(text: str, source_batch: Optional[List[str]] = None) -> List[str]:
    """Parse the entity audit and only keep candidates actually present in source text.

    The previous parser accepted any CJK-looking output from the audit model, which could
    accidentally turn ordinary Chinese words/phrases into KEEP_TERM markers.  V22 binds
    each returned candidate to its numbered source sentence and rejects full-sentence
    candidates or candidates that do not occur verbatim in the source.
    """
    sources = [str(x or '') for x in (source_batch or [])]
    candidates: List[tuple[int, str]] = []
    for line in str(text or '').splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(\d+)\s*[.:)\-]\s*(.*)$", line)
        if m:
            idx = max(0, int(m.group(1)) - 1)
            payload = m.group(2).strip()
        else:
            idx = -1
            payload = line
        # Accept the pipe-separated format requested by the audit prompt.
        parts = re.split(r"\s*[|｜]\s*", payload) if payload else []
        for part in parts:
            term = part.strip().strip('`"“”‘’[]()<>')
            if 2 <= len(term) <= 80 and not re.fullmatch(r"\d+[.:)]?", term):
                candidates.append((idx, term))

    bad = {"không có", "không rõ", "không chắc", "không chắc chắn", "none", "n/a", "null", "không", "无", "没有", "不确定"}
    out: List[str] = []
    seen = set()
    for idx, term in candidates:
        if term.casefold() in bad:
            continue
        # The candidate must occur in the source sentence it claims to belong to;
        # if numbering is malformed, allow a match anywhere in this audit batch.
        possible_sources = []
        if 0 <= idx < len(sources):
            possible_sources.append(sources[idx])
        possible_sources.extend(s for s in sources if s not in possible_sources)
        if not any(term in src for src in possible_sources):
            continue
        # Reject obvious full sentences / long CJK phrases.
        cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", term))
        if cjk_count >= 2 and cjk_count > 8:
            continue
        if len(re.findall(r"\s+", term)) >= 4:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def _discover_named_entities(texts: List[str], target_lang: str, api_key=None, provider="xkira", model_override=None) -> List[Dict[str, str]]:
    """High-precision one-time name/term audit used to lock proper nouns.

    The audit intentionally prefers false negatives over false positives: it asks the
    model to return only clearly identifiable named entities. The discovered terms are
    then placeholder-protected during the real translation call.
    """
    if not bool(getattr(config, "TRANSLATION_AUTO_ENTITY_AUDIT", True)) or not texts:
        return []
    # Limit each audit request to keep latency and prompt size bounded.
    max_items=int(getattr(config,"TRANSLATION_ENTITY_AUDIT_BATCH",60))
    source_batches=[texts[i:i+max_items] for i in range(0,len(texts),max_items)]
    all_terms=[]
    for batch_no,batch in enumerate(source_batches,1):
        numbered="\n".join(f"{i+1}. {t}" for i,t in enumerate(batch))
        prompt=(
            "Phân tích danh từ riêng trong các câu phụ đề tiếng Trung sau. CHỈ trả về các tên/thuật ngữ "
            "chắc chắn là tên riêng cần giữ nguyên, gồm: tên người, nhân vật, game/anime/phim, vật phẩm, "
            "kỹ năng, vũ khí, địa danh riêng, tổ chức, thương hiệu, sản phẩm, tài khoản/server/map/boss. "
            "Ưu tiên độ chính xác: KHÔNG đưa từ thông thường, đại từ, động từ, địa danh chung, mô tả chung vào danh sách. "
            "Nếu không chắc, bỏ qua. Không phiên âm, không dịch tên. Mỗi dòng tương ứng một câu nguồn và có thể có 0 hoặc nhiều mục, "
            "ngăn cách bằng |. Chỉ trả về các dòng đánh số, không giải thích.\n\n"+numbered)
        raw=None
        try:
            if provider=="xkira":
                keys=_get_xkira_api_keys(api_key)
                models=get_xkira_qwen_models(include_paid=XKIRA_USE_PAID_QWEN)
                mids=[str(m.get("id")) for m in models if m.get("id")]
                if model_override and str(model_override).strip():
                    mids=[str(model_override).strip()]
                model=mids[0] if mids else str(getattr(config,"XKIRA_STT_MODEL","qwen/qwen3.8-omni-flash:free"))
                key=keys[0]
                payload={"model":model,"messages":[
                    {"role":"system","content":"Bạn là chuyên gia nhận diện tên riêng cho phụ đề. Ưu tiên độ chính xác, bỏ qua nếu không chắc."},
                    {"role":"user","content":prompt}],"temperature":0.0,"max_tokens":2048,"stream":False}
                resp=_get_session().post(XKIRA_CHAT_URL,json=payload,headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},timeout=(10,XKIRA_TRANSLATE_TIMEOUT))
                if resp.status_code==200:
                    raw=((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content","") or ""
            else:
                keys=_get_groq_api_keys(api_key)
                models=_live_models()
                model=str(model_override or (models[0] if models else _PRIMARY_MODEL))
                payload={"model":model,"messages":[
                    {"role":"system","content":"Bạn là chuyên gia nhận diện tên riêng cho phụ đề. Ưu tiên độ chính xác, bỏ qua nếu không chắc."},
                    {"role":"user","content":prompt}],"temperature":0.0,"max_tokens":2048}
                resp=_get_session().post("https://api.groq.com/openai/v1/chat/completions",json=payload,headers={"Authorization":f"Bearer {keys[0]}","Content-Type":"application/json"},timeout=(10,TIMEOUT))
                if resp.status_code==200:
                    raw=((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content","") or ""
        except Exception as exc:
            logger.warning("Entity audit batch %d/%d bỏ qua: %s", batch_no,len(source_batches),exc)
            continue
        terms=_parse_entity_audit(raw)
        all_terms.extend(terms)
    uniq=[]; seen=set()
    for term in all_terms:
        key=term.casefold()
        if key not in seen:
            seen.add(key); uniq.append({"source":term,"target":term})
    if uniq:
        logger.info("Entity audit: khóa %d tên/thuật ngữ",len(uniq))
    return uniq


def _is_likely_discovered_name(term: str) -> bool:
    t = str(term or "").strip()
    if len(t) < 2:
        return False
    if _CJK_RE.search(t):
        # V22: do not treat every Chinese phrase as a proper noun.
        # Entity-audit candidates must be short, compact names/terms.
        core_cjk = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff]", "", t)
        if core_cjk.strip():
            return False
        return 2 <= len(t) <= 8
    core = t.strip(" .,!?:;，。！？；：\"'“”‘’()[]{}")
    if not core:
        return False
    words = [w for w in re.split(r"\s+", core) if w]
    lowered = [re.sub(r"[^A-Za-z]", "", w).lower() for w in words]
    if lowered and all((w in _COMMON_LATIN_WORDS or not w) for w in lowered):
        return False
    alpha = [c for c in core if c.isalpha()]
    has_digit = any(c.isdigit() for c in core)
    has_internal_upper = any(any(c.isupper() for c in w[1:] if c.isalpha()) for w in words)
    all_upper = bool(alpha) and all(c.isupper() for c in alpha)
    title_words = sum(1 for w in words if any(c.isalpha() for c in w) and next((c for c in w if c.isalpha()), '').isupper())
    title_phrase = len(words) >= 2 and title_words == len(words)
    dotted_or_branded = bool(re.search(r"[A-Za-z0-9]\.[A-Za-z0-9]", core) or re.search(r"[A-Za-z0-9][_-][A-Za-z0-9]", core))
    single_proper = len(words) == 1 and next((c for c in words[0] if c.isalpha()), '').isupper() and lowered[0] not in _COMMON_LATIN_WORDS
    return has_digit or has_internal_upper or all_upper or title_phrase or dotted_or_branded or single_proper


def _effective_glossary(texts: List[str], target_lang: str, api_key=None, provider="xkira", model_override=None, glossary=None):
    discovered = _discover_named_entities(texts,target_lang,api_key,provider,model_override)
    discovered = [item for item in discovered if isinstance(item, dict) and _is_likely_discovered_name(item.get("source", ""))]
    return _merge_unique_glossary(glossary,discovered)

def _build_translation_prompt(texts: List[str], target_lang: str, context_window: int = 0,
                              glossary=None, additional_prompt: str = "") -> str:
    protected_texts = [str(t or "") for t in texts]
    combined = "\n".join(f"{i+1}. {t}" for i, t in enumerate(protected_texts))
    glossary_block = ""
    if glossary:
        pairs = [f"- {str(x.get('source','')).strip()} => {str(x.get('target','')).strip()}"
                 for x in glossary if isinstance(x, dict) and str(x.get('source','')).strip()]
        if pairs:
            glossary_block = "\nTHUẬT NGỮ/TÊN RIÊNG BẮT BUỘC:\n" + "\n".join(pairs) + "\n"
    extra = str(additional_prompt or "").strip()
    extra_block = f"\nYÊU CẦU BỔ SUNG:\n{extra}\n" if extra else ""
    ctx_block = ""
    # Context is intentionally omitted here for xKiro batch prompts unless caller supplies it in additional_prompt.
    return f"""Dịch chính xác các câu sau sang {target_lang}. Đây là phụ đề hội thoại/video/game.
YÊU CẦU:
- Đúng nghĩa và đúng ngữ cảnh; ưu tiên tiếng Việt tự nhiên, dễ đọc thành lời.
- Giữ nguyên tên game, nhân vật, vật phẩm, kỹ năng, thương hiệu khi chúng là tên riêng.
- Không tự bịa hoặc đổi tên riêng.
- Không bỏ sót ý, không thêm ý không có trong nguồn.
- Mỗi câu nguồn tương ứng đúng 1 câu dịch.
- CHỈ trả về đúng {len(texts)} dòng.
- Mỗi dòng bắt đầu bằng đúng số thứ tự + dấu chấm.
- Không giải thích, không markdown, không gộp hoặc tách câu.
{glossary_block}{extra_block}{ctx_block}
VĂN BẢN:
{combined}
BẢN DỊCH:"""


def _pass_through_translation(source: str, target_lang: str) -> Optional[str]:
    """Return a valid translation for content that should not be sent to the LLM.
    Numeric-only / symbol-only subtitle tokens are semantically unchanged.
    Common short acknowledgements are localized for Vietnamese speech.
    """
    s = (source or "").strip()
    if not s:
        return None
    lang = str(target_lang or "").lower()
    compact = re.sub(r"\s+", "", s)
    # Pure numeric / punctuation tokens: 7.2, 2800, 1 2 3, 10, etc.
    if compact and re.fullmatch(r"[\d.,:+\-/%×xX()]+", compact):
        return s
    if "vi" in lang and re.fullmatch(r"(?i)ok(?:ay)?", s):
        return "Được"
    if "vi" in lang and re.fullmatch(r"(?i)yes", s):
        return "Vâng"
    if "vi" in lang and re.fullmatch(r"(?i)no", s):
        return "Không"
    return None


def _translate_with_xkira(texts: List[str], target_lang: str, api_key=None,
                          max_workers: int = 10, context_window: int = 0, force: bool = False,
                          glossary=None, additional_prompt: str = "", model_override: Optional[str] = None) -> List[str]:
    keys = _get_xkira_api_keys(api_key)
    result = [""] * len(texts)
    if not texts:
        return result
    model_meta = get_xkira_qwen_models(include_paid=XKIRA_USE_PAID_QWEN)
    model_ids = [str(m.get("id")) for m in model_meta if m.get("id")]
    if model_override and str(model_override).strip():
        requested = str(model_override).strip()
        model_ids = [requested] if requested in model_ids or not model_ids else [requested]
    if not model_ids:
        fallback = str(getattr(config, "XKIRA_STT_MODEL", "qwen/qwen3.8-omni-flash:free"))
        model_ids = [fallback]
    logger.info("xKiro Qwen: %s models | %s keys | workers<=%s", len(model_ids), len(keys), max_workers)
    batches = []
    start = 0
    chars = 0
    for i, t in enumerate(texts):
        chars += len(t or "") + 4
        if (i - start + 1) >= XKIRA_TRANSLATE_BATCH_SIZE or chars > XKIRA_TRANSLATE_MAX_BATCH_CHARS:
            batches.append((texts[start:i+1], start))
            start = i + 1
            chars = 0
    if start < len(texts):
        batches.append((texts[start:], start))

    key_index = 0
    model_index = 0
    lock = threading.RLock()
    key_cooldown = {k: 0.0 for k in keys}
    key_fail = {k: 0 for k in keys}
    model_fail = {m: 0 for m in model_ids}
    completed = 0

    def pick_key() -> str:
        nonlocal key_index
        while True:
            now = time.time()
            with lock:
                usable = [k for k in keys if key_cooldown.get(k, 0.0) <= now]
                if usable:
                    k = usable[key_index % len(usable)]
                    key_index += 1
                    return k
                wait = min((key_cooldown.get(k, now+1) for k in keys), default=now+1) - now
            time.sleep(max(0.05, min(wait, 2.0)))

    def pick_model() -> str:
        nonlocal model_index
        now=time.time()
        with lock:
            usable=[m for m in model_ids if _xkira_dead_models.get(m,0) <= now]
            if not usable:
                usable=list(model_ids)
            m=usable[model_index % len(usable)]
            model_index += 1
            return m

    def save_cache(src: str, dst: str):
        if not getattr(config, "CACHE_ENABLED", True): return
        try:
            basis=f"V22-NAMELOCK|{src}|{target_lang}|xkira|{json_safe_glossary(glossary)}|{additional_prompt}"
            (CACHE_DIR / f"{hashlib.md5(basis.encode('utf-8')).hexdigest()}.txt").write_text(dst, encoding='utf-8')
        except Exception: pass

    def _local_glossary(batch_sources: List[str]):
        joined = "\n".join(str(x or "") for x in batch_sources)
        local = []
        for item in (glossary or []):
            if not isinstance(item, dict):
                continue
            src = str(item.get("source", "") or "").strip()
            if src and src in joined:
                local.append(item)
        return local

    def _repair_one_segment(source: str, target_lang_: str, term_items: List[Dict[str, str]], key: str, model: str) -> Optional[str]:
        """Single-segment repair that does not depend on multi-segment placeholders.

        This is only used after a batch response fails term preservation.  The repair
        call gets the exact canonical terms for this one source sentence, so a missing
        marker in one sentence cannot poison/retry the whole batch.
        """
        term_lines = []
        for item in term_items:
            src = str(item.get("source", "") or "").strip()
            dst = str(item.get("target", "") or src).strip() or src
            if src and src in source:
                term_lines.append(f"- {src} => {dst}")
        prompt = (
            f"Dịch duy nhất câu sau sang {target_lang_}.\n"
            "Giữ nguyên tuyệt đối tên riêng/thuật ngữ được liệt kê; không phiên âm, không Hán-Việt hóa.\n"
            "Không thêm hoặc bớt ý. Chỉ trả về bản dịch, một dòng.\n"
            + ("THUẬT NGỮ BẮT BUỘC:\n" + "\n".join(term_lines) + "\n" if term_lines else "")
            + f"NGUỒN: {source}\nBẢN DỊCH:")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Bạn là dịch giả Trung→Việt. Giữ nguyên chính xác các tên/thuật ngữ được chỉ định."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0, "max_tokens": 512, "stream": False,
        }
        try:
            resp = _get_session().post(XKIRA_CHAT_URL, json=payload, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=(10, XKIRA_TRANSLATE_TIMEOUT))
            if resp.status_code != 200:
                return None
            raw = ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""
            val = str(raw).strip()
            # Strip accidental numbering/markdown from the one-line repair.
            m = re.match(r"^\s*(?:\d+\s*[.:)\-]\s*)?(.*?)\s*$", val, flags=re.S)
            if m:
                val = m.group(1).strip().strip('`')
            if not val:
                return None
            if _looks_same(source, val, target_lang_):
                pt = _pass_through_translation(source, target_lang_)
                if pt is not None:
                    return pt
            missing = _canonical_translation_terms_missing(source, val, term_items)
            if missing:
                return None
            return val
        except requests.RequestException:
            return None
        except Exception:
            return None

    def one_batch(batch_texts: List[str], start_idx: int):
        pending={i:t.strip() for i,t in enumerate(batch_texts) if str(t or '').strip()}
        out={i:'' for i in range(len(batch_texts)) if not str(batch_texts[i] or '').strip()}
        # Do not waste API calls on content whose meaning is unchanged.
        for local_i, src in list(pending.items()):
            passthrough = _pass_through_translation(src, target_lang)
            if passthrough is not None:
                out[local_i] = passthrough
                pending.pop(local_i, None)
                save_cache(src, passthrough)
                logger.debug("xKiro pass-through segment %s: %r -> %r", start_idx + local_i + 1, src, passthrough)
                continue
            basis=f"V22-NAMELOCK|{src}|{target_lang}|xkira|{json_safe_glossary(glossary)}|{additional_prompt}"
            cf=CACHE_DIR / f"{hashlib.md5(basis.encode('utf-8')).hexdigest()}.txt"
            if not force and getattr(config, "CACHE_ENABLED", True) and cf.exists():
                try:
                    cached=cf.read_text(encoding='utf-8').strip()
                    if cached and not (target_lang and 'vi' in target_lang.lower() and _CJK_RE.search(cached)):
                        out[local_i]=cached; pending.pop(local_i,None)
                except Exception: pass
            elif force and cf.exists():
                try: cf.unlink()
                except Exception: pass
        if not pending:
            return out
        attempts=0
        while pending and attempts < max(1, XKIRA_TRANSLATE_RETRY_COUNT + 1):
            attempts += 1
            model=pick_model(); key=pick_key()
            subset=list(pending.keys())
            batch_sources=[pending[i] for i in subset]
            local_glossary = _local_glossary(batch_sources)
            protected_texts, protected_map, expected_terms = _protect_terms_in_texts(batch_sources, local_glossary)
            prompt=_build_translation_prompt(protected_texts, target_lang, context_window, local_glossary, additional_prompt)
            payload={"model":model,"messages":[
                {"role":"system","content":"Bạn là dịch giả phụ đề chuyên nghiệp Trung→Việt. Độ chính xác và 1:1 là ưu tiên cao nhất. Tên nhân vật, tên riêng, game, vật phẩm, kỹ năng, thương hiệu và thuật ngữ được khóa phải được giữ nguyên tuyệt đối; không phiên âm, không dịch, không đổi cách viết. Không được xóa hoặc đổi token KEEPTERM001/KEEPTERM002/... nếu xuất hiện."},
                {"role":"user","content":prompt}],"temperature":0.0,"max_tokens":2048,"stream":False}
            try:
                resp=_get_session().post(XKIRA_CHAT_URL, json=payload,
                    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
                    timeout=(10, XKIRA_TRANSLATE_TIMEOUT))
                if resp.status_code == 200:
                    data=resp.json(); raw=((data.get('choices') or [{}])[0].get('message') or {}).get('content','') or ''
                    parsed=parse_response(raw)
                    got=0
                    for n,val in parsed.items():
                        if 0 <= n < len(subset) and val.strip():
                            raw_val = val.strip()
                            local_i = subset[n]
                            src = pending.get(local_i, '')
                            expected = expected_terms[n] if n < len(expected_terms) else []
                            missing_keep = _protected_terms_missing(raw_val, expected)
                            if missing_keep:
                                # V22: do not poison/retry the entire batch. Repair this one sentence
                                # with only its own canonical terms, then accept it if validation passes.
                                local_term_items = [g for g in local_glossary if isinstance(g, dict) and str(g.get("source", "") or "").strip() in src]
                                repaired = _repair_one_segment(src, target_lang, local_term_items, key, model) if local_term_items else None
                                if repaired:
                                    out[local_i] = repaired.strip()
                                    save_cache(src, repaired.strip())
                                    pending.pop(local_i, None)
                                    got += 1
                                else:
                                    logger.warning("xKiro segment %s cần sửa riêng vì không giữ đủ thuật ngữ khóa: %s", start_idx + local_i + 1, missing_keep)
                                continue
                            val = _restore_protected_terms(raw_val, protected_map)
                            canonical_missing = _canonical_translation_terms_missing(src, val, local_glossary)
                            if canonical_missing:
                                logger.warning("xKiro bỏ kết quả segment %s vì mất tên/thuật ngữ canonical sau restore: %s", start_idx + local_i + 1, canonical_missing)
                                continue
                            if _looks_same(src,val,target_lang):
                                passthrough = _pass_through_translation(src, target_lang)
                                if passthrough is None:
                                    continue
                                val = passthrough
                            out[local_i]=val.strip(); save_cache(src,val.strip()); pending.pop(local_i,None); got+=1
                    logger.debug("xKiro batch %s model=%s key=%s... parsed=%s/%s", start_idx, model, key[:8], got, len(subset))
                    if got == 0 and pending:
                        continue
                elif resp.status_code in (401,403,404):
                    _mark_xkira_model_dead(model, 600 if resp.status_code == 403 else 1800)
                    key_fail[key]=key_fail.get(key,0)+1
                    continue
                elif resp.status_code == 429:
                    ra=resp.headers.get('Retry-After','1')
                    try: wait=float(ra)
                    except Exception: wait=2.0
                    wait=max(1.0,min(wait,120.0))
                    with lock: key_cooldown[key]=time.time()+wait
                    continue
                elif resp.status_code >= 500:
                    time.sleep(min(2**(attempts-1),4.0)); continue
                else:
                    logger.warning("xKiro HTTP %s model=%s: %s", resp.status_code, model, resp.text[:180])
                    break
            except requests.RequestException as exc:
                logger.warning("xKiro request lỗi model=%s: %s", model, exc)
                if attempts < XKIRA_TRANSLATE_RETRY_COUNT + 1:
                    time.sleep(min(2**(attempts-1),4.0))
        return out

    effective_workers=max(1,min(int(max_workers), len(batches), XKIRA_TRANSLATE_MAX_CONCURRENCY, len(keys)))
    logger.info("xKiro dịch song song %s luồng | %s batch | Qwen=%s", effective_workers, len(batches), len(model_ids))
    with ThreadPoolExecutor(max_workers=effective_workers) as ex:
        futures={ex.submit(one_batch,b,s):s for b,s in batches}
        for fut in as_completed(futures):
            start_idx=futures[fut]
            try:
                batch_out=fut.result()
                for local_i,val in batch_out.items():
                    gi=start_idx+local_i
                    if 0 <= gi < len(result): result[gi]=val
                completed += 1
                logger.info("[DỊCH xKiro] batch %s/%s (%0.1f%%) | segment %s-%s", completed, len(batches), completed/len(batches)*100.0, start_idx+1, min(len(texts), start_idx+len(batches[next(i for i,x in enumerate(batches) if x[1]==start_idx)][0])))
            except Exception as exc:
                logger.error("xKiro batch %s lỗi: %s", start_idx, exc)
    return result


def _looks_same(source: str, translated: str, target_lang: str) -> bool:
    s=(source or '').strip(); t=(translated or '').strip()
    if not s or not t or s != t: return False
    return bool(target_lang and 'vi' not in str(target_lang).lower()) or not _CJK_RE.search(s)

def _direct_groq_completion(prompt: str, keys: List[str], system: str,
                            max_models: int = 2, max_keys: int = 3) -> Optional[str]:
    session = _get_session()
    url = "https://api.groq.com/openai/v1/chat/completions"
    models = _live_models()[:max_models]
    for model in models:
        for key in keys[:max_keys]:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0, "max_tokens": 4096,
            }
            try:
                resp = session.post(url, json=payload,
                                    headers={"Authorization": f"Bearer {key}",
                                             "Content-Type": "application/json"},
                                    timeout=(10, TIMEOUT))
            except requests.exceptions.RequestException:
                continue
            if resp.status_code == 200:
                try:
                    return (resp.json().get("choices", [{}])[0]
                            .get("message", {}).get("content", "") or "").strip()
                except Exception:
                    return None
            if resp.status_code == 404:
                with _dead_models_lock:
                    _dead_models.add(model)
                break
            if resp.status_code == 429:
                time.sleep(2.0)
                continue
    return None


def translate_segments_fast(segments, target_lang, api_key=None, provider="groq",
                            max_workers=MAX_WORKERS, context_window=2,
                            force=False, auto_retry=AUTO_RETRY, glossary=None,
                            additional_prompt="", model=None) -> List[str]:
    if not segments:
        return []
    logger.info(f"Dịch {len(segments)} đoạn với {provider} (batch={BATCH_SIZE}, workers={max_workers}, force={force})")
    start_time = time.time()
    texts = [str(s.get("text", "") or "").strip() for s in segments]
    result = [""] * len(texts)
    # V18: discover named entities once per translation run, then placeholder-lock them.
    effective_glossary = _effective_glossary(texts, target_lang, api_key, provider, model, glossary) if bool(getattr(config,"TRANSLATION_STRICT_NAME_PRESERVE",True)) else list(glossary or [])
    if provider not in {"groq", "xkira"}:
        raise ValueError(f"Provider {provider} không hỗ trợ. Chỉ hỗ trợ 'groq' hoặc 'xkira'.")
    try:
        if provider == "xkira":
            translated = _translate_with_xkira(texts, target_lang, api_key, max_workers, context_window, force=force, glossary=effective_glossary, additional_prompt=additional_prompt, model_override=model)
        else:
            translated = _translate_with_groq(texts, target_lang, api_key, max_workers, context_window, force=force, glossary=effective_glossary, additional_prompt=additional_prompt)
        for i, trans in enumerate(translated):
            if trans and trans.strip():
                result[i] = trans.strip()
    except Exception as e:
        logger.error(f"{provider} dịch thất bại (lần đầu): {e}")

    # Finalize pass-through segments before any retry/review pass.
    for i, src in enumerate(texts):
        if result[i].strip():
            continue
        passthrough = _pass_through_translation(src, target_lang)
        if passthrough is not None:
            result[i] = passthrough

    remaining = [i for i, r in enumerate(result) if not r or not r.strip()]
    effective_auto_retry = min(max(0, int(auto_retry)), 1) if provider == "xkira" else max(0, int(auto_retry))
    attempt = 0
    while remaining and attempt < effective_auto_retry:
        attempt += 1
        logger.warning(f"Tự động dịch lại {len(remaining)} đoạn (lần {attempt}/{effective_auto_retry}, force=True)...")
        retry_texts = [texts[i] for i in remaining]
        try:
            if provider == "xkira":
                retried = _translate_with_xkira(retry_texts, target_lang, api_key, max_workers, context_window, force=True, glossary=effective_glossary, additional_prompt=additional_prompt, model_override=model)
            else:
                retried = _translate_with_groq(retry_texts, target_lang, api_key, max_workers, context_window, force=True, glossary=effective_glossary, additional_prompt=additional_prompt)
        except Exception as e:
            logger.error(f"Tự động dịch lại thất bại (lần {attempt}): {e}")
            break
        still_remaining = []
        for i, trans in zip(remaining, retried):
            if trans and trans.strip():
                result[i] = trans.strip()
            else:
                still_remaining.append(i)
        remaining = still_remaining

    # V18: KHÔNG chạy "CJK transliteration rescue". Nó có thể biến tên nhân vật
    # thành cách viết khác. Tên/thuật ngữ được khóa sẽ được placeholder-protect trước khi dịch.

    elapsed = time.time() - start_time
    # Recompute from the final result so the progress count cannot become
    # inconsistent when a retry returns a partial list.
    remaining = [i for i, r in enumerate(result) if not str(r or '').strip()]
    translated_count = len(segments) - len(remaining)
    logger.info(f"Dịch xong {translated_count}/{len(segments)} đoạn trong {elapsed:.2f}s")
    if remaining:
        # Use 1-based segment numbers in logs so they match the editor/UI.
        details = []
        for i in sorted(remaining)[:20]:
            try:
                src = str(segments[i].get('text', '') or '').strip().replace('\n', ' ')[:140]
            except Exception:
                src = ''
            details.append(f"#{i+1} idx={i} source={src!r}")
        logger.error(f"{len(remaining)}/{len(segments)} đoạn KHÔNG dịch được sau {effective_auto_retry} lần retry: " + " | ".join(details))
    return result


def json_safe_glossary(glossary) -> str:
    try:
        if not glossary:
            return ""
        return ";".join(f"{str(x.get('source','')).strip()}=>{str(x.get('target','')).strip()}"
                        for x in glossary if isinstance(x, dict))
    except Exception:
        return str(glossary)

def _translate_with_groq(texts: List[str], target_lang: str, api_key: Optional[str],
                         max_workers: int, context_window: int = 0, force: bool = False,
                         glossary=None, additional_prompt="") -> List[str]:
    groq_keys = _get_groq_api_keys(api_key)
    if not groq_keys:
        raise RuntimeError("Groq: thiếu API key")
    result = [""] * len(texts)
    if not texts:
        return result

    batches = []
    start = 0
    chars = 0
    for i, t in enumerate(texts):
        chars += len(t or "") + 4
        if (i - start + 1) >= BATCH_SIZE or chars > MAX_BATCH_CHARS:
            batches.append((texts[start:i + 1], start))
            start = i + 1
            chars = 0
    if start < len(texts):
        batches.append((texts[start:], start))
    if not batches:
        return result

    key_index = 0
    key_fail_count = {k: 0 for k in groq_keys}
    MAX_KEY_FAIL = 5
    rate_lock = threading.Lock()
    key_cooldown_until = {k: 0.0 for k in groq_keys}
    global_pause_until = [0.0]
    batch_lock = threading.Lock()
    completed_batches = set()

    def get_next_key() -> str:
        nonlocal key_index
        while True:
            now = time.time()
            if global_pause_until[0] > now:
                time.sleep(min(global_pause_until[0] - now, 1.0))
                continue
            wait = 0.0
            with rate_lock:
                available_keys = [k for k in groq_keys
                                  if key_fail_count.get(k, 0) < MAX_KEY_FAIL
                                  and key_cooldown_until.get(k, 0.0) <= now]
                if not available_keys:
                    future_times = [key_cooldown_until.get(k, 0.0) for k in groq_keys
                                    if key_fail_count.get(k, 0) < MAX_KEY_FAIL]
                    future_times = [t for t in future_times if t > now]
                    if future_times:
                        wait = min(future_times) - now
                    else:
                        for k in groq_keys:
                            key_fail_count[k] = 0
                            key_cooldown_until[k] = 0.0
                        available_keys = groq_keys[:]
            if wait > 0:
                time.sleep(min(max(wait, 0.2), 5.0))
                continue
            with rate_lock:
                available_keys = [k for k in groq_keys
                                  if key_fail_count.get(k, 0) < MAX_KEY_FAIL
                                  and key_cooldown_until.get(k, 0.0) <= time.time()]
                if not available_keys:
                    for k in groq_keys:
                        key_fail_count[k] = 0
                        key_cooldown_until[k] = 0.0
                    available_keys = groq_keys[:]
                key = available_keys[key_index % len(available_keys)]
                key_index += 1
                return key

    def mark_key_success(key: str) -> None:
        with rate_lock:
            key_fail_count[key] = 0
            key_cooldown_until[key] = 0.0

    def cache_write(text: str, translated: str) -> None:
        if not getattr(config, "CACHE_ENABLED", True):
            return
        try:
            cache_basis = f"V22-NAMELOCK|{text}|{target_lang}|groq|{json_safe_glossary(glossary)}|{additional_prompt}"
            cache_file = CACHE_DIR / f"{hashlib.md5(cache_basis.encode('utf-8')).hexdigest()}.txt"
            cache_file.write_text(translated, encoding="utf-8")
        except Exception:
            pass

    def build_prompt(sub_texts: List[str], start_idx: int = 0) -> str:
        protected_texts, _protect_map, _expected = _protect_terms_in_texts(sub_texts, glossary)
        combined = "\n".join([f"{n + 1}. {t}" for n, t in enumerate(protected_texts)])
        ctx_lines = []
        if context_window and context_window > 0:
            before = texts[max(0, start_idx - context_window):start_idx]
            after = texts[start_idx + len(sub_texts):start_idx + len(sub_texts) + context_window]
            if before:
                ctx_lines.append("Các câu ngay trước (chỉ để hiểu ngữ cảnh, KHÔNG dịch):")
                ctx_lines += [f"## {t}" for t in before if t]
            if after:
                ctx_lines.append("Các câu ngay sau (chỉ để hiểu ngữ cảnh, KHÔNG dịch):")
                ctx_lines += [f"## {t}" for t in after if t]
        ctx_block = "\n".join(ctx_lines) + "\n\n" if ctx_lines else ""
        glossary_block = ""
        if glossary:
            pairs = [f"- {str(x.get('source','')).strip()} => {str(x.get('target','')).strip()}"
                     for x in glossary if isinstance(x, dict) and str(x.get('source','')).strip()]
            if pairs:
                glossary_block = "\nTHUẬT NGỮ/TÊN RIÊNG BẮT BUỘC:\n" + "\n".join(pairs) + "\n"
        extra = str(additional_prompt or "").strip()
        extra_block = f"\nYÊU CẦU BỔ SUNG CỦA NGƯỜI DÙNG:\n{extra}\n" if extra else ""
        return f"""Dịch các câu sau sang {target_lang}.
Đây là hội thoại trong video/game: dịch tự nhiên như lời nói nhân vật, đúng ngữ cảnh, đúng sắc thái.
YÊU CẦU:
- Dịch ĐÚNG nghĩa, không dịch từng chữ máy móc.
- Phân biệt tên game, tên nhân vật, vật phẩm, kỹ năng, địa danh, thương hiệu với từ ngữ thông thường.
- MỌI THUẬT NGỮ/TÊN RIÊNG ĐƯỢC ĐÁNH DẤU KEEPTERMXXX LÀ BẮT BUỘC GIỮ NGUYÊN VỊ TRÍ; KHÔNG dịch, không phiên âm, không đổi chính tả.
- Tên riêng chữ Hán đã được khóa phải được chép nguyên văn; tuyệt đối không Hán-Việt hóa, không phiên âm, không dịch.
- Nếu không chắc một cụm có phải tên riêng hay không, không tự sáng tạo tên mới; ưu tiên giữ nguyên cụm đó để kiểm tra.
- Không được suy đoán một từ không rõ thành tên riêng có thật. Nếu nguồn mơ hồ, giữ nguyên chuỗi nguồn để người dùng kiểm tra.
- Không bỏ sót nội dung thông thường cần dịch; chỉ giữ nguyên các tên/thuật ngữ thực sự là tên riêng hoặc được khóa.
- Không thêm ý không có trong nguồn.
- Mỗi câu KHỚP 1:1 với câu gốc, viết trên MỘT DÒNG DUY NHẤT.
- CHỈ trả về đúng {len(sub_texts)} dòng, mỗi dòng bắt đầu bằng đúng số thứ tự đã cho + dấu chấm + bản dịch.
- Không giải thích, không thêm chữ thừa, không lặp lại các câu ngữ cảnh.
{glossary_block}{extra_block}
{ctx_block}Văn bản cần dịch:
{combined}
Bản dịch:"""

    def _looks_untranslated(source: str, translated: str) -> bool:
        s = (source or "").strip()
        t = (translated or "").strip()
        return bool(s and t and s == t)

    def _should_accept_as_is(source: str, translated: str, lang: str = "") -> bool:
        s = (source or "").strip()
        t = (translated or "").strip()
        if s != t:
            return False
        if not s:
            return True
        if lang and "vi" in lang.lower() and _CJK_RE.search(s):
            return False
        if len(s) < 5:
            return True
        if not any(c.isalpha() for c in s):
            return True
        if re.search(r"[.!?。！？…]$", s):
            return False
        words = s.split()
        alpha_words = [w for w in words if any(c.isalpha() for c in w)]

        def first_alpha_upper(word: str) -> bool:
            for ch in word:
                if ch.isalpha():
                    return ch.isupper()
            return True

        if 1 <= len(words) <= 5 and len(s) <= 60 and alpha_words:
            if all(first_alpha_upper(w) for w in alpha_words):
                return True
            if s.isupper() and len(s) <= 30:
                return True
        cjk_chars = re.findall(_CJK_RE, s)
        if cjk_chars and len(s) <= max(0, int(CJK_ACCEPT_MAX_LEN)):
            return True
        return False

    def try_model(model: str, sub_indices: List[int], pending: Dict[int, str], global_start_idx: int = 0) -> Dict[int, str]:
        nonlocal key_index
        if not sub_indices:
            return {}
        batch_sources = [pending[i] for i in sub_indices]
        protected_texts, protected_map, expected_terms = _protect_terms_in_texts(batch_sources, glossary)
        prompt = build_prompt(batch_sources, global_start_idx + min(sub_indices))
        current_key = get_next_key()
        session = _get_session()
        url = "https://api.groq.com/openai/v1/chat/completions"
        key_rounds = min(max(2, len(groq_keys)), 6)
        for _ in range(key_rounds):
            headers = {"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Bạn là dịch giả chuyên nghiệp. Dịch chính xác, tự nhiên, giữ ngữ cảnh. Không giải thích. Không bỏ sót câu. Tên riêng đã khóa phải được giữ nguyên tuyệt đối. Không Hán-Việt hóa, không phiên âm, không đổi tên."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2, "max_tokens": 4096,
            }
            for attempt in range(RETRY_COUNT):
                try:
                    response = session.post(url, json=payload, headers=headers, timeout=(10, TIMEOUT))
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            translated_text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                        except Exception as e:
                            logger.error(f"Không parse được JSON từ Groq: {e}")
                            continue
                        parsed_global = {}
                        for n, translated in parse_response(translated_text).items():
                            if 0 <= n < len(sub_indices):
                                orig_i = sub_indices[n]
                                source_text = pending.get(orig_i, "")
                                raw_translated = translated.strip()
                                missing_keep = _protected_terms_missing(raw_translated, expected_terms[n] if n < len(expected_terms) else [])
                                if missing_keep:
                                    logger.warning("Groq bỏ kết quả segment %s vì mất thuật ngữ khóa: %s", orig_i + 1, missing_keep)
                                    continue
                                translated = _restore_protected_terms(raw_translated, protected_map)
                                canonical_missing = _canonical_translation_terms_missing(source_text, translated, glossary)
                                if canonical_missing:
                                    logger.warning("Groq bỏ kết quả segment %s vì mất tên/thuật ngữ canonical sau restore: %s", orig_i + 1, canonical_missing)
                                    continue
                                if _looks_untranslated(source_text, translated):
                                    if _should_accept_as_is(source_text, translated, target_lang):
                                        parsed_global[orig_i] = translated
                                        cache_write(source_text, translated)
                                    else:
                                        continue
                                else:
                                    parsed_global[orig_i] = translated
                                    cache_write(source_text, translated)
                        mark_key_success(current_key)
                        return parsed_global
                    elif response.status_code in (400, 404):
                        with _dead_models_lock:
                            _dead_models.add(model)
                        return {}
                    elif response.status_code in (401, 403):
                        with rate_lock:
                            key_fail_count[current_key] = MAX_KEY_FAIL
                            key_cooldown_until[current_key] = time.time() + 30
                        current_key = get_next_key()
                        break
                    elif response.status_code == 429:
                        retry_after = response.headers.get("Retry-After", "")
                        cooldown = 5.0
                        try:
                            if retry_after:
                                cooldown = float(retry_after)
                        except Exception:
                            cooldown = 5.0
                        cooldown = min(max(cooldown, 3.0), 30.0)
                        with rate_lock:
                            key_cooldown_until[current_key] = time.time() + cooldown
                            global_pause_until[0] = max(global_pause_until[0], time.time() + min(cooldown, 3.0))
                            key_fail_count[current_key] = MAX_KEY_FAIL
                        current_key = get_next_key()
                        headers["Authorization"] = f"Bearer {current_key}"
                        continue
                    else:
                        with rate_lock:
                            key_fail_count[current_key] = key_fail_count.get(current_key, 0) + 1
                        if response.status_code >= 500 and attempt < RETRY_COUNT - 1:
                            time.sleep(min(BACKOFF_BASE * (2 ** attempt), 5))
                            continue
                        break
                except requests.exceptions.Timeout:
                    with rate_lock:
                        key_fail_count[current_key] = key_fail_count.get(current_key, 0) + 1
                    if attempt < RETRY_COUNT - 1:
                        time.sleep(min(BACKOFF_BASE * (2 ** attempt), 5))
                        continue
                    break
                except requests.exceptions.RequestException as e:
                    logger.error(f"Groq request error key {current_key[:8]}...: {e}")
                    with rate_lock:
                        key_fail_count[current_key] = key_fail_count.get(current_key, 0) + 1
                    if attempt < RETRY_COUNT - 1:
                        time.sleep(min(BACKOFF_BASE * (2 ** attempt), 5))
                        continue
                    break
                except Exception as e:
                    logger.error(f"Groq lỗi không xác định key {current_key[:8]}...: {e}")
                    with rate_lock:
                        key_fail_count[current_key] = key_fail_count.get(current_key, 0) + 1
                    if attempt < RETRY_COUNT - 1:
                        time.sleep(min(BACKOFF_BASE * (2 ** attempt), 5))
                        continue
                    break
            current_key = get_next_key()
        return {}

    def translate_batch(batch_texts: List[str], start_idx: int) -> Dict[int, str]:
        results = {}
        pending = {}
        for i, text in enumerate(batch_texts):
            text = (text or "").strip()
            if not text:
                results[i] = ""
                continue
            cache_basis = f"V22-NAMELOCK|{text}|{target_lang}|groq|{json_safe_glossary(glossary)}|{additional_prompt}"
            cache_file = CACHE_DIR / f"{hashlib.md5(cache_basis.encode('utf-8')).hexdigest()}.txt"
            if not force and getattr(config, "CACHE_ENABLED", True) and cache_file.exists():
                try:
                    cached = cache_file.read_text(encoding="utf-8").strip()
                    if cached and cached != text:
                        if target_lang and "vi" in target_lang.lower() and _CJK_RE.search(cached):
                            try:
                                cache_file.unlink()
                            except Exception:
                                pass
                        else:
                            results[i] = cached
                            continue
                    elif cached and cached == text:
                        if _should_accept_as_is(text, cached, target_lang):
                            results[i] = cached
                            continue
                        try:
                            cache_file.unlink()
                        except Exception:
                            pass
                except Exception:
                    pass
            elif force and cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception:
                    pass
            pending[i] = text
        if not pending:
            return results
        for model in _live_models():
            if not pending:
                break
            parsed_global = try_model(model, list(pending.keys()), pending)
            for i, translated in parsed_global.items():
                if translated and translated.strip():
                    results[i] = translated
                    pending.pop(i, None)
        if pending:
            pending_ids = sorted(pending.keys())
            for model in _live_models():
                if not pending:
                    break
                for chunk_start in range(0, len(pending_ids), 8):
                    chunk_ids = [i for i in pending_ids[chunk_start:chunk_start + 8] if i in pending]
                    if not chunk_ids:
                        continue
                    parsed = try_model(model, chunk_ids, pending, start_idx)
                    for i, translated in parsed.items():
                        if translated and translated.strip():
                            results[i] = translated
                            pending.pop(i, None)
                if not pending:
                    break
        if pending:
            pending_ids = sorted(pending.keys())
            for model in _live_models():
                if not pending:
                    break
                for chunk_start in range(0, len(pending_ids), 3):
                    chunk_ids = [i for i in pending_ids[chunk_start:chunk_start + 3] if i in pending]
                    if not chunk_ids:
                        continue
                    parsed = try_model(model, chunk_ids, pending, start_idx)
                    for i, translated in parsed.items():
                        if translated and translated.strip():
                            results[i] = translated
                            pending.pop(i, None)
                if not pending:
                    break
        if pending:
            sample = [pending[i][:80] for i in list(pending.keys())[:5]]
            logger.error(f"Batch {start_idx}: vẫn còn {len(pending)} câu không dịch được, sample={sample}")
        for i in pending:
            results[i] = ""
        return results

    per_key = max(1, int(GROQ_MAX_CONCURRENT_PER_KEY))
    max_key_slots = max(1, len(groq_keys) * per_key)
    effective_workers = max(1, min(max_workers, len(batches), MAX_CONCURRENCY_CAP, max_key_slots))
    logger.info(f"Dịch song song {effective_workers} luồng | {len(batches)} batch | {len(groq_keys)} keys | model={_live_models()}")
    executor = ThreadPoolExecutor(max_workers=effective_workers)
    futures = {executor.submit(translate_batch, b, s): s for b, s in batches}
    total_timeout = max(1800, ((len(batches) // max(1, effective_workers)) + 1) * (TIMEOUT + 30) + 600)
    try:
        for future in as_completed(futures, timeout=total_timeout):
            start_idx = futures[future]
            with batch_lock:
                if start_idx in completed_batches:
                    continue
                completed_batches.add(start_idx)
                try:
                    batch_results = future.result(timeout=5)
                    for i, translated in batch_results.items():
                        global_idx = start_idx + i
                        if 0 <= global_idx < len(result):
                            result[global_idx] = translated
                except FuturesTimeoutError:
                    logger.error(f"Batch {start_idx} bị timeout")
                except Exception as e:
                    logger.error(f"Batch {start_idx} lỗi: {e}")
    except FuturesTimeoutError:
        logger.error("Vượt quá tổng thời gian dịch, hủy các batch chưa chạy")
        for f in futures:
            f.cancel()
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi dịch song song: {e}")
    finally:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)
    return result