"""AgentHub public batch runner. See README.md for setup and the offline example."""
from __future__ import annotations
import argparse
import base64
import collections
import concurrent.futures as cf
import datetime as dt
import json
import mimetypes
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
HUB = Path(os.environ.get("AGENTHUB_WORKSPACE", PROJECT / "runs")).resolve()
CFG = json.loads((PROJECT / "config.json").read_text(encoding="utf-8"))
JEV = CFG["jev"]
LOG = HUB / "logs" / "calls.jsonl"
EVALS = HUB / "logs" / "evals.jsonl"
IO_DIR = HUB / "logs" / "io"
TASKS = HUB / "tasks"
RECIPES = HUB / "recipes"
PROMPTS = HUB / "prompts"
BJ = dt.timezone(dt.timedelta(hours=8))
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
TXT_EXT = {".txt", ".md"}
CHAT_PROVIDERS = ("deepseek", "defapi")  # OpenAI 兼容的对话接口
API_EXECUTORS = ("deepseek", "defapi", "jev")  # 按量付费、计入每日预算的执行者
RECIPE_FIELDS = ("via", "model", "system", "json", "think", "detail", "max_tokens", "temperature")
HINTS = {
    "deepseek": {400: "请求格式错误", 401: "key 无效或已删除", 402: "余额不足，请到 DeepSeek 开放平台充值",
                 422: "参数错误", 429: "请求太快被限速", 500: "DeepSeek 服务器错误", 503: "DeepSeek 服务器繁忙"},
    "defapi": {400: "请求格式错误（模型名对吗？）", 401: "key 无效", 402: "余额不足，请到 defapi 充值",
               404: "模型或接口不存在", 429: "请求太快被限速", 500: "defapi 或上游服务器错误", 503: "上游繁忙"},
    "jev": {400: "请求违反规则", 401: "Jev key 无效", 413: "请求超过 32 MiB", 415: "压缩格式不支持",
            422: "字段不合法（看 detail）", 429: "请求太快被限速", 529: "TypeSafe 繁忙"},
}
_lock = threading.Lock()

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass





def now():
    return dt.datetime.now(BJ)


def text_or_file(text, path):
    if path:
        return Path(path).read_text(encoding="utf-8")
    if text == "-":
        return sys.stdin.read()
    return text


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def task_path(name):
    return Path(name) if Path(name).is_absolute() else TASKS / name


def task_dir(name):
    d = task_path(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_call(**rec):
    rec = {"ts": now().isoformat(timespec="seconds"), **{k: v for k, v in rec.items() if v is not None}}
    with _lock:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def read_log(since=""):
    return [r for r in read_jsonl(LOG) if r.get("ts", "") >= since]


def read_secret(name):
    return os.environ.get(name)


def money(v):
    """金额显示：小额不用科学计数法。"""
    if v is None:
        return "?"
    return f"{v:.6f}".rstrip("0").rstrip(".") if v < 0.01 else f"{v:.4f}"


def http_json(url, payload=None, headers=None, timeout=60, retries=3, hints=None):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < retries:
                time.sleep(3 * 2 ** attempt)
                continue
            body = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"HTTP {e.code} {(hints or {}).get(e.code, '')}：{body}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries:
                time.sleep(3 * 2 ** attempt)
                continue
            raise RuntimeError(f"网络错误：{e}") from None


def spent_today():
    day = now().date().isoformat()
    return sum(r.get("cost_cny") or 0 for r in read_log(day) if r.get("executor") in API_EXECUTORS)


def check_budget(force):
    cap, spent = CFG["budget"]["daily_cny"], spent_today()
    if spent >= cap and not force:
        sys.exit(f"今天 API 已花 ¥{spent:.2f}，到达每日上限 ¥{cap}（config.json → budget.daily_cny）。"
                 "先问用户；确需继续加 --force。")
    return spent


def batch_cap(args, spent0):
    budget = args.budget if args.budget is not None else CFG["budget"]["batch_run_cny"]
    return budget if args.force else min(budget, CFG["budget"]["daily_cny"] - spent0)


def get_key(required=True, via="deepseek"):
    name = CFG[via]["key_env"]
    value = read_secret(name)
    if required and not value:
        sys.exit(f"Set {name} in your shell before running this provider.")
    return value


def resolve_model(name, via="deepseek"):
    p = CFG[via]
    name = name or p.get("default_model")
    if not name:
        sys.exit(f"{via} 需要 --model")
    return p.get("aliases", {}).get(name, name)


def is_peak(t=None):
    t = (t or now()).astimezone(BJ)
    if t.weekday() >= 5:
        return False
    h = t.hour + t.minute / 60
    return any(a <= h < b for a, b in CFG["deepseek"]["peak_hours_bj"])


def wait_until_offpeak(stop=None):
    """高峰时段就睡到错峰（每 30 秒醒一次，方便中途停止）。"""
    announced = False
    while is_peak() and not (stop and stop["stop"]):
        if not announced:
            print(f"现在是 DeepSeek 高峰时段（价格翻倍），等到错峰再继续……（{now():%H:%M}）", file=sys.stderr)
            announced = True
        time.sleep(30)


def cached_tokens(u):
    return u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def price_known(via, model):
    return via == "deepseek" or model in CFG["defapi"]["models"]


def chat_cost(via, model, usage, peak):
    if via == "deepseek":
        price = CFG["deepseek"]["models"].get(model, {}).get("price_cny_per_m", {}).get("peak" if peak else "offpeak")
        if not price:
            return None
        hit = cached_tokens(usage)
        miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
        out = usage.get("completion_tokens", 0)
        return round((hit * price["in_hit"] + miss * price["in_miss"] + out * price["out"]) / 1e6, 6)
    p = CFG["defapi"]  # 没配置单价的模型按保守价估算（宁可高估，预算保护才有效）
    price = (p["models"].get(model) or {}).get("price_usd_per_m") or p["fallback_price_usd_per_m"]
    tin, out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return round((tin * price["in"] + out * price["out"]) * p["usd_cny"] / 1e6, 6)


def usage_fields(u):
    return {
        "in_tokens": u.get("prompt_tokens", 0),
        "cache_hit": cached_tokens(u),
        "out_tokens": u.get("completion_tokens", 0),
        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
    }


def image_block(src, detail=None):
    if src.startswith(("http://", "https://")):
        url = src
    else:
        p = Path(src)
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        url = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    block = {"type": "image_url", "image_url": {"url": url}}
    if detail:
        block["image_url"]["detail"] = detail
    return block


def build_messages(prompt, system=None, files=(), images=(), detail=None):
    # 固定内容在前、每项不同的内容在后，前缀才能命中缓存
    for f in files:
        p = Path(f)
        prompt += f'\n\n<file name="{p.name}">\n{p.read_text(encoding="utf-8", errors="replace")}\n</file>'
    content = ([{"type": "text", "text": prompt}] + [image_block(i, detail) for i in images]) if images else prompt
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": content})
    return msgs


def ensure_json_hint(prompt, system):
    # json_object 模式要求提示词里出现 "json"，否则模型可能一直输出空白
    if "json" in (prompt + (system or "")).lower():
        return prompt
    return prompt + "\n\n只输出 JSON。"


def chat(messages, model, think=None, json_mode=False, max_tokens=None, temperature=None, key=None, via="deepseek"):
    p = CFG[via]
    payload = {"model": model, "messages": messages, "stream": False}
    if via == "deepseek":
        payload["thinking"] = {"type": "enabled" if think else "disabled"}  # DeepSeek 默认开思考，不需要时关掉省钱
    if think:
        payload["reasoning_effort"] = think
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if temperature is not None and not think:
        payload["temperature"] = temperature
    peak, t0 = (is_peak() if via == "deepseek" else False), time.time()
    resp = http_json(p["base_url"].rstrip("/") + "/chat/completions", payload,
                     {"Authorization": f"Bearer {key or get_key(via=via)}"}, p["timeout_s"], p["retries"], HINTS[via])
    choice, usage = resp["choices"][0], resp.get("usage") or {}
    served = resp.get("model")
    actual = actual_usd(resp) if via == "defapi" else None  # 有实际扣费就用实际的，不靠单价估
    if actual is not None:
        cost = round(actual * CFG["defapi"]["usd_cny"], 6)
    else:
        cost = chat_cost(via, model, usage, peak)
    return {
        "content": choice["message"].get("content") or "",
        "reasoning": choice["message"].get("reasoning_content"),
        "finish": choice.get("finish_reason"),
        "usage": usage,
        "peak": peak,
        "served_model": served,
        # 中转平台可能"智能路由"到别的模型：实际返回的模型名和请求的不一致就标出来
        "mismatch": bool(via == "defapi" and served and not same_model(model, served)),
        "latency_s": round(time.time() - t0, 2),
        "cost_cny": cost,
        "estimated": actual is None and not price_known(via, model),  # 既没实际扣费、也没配置单价：按保守价估的
    }


def same_model(requested, served):
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().split("/")[-1])  # noqa: E731
    a, b = norm(requested), norm(served)
    return a == b or a in b or b in a


def model_notes(via, model):
    """把配置里记下的模型注意事项（上次验证不可用、固定开销大等）提醒出来。"""
    info = CFG[via].get("models", {}).get(model) or {}
    if info.get("available") is False:
        print(f"[提示] {model} 上次验证不可用：{info.get('note', '')}。可以先 python hub.py ping --all-models 重新验证。", file=sys.stderr)
    elif info.get("note"):
        print(f"[提示] {model}：{info['note']}", file=sys.stderr)


def check_vision(via, model, has_images):
    if not has_images:
        return
    info = CFG[via]["models"].get(model)
    if info is not None and not info.get("vision"):
        sys.exit(f"{model} 不支持图片输入")
    if info is None:
        print(f"[提示] 不确定 {model} 能否看图，先小量试一下。", file=sys.stderr)


def warn_estimated(model):
    fp = CFG["defapi"]["fallback_price_usd_per_m"]
    print(f"[提示] defapi 没返回 {model} 的实际扣费，也没配置单价，花费暂按保守价 ${fp['in']}/${fp['out']}（每百万 token）估算，会偏高。"
          "到 defapi 网站查实际单价，填进 config.json → defapi.models。", file=sys.stderr)


def redact_images(msgs):
    out = []
    for m in msgs:
        c = m["content"]
        if isinstance(c, list):
            c = [{"type": "image_url", "image_url": "<图片已省略>"} if b.get("type") == "image_url" else b for b in c]
        out.append({**m, "content": c})
    return out


def resolve_system(args):
    system = text_or_file(args.system, args.system_file)
    return CFG["deepseek"].get("default_system") if system is None else system


def jev_channel():
    """默认走 TypeSafe 官方（便宜一半）；官方 key 没设时自动退回 defapi，保证 Jev 随时可用。"""
    explicit = read_secret(JEV["channel_env"])
    name = explicit or JEV["channel"]
    ch = JEV["channels"][name]
    fb = JEV.get("fallback_channel")
    if not explicit and fb and not read_secret(ch.get("key_env", JEV["key_env"])):
        fch = JEV["channels"][fb]
        if read_secret(fch.get("key_env", JEV["key_env"])):
            return fb, fch
    return name, ch


def get_jev_key(required=True):
    _, channel = jev_channel()
    name = channel.get("key_env", JEV["key_env"])
    key = read_secret(name)
    if required and not key:
        sys.exit(f"Set {name} in your shell before running Jev.")
    return key


def jev_http(payload, key=None, images=False):
    _, ch = jev_channel()
    headers = {"Authorization": f"Bearer {key or get_jev_key()}"}
    if images:
        headers["TypeSafe-Danger-Zone-Images"] = "true"  # 图片功能官方标为实验性
    url = ch["base_url"].rstrip("/") + ch.get("path", "/v1/systemone")
    return http_json(url, payload, headers, JEV["timeout_s"], JEV["retries"], HINTS["jev"])


def actual_usd(resp):
    """defapi 会在返回里直接给出实际扣费（美元）：顶层 consumed，Jev 还有 usage.cost。"""
    for v in (resp.get("consumed"), (resp.get("usage") or {}).get("cost")):
        try:
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            pass
    return None


def jev_cost(resp):
    usd = actual_usd(resp)
    if usd is None:
        _, ch = jev_channel()
        usd = (resp.get("usage") or {}).get("input_tokens", 0) * ch.get("price_usd_per_m_in", JEV["price_usd_per_m_in"]) / 1e6
    return round(usd * JEV["usd_cny"], 8)


def jev_asset(path):
    p = Path(path)
    fmt = p.suffix.lower().lstrip(".")
    if fmt not in ("png", "jpg", "jpeg"):
        raise ValueError(f"Jev 只支持 png / jpg 图片：{p.name}")
    return {"type": "image", "format": fmt, "data_base64": base64.b64encode(p.read_bytes()).decode()}


def jev_ask(state, questions, images=(), key=None):
    _, ch = jev_channel()
    payload = {"model": ch["model"], "state": state, "questions": questions}
    if images:
        refs = {f"img-{uuid.uuid4().hex[:10]}": jev_asset(p) for p in images}
        payload["assets"] = refs
        base = state if isinstance(state, dict) else ({"text": state} if state else {})
        payload["state"] = {**base, "images": list(refs)}  # state 里写占位符，服务端替换成图片
    t0 = time.time()
    resp = jev_http(payload, key, images=bool(images))
    usage = resp.get("usage") or {}
    return {"answers": resp.get("answers") or {}, "model": resp.get("model"), "usage": usage,
            "latency_s": round(time.time() - t0, 3), "cost_cny": jev_cost(resp)}


def answer_confidence(a):
    if a.get("type") == "noul":
        return abs(a.get("noul", 0.5) - 0.5) * 2  # 0.5 最不确定，0 或 1 最确定
    return a.get("confidence")


def summarize_answer(a):
    t = a.get("type")
    if t == "noul":
        return f"是的概率 {a['noul']:.2f}"
    if t == "choice":
        top = sorted((a.get("probabilities") or {}).items(), key=lambda kv: -kv[1])[:3]
        return f"{a['choice']}（置信 {a['confidence']:.2f}；" + "，".join(f"{k} {v:.2f}" for k, v in top) + "）"
    if t == "score":
        return f"{a['score']:.2f} 分（置信 {a['confidence']:.2f}）"
    if t == "box":
        return f"位置 {a.get('box')}（置信 {a.get('confidence')}）"
    return json.dumps(a, ensure_ascii=False)


def load_questions(args, recipe=None):
    qs = {}
    if getattr(args, "questions_file", None):
        qs.update(json.loads(Path(args.questions_file).read_text(encoding="utf-8")))
    if getattr(args, "questions", None):
        qs.update(json.loads(args.questions))
    for i, text in enumerate(getattr(args, "yes", None) or [], 1):
        qs[f"yes{i}"] = {"type": "noul", "instructions": text}
    if not qs and recipe and recipe.get("questions"):
        qs = recipe["questions"]
    if not qs:
        sys.exit("需要问题：--questions-file q.json、--questions '{…}'、--yes \"问题\" 或 --recipe")
    return qs


def load_recipe(name):
    if not name:
        return None
    path = RECIPES / f"{name}.json"
    if not path.exists():
        sys.exit(f"没有这个配方：{name}（python hub.py recipe list 查看）")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_recipe(args):
    """用配方补全命令里没写的参数；命令里显式写的优先。"""
    rec = load_recipe(getattr(args, "recipe", None))
    if rec:
        for f in RECIPE_FIELDS:
            if getattr(args, f, None) in (None, False) and rec.get(f) not in (None, False):
                setattr(args, f, rec[f])
    if getattr(args, "via", None) is None:
        args.via = (rec or {}).get("executor") if (rec or {}).get("executor") in CHAT_PROVIDERS else "deepseek"
    return rec


def load_items(src):
    p = Path(src)
    items = []
    if p.is_dir():
        for f in sorted(p.iterdir()):
            ext = f.suffix.lower()
            if ext in IMG_EXT:
                items.append({"id": f.name, "images": [str(f)]})
            elif ext in TXT_EXT:
                items.append({"id": f.name, "text": f.read_text(encoding="utf-8", errors="replace")})
        return items
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            it = json.loads(line)
            it.setdefault("id", str(i))
            if "image" in it:
                it["images"] = [it.pop("image")]
            items.append(it)
    return items


def judge_items(task_name, with_images):
    """把一个批量任务的（指令、输入、输出）打包成待检查的项，用来自动质检。"""
    t = task_path(task_name)
    cfg = json.loads((t / "run_config.json").read_text(encoding="utf-8"))
    src = {}
    if cfg.get("input") and Path(cfg["input"]).exists():
        src = {str(it["id"]): it for it in load_items(cfg["input"])}
    items = []
    for r in read_jsonl(t / "results.jsonl"):
        if r.get("status") != "ok":
            continue
        orig = src.get(str(r["id"]), {})
        state = {"指令": cfg.get("prompt"), "输入": item_text(orig) or None, "输出": r.get("output")}
        items.append({"id": r["id"], "state": {k: v for k, v in state.items() if v},
                      "images": orig.get("images", []) if with_images else []})
    return items


def item_text(item):
    if item.get("text"):
        return item["text"]
    return json.dumps(item["state"], ensure_ascii=False) if item.get("state") is not None else ""


def render_prompt(template, item):
    text = item_text(item)
    if "{text}" in template or "{id}" in template:
        return template.replace("{text}", text).replace("{id}", str(item["id"]))
    return f"{template}\n\n{text}" if text else template


def compact_results(path, order):
    """重跑会追加新记录；每个 id 只保留最后一条，并按输入顺序排列。"""
    latest = {str(r.get("id")): r for r in read_jsonl(path)}
    rank = {i: n for n, i in enumerate(order)}
    rows = sorted(latest.values(), key=lambda r: rank.get(str(r.get("id")), len(rank)))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def results_stats(path):
    """整个任务（所有运行累计）的当前状态：配方和复盘用这个，不受某次补跑影响。"""
    rows = read_jsonl(path)
    ok = [r for r in rows if r.get("status") == "ok"]
    tin = sum(r.get("in_tokens", 0) for r in rows)
    return {
        "items": len(rows), "ok": len(ok),
        "statuses": dict(collections.Counter(r.get("status") for r in rows)),
        "cost_per_ok_item": round(sum(r.get("cost_cny") or 0 for r in ok) / len(ok), 8) if ok else None,
        "cache_hit_rate": round(sum(r.get("cache_hit", 0) for r in rows) / tin, 3) if tin else None,
        "out_tokens_per_item": round(sum(r.get("out_tokens", 0) for r in ok) / len(ok)) if ok else None,
    }


def run_pool(fn, todo, workers, state):
    pool = cf.ThreadPoolExecutor(max_workers=workers)
    try:
        list(pool.map(fn, todo))
    except KeyboardInterrupt:
        state["stop"] = True
        print("已中断，等待进行中的请求结束……", file=sys.stderr)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def pending(items, out, limit):
    done = {str(r["id"]) for r in read_jsonl(out) if r.get("status") == "ok"}
    todo = [it for it in items if str(it["id"]) not in done]
    return done, (todo[:limit] if limit else todo)


def print_recipes():
    files = sorted(RECIPES.glob("*.json")) if RECIPES.exists() else []
    if not files:
        print("- （还没有）")
        return
    uses = collections.Counter(r.get("recipe") for r in read_log() if r.get("recipe"))
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        per = (rec.get("stats") or {}).get("cost_per_ok_item")
        who = f"{rec.get('executor')}/{rec.get('model')}"
        print(f"- {f.stem}：{who}，每项约 ¥{money(per)}，质量：{rec.get('quality', '未评估')}，"
              f"已用 {uses.get(f.stem, 0)} 次。{rec.get('notes') or ''}")


def cmd_ds(args):
    recipe = apply_recipe(args)
    check_budget(args.force)
    via = args.via
    model = resolve_model(args.model, via)
    prompt = text_or_file(args.prompt, args.prompt_file)
    if recipe and recipe.get("prompt"):
        prompt = render_prompt(recipe["prompt"], {"id": "ds", "text": prompt or ""})
    if not prompt:
        sys.exit("需要提问内容：直接写在命令里，或用 --prompt-file")
    system = resolve_system(args)
    if args.json:
        prompt = ensure_json_hint(prompt, system)
    check_vision(via, model, args.image)
    model_notes(via, model)
    msgs = build_messages(prompt, system, args.file or [], args.image or [], args.detail)
    call_id = uuid.uuid4().hex[:8]
    base = dict(executor=via, model=model, mode="ds", task=args.task, caller=args.caller, purpose=args.purpose,
                recipe=args.recipe, id=call_id)
    try:
        r = chat(msgs, model, args.think, args.json, args.max_tokens, args.temperature, via=via)
    except RuntimeError as e:
        log_call(**base, status="error", error=str(e)[:300])
        sys.exit(f"调用失败：{e}")
    io_path = IO_DIR / now().date().isoformat() / f"{call_id}.json"
    write_json(io_path, {"id": call_id, "via": via, "model": model, "images": args.image, "files": args.file,
                         "request": redact_images(msgs), "response": r})
    out_path = None
    if args.out:
        out_path = Path(args.out)
    elif args.task:
        out_path = task_dir(args.task) / f"{via}_{call_id}.md"
    if out_path:
        out_path.write_text(r["content"], encoding="utf-8")
    u = usage_fields(r["usage"])
    log_call(**base, status="ok" if r["finish"] != "length" else "truncated", peak=r["peak"],
             served_model=r["served_model"], model_mismatch=r["mismatch"] or None,
             price_estimated=r["estimated"] or None,
             latency_s=r["latency_s"], cost_cny=r["cost_cny"], **u,
             io=str(io_path.relative_to(HUB)), out=str(out_path) if out_path else None)
    print(r["content"])
    timing = ("高峰" if r["peak"] else "错峰") if via == "deepseek" else f"返回模型 {r['served_model']}"
    print(f"\n[{via} {model} | 入 {u['in_tokens']}（缓存 {u['cache_hit']}）出 {u['out_tokens']} token | "
          f"¥{money(r['cost_cny'])}{'（估）' if r['estimated'] else ''} | {r['latency_s']}s | {timing} | id {call_id}]",
          file=sys.stderr)
    if r["estimated"]:
        warn_estimated(model)
    if r["mismatch"]:
        print(f"[警告] 请求的是 {model}，defapi 实际返回的是 {r['served_model']}", file=sys.stderr)
    if r["finish"] == "length":
        print("[警告] 输出被 max_tokens 截断", file=sys.stderr)


def cmd_batch(args):
    recipe = apply_recipe(args)
    via = args.via
    key = get_key(via=via)
    spent0 = check_budget(args.force)
    model = resolve_model(args.model, via)
    template_raw = text_or_file(args.prompt, args.prompt_file) or (recipe or {}).get("prompt")
    if not template_raw:
        sys.exit("需要 --prompt、--prompt-file 或 --recipe")
    system = resolve_system(args)
    template = ensure_json_hint(template_raw, system) if args.json else template_raw
    if args.judge:
        items = judge_items(args.judge, args.with_images)
    elif args.input:
        items = load_items(args.input)
    else:
        sys.exit("需要 --input 或 --judge")
    check_vision(via, model, any(it.get("images") for it in items))
    model_notes(via, model)
    task = task_dir(args.task or f"{now():%Y%m%d-%H%M}_batch")
    out = Path(args.out) if args.out else task / "results.jsonl"
    done, todo = pending(items, out, args.limit)
    cap = batch_cap(args, spent0)
    offpeak = args.wait_offpeak and via == "deepseek"  # 只有 DeepSeek 有错峰价
    run_cfg_path = task / "run_config.json"
    run_cfg = {"executor": via, "via": via, "model": model, "system": system, "prompt": template_raw,
               "json": args.json, "think": args.think, "detail": args.detail, "max_tokens": args.max_tokens,
               "temperature": args.temperature, "input": str(args.input) if args.input else None,
               "judge": args.judge, "recipe": args.recipe}
    write_json(run_cfg_path, run_cfg)
    timing = f"；{'高峰时段（价格翻倍）' if is_peak() else '错峰时段（半价）'}" if via == "deepseek" else ""
    print(f"共 {len(items)} 项，已完成 {len(done)}，本次处理 {len(todo)}；{via}/{model}；本次预算 ¥{cap:.2f}{timing}",
          file=sys.stderr)
    if via == "deepseek" and is_peak() and len(todo) > 100 and not offpeak:
        print("提示：大批量加 --wait-offpeak 可自动等到错峰再跑，费用减半。", file=sys.stderr)
    state = {"cost": 0.0, "peak_cost": 0.0, "ok": 0, "fail": 0, "in": 0, "hit": 0, "out": 0, "reason": 0,
             "mismatch": 0, "estimated": 0, "served": collections.Counter(), "statuses": collections.Counter(),
             "stop": False}
    if offpeak:
        wait_until_offpeak(state)
    t0 = time.time()

    def work(it):
        if state["stop"]:
            return
        if offpeak:
            wait_until_offpeak(state)  # 跑着跑着进入高峰也会暂停
            if state["stop"]:
                return
        rec = {"id": it["id"]}
        try:
            msgs = build_messages(render_prompt(template, it), system, (), it.get("images", []), args.detail)
            r = chat(msgs, model, args.think, args.json, args.max_tokens, args.temperature, key, via)
            rec.update(status="ok", output=r["content"], cost_cny=r["cost_cny"], latency_s=r["latency_s"],
                       peak=r["peak"], served_model=r["served_model"], **usage_fields(r["usage"]))
            if r["mismatch"]:
                rec["model_mismatch"] = True
            if r["estimated"]:
                rec["cost_estimated"] = True
            if args.json:
                try:
                    rec["output"] = json.loads(r["content"])
                except ValueError:
                    rec["status"] = "bad_json"
            if r["finish"] == "length":
                rec["status"] = "truncated"
        except Exception as e:  # 单项失败不影响整批，重跑时会自动重试
            rec.update(status="error", error=str(e)[:300])
        with _lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            c = rec.get("cost_cny") or 0
            state["cost"] += c
            state["peak_cost"] += c if rec.get("peak") else 0
            state["in"] += rec.get("in_tokens", 0)
            state["hit"] += rec.get("cache_hit", 0)
            state["out"] += rec.get("out_tokens", 0)
            state["reason"] += rec.get("reasoning_tokens", 0)
            state["mismatch"] += bool(rec.get("model_mismatch"))
            state["estimated"] += bool(rec.get("cost_estimated"))
            if rec.get("served_model"):
                state["served"][rec["served_model"]] += 1
            state["statuses"][rec["status"]] += 1
            state["ok" if rec["status"] == "ok" else "fail"] += 1
            n = state["ok"] + state["fail"]
            if n == 10 and len(todo) > 20:
                print(f"按前 10 项估算，本次总花费约 ¥{state['cost'] / n * len(todo):.2f}", file=sys.stderr)
            if n % 50 == 0 or n == len(todo):
                print(f"进度 {n}/{len(todo)}，成功 {state['ok']}，失败 {state['fail']}，已花 ¥{state['cost']:.4f}", file=sys.stderr)
            if state["cost"] >= cap and not state["stop"]:
                state["stop"] = True
                print(f"已达本次预算 ¥{cap:.2f}，停止派发。已完成的结果已保存，加大 --budget 后重跑同一命令即可续跑。", file=sys.stderr)

    run_pool(work, todo, args.workers or CFG["batch"]["workers"], state)
    compact_results(out, [str(it["id"]) for it in items])
    n = state["ok"] + state["fail"]
    run_cfg["last_run"] = {"ts": now().isoformat(timespec="seconds"), "items": n, "ok": state["ok"],
                           "failed": state["fail"], "statuses": dict(state["statuses"]),
                           "served_models": dict(state["served"]), "cost_cny": round(state["cost"], 6)}
    run_cfg["stats"] = stats = results_stats(out)
    write_json(run_cfg_path, run_cfg)
    elapsed = round(time.time() - t0, 1)
    log_call(executor=via, model=model, mode="batch", task=task.name, caller=args.caller, purpose=args.purpose,
             recipe=args.recipe, judge=args.judge, status="stopped" if state["stop"] else "ok", items=n, ok=state["ok"],
             failed=state["fail"], statuses=dict(state["statuses"]), in_tokens=state["in"], cache_hit=state["hit"],
             out_tokens=state["out"], reasoning_tokens=state["reason"], cost_cny=round(state["cost"], 6),
             peak_cost_cny=round(state["peak_cost"], 6), model_mismatch=state["mismatch"] or None,
             served_models=dict(state["served"]) or None, price_estimated=bool(state["estimated"]) or None,
             latency_s=elapsed, out=str(out))
    print(f"本次：成功 {state['ok']}，失败 {state['fail']}，花费 ¥{state['cost']:.4f}"
          f"{'（估）' if state['estimated'] else ''}，用时 {elapsed}s。结果：{out}")
    if state["estimated"]:
        warn_estimated(model)
    if state["mismatch"]:
        print(f"[警告] {state['mismatch']} 项实际返回的模型和请求的不一样：{dict(state['served'])}", file=sys.stderr)
    if stats["ok"]:
        cache = f"缓存命中 {stats['cache_hit_rate']:.0%}，" if stats["cache_hit_rate"] is not None else ""
        print(f"任务累计：{stats['ok']}/{stats['items']} 项成功，每项 ¥{money(stats['cost_per_ok_item'])}，"
              f"{cache}每项输出 {stats['out_tokens_per_item']} token。验收：python hub.py sample --task {task.name}")


def cmd_jev(args):
    recipe = load_recipe(args.recipe)
    check_budget(args.force)
    questions = load_questions(args, recipe)
    state = text_or_file(args.state, args.state_file)
    if state and args.state_json:
        state = json.loads(state)
    if not state and not args.image:
        sys.exit("需要 --state、--state-file 或 --image")
    call_id = uuid.uuid4().hex[:8]
    base = dict(executor="jev", mode="jev", task=args.task, caller=args.caller, purpose=args.purpose,
                recipe=args.recipe, id=call_id)
    try:
        r = jev_ask(state or "", questions, args.image or [])
    except (RuntimeError, ValueError) as e:
        log_call(**base, status="error", error=str(e)[:300])
        sys.exit(f"调用失败：{e}")
    io_path = IO_DIR / now().date().isoformat() / f"jev_{call_id}.json"
    write_json(io_path, {"id": call_id, "state": state, "images": args.image, "questions": questions, "response": r})
    low = [n for n, a in r["answers"].items() if (answer_confidence(a) or 1) < JEV["low_confidence"]]
    log_call(**base, model=r["model"], status="ok", latency_s=r["latency_s"], cost_cny=r["cost_cny"],
             in_tokens=r["usage"].get("input_tokens", 0), questions=len(questions), low_confidence=len(low),
             io=str(io_path.relative_to(HUB)))
    for name, a in r["answers"].items():
        print(f"{name}：{summarize_answer(a)}{'  ← 低置信度' if name in low else ''}")
    if args.raw:
        print(json.dumps(r["answers"], ensure_ascii=False, indent=2))
    print(f"\n[jev {r['model']} | 入 {r['usage'].get('input_tokens', 0)} token | ¥{money(r['cost_cny'])} | "
          f"{r['latency_s']}s | id {call_id}]", file=sys.stderr)


def cmd_jbatch(args):
    recipe = load_recipe(args.recipe)
    key = get_jev_key()
    spent0 = check_budget(args.force)
    questions = load_questions(args, recipe)
    if args.judge:
        items = judge_items(args.judge, args.with_images)
    elif args.input:
        items = load_items(args.input)
    else:
        sys.exit("需要 --input 或 --judge")
    task = task_dir(args.task or f"{now():%Y%m%d-%H%M}_jev")
    out = Path(args.out) if args.out else task / "jev_results.jsonl"
    done, todo = pending(items, out, args.limit)
    cap = batch_cap(args, spent0)
    run_cfg_path = task / "run_config.json"
    run_cfg = {"executor": "jev", "model": jev_channel()[1]["model"], "questions": questions,
               "input": str(args.input) if args.input else None, "judge": args.judge, "recipe": args.recipe}
    write_json(run_cfg_path, run_cfg)
    print(f"共 {len(items)} 项，已完成 {len(done)}，本次处理 {len(todo)}；{len(questions)} 个问题；本次预算 ¥{cap:.2f}",
          file=sys.stderr)
    state = {"cost": 0.0, "ok": 0, "fail": 0, "in": 0, "low": 0, "stop": False}
    t0 = time.time()

    def work(it):
        if state["stop"]:
            return
        rec = {"id": it["id"]}
        try:
            st = it.get("state") if it.get("state") is not None else it.get("text", "")
            r = jev_ask(st, questions, it.get("images", []), key)
            rec.update(status="ok", answers=r["answers"], cost_cny=r["cost_cny"], latency_s=r["latency_s"],
                       in_tokens=r["usage"].get("input_tokens", 0))
            low = [n for n, a in r["answers"].items() if (answer_confidence(a) or 1) < JEV["low_confidence"]]
            if low:
                rec["low_confidence"] = low
        except Exception as e:  # 单项失败不影响整批，重跑时会自动重试
            rec.update(status="error", error=str(e)[:300])
        with _lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["cost"] += rec.get("cost_cny") or 0
            state["in"] += rec.get("in_tokens", 0)
            state["low"] += bool(rec.get("low_confidence"))
            state["ok" if rec["status"] == "ok" else "fail"] += 1
            n = state["ok"] + state["fail"]
            if n % 200 == 0 or n == len(todo):
                print(f"进度 {n}/{len(todo)}，成功 {state['ok']}，失败 {state['fail']}，已花 ¥{money(state['cost'])}", file=sys.stderr)
            if state["cost"] >= cap and not state["stop"]:
                state["stop"] = True
                print(f"已达本次预算 ¥{cap:.2f}，停止派发。", file=sys.stderr)

    run_pool(work, todo, args.workers or JEV["workers"], state)
    compact_results(out, [str(it["id"]) for it in items])
    rows = read_jsonl(out)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    # 低置信度的项另存成 batch 能直接读的格式：可以交给 DeepSeek 精读（级联），或交给人复核
    by_id = {str(it["id"]): it for it in items}
    low_rows = [r for r in ok_rows if r.get("low_confidence")]
    with (task / "low_confidence.jsonl").open("w", encoding="utf-8") as f:
        for r in low_rows:
            it = by_id.get(str(r["id"]), {"id": r["id"]})
            line = {"id": it["id"]}
            if it.get("images"):
                line["image"] = it["images"][0]
            if it.get("text"):
                line["text"] = it["text"]
            if it.get("state") is not None:
                line["state"] = it["state"]
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    stats = results_stats(out)
    stats["low_confidence"] = len(low_rows)
    run_cfg["stats"] = stats
    write_json(run_cfg_path, run_cfg)
    elapsed = round(time.time() - t0, 1)
    log_call(executor="jev", model=run_cfg["model"], mode="jbatch", task=task.name, caller=args.caller,
             purpose=args.purpose, recipe=args.recipe, status="stopped" if state["stop"] else "ok",
             items=state["ok"] + state["fail"], ok=state["ok"], failed=state["fail"], low_confidence=state["low"],
             in_tokens=state["in"], cost_cny=round(state["cost"], 8), latency_s=elapsed, judge=args.judge, out=str(out))
    print(f"本次：成功 {state['ok']}，失败 {state['fail']}，花费 ¥{money(state['cost'])}，用时 {elapsed}s。结果：{out}")
    for name, q in questions.items():
        answers = [r["answers"][name] for r in ok_rows if name in r.get("answers", {})]
        if not answers:
            continue
        if q.get("type") == "choice":
            dist = collections.Counter(a["choice"] for a in answers)
            desc = "，".join(f"{k} {v}" for k, v in dist.most_common())
        elif q.get("type") == "noul":
            desc = f"判为「是」{sum(a['noul'] >= 0.5 for a in answers)}/{len(answers)}"
        elif q.get("type") == "score":
            desc = f"平均 {sum(a['score'] for a in answers) / len(answers):.2f} 分"
        else:
            desc = f"{len(answers)} 个回答"
        print(f"- {name}：{desc}")
    if low_rows:
        ids = "、".join(str(r["id"]) for r in low_rows[:15])
        print(f"低置信度 {len(low_rows)} 项（建议复核，或交给 DeepSeek 精读：--input {task / 'low_confidence.jsonl'}）：{ids}"
              + ("……" if len(low_rows) > 15 else ""))


def norm_label(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v).strip().lower()
    return {"是": "true", "否": "false", "yes": "true", "no": "false"}.get(s, s)


def task_results(name):
    t = task_path(name)
    path = t / "jev_results.jsonl" if (t / "jev_results.jsonl").exists() else t / "results.jsonl"
    cfg = json.loads((t / "run_config.json").read_text(encoding="utf-8")) if (t / "run_config.json").exists() else {}
    return t, path, cfg


def predict(r, field):
    if r.get("status") != "ok":
        return None
    if "answers" in r:
        a = r["answers"].get(field) or {}
        kind = a.get("type")
        value = a.get(kind)
        if kind == "choice":
            return value
        if isinstance(value, (int, float)):
            if kind == "noul":
                return value >= 0.5
            if kind == "score":
                return round(value)
        return None
    out = r.get("output")
    return out.get(field) if isinstance(out, dict) else out


def cmd_eval(args):
    reference = read_jsonl(Path(args.gold))
    gold = {str(r["id"]): r[args.gold_field] for r in reference}
    if not gold or len(gold) != len(reference):
        sys.exit("Reference labels require at least one record and unique IDs.")
    t, path, cfg = task_results(args.task)
    if args.file:
        path = Path(args.file)
    if not path.is_file():
        sys.exit(f"Result file does not exist: {path}")
    attempts = [r for r in read_jsonl(path) if str(r.get("id")) in gold]
    result_rows = {str(r["id"]):r for r in attempts}
    rows = [result_rows.get(i, {"id": i, "status":"missing"}) for i in gold]
    priced = [r.get("cost_cny") for r in attempts]
    who = f"{cfg.get('executor')}/{cfg.get('model')}"
    # 级联：后备任务只重做了被挑出来的项（pick 或 low_confidence.jsonl），这些项以后备的答案为准
    fb, used = {}, set()
    if args.fallback:
        _, fpath, fcfg = task_results(args.fallback)
        if not fpath.is_file():
            sys.exit(f"Review result file does not exist: {fpath}")
        reviews = [r for r in read_jsonl(fpath) if str(r.get("id")) in gold]
        fb = {str(r["id"]):r for r in reviews if r.get("status") == "ok"}
        priced.extend(r.get("cost_cny") for r in reviews)
        who = f"{who} → 挑出的项交 {fcfg.get('executor')}/{fcfg.get('model')}"

    def final(r):
        i = str(r["id"])
        if i in fb:
            used.add(i)
            return predict(fb[i], args.fallback_field or args.field)
        return predict(r, args.field)

    pairs = [(str(r["id"]), norm_label(gold[str(r["id"])]), final(r)) for r in rows]
    wrong = [(i, g, p) for i, g, p in pairs if p is None or norm_label(p) != g]
    acc = 1 - len(wrong) / len(pairs)
    for extra in args.plus or []:  # 组合方案的其他步骤（如质检）也要算进每项成本，否则组合会显得太便宜
        _, xpath, _ = task_results(extra)
        if not xpath.is_file():
            sys.exit(f"Additional result file does not exist: {xpath}")
        priced.extend(r.get("cost_cny") for r in read_jsonl(xpath) if str(r.get("id")) in gold)
    per = sum(priced) / len(pairs) if priced and all(v is not None for v in priced) else None
    rec = {"ts": now().isoformat(timespec="seconds"), "task": t.name, "family": args.family, "field": args.field,
           "executor": (cfg.get("executor") or "?") + (f"+{args.fallback}" if args.fallback else ""),
           "model": cfg.get("model"), "fallback": args.fallback, "fallback_used": len(used) if args.fallback else None,
           "plus": args.plus,
           "n": len(pairs), "accuracy": round(acc, 4), "cost_per_item": per, "note": args.note}
    EVALS.parent.mkdir(parents=True, exist_ok=True)
    with EVALS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    extra = f"（其中 {len(used)} 项用了后备答案）" if args.fallback else ""
    print(f"{t.name}（{who}）：{len(pairs)} 项，准确率 {acc:.1%}{extra}，每项 ¥{money(per)}")
    for i, g, p in wrong[:10]:
        print(f"  ✗ {i}：标准 {g}，预测 {p}")
    print("Saved evaluation to", EVALS)


def source_input(task_name):
    """找到一个任务最初的输入：质检任务要回到被检查的那个任务的输入。"""
    _, _, cfg = task_results(task_name)
    if cfg.get("judge"):
        return source_input(cfg["judge"])
    return cfg.get("input")


def cmd_pick(args):
    """从结果里挑出某些项，导出成 batch 能直接读的输入文件。用于级联：不合格 / 拿不准的交给更强的模型重做。"""
    t, path, _ = task_results(args.task)
    rows = read_jsonl(path)
    field, _, want = args.where.partition("=")
    if not want:
        sys.exit('--where 的格式是 字段=值，例如 pass=false')
    ids = {str(r["id"]) for r in rows
           if (r.get("status") != "ok" and args.include_failed) or
           (r.get("status") == "ok" and norm_label(predict(r, field)) == norm_label(want))}
    src = source_input(args.task)
    if not src or not Path(src).exists():
        sys.exit(f"找不到 {args.task} 的原始输入，无法导出")
    out = Path(args.out) if args.out else t / "picked.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for it in load_items(src):
            if str(it["id"]) not in ids:
                continue
            line = {"id": it["id"]}
            if it.get("images"):
                line["image"] = it["images"][0]
            for k in ("text", "state"):
                if it.get(k) is not None:
                    line[k] = it[k]
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            n += 1
    print(f"挑出 {n} 项（{args.where}{'，含失败项' if args.include_failed else ''}）→ {out}")
    print(f"交给更强的模型重做：python hub.py batch --via defapi --input {out} --task <新任务> …")


def cmd_sample(args):
    path = Path(args.file) if args.file else task_path(args.task) / "results.jsonl"
    if not args.file and not path.exists():
        path = task_path(args.task) / "jev_results.jsonl"
    rows = read_jsonl(path)
    if not rows:
        sys.exit(f"没有结果：{path}")
    print(f"{path}：共 {len(rows)} 项，状态 " + "，".join(f"{k} {v}" for k, v in collections.Counter(r.get('status') for r in rows).items()))
    if args.status:
        rows = [r for r in rows if r.get("status") == args.status]
    if args.low:
        rows = [r for r in rows if r.get("low_confidence")]
    for r in random.Random(args.seed).sample(rows, min(args.n, len(rows))):
        if "answers" in r:
            body = "；".join(f"{k} {summarize_answer(a)}" for k, a in r["answers"].items())
        elif "output" in r:
            body = json.dumps(r.get("output"), ensure_ascii=False)
        else:
            body = r.get("error", "")
        print(f"- {r.get('id')} [{r.get('status')}] {body[: args.width]}")


def cmd_recipe(args):
    if args.action == "list":
        print_recipes()
        return
    if not args.name:
        sys.exit("需要配方名")
    path = RECIPES / f"{args.name}.json"
    if args.action == "show":
        if not path.exists():
            sys.exit(f"没有这个配方：{args.name}")
        print(path.read_text(encoding="utf-8"))
        return
    if not args.task:
        sys.exit("保存配方需要 --task（从这个任务的设置和实测结果生成）")
    cfg_path = task_path(args.task) / "run_config.json"
    if not cfg_path.exists():
        sys.exit(f"{cfg_path} 不存在：只能从跑过 batch / jbatch 的任务保存配方")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("executor") == "jev":
        rec = {"executor": "jev", "model": cfg.get("model"), "questions": cfg.get("questions")}
    else:
        rec = {"executor": cfg.get("executor"), **{f: cfg.get(f) for f in RECIPE_FIELDS + ("prompt",)}}
    quality = args.quality
    if not quality:  # 没写就用这个任务最近一次擂台成绩
        ev = [e for e in read_jsonl(EVALS) if e.get("task") == Path(args.task).name]
        quality = f"准确率 {ev[-1]['accuracy']:.1%}（{ev[-1]['n']} 项标准答案）" if ev else "未评估"
    rec.update(saved=now().date().isoformat(), from_task=Path(args.task).name, stats=cfg.get("stats"),
               quality=quality, notes=args.notes)
    write_json(path, rec)
    print(f"已保存配方 {args.name}（质量：{quality}）→ {path}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--caller", default=os.environ.get("HUB_CALLER", "user"), help="谁在调度（写进日志）")
    common.add_argument("--purpose", help="这次调用为了什么（写进日志，复盘和优化建议用）")
    common.add_argument("--task", help="任务文件夹名（tasks/ 下），结果会存进去")
    common.add_argument("--force", action="store_true", help="越过每日预算上限（先问用户）")

    chatopt = argparse.ArgumentParser(add_help=False)
    chatopt.add_argument("--via", choices=CHAT_PROVIDERS, help="走哪家：deepseek（默认，最便宜）或 defapi（Claude 等，更贵）")
    chatopt.add_argument("--recipe", help="用已保存的配方（命令里显式写的参数优先）")
    chatopt.add_argument("--model", help="deepseek：flash（默认）/ pro；defapi：sonnet（默认）、gemini-flash、gpt-mini、"
                                          "gpt-terra、gemini-pro、opus、gpt-sol 或完整模型名")
    chatopt.add_argument("--system", help="系统提示词；传空字符串表示不用默认系统提示词")
    chatopt.add_argument("--system-file")
    chatopt.add_argument("--json", action="store_true", help="要求输出合法 JSON")
    chatopt.add_argument("--think", choices=["low", "high", "max"], help="开启思考 / 推理（默认关，开了更贵）")
    chatopt.add_argument("--max-tokens", type=int)
    chatopt.add_argument("--temperature", type=float)
    chatopt.add_argument("--detail", choices=["low", "high", "original", "auto"], help="图片精度，low 更省")

    qopt = argparse.ArgumentParser(add_help=False)
    qopt.add_argument("--recipe", help="用已保存的 Jev 配方（里面有问题定义）")
    qopt.add_argument("--questions-file", help="问题定义 JSON：{名字: {type: noul|choice|score, instructions, criteria}}")
    qopt.add_argument("--questions", help="直接写问题定义 JSON")
    qopt.add_argument("--yes", action="append", help="快捷是/否题（可多次）")

    ap = argparse.ArgumentParser(description="AgentHub | Batch model runs, selective review and evaluation",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ds", parents=[common, chatopt], help="单次调用（默认 DeepSeek，--via defapi 走 defapi）")
    p.add_argument("prompt", nargs="?", help="提问内容；写 - 表示从标准输入读")
    p.add_argument("--prompt-file")
    p.add_argument("--file", action="append", help="附带文本文件（可多次）")
    p.add_argument("--image", action="append", help="附带图片路径或 URL（可多次）")
    p.add_argument("--out", help="把回答另存到这个文件")

    p = sub.add_parser("batch", parents=[common, chatopt], help="批量调用（默认 DeepSeek，--via defapi 走 defapi）")
    p.add_argument("--input", help="图片/文本文件夹，或 .jsonl（每行 id、text、image、state）")
    p.add_argument("--judge", help="质检：检查这个任务的结果（把指令、输入、输出交给模型判断）")
    p.add_argument("--with-images", action="store_true", help="质检时把原图也带上")
    p.add_argument("--prompt", help="对每一项的指令；可用 {text} {id} 占位（放末尾更容易命中缓存）")
    p.add_argument("--prompt-file")
    p.add_argument("--out", help="结果文件，默认 tasks/<task>/results.jsonl")
    p.add_argument("--limit", type=int, help="只处理前 N 项（试点用）")
    p.add_argument("--workers", type=int, help="并发数，默认见 config.json")
    p.add_argument("--budget", type=float, help="本次花费上限（元），默认见 config.json")
    p.add_argument("--wait-offpeak", action="store_true", help="DeepSeek 高峰时段自动等到错峰再跑（半价）")

    p = sub.add_parser("jev", parents=[common, qopt], help="单次调用 Jev 判断（默认 TypeSafe 官方）")
    p.add_argument("--state", help="要判断的内容；写 - 表示从标准输入读")
    p.add_argument("--state-file")
    p.add_argument("--state-json", action="store_true", help="把 state 当 JSON 解析")
    p.add_argument("--image", action="append", help="附带 png/jpg 图片（实验性，可多次）")
    p.add_argument("--raw", action="store_true", help="同时打印完整概率")

    p = sub.add_parser("jbatch", parents=[common, qopt], help="批量调用 Jev 判断（默认 TypeSafe 官方）")
    p.add_argument("--input", help="图片/文本文件夹，或 .jsonl（每行 id 加 text、state 或 image）")
    p.add_argument("--judge", help="给这个批量任务的结果做质检（任务名）")
    p.add_argument("--with-images", action="store_true", help="质检时把原图也给 Jev（实验性）")
    p.add_argument("--out", help="结果文件，默认 tasks/<task>/jev_results.jsonl")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--budget", type=float)

    p = sub.add_parser("eval", help="擂台：对标准答案算准确率")
    p.add_argument("--task", required=True)
    p.add_argument("--gold", required=True, help="标准答案 .jsonl（每行 id 和 label）")
    p.add_argument("--field", required=True, help="比较哪个字段：输出 JSON 的字段名（Jev 是问题名）")
    p.add_argument("--gold-field", default="label")
    p.add_argument("--family", help="任务类别名（同类任务的擂台放一起比），如 街景老城新区")
    p.add_argument("--file", help="直接指定结果文件")
    p.add_argument("--fallback", help="级联：这个后备任务里有的项（被挑出来重做的），以它的答案为准")
    p.add_argument("--fallback-field", help="后备任务里对应的字段名（默认同 --field）")
    p.add_argument("--plus", action="append", help="组合方案里其他步骤的任务（如质检），花费一并计入每项成本（可多次）")
    p.add_argument("--note")

    p = sub.add_parser("pick", help="挑出某些项导出成输入文件（级联用）")
    p.add_argument("--task", required=True, help="从哪个任务的结果里挑（通常是质检任务）")
    p.add_argument("--where", required=True, help="字段=值，如 pass=false")
    p.add_argument("--include-failed", action="store_true", help="把执行失败的项也挑出来")
    p.add_argument("--out", help="默认 tasks/<task>/picked.jsonl")

    p = sub.add_parser("sample", help="抽查结果")
    p.add_argument("--task", help="任务文件夹名")
    p.add_argument("--file", help="或直接指定结果文件")
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--status", help="只看某种状态，如 bad_json / error / ok")
    p.add_argument("--low", action="store_true", help="只看 Jev 低置信度的项")
    p.add_argument("--width", type=int, default=300, help="每项最多显示多少字")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("recipe", help="配方：list / show 名 / save 名 --task T")
    p.add_argument("action", choices=["list", "show", "save"])
    p.add_argument("name", nargs="?")
    p.add_argument("--task")
    p.add_argument("--quality", help="实测质量；不写就用这个任务最近一次 eval 的成绩")
    p.add_argument("--notes", help="适用范围、注意事项")

    args = ap.parse_args()
    commands = {'ds': cmd_ds, 'batch': cmd_batch, 'jev': cmd_jev, 'jbatch': cmd_jbatch, 'eval': cmd_eval, 'sample': cmd_sample, 'pick': cmd_pick, 'recipe': cmd_recipe}
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
