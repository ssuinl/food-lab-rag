"""GMS 에서 실제로 쓸 수 있는 임베딩 모델을 최소 요청으로 확인한다.

    python check_gms.py            # 후보 모델을 순서대로 1회씩, 처음 성공하면 멈춤 (최대 3회 요청)
    python check_gms.py --save     # 성공한 모델명을 config.json 의 embedding_model 에 기록
    python check_gms.py --chat     # 채팅 모델도 1회 확인 (짧은 응답)

- 키 값은 출력하지 않는다. 입력은 짧은 문장 1개다.
- 지원되지 않으면 다른 유료 API로 바꾸지 않고 그대로 멈춘다.
- GMS 과금·쿼터 정책은 API로 알 수 없으므로 GMS 안내(https://gms.ssafy.io)에서 직접 확인해야 한다.
"""
import json
import sys
import time

from openai import OpenAI, APIStatusError

from foodlab.settings import PROJECT, gms_key, load_config, safe_error

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def main():
    cfg = load_config()
    key = gms_key()
    if not key:
        raise SystemExit("GMS_KEY 를 찾지 못했습니다 (.env).")
    print(f"GMS_KEY 확인 (길이 {len(key)}) — 값은 출력하지 않습니다")
    raw_client = OpenAI(api_key=key, base_url=cfg["gms_base_url"], max_retries=0, timeout=30)
    found = None
    for model in cfg["embedding_candidates"]:
        t = time.time()
        try:
            resp = raw_client.embeddings.with_raw_response.create(model=model, input=["두부와 달걀로 만드는 반찬"])
            data = resp.parse()
            hdr = {k: v for k, v in resp.headers.items() if "ratelimit" in k.lower() or "quota" in k.lower()}
            dim = len(data.data[0].embedding)
            usage = getattr(data.usage, "total_tokens", None)
            print(f"[성공] {model}  차원 {dim}  사용 토큰 {usage}  응답 {time.time()-t:.1f}s  한도 헤더 {hdr or '없음'}")
            found = {"model": model, "dim": dim}
            break
        except APIStatusError as e:
            print(f"[실패] {model}  HTTP {e.status_code}  {safe_error(e)[:160]}")
        except Exception as e:
            print(f"[실패] {model}  {safe_error(e)[:160]}")
    if "--chat" in sys.argv:
        try:
            r = raw_client.chat.completions.create(model=cfg["chat_model"], messages=[{"role": "user", "content": "한 단어로 답해: 안녕"}])
            print(f"[채팅 성공] {cfg['chat_model']}  사용 토큰 {getattr(r.usage, 'total_tokens', None)}")
        except Exception as e:
            print(f"[채팅 실패] {cfg['chat_model']}  {safe_error(e)[:160]}")
    if not found:
        raise SystemExit("사용 가능한 임베딩 모델을 찾지 못했습니다. 다른 유료 API로 바꾸지 말고 보고해 주세요.")
    if "--save" in sys.argv:
        p = PROJECT / "config.json"
        c = json.loads(p.read_text(encoding="utf-8")); c["embedding_model"] = found["model"]
        p.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"config.json embedding_model = {found['model']} 저장")
    else:
        print(f"다음 단계: python check_gms.py --save  (또는 config.json 의 embedding_model 에 '{found['model']}' 입력)")


if __name__ == "__main__":
    main()
