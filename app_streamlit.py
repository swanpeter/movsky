import os
import time
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional, Union

import streamlit as st
from openai import OpenAI
from openai.types.video import Video

import basic_setting

basic_setting.sync_cookie_controller()
basic_setting.require_login()
basic_setting.init_history()

endpoint = st.secrets.get("ENDPOINT_URL") or os.getenv(
    "ENDPOINT_URL", "https://kurokawa-sweden-resource.openai.azure.com/"
)
deployment = st.secrets.get("DEPLOYMENT_NAME") or os.getenv(
    "DEPLOYMENT_NAME", "sora-2_kuro"
)
subscription_key = st.secrets.get("AZURE_OPENAI_API_KEY") or os.getenv(
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"
)
api_version = st.secrets.get("API_VERSION") or os.getenv("API_VERSION", "preview")

st.session_state.setdefault("endpoint", endpoint)
st.session_state.setdefault("deployment", deployment)
st.session_state.setdefault("api_version", api_version)

prompt = st.text_area("Prompt", value="text prompt", height=120)
seconds = st.selectbox("Seconds", [4, 8, 12], index=1)
size = st.selectbox("Size", ["1280x720", "720x1280"], index=0)

st.subheader("Reference Image (optional)")
uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
input_image_path = None
match_ref_size = True
resize_ref = st.checkbox("Resize reference to selected size", value=False)

if subscription_key == "AZURE_OPENAI_API_KEY":
    st.warning("AZURE_OPENAI_API_KEY が未設定です。環境変数を設定してください。")


def build_client() -> OpenAI:
    return OpenAI(
        api_key=subscription_key,
        base_url=f"{st.session_state.endpoint}openai/v1/",
        default_headers={"api-key": subscription_key},
        default_query={"api-version": st.session_state.api_version},
    )


def resolve_input_reference() -> tuple[Optional[Union[BinaryIO, Path]], Optional[Path]]:
    if uploaded is not None:
        # Ensure a valid file extension so the SDK can infer mimetype.
        name = uploaded.name or "upload.png"
        _, ext = os.path.splitext(name)
        if ext.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".mp4"}:
            ext = ".png"
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp.write(uploaded.getvalue())
        temp.flush()
        temp.close()
        return Path(temp.name), Path(temp.name)
    if input_image_path and os.path.isfile(input_image_path):
        return Path(input_image_path), None
    return None, None


if st.button("Generate Video", type="primary"):
    if subscription_key == "AZURE_OPENAI_API_KEY":
        st.error("API キーが未設定です。")
        st.stop()

    client = build_client()

    input_reference: Optional[Union[BinaryIO, Path]] = None
    temp_path: Optional[Path] = None
    try:
        input_reference, temp_path = resolve_input_reference()
        effective_size = size
        resized_path: Optional[Path] = None

        if input_reference:
            try:
                from PIL import Image
            except Exception:
                st.error("Pillow が必要です。`pip install pillow` を実行してください。")
                st.stop()
            if match_ref_size and not resize_ref:
                try:
                    with Image.open(input_reference) as img:
                        w, h = img.size
                    effective_size = f"{w}x{h}"
                except Exception as e:
                    st.error(f"参考画像のサイズ取得に失敗しました: {e}")
                    st.stop()

                if effective_size not in {"1280x720", "720x1280"}:
                    st.error(
                        f"参考画像サイズ {effective_size} は未対応です。"
                        " 対応サイズの画像を用意するか、サイズ自動一致をOFFにしてください。"
                    )
                    st.stop()

            if resize_ref:
                try:
                    # Resize to the user-selected size (not the original ref size).
                    w, h = map(int, size.split("x"))
                    with Image.open(input_reference) as img:
                        resized = img.resize((w, h), Image.LANCZOS)
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        resized.save(tmp.name, format="PNG")
                    resized_path = Path(tmp.name)
                    input_reference = resized_path
                    effective_size = size
                except Exception as e:
                    st.error(f"参考画像のリサイズに失敗しました: {e}")
                    st.stop()

        with st.status("Requesting video job...", expanded=False) as status_box:
            job: Video = client.videos.create(
                model=st.session_state.deployment,
                prompt=prompt,
                seconds=int(seconds),
                size=effective_size,
                input_reference=input_reference,
            )
            status_box.update(label=f"Job created: {job.id}", state="running")

        st.info(f"Polling job status for ID: {job.id}")
        status_placeholder = st.empty()
        spinner = st.empty()
        status = "queued"
        tick = 0

        if hasattr(client.videos, "retrieve"):
            while status in {"queued", "running", "in_progress"}:
                result = client.videos.retrieve(job.id)
                status = result.status
                dots = "・" * (tick % 4)
                spinner.write(f"処理中{dots}")
                status_placeholder.write(f"Status: {status}")
                if getattr(result, "progress", None) is not None:
                    prog = result.progress
                    if isinstance(prog, (int, float)):
                        pct = int(prog * 100) if prog <= 1 else int(prog)
                        pct = max(0, min(99, pct))
                if status in {"queued", "running", "in_progress"}:
                    time.sleep(5)
                    tick += 1
        else:
            result = client.videos.poll(job.id, poll_interval_ms=5_000)
            status = result.status
            status_placeholder.write(f"Status: {status}")

        if status == "completed":
            st.success("Video generation succeeded.")
            spinner.write("✅ 完了")
            content = client.videos.download_content(result.id, variant="video")
            filename = f"output_{int(time.time())}.mp4"
            content.write_to_file(filename)
            st.session_state["last_video"] = filename
            history = st.session_state.get("history", [])
            if isinstance(history, list):
                history.append(
                    {
                        "id": result.id,
                        "prompt": prompt,
                        "size": effective_size,
                        "seconds": int(seconds),
                        "created_at": int(time.time()),
                        "video_path": filename,
                    }
                )
                st.session_state["history"] = history
                basic_setting.persist_history_to_storage()
            st.video(filename)
            with open(filename, "rb") as f:
                st.download_button("Download video", data=f, file_name=filename)
        elif status == "failed":
            st.error("Video generation failed.")
            spinner.write("❌ 失敗")
            st.json(result.model_dump())
        elif status in {"queued", "running", "in_progress"}:
            st.info("まだ処理中です。しばらくお待ちください。")
            st.json(result.model_dump())
        else:
            st.warning("Unexpected status returned.")
            st.json(result.model_dump())
    finally:
        if input_reference and hasattr(input_reference, "close"):
            input_reference.close()
        if temp_path and temp_path.is_file():
            temp_path.unlink()
        if resized_path and resized_path.is_file():
            resized_path.unlink()

if "last_video" in st.session_state and os.path.isfile(st.session_state["last_video"]):
    st.subheader("Last generated video")
    st.video(st.session_state["last_video"])

history = st.session_state.get("history", [])
if isinstance(history, list) and history:
    st.subheader("History")
    for item in reversed(history[-10:]):
        label = item.get("prompt") or "(no prompt)"
        st.write(label)
        video_path = item.get("video_path")
        if video_path and os.path.isfile(video_path):
            st.video(video_path)
