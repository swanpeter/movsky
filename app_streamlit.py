import os
import time
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional, Union

import base64
import datetime
import json
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from openai.types.video import Video

try:
    from streamlit_cookies_controller import CookieController
except ImportError:
    CookieController = None

try:
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except ImportError:
    StreamlitSecretNotFoundError = Exception

def get_secret_value(key: str) -> Optional[str]:
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None
    except Exception:
        return None
    try:
        return secrets_obj[key]
    except (KeyError, TypeError, StreamlitSecretNotFoundError):
        pass
    get_method = getattr(secrets_obj, "get", None)
    if callable(get_method):
        try:
            return get_method(key)
        except Exception:
            return None
    return None


def rerun_app() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


def decode_image_data(data: Optional[object]) -> Optional[bytes]:
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except (ValueError, TypeError):
            return None
    return None


def _normalize_credential(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def get_secret_auth_credentials() -> tuple[Optional[str], Optional[str]]:
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None, None
    except Exception:
        return None, None

    auth_section: Optional[dict] = None
    if isinstance(secrets_obj, dict):
        auth_section = secrets_obj.get("auth")
    else:
        auth_section = getattr(secrets_obj, "get", lambda _key, _default=None: None)("auth")

    def _get_from_container(container: object, key: str) -> Optional[object]:
        if isinstance(container, dict):
            return container.get(key)
        getter = getattr(container, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except TypeError:
                try:
                    return getter(key, None)
                except TypeError:
                    return None
        try:
            return getattr(container, key)
        except AttributeError:
            return None

    def _extract_credential(container: object, keys: tuple[str, ...]) -> Optional[object]:
        for key in keys:
            value = _get_from_container(container, key)
            if value is not None:
                return value
        return None

    username = None
    password = None
    if auth_section is not None:
        username = _extract_credential(auth_section, ("username", "id", "user", "name"))
        password = _extract_credential(auth_section, ("password", "pass", "pwd"))

    if username is None:
        username = get_secret_value("USERNAME") or get_secret_value("ID")
    if password is None:
        password = get_secret_value("PASSWORD") or get_secret_value("PASS")

    normalized_username = _normalize_credential(str(username)) if username is not None else None
    normalized_password = _normalize_credential(str(password)) if password is not None else None
    return normalized_username, normalized_password


class BasicSetting:
    def __init__(
        self,
        cookie_key: str = "logged_in",
        session_cookie_key: str = "browser_session_id",
        history_dir: Optional[str] = None,
        history_state_key: str = "history",
        history_loaded_key: str = "_history_loaded",
        auth_state_key: str = "authenticated",
        default_username: str = "mezamashi",
        default_password: str = "mezamashi",
        login_title: str = "ログイン",
        cookie_controller_state_key: str = "_cookie_controller",
        cookies_sync_stage_key: str = "_cookies_sync_stage",
    ) -> None:
        self.cookie_key = cookie_key
        self.session_cookie_key = session_cookie_key
        self.history_dir = history_dir or os.path.join(tempfile.gettempdir(), "nanobanana_history")
        self.history_state_key = history_state_key
        self.history_loaded_key = history_loaded_key
        self.auth_state_key = auth_state_key
        self.default_username = default_username
        self.default_password = default_password
        self.login_title = login_title
        self.cookie_controller_state_key = cookie_controller_state_key
        self.cookies_sync_stage_key = cookies_sync_stage_key

    def get_configured_auth_credentials(self) -> tuple[str, str]:
        secret_username, secret_password = get_secret_auth_credentials()
        if secret_username and secret_password:
            return secret_username, secret_password
        return self.default_username, self.default_password

    def _get_cookie_controller(self) -> Optional[object]:
        if CookieController is None:
            return None
        controller = st.session_state.get(self.cookie_controller_state_key)
        if controller is None:
            try:
                controller = CookieController()
            except Exception:
                return None
            st.session_state[self.cookie_controller_state_key] = controller
        return controller

    def cookie_controller_available(self) -> bool:
        return self._get_cookie_controller() is not None

    def sync_cookie_controller(self) -> None:
        controller = self._get_cookie_controller()
        if controller is None:
            return
        sync_stage = st.session_state.get(self.cookies_sync_stage_key, 0)
        if sync_stage == 0:
            try:
                controller.refresh()
            except Exception:
                return
            st.session_state[self.cookies_sync_stage_key] = 1
            rerun_app()
            return
        if sync_stage == 1:
            try:
                controller.refresh()
            except Exception:
                return
            st.session_state[self.cookies_sync_stage_key] = 2

    def restore_login_from_cookie(self) -> bool:
        controller = self._get_cookie_controller()
        if controller is None:
            return False
        for _ in range(2):
            try:
                controller.refresh()
                if controller.get(self.cookie_key) == "1":
                    return True
            except Exception:
                return False
            time.sleep(0.3)
        return False

    def persist_login_to_cookie(self, value: bool) -> None:
        controller = self._get_cookie_controller()
        if controller is None:
            return
        try:
            if value:
                controller.set(self.cookie_key, "1")
                time.sleep(0.6)
            else:
                controller.remove(self.cookie_key)
        except Exception:
            return

    def _get_history_path(self, session_id: str) -> str:
        os.makedirs(self.history_dir, exist_ok=True)
        safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
        return os.path.join(self.history_dir, f"{safe_id}.json")

    def get_browser_session_id(self, create: bool = True) -> Optional[str]:
        controller = self._get_cookie_controller()
        if controller is None:
            return None
        try:
            controller.refresh()
            session_id = controller.get(self.session_cookie_key)
        except Exception:
            session_id = None
        if session_id:
            return str(session_id)
        if not create:
            return None
        new_id = uuid.uuid4().hex
        try:
            controller.set(self.session_cookie_key, new_id)
            time.sleep(0.6)
        except Exception:
            return None
        return new_id

    def _serialize_history(self, history: list[dict[str, object]]) -> list[dict[str, object]]:
        serialized: list[dict[str, object]] = []
        for entry in history:
            image_bytes = entry.get("image_bytes")
            if isinstance(image_bytes, (bytes, bytearray, memoryview)):
                image_b64 = base64.b64encode(bytes(image_bytes)).decode("utf-8")
            else:
                image_b64 = None
            serialized.append(
                {
                    "id": entry.get("id"),
                    "prompt": entry.get("prompt"),
                    "model": entry.get("model"),
                    "no_text": entry.get("no_text"),
                    "image_b64": image_b64,
                    "size": entry.get("size"),
                    "seconds": entry.get("seconds"),
                    "created_at": entry.get("created_at"),
                    "video_path": entry.get("video_path"),
                }
            )
        return serialized

    def _deserialize_history(self, payload: list[dict[str, object]]) -> list[dict[str, object]]:
        history: list[dict[str, object]] = []
        for entry in payload:
            image_b64 = entry.get("image_b64")
            image_bytes = decode_image_data(image_b64) if image_b64 else None
            history.append(
                {
                    "id": entry.get("id"),
                    "prompt": entry.get("prompt"),
                    "model": entry.get("model"),
                    "no_text": entry.get("no_text"),
                    "image_bytes": image_bytes,
                    "size": entry.get("size"),
                    "seconds": entry.get("seconds"),
                    "created_at": entry.get("created_at"),
                    "video_path": entry.get("video_path"),
                }
            )
        return history

    def load_history_from_storage(self) -> Optional[list[dict[str, object]]]:
        session_id = self.get_browser_session_id(create=False)
        if not session_id:
            return None
        history_path = self._get_history_path(session_id)
        if not os.path.exists(history_path):
            return None
        try:
            with open(history_path, "r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        entries = payload.get("history")
        if not isinstance(entries, list):
            return None
        return self._deserialize_history(entries)

    def persist_history_to_storage(self) -> None:
        session_id = self.get_browser_session_id(create=True)
        if not session_id:
            return
        history_path = self._get_history_path(session_id)
        history = st.session_state.get(self.history_state_key, [])
        if not isinstance(history, list):
            return
        payload = {
            "updated_at": datetime.datetime.utcnow().isoformat(),
            "history": self._serialize_history(history),
        }
        try:
            with open(history_path, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle)
        except Exception:
            return

    def clear_history_storage(self) -> None:
        session_id = self.get_browser_session_id(create=False)
        if not session_id:
            return
        history_path = self._get_history_path(session_id)
        try:
            if os.path.exists(history_path):
                os.remove(history_path)
        except Exception:
            return

    def logout(self) -> None:
        st.session_state[self.auth_state_key] = False
        self.persist_login_to_cookie(False)
        self.clear_history_storage()
        if self.history_state_key in st.session_state:
            st.session_state[self.history_state_key] = []
        rerun_app()

    def inject_login_autofill_js(self) -> None:
        components.html(
            """
        <script>
        (function () {
            const parent = window.parent;
            if (!parent || !parent.document) {
                return;
            }
            const doc = parent.document;
            const inputs = Array.from(doc.querySelectorAll("input"));
            if (!inputs.length) {
                return;
            }
            let userInput = null;
            let passInput = null;
            for (const input of inputs) {
                const label = (input.getAttribute("aria-label") || "").toLowerCase();
                if (!userInput && (label === "id" || label === "user" || label === "username")) {
                    userInput = input;
                }
                if (!passInput && (label === "pass" || label === "password")) {
                    passInput = input;
                }
            }
            if (userInput) {
                userInput.setAttribute("name", "username");
                userInput.setAttribute("autocomplete", "username");
            }
            if (passInput) {
                passInput.setAttribute("name", "password");
                passInput.setAttribute("autocomplete", "current-password");
            }
            const form = userInput ? userInput.form : null;
            if (form) {
                form.setAttribute("autocomplete", "on");
            }
        })();
        </script>
        """,
            height=0,
            scrolling=False,
        )

    def require_login(self) -> None:
        if self.auth_state_key not in st.session_state:
            st.session_state[self.auth_state_key] = False

        if not st.session_state[self.auth_state_key] and self.restore_login_from_cookie():
            st.session_state[self.auth_state_key] = True
            self.get_browser_session_id(create=True)

        if st.session_state[self.auth_state_key]:
            return

        st.title(self.login_title)

        username, password = self.get_configured_auth_credentials()
        if not username or not password:
            st.info("ログイン情報が未設定です。管理者に連絡してください。")
            st.stop()
            return

        with st.form("login_form", clear_on_submit=False):
            input_username = st.text_input("ID")
            input_password = st.text_input("PASS", type="password")
            submitted = st.form_submit_button("ログイン")

        self.inject_login_autofill_js()

        if submitted:
            if input_username == username and input_password == password:
                st.session_state[self.auth_state_key] = True
                self.persist_login_to_cookie(True)
                self.get_browser_session_id(create=True)
                st.success("ログインしました。")
                rerun_app()
                return
            st.error("IDまたはPASSが正しくありません。")
        st.stop()

    def init_history(self) -> None:
        if self.history_state_key not in st.session_state:
            st.session_state[self.history_state_key] = []
        if not st.session_state.get(self.history_loaded_key):
            restored = self.load_history_from_storage()
            if restored is not None:
                st.session_state[self.history_state_key] = restored
                st.session_state[self.history_loaded_key] = True
            else:
                if self.get_browser_session_id(create=False) is not None or not self.cookie_controller_available():
                    st.session_state[self.history_loaded_key] = True


_default_container = BasicSetting()


def sync_cookie_controller() -> None:
    return _default_container.sync_cookie_controller()


def persist_history_to_storage() -> None:
    return _default_container.persist_history_to_storage()


def require_login() -> None:
    return _default_container.require_login()


def init_history() -> None:
    return _default_container.init_history()


sync_cookie_controller()
require_login()
init_history()

endpoint = st.secrets.get("ENDPOINT_URL")
deployment = st.secrets.get("DEPLOYMENT_NAME")
subscription_key = st.secrets.get("AZURE_OPENAI_API_KEY")
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
            create_kwargs = {
                "model": st.session_state.deployment,
                "prompt": prompt,
                "seconds": int(seconds),
                "size": effective_size,
            }
            if input_reference is not None:
                create_kwargs["input_reference"] = input_reference
            job: Video = client.videos.create(**create_kwargs)
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
                persist_history_to_storage()
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
