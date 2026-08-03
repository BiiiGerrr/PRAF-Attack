import os
import argparse
import base64
import mimetypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import torchvision.datasets as datasets


class ImageCaptioner:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4o"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._local = threading.local()

    def _get_client(self) -> OpenAI:
        if not hasattr(self._local, "client"):
            self._local.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._local.client

    def encode_image_with_mime(self, image_path: str):
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            ext = os.path.splitext(image_path)[1].lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
            mime_type = mime_map.get(ext, "image/jpeg")

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return b64, mime_type

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def generate_caption(self, image_path: str, prompt_text: str) -> str:
        b64, mime = self.encode_image_with_mime(image_path)

        resp = self._get_client().chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
            # max_tokens=100,
            temperature=0,
        )

        # 更稳一点的解析
        if not resp.choices:
            raise ValueError(f"No choices returned | image={image_path}")

        choice = resp.choices[0]
        msg = choice.message
        content = getattr(msg, "content", None)
        finish_reason = getattr(choice, "finish_reason", None)
        refusal = getattr(msg, "refusal", None)

        # 正常返回
        if isinstance(content, str) and content.strip():
            return content.strip()

        error_msg = (
            f"Empty content | image={image_path} | "
            f"finish_reason={finish_reason} | refusal={refusal} | message={msg}"
        )
        print(f"[Error] {error_msg}")  # 添加打印
        raise ValueError(error_msg)

def folder_tag(folder: str) -> str:
    return os.path.basename(os.path.normpath(folder))

def process_one_image(captioner: ImageCaptioner, img_path: str, prompt_text: str):
    try:
        cap = captioner.generate_caption(img_path, prompt_text)
        return cap.replace("\n", " ").strip()
    except Exception as e:
        err = str(e).replace("\n", " ")[:500]
        tqdm.write(f"[Final Error] image={img_path} | {type(e).__name__}: {err}")
        return f"[Error: {type(e).__name__}: {err}]"

def run_one_model_one_folder(args, model_name: str, folder: str):
    captioner = ImageCaptioner(api_key=args.api_key, base_url=args.base_url, model=model_name)

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = f"{model_name.replace('/', '_')}_{os.path.basename(os.path.normpath(folder))}.txt"
    out_path = os.path.join(args.output_dir, out_name)

    ds = datasets.ImageFolder(folder, transform=None)
    paths = [p for (p, _) in ds.samples]
    if args.max_samples > 0:
        paths = paths[:args.max_samples]

    print(f"\n[Running] Model: {model_name} | Folder: {folder} | Total: {len(paths)}")

    caps = ["[Pending]"] * len(paths)

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(process_one_image, captioner, p, args.prompt): i for i, p in enumerate(paths)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Progress", leave=False):
            idx = futs[fut]
            try:
                result = fut.result()
                caps[idx] = result if result and result.strip() else "[Error: Null Response]"
            except Exception as e:
                caps[idx] = f"[Error: {str(e)}]"

    # 写入文件
    with open(out_path, "w", encoding="utf-8") as f:
        for p, c in zip(paths, caps):

            final_content = c if c else "[Error: Final Fallback]"
            if args.write_path:
                f.write(f"{p}\t{final_content}\n")
            else:
                f.write(f"{final_content}\n")

    return out_path, len(caps)



def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions: save per-folder per-model as {model}_{file}.txt")

    parser.add_argument(
        "--img_folders",
        nargs="+",
        default=[
            "./PRAF_Attack",
            "resources/images/target_images"
        ],
        help="One or more image folders (ImageFolder structure: root/class_x/*.jpg)."
    )

    parser.add_argument("--output_dir", default="./Black-box-description", type=str)
    parser.add_argument("--max_samples", default=100, type=int, help="Max samples per folder (<=0 means no limit)")
    parser.add_argument("--num_workers", default=50, type=int, help="Concurrent API workers per folder")

    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")

    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in one concise sentence, no longer than 20 words.",
        help="Prompt for captioning."
    )

    parser.add_argument("--write_path", action="store_true",
                        help="If set, each line: <img_path>\\t<caption>")

    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.4"],
        help="Model names to test."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[Config] num_workers={args.num_workers}, max_samples={args.max_samples}")
    print(f"[Config] base_url={args.base_url}")
    print(f"[Config] models={args.models}")
    print(f"[Config] output_dir={args.output_dir}")
    summary = []
    for m in args.models:
        for folder in args.img_folders:
            try:
                out_path, n = run_one_model_one_folder(args, m, folder)
                summary.append((m, folder, n, out_path))
            except Exception as e:
                print(f"[Error] model={m}, folder={folder}: {e}")
                summary.append((m, folder, 0, "ERROR"))

    print("\n" + "=" * 70)
    print("Done. Summary:")
    for m, folder, n, out_path in summary:
        print(f"\"{out_path}\",")
    print("=" * 70)


if __name__ == "__main__":
    main()

