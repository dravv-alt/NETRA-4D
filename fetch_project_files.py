import os
import requests
import gdown

FILE_IDS = [
    ("1KJnKnvIYux_JGU107iRPg8DK4UZGItEz", None),
    ("1LsScqj56-Tii8d9KPQ1IFhQQta0FPlNE", None),
    ("1Q-rsUlH3_5GUIESjSAop7-h5SFnhQ-Cf", None),
    ("1RN1sC_HvgIjIFJQl2TyZDcnhHkDr5wLi", None),
    ("1StrGyzzTlfZqauXL91BZ2B6-ssngKDNw", None),
    ("1WB0Dil1XPernLH43xmKfS1eT71iVGG4M", None),
    ("1XIKvjt6ewIXmi24-ll4TyeyQjzE2byjV", None),
    ("1hOEG2Ln2PDhNqzBBjAi2vsfoRxV_chb9", None),
    ("1krBr2AD_B0dVaOKFdtncnP61eW5i5at7", None),
    ("1nlC6KBqxpG0bkCyb_7gqsHSBeGMp5J-t", None),
    ("1qLc2qAUI1gaje7lzo0F5wk-GPjiJ0TvW", "presentation.pptx"),
    ("1whqdsa_6765uxbr-5D5DWijZtv61gECf", None),
]

def main():
    print(f"[*] Starting download of {len(FILE_IDS)} project files...")
    for fid, name in FILE_IDS:
        print(f"\n[+] Fetching file ID: {fid}...")
        url = f"https://drive.google.com/uc?id={fid}"
        try:
            if name:
                export_url = f"https://docs.google.com/presentation/d/{fid}/export/pptx"
                res = requests.get(export_url, allow_redirects=True)
                if res.status_code == 200 and len(res.content) > 1000:
                    with open(name, "wb") as f:
                        f.write(res.content)
                    print(f"[✔] Successfully exported {name} ({len(res.content)/(1024*1024):.2f} MB)")
                else:
                    out = gdown.download(id=fid, output=name, quiet=False)
                    print(f"[✔] Fetched: {out}")
            else:
                out = gdown.download(id=fid, quiet=False)
                print(f"[✔] Fetched: {out}")
        except Exception as e:
            print(f"[!] Error fetching {fid}: {e}")

    print("\n[★] All downloads finished.")

if __name__ == "__main__":
    main()
