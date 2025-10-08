from huggingface_hub import snapshot_download


def main():
    snapshot_download(
        repo_id="momergul/mash_data",
        repo_type="dataset",
        local_dir="./mash_data"
    )

if __name__ == "__main__":
    main()
