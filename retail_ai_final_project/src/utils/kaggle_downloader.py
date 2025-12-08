"""
Kaggle Dataset Downloader Utility
Provides tools for searching and downloading datasets from Kaggle.
"""
import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import json
import pandas as pd


class KaggleDownloader:
    """
    Utility class for interacting with Kaggle API to download datasets.

    Prerequisites:
    1. Install kaggle package: pip install kaggle
    2. Set up Kaggle API credentials:
       - Go to https://www.kaggle.com/account
       - Create API token (downloads kaggle.json)
       - Place kaggle.json in ~/.kaggle/ directory
       - Set permissions: chmod 600 ~/.kaggle/kaggle.json
    """

    def __init__(self, download_dir: str = "data/raw"):
        """
        Initialize Kaggle downloader.

        Args:
            download_dir: Directory to download datasets to
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._verify_kaggle_setup()

    def _verify_kaggle_setup(self) -> bool:
        """Verify Kaggle API is properly configured."""
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

        if not kaggle_json.exists():
            print("⚠️  Kaggle API not configured!")
            print("📋 Setup Instructions:")
            print("   1. Go to https://www.kaggle.com/account")
            print("   2. Click 'Create New API Token'")
            print("   3. Move downloaded kaggle.json to ~/.kaggle/")
            print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False

        print("✅ Kaggle API configured")
        return True

    def search_datasets(
        self,
        query: str,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        file_type: str = "csv",
        sort_by: str = "hottest"
    ) -> List[Dict]:
        """
        Search for datasets on Kaggle.

        Args:
            query: Search query string
            min_size: Minimum dataset size in bytes
            max_size: Maximum dataset size in bytes
            file_type: File type filter (csv, json, etc.)
            sort_by: Sort order (hottest, votes, updated, active)

        Returns:
            List of dataset information dictionaries
        """
        try:
            # Run kaggle datasets list command
            cmd = [
                "kaggle", "datasets", "list",
                "--search", query,
                "--sort-by", sort_by,
                "--csv"
            ]

            if min_size:
                cmd.extend(["--min-size", str(min_size)])
            if max_size:
                cmd.extend(["--max-size", str(max_size)])
            if file_type:
                cmd.extend(["--file-type", file_type])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse CSV output
            from io import StringIO
            df = pd.read_csv(StringIO(result.stdout))

            datasets = []
            for _, row in df.iterrows():
                datasets.append({
                    "ref": row["ref"],
                    "title": row["title"],
                    "size": row["size"],
                    "lastUpdated": row["lastUpdated"],
                    "downloadCount": row["downloadCount"],
                    "voteCount": row["voteCount"],
                    "usabilityRating": row["usabilityRating"]
                })

            print(f"✅ Found {len(datasets)} datasets matching '{query}'")
            return datasets

        except subprocess.CalledProcessError as e:
            print(f"❌ Error searching datasets: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return []

    def download_dataset(
        self,
        dataset_ref: str,
        unzip: bool = True,
        force: bool = False
    ) -> Optional[Path]:
        """
        Download a dataset from Kaggle.

        Args:
            dataset_ref: Kaggle dataset reference (owner/dataset-name)
            unzip: Whether to unzip downloaded files
            force: Force download even if file exists

        Returns:
            Path to downloaded dataset directory, or None if failed
        """
        try:
            print(f"📥 Downloading dataset: {dataset_ref}")

            cmd = [
                "kaggle", "datasets", "download",
                dataset_ref,
                "--path", str(self.download_dir)
            ]

            if unzip:
                cmd.append("--unzip")
            if force:
                cmd.append("--force")

            subprocess.run(cmd, check=True)

            print(f"✅ Dataset downloaded to: {self.download_dir}")
            return self.download_dir

        except subprocess.CalledProcessError as e:
            print(f"❌ Error downloading dataset: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def list_dataset_files(self, dataset_ref: str) -> List[str]:
        """
        List files in a Kaggle dataset without downloading.

        Args:
            dataset_ref: Kaggle dataset reference

        Returns:
            List of file names in the dataset
        """
        try:
            cmd = [
                "kaggle", "datasets", "files",
                dataset_ref,
                "--csv"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            from io import StringIO
            df = pd.read_csv(StringIO(result.stdout))
            files = df["name"].tolist()

            print(f"✅ Dataset contains {len(files)} files")
            return files

        except subprocess.CalledProcessError as e:
            print(f"❌ Error listing files: {e}")
            return []

    def get_dataset_metadata(self, dataset_ref: str) -> Optional[Dict]:
        """
        Get metadata for a Kaggle dataset.

        Args:
            dataset_ref: Kaggle dataset reference

        Returns:
            Dictionary with dataset metadata
        """
        try:
            cmd = [
                "kaggle", "datasets", "metadata",
                dataset_ref,
                "--path", "/tmp"
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            metadata_file = Path("/tmp") / "dataset-metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                metadata_file.unlink()  # Clean up temp file
                return metadata

            return None

        except Exception as e:
            print(f"❌ Error getting metadata: {e}")
            return None

    def recommend_datasets(self, category: str = "retail") -> List[Dict]:
        """
        Get recommended datasets for common categories.

        Args:
            category: Category name (retail, ecommerce, sales, etc.)

        Returns:
            List of recommended datasets
        """
        recommendations = {
            "retail": [
                "carrie1/ecommerce-data",
                "mkechinov/ecommerce-behavior-data-from-multi-category-store",
                "olistbr/brazilian-ecommerce",
                "uciml/online-retail-ii-data-set"
            ],
            "sales": [
                "rohitsahoo/sales-forecasting",
                "kyanyoga/sample-sales-data",
                "shivan118/big-mart-sales-prediction-datasets"
            ],
            "customer": [
                "imakash3011/customer-personality-analysis",
                "vjchoudhary7/customer-segmentation-tutorial-in-python"
            ]
        }

        refs = recommendations.get(category.lower(), [])

        print(f"📚 Recommended {category} datasets:")
        results = []
        for ref in refs:
            print(f"   - {ref}")
            results.append({"ref": ref})

        return results


def main():
    """Example usage of KaggleDownloader."""
    downloader = KaggleDownloader()

    # Search for retail datasets
    print("\n🔍 Searching for retail datasets...")
    datasets = downloader.search_datasets("retail", sort_by="hottest")

    if datasets:
        print(f"\nTop 5 Results:")
        for i, ds in enumerate(datasets[:5], 1):
            print(f"\n{i}. {ds['title']}")
            print(f"   Ref: {ds['ref']}")
            print(f"   Size: {ds['size']}")
            print(f"   Downloads: {ds['downloadCount']}")
            print(f"   Rating: {ds['usabilityRating']}")

    # Example: Download a specific dataset
    # dataset_ref = "carrie1/ecommerce-data"
    # print(f"\n📥 Downloading {dataset_ref}...")
    # downloader.download_dataset(dataset_ref)


if __name__ == "__main__":
    main()
