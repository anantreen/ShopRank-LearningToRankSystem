# Getting the ESCI data

You do **not** need the large dataset for your first run. `python run_experiment.py` uses the
built-in 30-query catalog. Once that makes sense, download the official data:

```bash
git lfs install
git clone https://github.com/amazon-science/esci-data.git data/esci
```

The two important files are under `data/esci/shopping_queries_dataset/`:

- `shopping_queries_dataset_examples.parquet`: query IDs, product IDs, E/S/C/I labels and splits.
- `shopping_queries_dataset_products.parquet`: title, description, bullets, brand and color.

Start with English Task 1 only:

```python
import pandas as pd

root = "data/esci/shopping_queries_dataset"
examples = pd.read_parquet(f"{root}/shopping_queries_dataset_examples.parquet")
products = pd.read_parquet(f"{root}/shopping_queries_dataset_products.parquet")
rows = examples.query("small_version == 1 and product_locale == 'us'").merge(
    products, on=["product_id", "product_locale"], how="left"
)
```

The official repository reports 48,300 queries and 1,118,011 judgments in the reduced ranking
version. It is still large: first sample complete query groups, never random individual rows.
The data is Apache-2.0 licensed; cite the dataset paper when publishing results.

Official source: https://github.com/amazon-science/esci-data

