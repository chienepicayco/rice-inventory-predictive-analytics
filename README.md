# Rice Inventory Predictive Analytics Website

This Python Streamlit website is preloaded with the CSV from the supplied `RICE_INVENTORY_UNSTRUCTURED-main.zip` repository. You do not need to upload the dataset each time.

## Run in VS Code

1. Extract this project ZIP.
2. Open the extracted `RiceInventoryPredictiveWebsite` folder in VS Code.
3. Open Terminal in VS Code.
4. Install dependencies:

   `pip install -r requirements.txt`

5. Run:

   `streamlit run app.py`

6. Streamlit will open the website in your browser.

## Included source data

`data/rice_inventory.csv` is the repository CSV:
`Rice Monthly Total Stocks Inventory by Sector Year (2000-2025) Month (January-December).csv`

The source contains 104 rows and 14 columns: Sector, Year, and January through December. The app reshapes it to monthly long format with Sector, Year, Month, Month_Num, and Rice_Inventory.

## Pages

- Dashboard
- Dataset (original and structured monthly view)
- Visualization (sector bar, sector-share pie, yearly trend, box plot, monthly average)
- Descriptive Analytics
- SVR Regression
- Classification Models (Gaussian Naive Bayes, Logistic Regression, Decision Tree, Random Forest)
- Model Comparison
- Future Predictions (2026–2030 classes by month and sector)

## Relationship to the supplied Colab notebook

The classification workflow follows the notebook's qcut Low/Medium/High target, one-hot sector representation, 80/20 train-test split, random_state=42, and the four classifier settings.

The notebook's SVR cell intends to use Sector, Year, and Month as predictors, but StandardScaler cannot directly process text values. The app preserves the intended predictors and SVR parameters (RBF, C=100000, epsilon=1000) while encoding categorical fields before model training. The future classification page follows the notebook's 2026–2030, 12-month, four-sector prediction grid.

All model scores are recalculated from the bundled source data at app startup; the app does not hardcode performance numbers.


## Sidebar alignment update
This version adjusts sidebar width and inner spacing so the logo, title, navigation labels, and source note remain visible. Run locally with:

```bash
python -m streamlit run app.py
```


## Full-page design update
This version applies consistent page headers, typography, controls, data-table styling, chart colors, and spacing across Dataset, Visualization, Descriptive Analytics, SVR Regression, Classification Models, Model Comparison, and Future Predictions. The underlying dataset and model workflow are preserved.

Run with:

```bash
python -m streamlit run app.py
```


## Dark mode
Use the **Dark mode** switch in the left sidebar to change between the light and dark themes. The selection is interactive and updates the page and Plotly chart theme.
