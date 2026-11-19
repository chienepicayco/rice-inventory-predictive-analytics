import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)

st.set_page_config(page_title="Rice Inventory Predictive Analytics", page_icon="🌾", layout="wide")

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
MONTH_MAP = {m: i + 1 for i, m in enumerate(MONTHS)}
SECTORS = ["Rice: Total Stock", "Rice: Household Stock",
           "Rice: Commercial Stock", "Rice: NFA Stock"]
DATA_PATH = "data/rice_inventory.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root { --forest:#123b2d; --forest2:#1d5a40; --leaf:#2f8055; --mint:#e8f2eb; --paper:#f6f8f5; --ink:#17251e; --muted:#66766d; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stAppViewContainer"] > .main { background:var(--paper); }
.block-container { max-width:1500px; padding:2rem clamp(1.1rem,3vw,2.8rem) 3rem; }
/* Sidebar alignment fix: consistent width and safe inner padding. */
section[data-testid="stSidebar"], [data-testid="stSidebar"] {
    background:linear-gradient(180deg,#103729 0%,#174b36 100%);
    border-right:1px solid rgba(255,255,255,.08);
    width:19rem !important; min-width:19rem !important; max-width:19rem !important;
    box-sizing:border-box !important; overflow-x:hidden !important;
}
section[data-testid="stSidebar"] > div, [data-testid="stSidebar"] > div {
    width:100% !important; min-width:0 !important; box-sizing:border-box !important;
    padding:1rem 0 1.4rem !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    width:100% !important; min-width:0 !important; box-sizing:border-box !important;
    padding:1rem 1.2rem 1.2rem !important; margin:0 !important;
}
[data-testid="stSidebar"] .block-container {
    width:100% !important; max-width:100% !important;
    padding:0 1.2rem 1.2rem !important; margin:0 !important;
    box-sizing:border-box !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    width:100% !important; min-width:0 !important;
    margin-left:0 !important; padding-left:0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    width:100% !important; min-width:0 !important; margin-left:0 !important;
    box-sizing:border-box !important;
}
[data-testid="stSidebar"] * { color:#f5fbf6; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { letter-spacing:-.02em; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { color:#c7dfd0; font-size:.76rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin:.6rem 0 .35rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap:.28rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label { padding:.48rem .65rem; border-radius:10px; border:1px solid transparent; transition:background .15s ease,border .15s ease; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover { background:rgba(255,255,255,.10); border-color:rgba(255,255,255,.10); }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p { font-weight:600; font-size:.9rem; }
[data-testid="stSidebar"] [data-testid="stInfo"] { background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.15); border-radius:12px; padding:.75rem; }
[data-testid="stSidebar"] [data-testid="stInfo"] p { color:#f5fbf6; font-size:.82rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#c3d8ca; font-size:.78rem; line-height:1.5; }
.hero { position:relative; overflow:hidden; padding:clamp(1.5rem,3vw,2.5rem); border-radius:22px; background:linear-gradient(115deg,#103729 0%,#1f6545 58%,#3c8757 100%); color:white; margin:0 0 1.7rem; box-shadow:0 12px 30px rgba(18,59,45,.13); }
.hero:after { content:'🌾'; position:absolute; right:3rem; top:50%; transform:translateY(-50%); font-size:5.2rem; opacity:.17; }
.hero h1 { color:white; font-family:'Manrope',sans-serif; font-size:clamp(1.8rem,3vw,2.65rem); line-height:1.15; margin:0; font-weight:800; }
.hero p { color:#e1f0e5; font-size:1rem; margin:.65rem 0 0; max-width:780px; }
.eyebrow { color:var(--leaf); font-size:.75rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.35rem; }
.section-title { color:var(--ink); font-family:'Manrope',sans-serif; font-weight:800; font-size:1.45rem; margin:.2rem 0 .25rem; }
.section-subtitle { color:var(--muted); font-size:.92rem; margin:0 0 1.1rem; }
.kpi-card { background:#fff; border:1px solid #e3ebe4; border-radius:16px; padding:1.05rem 1.15rem; min-height:118px; height:100%; box-shadow:0 4px 14px rgba(23,50,35,.035); transition:transform .18s ease,box-shadow .18s ease; }
.kpi-card:hover { transform:translateY(-2px); box-shadow:0 9px 22px rgba(23,50,35,.08); }
.kpi-label { color:#718078; font-size:.78rem; font-weight:600; margin-bottom:.5rem; }
.kpi-value { color:#183b2c; font-family:'Manrope',sans-serif; font-size:1.75rem; font-weight:800; line-height:1.2; }
.kpi-note { color:#2f8055; font-size:.78rem; margin-top:.4rem; }
.insight-card { background:#eaf3ec; border:1px solid #d8e8dc; border-left:4px solid #3b8a5c; padding:1rem 1.15rem; border-radius:13px; color:#244b35; margin:.5rem 0 1.2rem; }
div[data-testid="stMetric"] { background:#fff; padding:1rem 1.1rem; border-radius:15px; border:1px solid #e3ebe4; box-shadow:0 4px 14px rgba(23,50,35,.035); }
[data-testid="stMetricLabel"] { color:#718078; font-weight:600; }
[data-testid="stMetricValue"] { color:#183b2c; font-family:'Manrope',sans-serif; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #e5ece6; border-radius:16px; padding:.5rem .55rem; box-shadow:0 4px 14px rgba(23,50,35,.025); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #e5ece6; border-radius:13px; overflow:hidden; }
.stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid #2f8055; font-weight:600; }
.stButton>button[kind="primary"] { background:#216344; color:white; }
hr { border-color:#e2eae3; }
footer { visibility:hidden; }
@media (max-width:900px) {
    section[data-testid="stSidebar"], [data-testid="stSidebar"] {
        width:18rem !important; min-width:18rem !important; max-width:18rem !important;
    }
    .block-container { padding:1.2rem 1rem 2rem; }
    .hero { border-radius:16px; }
    .hero h1 { font-size:1.8rem; }
    .kpi-value { font-size:1.45rem; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    source = pd.read_csv(DATA_PATH)
    long = source.melt(id_vars=["Sector","Year"], value_vars=MONTHS,
                       var_name="Month", value_name="Rice_Inventory")
    long["Month_Num"] = long["Month"].map(MONTH_MAP).astype(int)
    long["Year"] = pd.to_numeric(long["Year"], errors="raise").astype(int)
    long["Rice_Inventory"] = pd.to_numeric(long["Rice_Inventory"], errors="coerce")
    long = long.dropna(subset=["Rice_Inventory"]).reset_index(drop=True)
    return source, long

@st.cache_resource
def train_all(data):
    # Classification follows the notebook's qcut Low/Medium/High target,
    # one-hot sector features, and 80/20 random_state=42 split.
    cls = data.copy()
    cls["Inventory_Class"] = pd.qcut(
        cls["Rice_Inventory"], q=3, labels=["Low","Medium","High"]
    )
    X = cls[["Year","Month_Num","Sector"]]
    y = cls["Inventory_Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), ["Year","Month_Num"]),
        ("sector", OneHotEncoder(handle_unknown="ignore"), ["Sector"])
    ])
    model_specs = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(max_depth=6, n_estimators=100, random_state=42)
    }
    models, predictions, metric_rows = {}, {}, []
    for name, estimator in model_specs.items():
        if name == "Gaussian Naive Bayes":
            prep = ColumnTransformer([
                ("numeric", StandardScaler(), ["Year","Month_Num"]),
                ("sector", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Sector"])
            ])
        else:
            prep = preprocessor
        pipe = Pipeline([("prep", prep), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        models[name] = pipe
        predictions[name] = pred
        metric_rows.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, pred, average="weighted", zero_division=0)
        })

    # SVR uses the notebook's intended predictors and parameters.
    # Categorical Sector and Month are encoded before scaling to avoid
    # passing text values directly into StandardScaler.
    Xr = data[["Sector","Year","Month"]]
    yr = data["Rice_Inventory"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    svr_prep = ColumnTransformer([
        ("numeric", StandardScaler(), ["Year"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["Sector","Month"])
    ])
    svr = Pipeline([
        ("prep", svr_prep),
        ("model", SVR(kernel="rbf", C=100000, epsilon=1000))
    ])
    svr.fit(Xr_train, yr_train)
    svr_pred = svr.predict(Xr_test)
    return {
        "models": models, "predictions": predictions, "X_test": X_test,
        "y_test": y_test, "metrics": pd.DataFrame(metric_rows),
        "svr": svr, "Xr_test": Xr_test, "yr_test": yr_test,
        "svr_pred": svr_pred,
        "rmse": float(np.sqrt(mean_squared_error(yr_test, svr_pred))),
        "r2": float(r2_score(yr_test, svr_pred))
    }

source_df, df = load_data()
try:
    results = train_all(df)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

st.sidebar.markdown("<div style='font-size:2rem;margin:.2rem 0 .45rem'>🌾</div><div style='font-family:Manrope,sans-serif;font-size:1.35rem;font-weight:800;line-height:1.2'>Rice Inventory</div><div style='color:#c3d8ca;font-size:.78rem;margin:.35rem 0 1.1rem'>Predictive Analytics Portal</div><div style='height:1px;background:rgba(255,255,255,.16);margin:.7rem 0 1rem'></div>", unsafe_allow_html=True)
dark_mode = st.sidebar.toggle("Dark mode", value=st.session_state.get("dark_mode", False), key="dark_mode")
if dark_mode:
    px.defaults.template = "plotly_dark"
    st.markdown("""
    <style>
    :root { --paper:#101a16; --ink:#edf5ef; --muted:#aabbb0; --leaf:#8ed0a5; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] { background:#101a16 !important; color:#edf5ef !important; }
    .block-container { color:#edf5ef !important; }
    h1,h2,h3,h4,p,li,label,[data-testid="stMarkdownContainer"] { color:#edf5ef; }
    .eyebrow { color:#91d2a8 !important; }
    .section-title { color:#f0f7f2 !important; }
    .section-subtitle, [data-testid="stCaptionContainer"] p { color:#aabbb0 !important; }
    .kpi-card, div[data-testid="stMetric"], [data-testid="stDataFrame"],
    [data-testid="stTable"], [data-testid="stPlotlyChart"],
    [data-testid="stExpander"] { background:#192820 !important; border-color:#30483a !important; color:#edf5ef !important; }
    .kpi-label, [data-testid="stMetricLabel"] { color:#b5c8bb !important; }
    .kpi-value, [data-testid="stMetricValue"] { color:#c5f0d1 !important; }
    .kpi-note { color:#91d2a8 !important; }
    .insight-card, .page-intro { background:#1a3024 !important; border-color:#355b43 !important; color:#e5f3e9 !important; }
    .insight-card b, .page-intro p { color:#e5f3e9 !important; }
    [data-testid="stSelectbox"] > div > div, [data-testid="stMultiSelect"] > div > div,
    [data-testid="stTextInput"] > div > div, [data-testid="stNumberInput"] > div > div,
    [data-testid="stDateInput"] > div > div { background:#192820 !important; border-color:#405b49 !important; color:#edf5ef !important; }
    input, textarea { color:#edf5ef !important; }
    .stButton>button, .stDownloadButton>button { background:#203c2c !important; color:#eaf6ed !important; border-color:#4d8060 !important; }
    .stButton>button:hover, .stDownloadButton>button:hover { background:#34764d !important; }
    [data-testid="stAlert"] { background:#1a3024 !important; border-color:#355b43 !important; }
    hr { border-color:#34483b !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    px.defaults.template = "plotly_white"
page = st.sidebar.radio("Explore", [
    "Dashboard","Dataset","Visualization","Descriptive Analytics",
    "SVR Regression","Classification Models","Model Comparison",
    "Future Predictions"
])
st.sidebar.markdown("<div style='height:1px;background:rgba(255,255,255,.16);margin:1.1rem 0 .8rem'></div>", unsafe_allow_html=True)
st.sidebar.caption("DATA SOURCE  •  SUPPLIED REPOSITORY CSV")
st.sidebar.info("🌱 Dataset included. No upload required.")

st.markdown('<div class="hero"><div class="eyebrow" style="color:#b8dfc4">DATA INSIGHTS&nbsp;&nbsp;•&nbsp;&nbsp;MACHINE LEARNING</div><h1>Rice Inventory Predictive Analytics</h1><p>Explore historical rice stock patterns, compare machine learning models, and review future inventory classifications by sector and month.</p></div>', unsafe_allow_html=True)

metrics = results["metrics"]
best = metrics.loc[metrics["Accuracy"].idxmax()]
total = df["Rice_Inventory"].sum()
year_min, year_max = int(df["Year"].min()), int(df["Year"].max())

if page == "Dashboard":
    st.markdown('<div class="eyebrow">OVERVIEW</div><div class="section-title">Dashboard at a glance</div><div class="section-subtitle">A quick summary of the historical inventory dataset and model performance.</div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    cards=[
        ("MONTHLY RECORDS",f"{len(df):,}","4 sectors × monthly observations"),
        ("TOTAL INVENTORY",f"{total:,.0f}","Sum of recorded monthly values"),
        ("YEAR COVERAGE",f"{year_min}–{year_max}","Historical data period"),
        ("TOP MODEL ACCURACY",f"{best['Accuracy']:.1%}",str(best["Algorithm"]))
    ]
    for col,(label,value,note) in zip([c1,c2,c3,c4],cards):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="insight-card"><b>Quick insight</b><br>Use the navigation panel to explore sector-level trends, descriptive statistics, regression results, classification performance, and 2026–2030 predictions.</div>',unsafe_allow_html=True)
    annual=df.groupby(["Year","Sector"],as_index=False)["Rice_Inventory"].sum()
    latest=df[df["Year"]==year_max].groupby("Sector",as_index=False)["Rice_Inventory"].sum()
    left,right=st.columns([1.65,1])
    with left:
        fig=px.line(annual,x="Year",y="Rice_Inventory",color="Sector",markers=True,title="Annual inventory trend")
        fig.update_layout(template="plotly_white",height=390,margin=dict(l=18,r=18,t=58,b=18),legend_title_text="Sector",hovermode="x unified")
        fig.update_xaxes(dtick=2,showgrid=False)
        fig.update_yaxes(title="Inventory",gridcolor="#edf1ed",zeroline=False)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=px.bar(latest,x="Rice_Inventory",y="Sector",color="Sector",orientation="h",title=f"Sector totals • {year_max}")
        fig.update_layout(template="plotly_white",height=390,margin=dict(l=10,r=18,t=58,b=18),showlegend=False)
        fig.update_xaxes(title="Inventory",gridcolor="#edf1ed")
        fig.update_yaxes(title="",categoryorder="total ascending",showgrid=False)
        st.plotly_chart(fig,use_container_width=True)
    st.caption("Source: supplied rice inventory dataset. Inventory totals are calculated from the monthly values in the source file.")

elif page == "Dataset":
    st.subheader("Original Dataset")
    st.write(f"Original file: **Rice Monthly Total Stocks Inventory by Sector Year (2000–2025) Month (January–December).csv**")
    a,b,c=st.columns(3)
    a.metric("Original rows",len(source_df))
    b.metric("Original columns",len(source_df.columns))
    c.metric("Melted monthly records",len(df))
    st.dataframe(source_df,use_container_width=True,height=330)
    st.subheader("Structured monthly dataset")
    st.dataframe(df[["Sector","Year","Month","Month_Num","Rice_Inventory"]],
                 use_container_width=True,height=380)
    st.download_button("Download structured monthly CSV",
        df[["Sector","Year","Month","Month_Num","Rice_Inventory"]].to_csv(index=False).encode(),
        "rice_inventory_structured_monthly.csv","text/csv")

elif page == "Visualization":
    st.subheader("Data Visualization")
    totals=df.groupby("Sector",as_index=False)["Rice_Inventory"].sum()
    annual=df.groupby(["Year","Sector"],as_index=False)["Rice_Inventory"].sum()
    pie=totals[totals["Sector"].isin(SECTORS[1:])]
    left,right=st.columns(2)
    with left:
        st.plotly_chart(px.bar(totals,x="Sector",y="Rice_Inventory",color="Sector",
                               title="Total Rice Inventory per Sector (All Sectors)"),use_container_width=True)
    with right:
        st.plotly_chart(px.pie(pie,names="Sector",values="Rice_Inventory",
                               title="Distribution: Household, Commercial & NFA Stocks"),use_container_width=True)
    st.plotly_chart(px.line(annual,x="Year",y="Rice_Inventory",color="Sector",markers=True,
                            title="Yearly Inventory Trend by Sector"),use_container_width=True)
    st.plotly_chart(px.box(df,x="Sector",y="Rice_Inventory",color="Sector",
                           title="Monthly Inventory Distribution by Sector"),use_container_width=True)
    month_avg=df.groupby(["Month_Num","Month"],as_index=False)["Rice_Inventory"].mean().sort_values("Month_Num")
    st.plotly_chart(px.bar(month_avg,x="Month",y="Rice_Inventory",
                           title="Average Inventory by Month"),use_container_width=True)

elif page == "Descriptive Analytics":
    st.subheader("Descriptive Analytics")
    st.dataframe(df["Rice_Inventory"].describe().to_frame("Rice_Inventory").round(2),use_container_width=True)
    by_sector=df.groupby("Sector")["Rice_Inventory"].agg(
        Count="count",Mean="mean",Median="median",Standard_Deviation="std",
        Minimum="min",Maximum="max",Total="sum").reset_index()
    st.subheader("Descriptive Statistics by Sector")
    st.dataframe(by_sector.round(2),use_container_width=True)
    st.plotly_chart(px.box(df,x="Sector",y="Rice_Inventory",color="Sector",
                           title="Inventory Spread by Sector"),use_container_width=True)

elif page == "SVR Regression":
    st.subheader("Support Vector Regression (SVR)")
    st.write("Predictors: Sector, Year, and Month. Target: Rice_Inventory.")
    a,b=st.columns(2)
    a.metric("Root Mean Squared Error (RMSE)",f"{results['rmse']:,.2f}")
    b.metric("R-squared (R²)",f"{results['r2']:.4f}")
    actual=results["yr_test"].to_numpy()
    pred=results["svr_pred"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=actual,y=pred,mode="markers",name="Predicted"))
    lo,hi=float(actual.min()),float(actual.max())
    fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Ideal line",line=dict(dash="dash")))
    fig.update_layout(title="Actual vs. Predicted Rice Inventory",xaxis_title="Actual",yaxis_title="Predicted")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("The notebook's intended SVR predictors include text categories. This website encodes Sector and Month before scaling so the model can train correctly.")

elif page == "Classification Models":
    st.subheader("Classification Model Evaluation")
    chosen=st.selectbox("Select classifier",list(results["models"].keys()))
    y=results["y_test"]
    pred=results["predictions"][chosen]
    row=metrics[metrics["Algorithm"]==chosen].set_index("Algorithm")
    st.dataframe(row.style.format("{:.2%}"),use_container_width=True)
    classes=["Low","Medium","High"]
    cm=confusion_matrix(y,pred,labels=classes)
    st.plotly_chart(px.imshow(cm,x=classes,y=classes,text_auto=True,color_continuous_scale="Blues",
                              labels={"x":"Predicted","y":"Actual","color":"Count"},
                              title=f"{chosen} Confusion Matrix"),use_container_width=True)
    st.subheader("Classification Report")
    report=classification_report(y,pred,labels=classes,output_dict=True,zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3),use_container_width=True)
    model=results["models"][chosen]
    class_order=list(model.named_steps["model"].classes_)
    y_bin=label_binarize(y,classes=class_order)
    probs=model.predict_proba(results["X_test"])
    all_fpr=np.unique(np.concatenate([roc_curve(y_bin[:,i],probs[:,i])[0] for i in range(len(class_order))]))
    mean_tpr=np.zeros_like(all_fpr)
    aucs=[]
    for i in range(len(class_order)):
        fpr,tpr,_=roc_curve(y_bin[:,i],probs[:,i])
        mean_tpr += np.interp(all_fpr,fpr,tpr)
        aucs.append(auc(fpr,tpr))
    mean_tpr/=len(class_order)
    macro_auc=auc(all_fpr,mean_tpr)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=all_fpr,y=mean_tpr,mode="lines",
                             name=f"Macro-average ROC (AUC={macro_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Chance",line=dict(dash="dash")))
    fig.update_layout(title=f"{chosen} — Multiclass Macro-Average ROC Curve",
                      xaxis_title="False Positive Rate",yaxis_title="True Positive Rate")
    st.plotly_chart(fig,use_container_width=True)

elif page == "Model Comparison":
    st.subheader("Four-Algorithm Model Comparison")
    st.dataframe(metrics.set_index("Algorithm").style.format("{:.2%}"),use_container_width=True)
    melted=metrics.melt(id_vars="Algorithm",var_name="Metric",value_name="Score")
    st.plotly_chart(px.bar(melted,x="Metric",y="Score",color="Algorithm",barmode="group",
                           range_y=[0,1],title="Accuracy, Precision, Recall and F1-Score"),
                    use_container_width=True)
    st.success(f"Highest accuracy: {best['Algorithm']} ({best['Accuracy']:.2%}).")
    st.caption("Metrics are computed live using the uploaded repository dataset, qcut class labels, and the notebook's 80/20 split with random_state=42.")

elif page == "Future Predictions":
    st.subheader("Future Inventory Classification — 2026 to 2030")
    st.write("Random Forest predicts the Low, Medium, or High class for each month and sector.")
    future=pd.DataFrame([
        {"Year":y,"Month_Num":m,"Sector":s}
        for y in range(2026,2031) for m in range(1,13) for s in SECTORS
    ])
    future["Predicted_Class"]=results["models"]["Random Forest"].predict(
        future[["Year","Month_Num","Sector"]])
    future["Month"]=future["Month_Num"].map({v:k for k,v in MONTH_MAP.items()})
    st.dataframe(future[["Year","Month","Sector","Predicted_Class"]],
                 use_container_width=True,height=400)
    summary=future.groupby(["Year","Sector","Predicted_Class"]).size().reset_index(name="Months")
    st.plotly_chart(px.bar(summary,x="Year",y="Months",color="Predicted_Class",
                           facet_col="Sector",facet_col_wrap=2,barmode="stack",
                           category_orders={"Predicted_Class":["Low","Medium","High"]},
                           title="Predicted Inventory Class Distribution by Sector"),
                    use_container_width=True)
    st.download_button("Download 2026–2030 predictions",
        future.to_csv(index=False).encode(),"rice_inventory_predictions_2026_2030.csv","text/csv")

st.divider()
st.caption("Rice Inventory Predictive Analytics | Python • Streamlit • Scikit-learn • Plotly")
