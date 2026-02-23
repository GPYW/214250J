from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from model_service import load_or_train_artifacts, predict_batch, predict_single


st.set_page_config(page_title="MLPR Model Dashboard", page_icon="📊", layout="wide")

MANDATORY_FIELDS = [
    "t_lines",
    "churn",
    "lines_added",
    "total_LOC",
    "n_fix",
    "n_auth",
    "age",
    "size",
]

FIELD_UI_META = {
    "project_name": {
        "label": "Repository Identifier (org#repo)",
        "help": "Repository identifier used during mining (e.g., organization#repository).",
    },
    "n_releases": {
        "label": "Number of Releases",
        "help": "Total releases or tags observed for this project.",
    },
    "total_LOC": {
        "label": "Total Lines of Code (LOC)",
        "help": "Total source lines of code in the repository snapshot.",
    },
    "t_lines": {
        "label": "Total Lines Touched",
        "help": "Total lines considered in the mined release window (lines touched/modified).",
    },
    "lines_added": {
        "label": "Lines Added",
        "help": "Number of lines added in the release change set.",
    },
    "max_lines_added": {
        "label": "Max Lines Added (Single Commit)",
        "help": "Highest lines-added value among commits in the release window.",
    },
    "avg_lines_added": {
        "label": "Average Lines Added per Commit",
        "help": "Mean lines added per commit in the release window.",
    },
    "weighted_age": {
        "label": "Weighted Change Age",
        "help": "Recency-weighted age of modified files; higher values indicate older changes.",
    },
    "n_fix": {
        "label": "Fix-related Commits",
        "help": "Count of fix-related commits in the release window.",
    },
    "n_auth": {
        "label": "Distinct Authors/Contributors",
        "help": "Number of distinct authors/contributors in the release window.",
    },
    "churn": {
        "label": "Code Churn",
        "help": "Code churn in the release window (added + deleted or net change by mining definition).",
    },
    "max_churn": {
        "label": "Max Churn (Single Commit)",
        "help": "Highest churn value among commits in the release window.",
    },
    "avg_churn": {
        "label": "Average Churn per Commit",
        "help": "Mean churn per commit in the release window.",
    },
    "max_change_set": {
        "label": "Max Changed Files (Single Commit)",
        "help": "Largest change set size (files changed) among commits.",
    },
    "avg_change_set": {
        "label": "Average Changed Files per Commit",
        "help": "Average number of files changed per commit.",
    },
    "age": {
        "label": "Project Age (Days)",
        "help": "Repository/project age at release time (days since first activity).",
    },
    "size": {
        "label": "Repository Size (KB)",
        "help": "Repository size reported by GitHub.",
    },
    "language": {
        "label": "Primary Language",
        "help": "Primary language reported by GitHub at collection time.",
    },
    "forks_count": {
        "label": "Fork Count",
        "help": "Number of forks reported by GitHub.",
    },
    "stargazers_count": {
        "label": "Stargazer Count",
        "help": "Number of stars reported by GitHub.",
    },
    "watchers_count": {
        "label": "Watcher Count",
        "help": "Number of watchers/subscribers reported by GitHub.",
    },
    "has_issues": {
        "label": "Issues Enabled",
        "help": "Whether GitHub Issues is enabled for the repository.",
    },
    "archived": {
        "label": "Archived Repository",
        "help": "Whether the repository is archived.",
    },
    "disabled": {
        "label": "Disabled Repository",
        "help": "Whether the repository is disabled on GitHub.",
    },
    "fork": {
        "label": "Is Forked Repository",
        "help": "Whether this repository is a fork of another repository.",
    },
    "num_contributors": {
        "label": "Total Contributors",
        "help": "Number of contributors reported by GitHub/collection process.",
    },
    "sbom_flag": {
        "label": "SBOM Artifact Available",
        "help": "Whether an SBOM-related artifact was detected during collection.",
    },
    "num_commits": {
        "label": "Total Commits",
        "help": "Total commits observed/collected for the repository.",
    },
    "num_buggy_commits": {
        "label": "Buggy Commits",
        "help": "Commits identified as buggy by mining heuristics in repository history.",
    },
    "lifetime_days": {
        "label": "Repository Lifetime (Days)",
        "help": "Repository lifetime in days over the collected history window.",
    },
    "num_languages": {
        "label": "Number of Languages",
        "help": "Distinct programming languages detected in the repository.",
    },
    "num_stars": {
        "label": "Alternative Star Count",
        "help": "Alternative star count metric from merged sources.",
    },
    "main_language": {
        "label": "Main Language (Preprocessed)",
        "help": "Main language label after preprocessing/standardization.",
    },
    "num_files": {
        "label": "Number of Files",
        "help": "Files detected in the repository snapshot.",
    },
    "num_methods": {
        "label": "Number of Methods/Functions",
        "help": "Methods or functions detected by static analysis.",
    },
    "num_buggy_files": {
        "label": "Buggy Files",
        "help": "Files identified as buggy in repository history.",
    },
    "num_buggy_methods": {
        "label": "Buggy Methods",
        "help": "Methods identified as buggy in repository history.",
    },
    "num_bugs": {
        "label": "Total Bug Count",
        "help": "Total bug instances mined for the repository.",
    },
}


def get_field_label(col: str) -> str:
    return FIELD_UI_META.get(col, {}).get("label", col.replace("_", " ").title())


def get_field_help(col: str) -> str | None:
    return FIELD_UI_META.get(col, {}).get("help")


def format_importance_table(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    display_df["feature"] = display_df["feature"].map(get_field_label)
    return display_df


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_or_train_artifacts()


def get_risk_band(probability: float) -> str:
    if probability < 0.33:
        return "Low risk"
    if probability < 0.66:
        return "Medium risk"
    return "High risk"


@st.cache_data(show_spinner=False)
def get_effort_thresholds() -> tuple[float, float]:
    effort = pd.read_csv("final_dataset_6000.csv", usecols=["effort_next_lines_added"])[
        "effort_next_lines_added"
    ].dropna()
    q33 = float(effort.quantile(0.33))
    q66 = float(effort.quantile(0.66))
    return q33, q66


def get_effort_band(predicted_effort: float, q33: float, q66: float) -> str:
    if predicted_effort < q33:
        return "Low development effort"
    if predicted_effort < q66:
        return "Medium development effort"
    return "High development effort"


def render_single_prediction():
    artifacts = get_artifacts()
    st.subheader("Single Release Assessment")
    st.caption(
        "Enter release and repository metrics to estimate defect-proneness and forecast next-release effort."
    )

    with st.form("single_prediction_form"):
        numeric_values = {}
        categorical_values = {}

        mandatory_numeric = [c for c in MANDATORY_FIELDS if c in artifacts.numeric_cols]
        mandatory_categorical = [c for c in MANDATORY_FIELDS if c in artifacts.categorical_cols]
        optional_numeric = [c for c in artifacts.numeric_cols if c not in MANDATORY_FIELDS]
        optional_categorical = [c for c in artifacts.categorical_cols if c not in MANDATORY_FIELDS]

        st.markdown("**Mandatory Fields**")
        st.caption("Fill these core inputs first. Use Advanced options for all other optional fields.")

        if mandatory_numeric:
            st.markdown("**Mandatory Numeric Features**")
            num_cols_a, num_cols_b = st.columns(2)
            for idx, col in enumerate(mandatory_numeric):
                default_val = float(artifacts.numeric_defaults.get(col, 0.0))
                holder = num_cols_a if idx % 2 == 0 else num_cols_b
                numeric_values[col] = holder.number_input(
                    label=get_field_label(col),
                    help=get_field_help(col),
                    value=default_val,
                    step=1.0,
                    format="%.6f",
                )

        if mandatory_categorical:
            st.markdown("**Mandatory Categorical Features**")
            cat_cols_a, cat_cols_b = st.columns(2)
            for idx, col in enumerate(mandatory_categorical):
                default_val = artifacts.categorical_defaults.get(col, "")
                options = artifacts.categorical_options.get(col, [])
                holder = cat_cols_a if idx % 2 == 0 else cat_cols_b
                if options:
                    choices = [default_val] + [x for x in options if x != default_val]
                    categorical_values[col] = holder.selectbox(
                        label=get_field_label(col),
                        options=choices,
                        help=get_field_help(col),
                    )
                else:
                    categorical_values[col] = holder.text_input(
                        label=get_field_label(col),
                        value=default_val,
                        help=get_field_help(col),
                    )

        with st.expander("Advanced options (optional)", expanded=False):
            if optional_numeric:
                st.markdown("**Optional Numeric Features**")
                opt_num_cols_a, opt_num_cols_b = st.columns(2)
                for idx, col in enumerate(optional_numeric):
                    default_val = float(artifacts.numeric_defaults.get(col, 0.0))
                    holder = opt_num_cols_a if idx % 2 == 0 else opt_num_cols_b
                    numeric_values[col] = holder.number_input(
                        label=get_field_label(col),
                        help=get_field_help(col),
                        value=default_val,
                        step=1.0,
                        format="%.6f",
                    )

            if optional_categorical:
                st.markdown("**Optional Categorical Features**")
                opt_cat_cols_a, opt_cat_cols_b = st.columns(2)
                for idx, col in enumerate(optional_categorical):
                    default_val = artifacts.categorical_defaults.get(col, "")
                    options = artifacts.categorical_options.get(col, [])
                    holder = opt_cat_cols_a if idx % 2 == 0 else opt_cat_cols_b
                    if options:
                        choices = [default_val] + [x for x in options if x != default_val]
                        categorical_values[col] = holder.selectbox(
                            label=get_field_label(col),
                            options=choices,
                            help=get_field_help(col),
                        )
                    else:
                        categorical_values[col] = holder.text_input(
                            label=get_field_label(col),
                            value=default_val,
                            help=get_field_help(col),
                        )

        submit = st.form_submit_button("Predict")

    if submit:
        payload = {}
        payload.update(numeric_values)
        payload.update(categorical_values)
        output = predict_single(payload, artifacts)
        risk_band = get_risk_band(output["defect_probability"])
        effort_q33, effort_q66 = get_effort_thresholds()
        effort_band = get_effort_band(output["effort_next_lines_added_prediction"], effort_q33, effort_q66)

        pred_label = "Defect-prone" if output["defect_prone_prediction"] == 1 else "Not defect-prone"
        c1, c2, c3 = st.columns(3)
        c1.metric("Classification", pred_label)
        c2.metric("Defect Probability", f'{output["defect_probability"]:.4f}')
        c2.caption(f"({risk_band})")
        c3.metric(
            "Predicted Effort (next lines added)",
            f'{output["effort_next_lines_added_prediction"]:.2f}',
        )
        c3.caption(f"({effort_band})")
        st.caption(
            f"Effort band thresholds from dataset quantiles: low < {effort_q33:.2f}, "
            f"medium < {effort_q66:.2f}, high >= {effort_q66:.2f}."
        )

        st.markdown("**Explanation (Global Feature Importance)**")
        left, right = st.columns(2)
        left.write("Top features for defect classification:")
        left.dataframe(format_importance_table(artifacts.qda_importance.head(10)), use_container_width=True)
        right.write("Top features for effort regression:")
        right.dataframe(format_importance_table(artifacts.krr_importance.head(10)), use_container_width=True)


def render_batch_prediction():
    artifacts = get_artifacts()
    st.subheader("Batch Assessment (CSV)")
    st.caption(
        "Upload a CSV of releases for bulk scoring. Missing input columns are automatically filled using training defaults."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return

    input_df = pd.read_csv(uploaded)
    st.write("Input preview:")
    st.dataframe(input_df.head(20), use_container_width=True)

    predicted = predict_batch(input_df, artifacts)
    st.write("Prediction output:")
    st.dataframe(predicted.head(50), use_container_width=True)

    csv_buffer = io.StringIO()
    predicted.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Predictions CSV",
        data=csv_buffer.getvalue(),
        file_name="predictions.csv",
        mime="text/csv",
    )


def main():
    st.title("Software Quality and Effort Forecasting Dashboard")
    st.write(
        "This application operationalizes the MLPR project models on `final_dataset_6000.csv` to support "
        "release-level risk screening and effort planning. It predicts whether a release is defect-prone and "
        "estimates next-release development effort (lines added), with feature-importance explanations for both tasks."
    )

    artifacts = get_artifacts()
    with st.expander("Model Details", expanded=False):
        st.write("Project scope: release-level software quality classification and effort forecasting.")
        st.write("Dataset: `final_dataset_6000.csv` (process metrics + repository metadata).")
        st.write("Targets:")
        st.write("- `defect_prone` (classification)")
        st.write("- `effort_next_lines_added` (regression)")
        st.write(f"Input feature count: {len(artifacts.feature_cols)}")
        st.write(f"Numeric features: {len(artifacts.numeric_cols)}")
        st.write(f"Categorical features: {len(artifacts.categorical_cols)}")
        st.write("Classification model: Quadratic Discriminant Analysis (tuned)")
        st.write("Effort model: Kernel Ridge Regression with RBF kernel (tuned)")

    tabs = st.tabs(["Single Input", "Batch CSV"])
    with tabs[0]:
        render_single_prediction()
    with tabs[1]:
        render_batch_prediction()


if __name__ == "__main__":
    main()
