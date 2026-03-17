# github/basic-scripts/pipeline/rfe_survival_cv.py
import argparse, json, os, sys, time, hashlib, math, random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import ast
import csv
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.feature_selection import VarianceThreshold
import warnings
from lifelines import CoxPHFitter
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")
from collections import Counter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.utils import resample
from scipy.stats import mannwhitneyu


def load_yaml(p): 
    with open(p, "r") as f: 
        return yaml.safe_load(f)

def load_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    sep = "," if p.suffix.lower()==".csv" else "\t"
    return pd.read_csv(p, sep=sep)


def load_feature_list(dir_: Path, name: str):
    txt = (dir_ / f"{name}.txt").read_text().splitlines()
    return [x.strip() for x in txt if x.strip()]





def create_XY_response(Main_df, feature_index, event_col=None, time_col=None, response_col=None):  
    ## if response_col does not equal NA
    if response_col is not None:
        # join response and features
        response_and_features = list(set(feature_index) | {response_col})
        # 'clean df by removing any rows with NA
        Main_df_cleaned_response = Main_df.dropna(subset=response_and_features)
        # return clean target array and features_array
        target_array = Main_df_cleaned_response[response_col].values
        features_array_response = Main_df_cleaned_response[feature_index]
    
    # if event and time cols do not equal NA
    if event_col is not None and time_col is not None:
        # define event and type
        event_time = [event_col, time_col]
        event_type = [(event_col, '?'), (time_col, '<f8')]
        # join features and outcomes
        survival_and_features = list(set(feature_index) | set(event_time))
        # remove rows with NA
        Main_df_cleaned_survival = Main_df.dropna(subset=survival_and_features)
        # create time array and features array
        time_array = Main_df_cleaned_survival[event_time].to_numpy()
        time_array_new = np.array([tuple(row) for row in time_array], dtype=event_type)
        event_indicator = time_array_new[event_col]
        time_to_event = time_array_new[time_col]
        features_array_survival = Main_df_cleaned_survival[feature_index]
        
    # Combine features only if both response and survival are requested
    if event_col and time_col and response_col:
        common_features = features_array_response.columns.intersection(features_array_survival.columns)
        features_array = Main_df_cleaned_response[common_features]
    elif response_col:
        features_array = features_array_response
        event_indicator = None
        time_to_event = None
    elif event_col and time_col:
        features_array = features_array_survival
        target_array = None

    return event_indicator, time_to_event, features_array, target_array


# def remove_correlated():
# easier pre-built function:
tr = DropCorrelatedFeatures(variables=None,
                            method='spearman',
                            threshold=0.9)


# def transformations
def log_transforms(df, featureset, suffix="_log"):
    # Adds log-transformed versions of selected features
    # keeps original features
    log_cols = {}

    for each in featureset:
        x = df[each]

        # fix if negative
        min_val = x.min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-6
            x = x + shift

        log_cols[each + suffix] = np.log(x)

    out_df = pd.concat([df, pd.DataFrame(log_cols, index=df.index)], axis=1)
    return out_df


def bootstrap_auc(y_true, probs, n_boot=1000, random_state=0):
    rng = np.random.default_rng(random_state)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # sample with replacement
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], probs[idx]))
    aucs = np.array(aucs)
    return np.percentile(aucs, [2.5, 50, 97.5]), aucs


def bootstrap_cindex(time, event, risk, n_boot=1000, random_state=0):
    rng = np.random.default_rng(random_state)
    cidxs = []
    n = len(time)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # compute c-index on bootstrap sample
        cidx = concordance_index(time[idx], -risk[idx], event[idx])
        cidxs.append(cidx)
    cidxs = np.array(cidxs)
    return np.percentile(cidxs, [2.5, 50, 97.5]), cidxs



# def cross_validation_RFE():
def stability_selection_cox(X_surv, y_surv, X_logistic, y_logistic,
    output_file, num_features):

    out_file_models = f"{output_file}_models.csv"
    out_file_fractions = f"{output_file}_fractions.csv"

    # Initialize Logistic Regression

    # Scale upfront
    scaler = StandardScaler()
    scaler.fit(X_logistic) # Fit scaler on training data only

    X_logit_scaled = pd.DataFrame(scaler.transform(X_logistic),
                                  columns=X_logistic.columns,
                                  index=X_logistic.index)
    X_surv_scaled = pd.DataFrame(scaler.transform(X_surv),
                                 columns=X_surv.columns,
                                 index=X_surv.index)


    Included_Features = []

    # Open CSV file for writing results
    with open(out_file_models, mode='w', newline='') as file:
        writer = csv.writer(file)
        out_row_names = ['Iteration']
        for feat_rank in range(1, num_features + 1):
            # print(feat_rank)
            out_row_names.append(f"Feature {feat_rank}")
            out_row_names.append(f"Coeff  {feat_rank}")

        writer.writerow(out_row_names) #write headers to output file


        for random_state in range(1, 101):
            # print(f'Running stability selection with random state: {random_state}')
            ## Randomly select half of the data
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
            for index, _ in cv.split(X_logit_scaled, y_surv['event']):
                X_sub = X_logit_scaled.iloc[index]
                Y_sub = y_logistic[index]
                df = X_sub.copy()
                
                #Initialize Logistic Regression
                log_reg = LogisticRegression(penalty=None,
                                             solver='lbfgs',
                                             max_iter=1000)


                log_reg.fit(X_sub, Y_sub)

                beta = pd.Series(log_reg.coef_.ravel(), index=X_sub.columns)

                ordered_beta = beta.loc[
                    beta.abs().sort_values(ascending=False).index]

                ordered_list = list(ordered_beta.items())

                output_format = []
                for each_feature in range(0, num_features):
                    output_format.append(ordered_list[each_feature][0])
                    output_format.append(ordered_list[each_feature][1])
                    Included_Features.append(ordered_list[each_feature][0])

                writer.writerow([random_state] + output_format)

        counts = Counter(Included_Features)
        ordered_counts = dict(counts.most_common())

        fractions = {}
        for item in ordered_counts:
            fractions[item] = ordered_counts[item] / 200

        ordered_fractions = dict(
            sorted(fractions.items(), key=lambda x: x[1], reverse=True))

        with open(out_file_fractions, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Feature", "Count", "Fraction"])

            for key, value in ordered_fractions.items():
                writer.writerow([key, ordered_counts[key], value])

        ## re-fit a logistic regression model with the first 3 features
        selected_features = list(ordered_counts.keys())[0:3]
        print(selected_features)
        X_select = X_logit_scaled[selected_features]


        #Initialize Logistic Regression
        log_reg = LogisticRegression(penalty=None,
                                     solver='lbfgs',
                                     max_iter=1000)

        #Fit the log reg model
        # align y to X_select index in case indices differ
        y_aligned = y_logistic.loc[X_select.index] if hasattr(y_logistic, "loc") else pd.Series(y_logistic, index=X_select.index)
        log_reg.fit(X_select, y_aligned)

        ## Pull effect size from the model
        effect = pd.Series(log_reg.coef_.ravel(), index=selected_features)
        print("Coefficients (effect size):")
        print(effect)

        # USE RISK SCORES TO FIT MODEL..
        risk_linear = X_select.values @ effect.values   # shape (20,)
        risk_log_reg = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
        risk_log_reg.fit(risk_linear.reshape(-1, 1), y_aligned)

        print("Logistic-on-risk coefficients (intercept, slope):")
        print(risk_log_reg.intercept_, risk_log_reg.coef_.ravel())



        # Predictions/probabilities
        probs = risk_log_reg.predict_proba(risk_linear.reshape(-1, 1))[:, 1]

        auc = roc_auc_score(y_aligned, probs)
        print(f"Logistic AUC on full training set: {auc:.3f}")

        ## Save Predicted response
        probs_df = pd.DataFrame({"Response": y_aligned,
                                 "RiskLinear": risk_linear,
                                 "Prob": probs}, index=X_select.index)

        probs_df.to_csv(f"{output_file}_ResponseProbs.csv", index=False)

        ## Add boxplot of risk scores by response
        risk0 = probs_df.loc[probs_df["Response"] == 0, "RiskScore"]
        risk1 = probs_df.loc[probs_df["Response"] == 1, "RiskScore"]

        stat, p_value = mannwhitneyu(risk0, risk1, alternative="two-sided")

        plt.figure(figsize=(5, 5))

        plt.boxplot([risk0, risk1],
                    labels=["NR", "R"],
                    showfliers=True)

        plt.ylabel("Risk score")
        plt.title("Risk score by group")
        # y-position for annotation
        y_max = max(risk0.max(), risk1.max())
        y_min = min(risk0.min(), risk1.min())
        y = y_max + 0.1 * (y_max - y_min)

        # draw line
        plt.plot([1, 1, 2, 2], [y, y*1.02, y*1.02, y], lw=1.5)

        # add text
        plt.text(1.5, y*1.03,
                 f"p = {p_value:.3f}",
                 ha="center",
                 va="bottom")

        plt.tight_layout()
        plt.savefig(f"{output_file}_boxplot.png")









        ## Calculate C Index for Holdout data!
        risk_surv = X_select[selected_features].values @ effect.values
        print(risk_surv)

        df_cph = pd.DataFrame({"time": y_surv["time"],
                               "event": y_surv["event"],
                               "risk": risk_surv}, index=X_surv_scaled.index)

        # Fit CPH model: outcome ~ risk
        cph = CoxPHFitter()
        cph.fit(df_cph[["time", "event", "risk"]], duration_col="time", event_col="event")


        c_index = concordance_index(df_cph["time"], - df_cph["risk"], df_cph["event"])
        print(f"C-index (survival set): {c_index:.3f}")


        ## Create survival curves 
        # median split
        median_risk = np.median(risk_surv)
        group_low = risk_surv <= median_risk
        group_high = ~group_low

        # get arrays for times/events
        time = df_cph["time"].values
        event = df_cph["event"].values

        ## Calculate logrank test p value
        res = logrank_test(time[group_low], time[group_high],
                           event_observed_A=event[group_low],
                           event_observed_B=event[group_high])
        print("Log-rank p-value:", res.p_value)

        kmf_low = KaplanMeierFitter()
        kmf_high = KaplanMeierFitter()

        plt.figure(figsize=(7,5))
        kmf_low.fit(time[group_low], event_observed=event[group_low], label="Low risk (<= median)")
        kmf_high.fit(time[group_high], event_observed=event[group_high], label="High risk (> median)")
        kmf_low.plot(ci_show=True)
        kmf_high.plot(ci_show=True)



        plt.title("KM curves by median-split risk")
        plt.xlabel("Time")
        plt.ylabel("Survival probability")
        plt.legend()

        textstr = (f"C-index = {c_index:.2f}\n"
                   f"Log-rank p = {res.p_value:.3f}")

        plt.text(0.02, 0.02, textstr,
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 verticalalignment="bottom",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()

        plt.savefig(f"{output_file}_KM.png")



      

        ## Bootstrap C Index, AUC with re-samples
        # Example: 1000 boots (can be slow)
        ci_auc, aucs = bootstrap_auc(y_logistic, probs, n_boot=100, random_state=2)
        print("AUC 95% CI:", ci_auc[[0,2]])

        ci_cindex, cidxs = bootstrap_cindex(time, event, risk_surv, n_boot=100, random_state=2)
        print("C-index 95% CI:", ci_cindex[[0,2]])



        








def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--settings", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--out_root", default="../runs")
    args = ap.parse_args()

    data_cfg = load_yaml(args.data)
    matrix = pd.read_csv(args.matrix)


    # print("Matrix shape:", matrix.shape)

    # print(matrix.head(2).to_string(index=False))

    for idx, row in matrix.iterrows():

        dataset = str(row["dataset"])
        featureset = str(row["featureset"])

        data_path_surv = Path(data_cfg["datasets"][str(row["dataset"])]["path"])
        df_surv = load_table(data_path_surv)

        data_path_logit = Path(data_cfg["datasets"][str(row["logitdata"])]["path"])
        df_logit = load_table(data_path_logit)

        feats = load_feature_list(Path(args.feature_dir), featureset)
        feats_present = [c for c in feats if c in df_surv.columns]

        df_surv = df_surv.loc[df_surv['Arm'] == 'Arm 1']
        # print(df_surv['Arm'])

        event, time, X_surv_df, y_surv = create_XY_response(df_surv, feats_present,
                                                     event_col=row["event"], 
                                                     time_col=row["time"])
        
        _, _, X_lr_df, y_lr = create_XY_response(df_logit, feats_present,
                                        response_col=row["logit_response"])

       ## Remove correlated features
        ## Make sure to keep 'transf HPV16/18 copies per ml of plasma D1'
        #no_corr = ['transf HPV16/18 copies per ml of plasma D1', 'HPV16/18 copies per ml of plasma D1']
        #cols_for_corr = [c for c in X_train.columns if c not in no_corr]

        X_surv_drop = tr.fit_transform(X_surv_df[X_surv_df.columns])
        #drop = ['LAP TGF-beta-1', 'Eosinophil Abs  D1', 'WBC D1', 'sCD27/IL8 D1']
        #X_train_drop_corr = X_train.drop(columns=drop, errors='ignore')
        X_lr_drop = X_lr_df[X_surv_drop.columns]

        dropped_features = list(set(X_surv_df.columns) - set(X_surv_drop.columns))
        print("Dropped features:", dropped_features)

        corr = X_surv_df.corr(method="spearman")

        # Print the top absolute correlations
        tri = corr.where(~np.tril(np.ones(corr.shape), k=0).astype(bool))
        pairs = (
            tri.stack()
            .abs()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"level_0":"feat1","level_1":"feat2",0:"abs_rho"})
        )
        print("top correlations:")
        print(pairs.head(10).to_string(index=False))

        ## Transformation??
        # X_train_with_logs = log_transforms(X_train_drop, cols_for_transform)
        # print(X_train_with_logs.columns.tolist())
        # X_test_with_logs = log_transforms(X_test_drop, cols_for_transform)

        y_surv = np.array(list(zip(event, time)),
                   dtype=[('event', '?'), ('time', '<f8')])

        stamp = datetime.now().strftime("%Y-%m-%d_%H")
        out_dir = Path(args.out_root) / "metrics" / stamp  / f"Stability_Selection_CPH_{dataset}__{featureset}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"Stability_Selection_OUT"


        ## CPH-based RFE
        folds = int(row["folds"])

        stability_selection_cox(X_surv_drop, y_surv, 
                                X_lr_drop, y_lr,
                                out_file, folds)

 
        save_dir = Path("data-wrangle/TrainTestSets")     
        save_dir.mkdir(parents=True, exist_ok=True)

        #X_train_drop.to_csv(save_dir / f"{featureset}_X_train_soluble_logs_raw_ctDNA_surv.csv", index=False)
        #pd.DataFrame(y_train).to_csv(save_dir / f"{featureset}_y_train_surv.csv", index=False)
        #X_test_drop.to_csv(save_dir / f"{featureset}_X_test_soluble_logs_raw_ctDNA_surv.csv", index=False)
        #pd.DataFrame(y_test).to_csv(save_dir / f"{featureset}_y_test_surv.csv", index=False)



if __name__ == "__main__":
    main()


