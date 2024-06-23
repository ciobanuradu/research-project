import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def read_result(filepaths):
    df1 = pd.read_csv(filepaths[0])
    df2 = pd.read_csv(filepaths[1])
    df3 = pd.read_csv(filepaths[2])
    df4 = pd.read_csv(filepaths[3])
    df = pd.concat([df1, df2, df3, df4])
    return df.melt(id_vars=["env"], value_vars=["continuous_low_gamma", "continuous_high_gamma", "parameter_drift", "induced_decoherence"], var_name="Environment used for training", value_name="Episode Length")

mapdict1 = {
    "continuous_low_gamma": "Continuous low gamma",
    "continuous_high_gamma": "Continuous high gamma",
    "parameter_drift": "Parameter drift",
    "induced_decoherence": "Induced decoherence"
}
mapdict2 = {
    "Continuous_low_gamma": "Continuous low gamma",
    "Continuous_high_gamma": "Continuous high gamma",
    "parameter_drift": "Parameter drift",
    "induced_decoherence": "Induced decoherence"
}

for j in [0, 2, 4, 5, 6, 7]:
    df = read_result([os.path.join(os.curdir, "cross_evaluations", f"cross_eval_{j}_{i}") for i in range(4)])
    df["Episode Length"] = df["Episode Length"].abs()
    
    df = df.replace({"Environment used for training": mapdict1, "env": mapdict2}).rename(columns = {"env": "Environment used for evaluation"})
    
    
    # print(df)
    # sns.set_theme(style="ticks", palette="pastel")

    ax = sns.boxplot(x="Environment used for evaluation", y="Episode Length",
                hue="Environment used for training", palette=["b", "orange", "g", "r"],
                data=df)
    sns.move_legend(ax, loc="upper right")
    sns.despine(offset=10, trim=True)
    plt.show()

    
    df["Episode Length"] = df.groupby("Environment used for evaluation")["Episode Length"].transform(lambda x: (x / (x.max())))
    # print(df)
    ax = sns.boxplot(x="Environment used for evaluation", y="Episode Length",
                hue="Environment used for training", palette=["b", "orange", "g", "r"],
                data=df)
    ax.set(ylabel="Episode Length (Normalized)")
    sns.move_legend(ax, loc="upper right")
    sns.despine(offset=10, trim=True)
    plt.show()
    # tips = sns.load_dataset("tips")
    # print(tips)
