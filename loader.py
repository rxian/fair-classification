import os
import pickle
import urllib.request

import datasets
import folktables
import numpy as np
import pandas as pd


def load_acsincome(data_dir,
                   n_classes=2,
                   sensitive_attr='SEX',
                   remove_sensitive_attr=False):
  target = 'PINCP'
  features = [
      'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX',
      'RAC1P'
  ]
  categories = {
      "COW": {
          1.0: ("Employee of a private for-profit company or"
                "business, or of an individual, for wages,"
                "salary, or commissions"),
          2.0: ("Employee of a private not-for-profit, tax-exempt,"
                "or charitable organization"),
          3.0:
              "Local government employee (city, county, etc.)",
          4.0:
              "State government employee",
          5.0:
              "Federal government employee",
          6.0: ("Self-employed in own not incorporated business,"
                "professional practice, or farm"),
          7.0: ("Self-employed in own incorporated business,"
                "professional practice or farm"),
          8.0:
              "Working without pay in family business or farm",
          9.0:
              "Unemployed and last worked 5 years ago or earlier or never worked",
      },
      "SCHL": {
          1.0: "No schooling completed",
          2.0: "Nursery school, preschool",
          3.0: "Kindergarten",
          4.0: "Grade 1",
          5.0: "Grade 2",
          6.0: "Grade 3",
          7.0: "Grade 4",
          8.0: "Grade 5",
          9.0: "Grade 6",
          10.0: "Grade 7",
          11.0: "Grade 8",
          12.0: "Grade 9",
          13.0: "Grade 10",
          14.0: "Grade 11",
          15.0: "12th grade - no diploma",
          16.0: "Regular high school diploma",
          17.0: "GED or alternative credential",
          18.0: "Some college, but less than 1 year",
          19.0: "1 or more years of college credit, no degree",
          20.0: "Associate's degree",
          21.0: "Bachelor's degree",
          22.0: "Master's degree",
          23.0: "Professional degree beyond a bachelor's degree",
          24.0: "Doctorate degree",
      },
      "MAR": {
          1.0: "Married",
          2.0: "Widowed",
          3.0: "Divorced",
          4.0: "Separated",
          5.0: "Never married or under 15 years old",
      },
      "SEX": {
          1.0: "Male",
          2.0: "Female"
      },
      "RAC1P": {
          1.0: "White alone",
          2.0: "Black or African American alone",
          3.0: "American Indian alone",
          4.0: "Alaska Native alone",
          5.0: ("American Indian and Alaska Native tribes specified;"
                "or American Indian or Alaska Native,"
                "not specified and no other"),
          6.0: "Asian alone",
          7.0: "Native Hawaiian and Other Pacific Islander alone",
          8.0: "Some Other Race alone",
          9.0: "Two or More Races",
      },
  }

  # Download or load the dataset
  get_data_fn = lambda: folktables.ACSDataSource(
      survey_year='2018',
      horizon='1-Year',
      survey='person',
  ).get_data(download=True)
  raw_dataset = cache_dataset(f"{data_dir}/raw_dataset.pkl", get_data_fn)
  df = folktables.adult_filter(raw_dataset)

  if n_classes == 2:
    label_names = ["<=50K", ">50K"]
    target_transform = lambda x: (x > 50000).astype(int)

  else:
    # Compute empirical CDF of PINCP
    x = np.sort(df[target])
    y = np.arange(len(x)) / float(len(x))

    # Partition into bins containing roughly the same number of samples
    partitions = np.array([
        x[np.argmax(y >= q)] for q in np.arange(1 / n_classes, 1, 1 / n_classes)
    ] + [np.inf])

    label_names = [f'[0, {partitions[0]})'] + [
        f'[{partitions[i]}, {partitions[i+1]})'
        for i in range(len(partitions) - 1)
    ]
    target_transform = lambda x: np.argmax(
        np.array(x)[:, None] < partitions[None, :], axis=1)

  if sensitive_attr == 'RAC1P':
    # Combine RAC1P categories 3, 4, 5, and 6, 7, and 8, 9 into new categories
    # 10, 11, and 12 respectively, due to small sample size in some groups.
    # This is also consistent with the UCI Adult dataset.
    categories['RAC1P'][10.0] = "American Indian or Alaska Native alone"
    categories['RAC1P'][
        11.0] = "Asian, Native Hawaiian or Other Pacific Islander alone"
    categories['RAC1P'][12.0] = "Other"
    df['RAC1P'] = df['RAC1P'].replace([3.0, 4.0, 5.0], 10.0)
    df['RAC1P'] = df['RAC1P'].replace([6.0, 7.0], 11.0)
    df['RAC1P'] = df['RAC1P'].replace([8.0, 9.0], 12.0)

  data, labels, groups = folktables.BasicProblem(
      features=features,
      target=target,
      target_transform=target_transform,
      group=sensitive_attr,
      postprocess=lambda x: np.nan_to_num(x, -1),
  ).df_to_pandas(df, categories=categories, dummies=True)

  labels = labels.values.squeeze()
  groups = groups.values.squeeze()

  group_names, groups = np.unique(groups, return_inverse=True)
  group_names = [categories[sensitive_attr][n] for n in group_names]

  if remove_sensitive_attr:
    data.drop(columns=list(data.filter(regex=f'^{sensitive_attr}')),
              inplace=True)

  return data, labels, label_names, groups, group_names


def load_biasbios(data_dir, add_sensitive_attribute=False):
  label_names = [
      "accountant", "architect", "attorney", "chiropractor", "comedian",
      "composer", "dentist", "dietitian", "dj", "filmmaker",
      "interior_designer", "journalist", "model", "nurse", "painter",
      "paralegal", "pastor", "personal_trainer", "photographer", "physician",
      "poet", "professor", "psychologist", "rapper", "software_engineer",
      "surgeon", "teacher", "yoga_teacher"
  ]
  group_names = ["female", "male"]

  features = datasets.Features({
      "bio": datasets.Value("string"),
      "title": datasets.ClassLabel(names=label_names),
      "gender": datasets.ClassLabel(names=group_names),
  })

  train_path = f"{data_dir}/train.pickle"
  test_path = f"{data_dir}/test.pickle"
  dev_path = f"{data_dir}/dev.pickle"
  if any(not os.path.exists(p) for p in [train_path, test_path, dev_path]):
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle",
        train_path)
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle",
        test_path)
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle",
        dev_path)

  rows = {k: [] for k in features}
  for split, path in zip(["train", "test", "dev"],
                         [train_path, test_path, dev_path]):
    with open(path, "rb") as pickle_file:
      for row in pickle.load(pickle_file):
        rows["gender"].append("female" if row["g"] == "f" else "male")
        rows["title"].append(row["p"])
        if add_sensitive_attribute:
          rows["bio"].append(rows["gender"][-1].capitalize() + ". " +
                             row["hard_text_untokenized"])
        else:
          rows["bio"].append(row["hard_text_untokenized"])

  raw_dataset = datasets.Dataset.from_dict(rows, features=features)
  labels = np.array(raw_dataset["title"])
  groups = np.array(raw_dataset["gender"])

  return raw_dataset, labels, label_names, groups, group_names


def load_adult(data_dir, sensitive_attrs=['Sex'], remove_sensitive_attr=False):
  features = [
      "Age", "Workclass", "fnlwgt", "Education", "Education-Num",
      "Martial Status", "Occupation", "Relationship", "Race", "Sex",
      "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"
  ]

  # Download data
  train_path = f"{data_dir}/adult.data"
  test_path = f"{data_dir}/adult.test"
  if any([not os.path.exists(p) for p in [train_path, test_path]]):
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        train_path)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        test_path)

  original_train = pd.read_csv(train_path,
                               names=features,
                               sep=r"\s*,\s*",
                               engine="python",
                               na_values="?")
  original_test = pd.read_csv(test_path,
                              names=features,
                              sep=r"\s*,\s*",
                              engine="python",
                              na_values="?",
                              skiprows=1)
  original = pd.concat([original_train, original_test])
  original.drop(["fnlwgt"], inplace=True, axis=1)

  # Binarize class labels, and remove it from the input data
  labels_original = original[["Target"
                             ]].replace("<=50K.",
                                        "<=50K").replace(">50K.", ">50K")
  original.drop(["Target"], inplace=True, axis=1)

  groups = original[sensitive_attrs[0]]
  for attribute in sensitive_attrs[1:]:
    groups = np.add(np.add(groups, ", "), original[attribute])

  # Encode labels and groups
  label_names, labels = np.unique(labels_original, return_inverse=True)
  group_names, groups = np.unique(groups, return_inverse=True)

  if remove_sensitive_attr:
    for sensitive_attr in sensitive_attrs:
      original.drop(columns=list(original.filter(regex=f'^{sensitive_attr}')),
                    inplace=True)

  # Encode categorical columns
  data = pd.get_dummies(original)

  return data, labels, label_names, groups, group_names


def load_communities(data_dir, n_classes=5, remove_sensitive_attr=False):
  features = [
      "state", "county", "community", "communityname", "fold", "population",
      "householdsize", "racepctblack", "racePctWhite", "racePctAsian",
      "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up",
      "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf",
      "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc",
      "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap",
      "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov",
      "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed",
      "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu",
      "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv",
      "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
      "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom",
      "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5",
      "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
      "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly",
      "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup",
      "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous",
      "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR",
      "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded",
      "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb",
      "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ",
      "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc",
      "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet",
      "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85",
      "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps",
      "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop",
      "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite",
      "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor",
      "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked",
      "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg",
      "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn",
      "PolicBudgPerPop", "ViolentCrimesPerPop"
  ]

  data_path = f"{data_dir}/communities.data"
  if not os.path.exists(data_path):
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
        data_path)

  original = pd.read_csv(data_path,
                         names=features,
                         sep=r",",
                         engine="python",
                         na_values="?")

  # Drop community name, state, and county, and columns with missing values
  original = original.drop(["communityname", "state", "county"],
                           axis=1).dropna(axis=1)

  # Create equidistance bins for ViolentCrimesPerPop column
  labels_original = pd.cut(original["ViolentCrimesPerPop"], n_classes)

  # Define a new sensitive attribute called "MinorityPresence"
  minority_pct = np.stack([
      original[a].to_numpy()
      for a in ["racePctHisp", "racePctAsian", "racepctblack"]
  ],
                          axis=1)
  minority_presence = np.array(["hispanic", "asian",
                                "black"])[minority_pct.argmax(axis=1)]
  minority_presence[original["racePctWhite"] > 0.95] = "white"
  original["MinorityPresence"] = minority_presence

  # Remove fold and target
  original.drop(columns=["ViolentCrimesPerPop", "fold"], inplace=True)

  # Encode labels and groups
  label_names, labels = np.unique(labels_original, return_inverse=True)
  group_names, groups = np.unique(original["MinorityPresence"],
                                  return_inverse=True)

  if remove_sensitive_attr:
    original.drop(
        columns=[
            "MinorityPresence", "racePctHisp", "racePctAsian", "racepctblack"
        ],
        inplace=True,
    )

  # Encode categorical columns
  data = pd.get_dummies(original)

  return data, labels, label_names, groups, group_names


def load_compas(data_dir, remove_sensitive_attr=False):
  data_path = f"{data_dir}/compas-scores-two-years.csv"
  if not os.path.exists(data_path):
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        data_path)

  df = pd.read_csv(data_path)

  # select features for analysis
  df = df[[
      'age', 'c_charge_degree', 'race', 'sex', 'priors_count',
      'days_b_screening_arrest', 'is_recid', 'c_jail_in', 'c_jail_out'
  ]]

  # drop missing/bad features (following ProPublica's analysis)
  # ix is the index of variables we want to keep.

  # Remove entries with inconsistent arrest information.
  ix = df['days_b_screening_arrest'] <= 30
  ix = (df['days_b_screening_arrest'] >= -30) & ix

  # remove entries entries where compas case could not be found.
  ix = (df['is_recid'] != -1) & ix

  # remove traffic offenses.
  ix = (df['c_charge_degree'] != "O") & ix

  # trim dataset
  df = df.loc[ix, :]

  # create new attribute "length of stay" with total jail time.
  df['length_of_stay'] = (
      pd.to_datetime(df['c_jail_out']) -
      pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)

  # drop 'c_jail_in' and 'c_jail_out'
  # drop columns that won't be used
  dropCol = ['c_jail_in', 'c_jail_out', 'days_b_screening_arrest']
  df.drop(dropCol, inplace=True, axis=1)

  # keep only African-American and Caucasian
  df = df.loc[df['race'].isin(['African-American', 'Caucasian']), :]

  # reset index
  df.reset_index(inplace=True, drop=True)

  # Binarize class labels, and remove it from the input data
  labels_original = df["is_recid"].replace(0, "No").replace(1, "Yes")
  df.drop(["is_recid"], inplace=True, axis=1)

  # Encode labels and groups
  label_names, labels = np.unique(labels_original, return_inverse=True)
  group_names, groups = np.unique(df["race"], return_inverse=True)

  if remove_sensitive_attr:
    df.drop(columns=["race"], inplace=True)

  # Encode categorical columns
  data = pd.get_dummies(df)

  return data, labels, label_names, groups, group_names


def cache_dataset(path, get_data_fn):
  if os.path.exists(path):
    with open(path, "rb") as f:
      data = pickle.load(f)
  else:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    data = get_data_fn()
    with open(path, "wb") as f:
      pickle.dump(data, f)
  return data


def dataset_stats(labels, label_names, groups, group_names):
  df = pd.DataFrame(np.stack([groups, labels], axis=1),
                    columns=["Group", "Target"])
  df_grouped = df.groupby(["Target", "Group"]).size().unstack()
  df_grouped.rename(
      index=dict(enumerate(label_names)),
      columns=dict(enumerate(group_names)),
      inplace=True,
  )
  return df_grouped
