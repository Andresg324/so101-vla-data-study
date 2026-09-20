#!/usr/bin/env python3
"""
tools/export_results.py

Regenerate every derived results file from the master workbook. The workbook is
the only file that should ever be edited by hand; everything under documents/
that ends in .csv is produced here.

    documents/results_raw_two_seeds.xlsx
    ├── sheet "results"              -> results_full.csv     600 rows, all columns
    │                                -> results.csv          600 rows, 5 columns
    │                                -> results_seed1000.csv
    │                                -> results_seed2000.csv
    └── sheet "exploratory results"  -> exploratory.csv      107 rows
    └── sheet "desnity results"      -> density.csv      610 rows

                                     -> results_all.csv      707 rows, both concatenated

Validation runs before anything is written. If a check fails nothing is
overwritten, so a bad hand edit cannot propagate.

RUN: python tools/export_results.py
"""

import os
import sys

import pandas as pd

'''

'''

DATASETS = []

XL = "documents/results_raw_two_seeds.xlsx"
OUT = "documents"
SHEET_REG = "results"
SHEET_EXP = "exploratory results"
SHEET_DEN = "density_sweep"

ROLLOUTS = [
    # Fixed Step Rollouts
    "density50-fixedstep_new_positions_20260911_013555"     ,
    "density25-fixedstep_new_positions_20260911_010438"     ,

    "density50-fixedstep_spot_check_20260910_223224"        ,
    "density25-fixedstep_spot_check_20260910_222647"        ,

    "density50-fixedstep_distractors_20260910_213949"       ,
    "density25-fixedstep_distractors_20260910_211803"       ,

    "density50-fixedstep_in_distribution_20260910_191903"   ,
    "density25-fixedstep_in_distribution_20260910_190343"   ,

    # Density Sweep Seed 2000 rollouts
    "density50-seed2000_new_positions_20260911_003419"      ,
    "density25-seed2000_new_positions_20260911_000222"      ,
    "density10-seed2000_new_positions_20260910_231537"      ,
    "density5-seed2000_new_positions_20260910_224247"       ,

    "density50-seed2000_spot_check_20260910_222115"         ,
    "density25-seed2000_spot_check_20260910_221548"         ,
    "density10-seed2000_spot_check_20260910_220954"         ,
    "density5-seed2000_spot_check_20260910_220229"          ,

    "density50-seed2000_distractors_20260910_202819"        ,
    "density25-seed2000_distractors_20260910_201628"        ,
    "density10-seed2000_distractors_20260910_195455"        ,
    "density5-seed2000_distractors_20260910_193509"         ,

    "density50-seed2000_in_distribution_20260910_184945"    ,
    "density25-seed2000_in_distribution_20260910_183836"    ,
    "density10-seed2000_in_distribution_20260910_182627"    ,
    "density5-seed2000_in_distribution_20260910_181456"     ,

    # First Density Sweep Rollout
    "density50_new_positions_20260910_162329"               ,
    "density25_new_positions_20260910_154959"               ,
    "density10_new_positions_20260910_151501"               ,
    "density5_new_positions_20260910_143853"                ,

    "density50_spot_check_20260910_143146"                  ,
    "density25_spot_check_20260910_141710"                  ,
    "density10_spot_check_20260910_141111"                  ,
    "density5_spot_check_20260910_140101"                   ,

    "density50_distractors_20260910_134413"                 ,
    "density25_distractors_20260910_132639"                 ,
    "density10_distractors_20260910_130836"                 ,
    "density5_distractors_20260910_124827"                  ,

    "density50_in_distribution_20260910_123053"             ,
    "density25_in_distribution_20260910_121957"             ,
    "density10_in_distribution_20260910_120741"             ,
    "density5_in_distribution_20260910_115543"              ,

    # Density Probe
    "density_new_positions_20260822_214327"                 ,
    "density_trained_t2_20260822_213229"                    ,
    "density_in_distribution_20260822_212155"               ,

    # Distance Probe
    "clean-seed2000_near_2in_20260811_173531"               ,
    "clean-seed2000_near_1in_20260811_170419"               ,
    "clean_near_2in_20260811_171431"                        ,
    "clean_near_1in_20260811_165307"                        ,

    # Color Slowpace
    "color-slowpace_different_object_20260811_163250"       ,
    "color-slowpace_in_distribution_20260811_161636"        ,

    # Seed 2 of registered experiments
    "color-seed2000_new_positions_20260811_155556"          ,
    "recovery-seed2000_new_positions_20260811_153653"       ,
    "randomized-seed2000_new_positions_20260811_151815"     ,
    "clean-seed2000_new_positions_20260811_145926"          ,

    "color-seed2000_reduced_lighting_20260811_141750"      ,
    "recovery-seed2000_reduced_lighting_20260811_135938"   ,
    "randomized-seed2000_reduced_lighting_20260811_134154" ,
    "clean-seed2000_reduced_lighting_20260811_132543"      ,

    "color-seed2000_distractors_20260811_131347"           ,
    "recovery-seed2000_distractors_20260811_125939"        ,
    "randomized-seed2000_distractors_20260811_124023"      ,
    "clean-seed2000_distractors_20260811_122707"           ,

    "color-seed2000_different_object_20260811_121401"      ,
    "recovery-seed2000_different_object_20260811_114348"   ,
    "randomized-seed2000_different_object_20260811_112537" ,
    "clean-seed2000_different_object_20260811_110857"      ,

    "color-seed2000_in_distribution_20260811_104851"      ,
    "recovery-seed2000_in_distribution_20260811_103305"   ,
    "randomized-seed2000_in_distribution_20260811_101603" ,
    "clean-seed2000_in_distribution_20260811_100456"      ,

    # Seed 1 of registered experiments
    "color_new_positions_20260810_140805"                  ,
    "recovery_new_positions_20260810_134854"               ,
    "randomized_new_positions_20260810_132839"             ,
    "clean_new_positions_20260810_130902"                  ,

    "color_reduced_lighting_20260810_171536"               ,
    "recovery_reduced_lighting_20260810_165851"            ,
    "randomized_reduced_lighting_20260810_164056"          ,
    "clean_reduced_lighting_20260810_162616"               ,

    "color_distractors_20260810_161402"                    ,
    "recovery_distractors_20260810_160107"                 ,
    "randomized_distractors_20260810_154140"               ,
    "clean_distractors_20260810_152709"                    ,

    "color_different_object_20260810_151625"               ,
    "recovery_different_object_20260810_150115"            ,
    "randomized_different_object_20260810_144117"          ,
    "clean_different_object_20260810_142829"               ,

    "color_in_distribution_20260810_123243"                ,
    "recovery_in_distribution_20260810_124638"             ,
    "randomized_in_distribution_20260810_121255"           ,
    "clean_in_distribution_20260810_120038"                ,
]

TRAININGS = {
    # Regustered datasets
    "color": "cube-pickup-color_20260809_183224"                ,         
    "recovery": "cube-pickup-recovery_20260809_141725"          ,
    "randomized": "cube-pickup-randomized_20260809_115825"      ,
    "clean": "cube-pickup-clean_20260809_105745"                ,
}

EXPLORATORY_TRAININGS = {
    # Density (sweep and probe)
    "density_sweep": "cube-pickup-densitypool_20260908_125700"   ,
    "density_probe": "cube-pickup-density_20260822_194111"       ,
    # Slowpace
    "slowpace":       "cube-pickup-color_20260809_130649"        , 
}

FIVE = ["condition", "eval_cell", "seed", "episode", "success"]

CONDITIONS = {"clean", "randomized", "recovery", "color"}

CELLS = {"in_distribution", "new_positions", "reduced_lighting",
         "different_object", "distractors"}

CELLS_DEN = {"in_distribution", "new_positions", "distractors", "spot_check"}

SEEDS = {1000, 2000}

VOCAB = {"success", "success_after_missed_grasp", "success_after_drop",
         "no_departure", "contact_no_grasp", "grasp_drop", "deliberate_drop",
         "cube_out_of_bounds", "cup_knocked", "timeout_other"}
SUCCESS_LABELS = {"success", "success_after_missed_grasp", "success_after_drop"}

# PROTOCOL.md §8.21 renames
LABEL_RENAMES = {
    "success_after_recovery": "success_after_missed_grasp",
    "success_after_regrasp": "success_after_drop",
}

def load(sheet):
    df = pd.read_excel(XL, sheet_name=sheet)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "failure_mode" in df:
        n = int(df.failure_mode.isin(LABEL_RENAMES.keys()).sum())
        if n:
            print(f" renamed {n} legacy failure_mode labels in sheet '{sheet}' (per §8.21)")
        df["failure_mode"] = df["failure_mode"].replace(LABEL_RENAMES)
    return df


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def validate_registered(df):
    print("Registered Grid:")
    good = True

    good &= check("600 rows", len(df) == 600, f"got {len(df)}")
    good &= check("columns present", set(FIVE).issubset(df.columns),
                  f"missing {set(FIVE) - set(df.columns)}")

    counts = df.groupby(["condition", "eval_cell", "seed"]).size()
    good &= check("20 cells x 2 seeds at 15 episodes", (counts == 15).all() and len(counts) == 40,
                  f"{len(counts)} groups, sizes {sorted(counts.unique())}")

    good &= check("conditions", set(df.condition) == CONDITIONS, str(set(df.condition)))
    good &= check("cells", set(df.eval_cell) == CELLS, str(set(df.eval_cell)))
    good &= check("seeds", set(df.seed) == SEEDS, str(set(df.seed)))
    good &= check("success is 0/1", set(df.success.unique()) <= {0, 1},
                  str(set(df.success.unique())))
    good &= check("episodes 1-15", set(df.episode.unique()) == set(range(1, 16)),
                  str(sorted(set(df.episode.unique()))))
    dup = df[df.duplicated(["condition", "eval_cell", "seed", "episode"], keep=False)]
    good &= check("no duplicate (condition, cell, seed, episode)", len(dup) == 0, f"{len(dup)} rows")

    if len(dup):
        print(dup[["condition", "eval_cell", "seed", "episode"]].to_string(index=False))

    if "failure_mode" in df:
        bad = set(df.failure_mode.dropna()) - VOCAB
        good &= check("failure_mode vocabulary", not bad, f"unknown: {bad}")
        mism = df[(df.success == 1) != df.failure_mode.isin(SUCCESS_LABELS)]
        good &= check("success agrees with failure_mode", len(mism) == 0,
                      f"{len(mism)} disagreements")
        if len(mism):
            print(mism[["condition", "eval_cell", "seed", "episode",
                        "success", "failure_mode"]].to_string(index=False))

    if "instance" in df:
        np_rows = df[df.eval_cell == "new_positions"]
        good &= check("new_positions instances are E1-E5",
                      set(np_rows.instance) == {"E1", "E2", "E3", "E4", "E5"},
                      str(set(np_rows.instance)))
    return good


def validate_exploratory(df):
    print("Exploratory:")
    good = True
    for k, g in df.groupby(["condition", "eval_cell", "seed"]):
        good &= check(f"episodes contiguous {k}", sorted(g.episode) == list(range(1, len(g) + 1)), str(sorted(g.episode)))
    good &= check("107 rows", len(df) == 107, f"got {len(df)}")
    counts = df.groupby(["condition", "eval_cell", "seed"]).size().to_dict()

    # NOTE: "color-slowpace".startswith("color") is True. Any downstream filter that
    # selects the Color condition by prefix will silently include the exploratory policy.
    # Match conditions exactly, not by prefix.
    expected = {("color-slowpace", "in_distribution", 1000): 15,
                ("color-slowpace", "different_object", 1000): 15,
                ("clean", "near_1in", 1000): 8, ("clean", "near_1in", 2000): 8,
                ("clean", "near_2in", 1000): 8, ("clean", "near_2in", 2000): 8,
                ("density", "in_distribution", 1000): 15,
                ("density", "trained_t2", 1000): 15,
                ("density", "new_positions", 1000): 15}
    good &= check("cell sizes", counts == expected, f"got {counts}")
    good &= check("success is 0/1", set(df.success.unique()) <= {0, 1},
                  str(set(df.success.unique())))

    dup = df[df.duplicated(["condition", "eval_cell", "seed", "episode"], keep=False)]
    good &= check("no duplicate (condition, cell, seed, episode)", len(dup) == 0, f"{len(dup)} rows")

    if len(dup):
        print(dup[["condition", "eval_cell", "seed", "episode"]].to_string(index=False))

    if "failure_mode" in df:
        bad = set(df.failure_mode.dropna()) - VOCAB
        good &= check("failure_mode vocabulary", not bad, f"unknown: {bad}")
        mism = df[(df.success == 1) != df.failure_mode.isin(SUCCESS_LABELS)]
        good &= check("success agrees with failure_mode", len(mism) == 0,
                      f"{len(mism)} disagreements")
        if len(mism):
            print(mism[["condition", "eval_cell", "seed", "episode",
                        "success", "failure_mode"]].to_string(index=False))
            
    return good

def validate_density(df):
    print("Density Sweep:")
    good = True
    for k, g in df.groupby(["condition", "eval_cell", "seed"]):
        good &= check(f"episodes contiguous {k}", sorted(g.episode) == list(range(1, len(g) + 1)), str(sorted(g.episode)))
    good &= check("610 rows", len(df) == 610, f"got {len(df)}")
    good &= check("cells", set(df.eval_cell) == CELLS_DEN, str(set(df.eval_cell)))

    counts = df.groupby(["condition", "eval_cell", "seed"]).size().to_dict()

    # NOTE: Density-fixedstep rollouts start with density, but had different epoch's in training
    # density10, etc.) Any downstream filter that selects the Density condition by prefix 
    # will include the ones with distinct training conditions. Match conditions exactly, not by prefix.
    
    expected = {("density5", "in_distribution", 1000): 15,
                ("density5", "distractors", 1000): 15,
                ("density5", "spot_check", 1000): 6,
                ("density5", "new_positions", 1000): 25,
                ("density10", "in_distribution", 1000): 15,
                ("density10", "distractors", 1000): 15,
                ("density10", "spot_check", 1000): 6,
                ("density10", "new_positions", 1000): 25,
                ("density25", "in_distribution", 1000): 15,
                ("density25", "distractors", 1000): 15,
                ("density25", "spot_check", 1000): 6,
                ("density25", "new_positions", 1000): 25,
                ("density50", "in_distribution", 1000): 15,
                ("density50", "distractors", 1000): 15,
                ("density50", "spot_check", 1000): 6,
                ("density50", "new_positions", 1000): 25,
                ("density5", "in_distribution", 2000): 15,
                ("density5", "distractors", 2000): 15,
                ("density5", "spot_check", 2000): 6,
                ("density5", "new_positions", 2000): 25,
                ("density10", "in_distribution", 2000): 15,
                ("density10", "distractors", 2000): 15,
                ("density10", "spot_check", 2000): 6,
                ("density10", "new_positions", 2000): 25,
                ("density25", "in_distribution", 2000): 15,
                ("density25", "distractors", 2000): 15,
                ("density25", "spot_check", 2000): 6,
                ("density25", "new_positions", 2000): 25,
                ("density50", "in_distribution", 2000): 15,
                ("density50", "distractors", 2000): 15,
                ("density50", "spot_check", 2000): 6,
                ("density50", "new_positions", 2000): 25,
                ("density25-fixedstep", "in_distribution", 1000): 15,
                ("density25-fixedstep", "distractors", 1000): 15,
                ("density25-fixedstep", "spot_check", 1000): 6,
                ("density25-fixedstep", "new_positions", 1000): 25,
                ("density50-fixedstep", "in_distribution", 1000): 15,
                ("density50-fixedstep", "distractors", 1000): 15,
                ("density50-fixedstep", "spot_check", 1000): 6,
                ("density50-fixedstep", "new_positions", 1000): 25}
    good &= check("cell sizes", counts == expected, f"got {counts}")
    good &= check("success is 0/1", set(df.success.unique()) <= {0, 1},
                  str(set(df.success.unique())))

    dup = df[df.duplicated(["condition", "eval_cell", "seed", "episode"], keep=False)]
    good &= check("no duplicate (condition, cell, seed, episode)", len(dup) == 0, f"{len(dup)} rows")

    if len(dup):
        print(dup[["condition", "eval_cell", "seed", "episode"]].to_string(index=False))

    if "failure_mode" in df:
        bad = set(df.failure_mode.dropna()) - VOCAB
        good &= check("failure_mode vocabulary", not bad, f"unknown: {bad}")
        mism = df[(df.success == 1) != df.failure_mode.isin(SUCCESS_LABELS)]
        good &= check("success agrees with failure_mode", len(mism) == 0,
                      f"{len(mism)} disagreements")
        if len(mism):
            print(mism[["condition", "eval_cell", "seed", "episode",
                        "success", "failure_mode"]].to_string(index=False))


            
    if "instance" in df:
        np_rows = df[df.eval_cell == "new_positions"]
        good &= check("new_positions instances are E1-E5",
                      set(np_rows.instance) == {"E1", "E2", "E3", "E4", "E5"},
                      str(set(np_rows.instance)))

        sp_rows = df[df.eval_cell == "spot_check"]
        good &= check("spot_check instances are T1, T3, T10",
                      set(sp_rows.instance) == {"T1", "T3", "T10"},
                      str(set(sp_rows.instance)))
        good &= check("spot_check is 2 per position",
                      sp_rows.groupby(["condition", "seed", "instance"]).size().eq(2).all(),
                      str(sp_rows.groupby("instance").size().to_dict()))
            
    return good


def main():
    if not os.path.exists(XL):
        sys.exit(f"missing {XL}")

    reg, exp, den = load(SHEET_REG), load(SHEET_EXP), load(SHEET_DEN)
    ok = validate_registered(reg)
    print()
    ok &= validate_exploratory(exp)
    print()
    ok &= validate_density(den)

    if not ok:
        sys.exit("\nvalidation failed, nothing written")

    reg.to_csv(f"{OUT}/results_full.csv", index=False)
    reg[FIVE].to_csv(f"{OUT}/results.csv", index=False)
    for s in sorted(SEEDS):
        reg[reg.seed == s][FIVE].to_csv(f"{OUT}/results_seed{s}.csv", index=False)
    exp.to_csv(f"{OUT}/exploratory.csv", index=False)
    den.to_csv(f"{OUT}/density.csv", index=False)

    pd.concat([reg.assign(source="registered"), exp.assign(source="exploratory"), den.assign(source="density")], ignore_index=True).to_csv(f"{OUT}/results_all.csv", index=False)

    print("\nwrote results_full.csv, results.csv, results_seed1000.csv, "
          "results_seed2000.csv, exploratory.csv, density.csv, results_all.csv")
    print(f"registered {len(reg)} rows, exploratory {len(exp)} rows, density {len(den)} rows, "
          f"combined {len(reg) + len(exp) + len(den)}")


if __name__ == "__main__":
    main()