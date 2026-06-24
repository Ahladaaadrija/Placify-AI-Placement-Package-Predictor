"""
generate_data.py
-----------------
Creates a synthetic dataset of student academic/skill records and their
placement package (in LPA - Lakhs Per Annum). In a real project you would
replace this with actual placement-cell data from your college. This script
exists so you have something to train on immediately.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200  # number of student records

cgpa = np.round(np.random.normal(7.5, 0.8, N).clip(5.0, 10.0), 2)
internships = np.random.poisson(1.2, N).clip(0, 5)
projects = np.random.poisson(2.5, N).clip(0, 8)
certifications = np.random.poisson(1.5, N).clip(0, 6)
coding_score = np.round(np.random.normal(65, 15, N).clip(0, 100), 1)   # e.g. HackerRank/LeetCode style score
communication_score = np.round(np.random.normal(70, 12, N).clip(0, 100), 1)
backlogs = np.random.choice([0, 0, 0, 1, 2], N)  # mostly 0, occasional backlogs

# True underlying relationship (this is what Linear Regression will try to learn)
package = (
    1.5
    + 0.85 * (cgpa - 5)          # higher CGPA -> higher package
    + 0.9 * internships
    + 0.35 * projects
    + 0.25 * certifications
    + 0.04 * coding_score
    + 0.02 * communication_score
    - 0.6 * backlogs
    + np.random.normal(0, 0.8, N)   # random noise, like real life
)
package = np.round(package.clip(2.0, 45.0), 2)  # clip to realistic LPA range

df = pd.DataFrame({
    "CGPA": cgpa,
    "Internships": internships,
    "Projects": projects,
    "Certifications": certifications,
    "Coding_Score": coding_score,
    "Communication_Score": communication_score,
    "Backlogs": backlogs,
    "Package_LPA": package,
})

df.to_csv("placement_data.csv", index=False)
print(f"Generated placement_data.csv with {len(df)} rows")
print(df.head())