import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('/Users/prayagadage/Desktop/Python_By_CJC/100_days_Ml/Projects/house_price_prediction/notebook/dataset/House_Rent_Dataset.csv')

# Create report file
report_path = '/Users/prayagadage/Desktop/Python_By_CJC/100_days_Ml/Projects/house_price_prediction/notebook/EDA_Report.txt'
report = open(report_path, 'w')

def write_section(title, content=""):
    report.write("\n" + "="*80 + "\n")
    report.write(f"{title.center(80)}\n")
    report.write("="*80 + "\n")
    if content:
        report.write(content + "\n")

def write_subsection(title):
    report.write("\n" + "-"*80 + "\n")
    report.write(f"{title}\n")
    report.write("-"*80 + "\n")

# Header
report.write("*"*80 + "\n")
report.write("HOUSE RENT DATASET - EXPLORATORY DATA ANALYSIS REPORT".center(80) + "\n")
report.write("*"*80 + "\n")
report.write(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.write(f"Dataset: House_Rent_Dataset.csv\n")

# 1. DATASET OVERVIEW
write_section("1. DATASET OVERVIEW")

write_subsection("1.1 Dataset Dimensions")
report.write(f"Number of Rows: {df.shape[0]}\n")
report.write(f"Number of Columns: {df.shape[1]}\n")
report.write(f"Total Data Points: {df.shape[0] * df.shape[1]}\n")

write_subsection("1.2 Column Information")
report.write(f"{'Column Name':<20} {'Data Type':<15} {'Non-Null Count':<15} {'Null Count':<15}\n")
report.write("-"*65 + "\n")
for col in df.columns:
    dtype = str(df[col].dtype)
    non_null = df[col].count()
    null_count = df[col].isnull().sum()
    report.write(f"{col:<20} {dtype:<15} {non_null:<15} {null_count:<15}\n")

write_subsection("1.3 Memory Usage")
report.write(f"Total Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

# 2. MISSING VALUES ANALYSIS
write_section("2. MISSING VALUES ANALYSIS")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing_Count': missing, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    report.write("Columns with Missing Values:\n\n")
    report.write(missing_df.to_string())
else:
    report.write("✓ NO MISSING VALUES FOUND IN THE DATASET\n")

# 3. DUPLICATE ANALYSIS
write_section("3. DUPLICATE ANALYSIS")
duplicates = df.duplicated().sum()
report.write(f"Number of Duplicate Rows: {duplicates}\n")
report.write(f"Percentage of Duplicates: {(duplicates/len(df)*100):.2f}%\n")
if duplicates == 0:
    report.write("✓ NO DUPLICATE ROWS FOUND\n")

# 4. UNIVARIATE ANALYSIS - NUMERICAL FEATURES
write_section("4. UNIVARIATE ANALYSIS - NUMERICAL FEATURES")

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numerical_cols:
    write_subsection(f"4.{numerical_cols.index(col)+1} {col}")
    
    stats = df[col].describe()
    report.write(f"\nDescriptive Statistics:\n")
    report.write(f"  Count:        {stats['count']:.0f}\n")
    report.write(f"  Mean:         {stats['mean']:.2f}\n")
    report.write(f"  Std Dev:      {stats['std']:.2f}\n")
    report.write(f"  Min:          {stats['min']:.2f}\n")
    report.write(f"  25th %ile:    {stats['25%']:.2f}\n")
    report.write(f"  Median:       {stats['50%']:.2f}\n")
    report.write(f"  75th %ile:    {stats['75%']:.2f}\n")
    report.write(f"  Max:          {stats['max']:.2f}\n")
    
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    report.write(f"\nDistribution Metrics:\n")
    report.write(f"  Skewness:     {skewness:.4f}")
    if abs(skewness) < 0.5:
        report.write(" (Fairly Symmetrical)\n")
    elif abs(skewness) < 1:
        report.write(" (Moderately Skewed)\n")
    else:
        report.write(" (Highly Skewed)\n")
    
    report.write(f"  Kurtosis:     {kurtosis:.4f}")
    if kurtosis < 3:
        report.write(" (Platykurtic - Flatter)\n")
    elif kurtosis > 3:
        report.write(" (Leptokurtic - Peaked)\n")
    else:
        report.write(" (Mesokurtic - Normal)\n")
    
    # Unique values
    unique_count = df[col].nunique()
    report.write(f"\nUnique Values: {unique_count}\n")
    
    if unique_count <= 10:
        value_counts = df[col].value_counts().head(10)
        report.write("\nValue Distribution:\n")
        for val, count in value_counts.items():
            pct = (count/len(df))*100
            report.write(f"  {val}: {count} ({pct:.2f}%)\n")

# 5. UNIVARIATE ANALYSIS - CATEGORICAL FEATURES
write_section("5. UNIVARIATE ANALYSIS - CATEGORICAL FEATURES")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    write_subsection(f"5.{categorical_cols.index(col)+1} {col}")
    
    unique_count = df[col].nunique()
    report.write(f"Unique Values: {unique_count}\n")
    
    # Show top categories if reasonable number
    if unique_count <= 50:
        value_counts = df[col].value_counts().head(20)
        report.write(f"\nTop {min(20, unique_count)} Categories:\n\n")
        report.write(f"{'Category':<30} {'Count':<10} {'Percentage':<10}\n")
        report.write("-"*50 + "\n")
        for val, count in value_counts.items():
            pct = (count/len(df))*100
            val_str = str(val)[:28]
            report.write(f"{val_str:<30} {count:<10} {pct:>6.2f}%\n")
    else:
        report.write(f"\nToo many unique values ({unique_count}) - showing top 10:\n\n")
        value_counts = df[col].value_counts().head(10)
        for val, count in value_counts.items():
            pct = (count/len(df))*100
            report.write(f"  {str(val)[:40]}: {count} ({pct:.2f}%)\n")

print("Step 1/3: Basic analysis completed...")

# 6. BIVARIATE ANALYSIS - CORRELATIONS
write_section("6. CORRELATION ANALYSIS")

corr_matrix = df[numerical_cols].corr()
report.write("\nCorrelation Matrix (Numerical Features):\n\n")
report.write(corr_matrix.to_string())

# Strong correlations with Rent
if 'Rent' in numerical_cols:
    write_subsection("6.1 Correlations with Rent (Target Variable)")
    rent_corr = corr_matrix['Rent'].sort_values(ascending=False)
    report.write("\n")
    for feature, corr in rent_corr.items():
        if feature != 'Rent':
            strength = "Very Strong" if abs(corr) > 0.8 else "Strong" if abs(corr) > 0.6 else "Moderate" if abs(corr) > 0.4 else "Weak"
            direction = "Positive" if corr > 0 else "Negative"
            report.write(f"  {feature:<20}: {corr:>7.4f} ({strength} {direction})\n")

print("Step 2/3: Correlation analysis completed...")

# 7. OUTLIER DETECTION
write_section("7. OUTLIER DETECTION (IQR Method)")

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100
    
    if outlier_count > 0:
        write_subsection(f"{col}")
        report.write(f"Lower Bound: {lower_bound:.2f}\n")
        report.write(f"Upper Bound: {upper_bound:.2f}\n")
        report.write(f"Number of Outliers: {outlier_count} ({outlier_pct:.2f}%)\n")
        
        if outlier_count <= 10:
            report.write(f"\nOutlier Values:\n")
            for val in outliers[col].values:
                report.write(f"  {val}\n")

print("Step 3/3: Outlier detection completed...")

# Close the report
report.close()

print(f"\n✓ Comprehensive EDA Report Generated Successfully!")
print(f"Report saved at: {report_path}")
