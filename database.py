import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_all_tables():
    """Load all CSV files from the current directory"""
    print("Loading tables...")
    
    # Load all tables
    tables = {
        'product_label': pd.read_csv('product_label.csv'),
        'product_adverse_effect': pd.read_csv('product_adverse_effect.csv'),
        'vocab_meddra_adverse_effect': pd.read_csv('vocab_meddra_adverse_effect.csv'),
        'product_to_rxnorm': pd.read_csv('product_to_rxnorm.csv'),
        'vocab_rxnorm_product': pd.read_csv('vocab_rxnorm_product.csv'),
        'vocab_rxnorm_ingredient_to_product': pd.read_csv('vocab_rxnorm_ingredient_to_product.csv'),
        'vocab_rxnorm_ingredient': pd.read_csv('vocab_rxnorm_ingredient.csv')
    }
    
    # Print shapes
    for name, df in tables.items():
        print(f"{name}: {df.shape}")
    
    return tables

def create_unified_table(tables):
    """Merge all tables following the ER diagram relationships"""
    print("\n" + "="*50)
    print("Creating unified table...")
    print("="*50)
    
    # Step 1: Start with product_adverse_effect (core adverse events)
    unified = tables['product_adverse_effect'].copy()
    print(f"\n1. Starting with product_adverse_effect: {unified.shape}")
    
    # Step 2: Join with vocab_meddra_adverse_effect (get adverse effect names)
    unified = unified.merge(
        tables['vocab_meddra_adverse_effect'],
        left_on='effect_meddra_id',
        right_on='meddra_id',
        how='left',
        suffixes=('', '_meddra')
    )
    print(f"2. After joining adverse effect vocabulary: {unified.shape}")
    
    # Step 3: Join with product_label (get product label info)
    unified = unified.merge(
        tables['product_label'],
        left_on='product_label_id',
        right_on='label_id',
        how='left',
        suffixes=('', '_label')
    )
    print(f"3. After joining product_label: {unified.shape}")
    
    # Step 4: Join with product_to_rxnorm (link to RxNorm standardized names)
    unified = unified.merge(
        tables['product_to_rxnorm'],
        on='label_id',
        how='left',
        suffixes=('', '_rxnorm_link')
    )
    print(f"4. After joining product_to_rxnorm: {unified.shape}")
    
    # Step 5: Join with vocab_rxnorm_product (get standardized product names)
    unified = unified.merge(
        tables['vocab_rxnorm_product'],
        left_on='rxnorm_product_id',
        right_on='rxnorm_id',
        how='left',
        suffixes=('', '_rxnorm_product')
    )
    print(f"5. After joining vocab_rxnorm_product: {unified.shape}")
    
    # Step 6: Join with ingredient-to-product mapping
    unified = unified.merge(
        tables['vocab_rxnorm_ingredient_to_product'],
        left_on='rxnorm_product_id',
        right_on='product_id',
        how='left',
        suffixes=('', '_ingredient_link')
    )
    print(f"6. After joining ingredient_to_product mapping: {unified.shape}")
    
    # Step 7: Join with vocab_rxnorm_ingredient (get ingredient names)
    unified = unified.merge(
        tables['vocab_rxnorm_ingredient'],
        left_on='ingredient_id',
        right_on='rxnorm_id',
        how='left',
        suffixes=('', '_ingredient')
    )
    print(f"7. After joining vocab_rxnorm_ingredient: {unified.shape}")
    
    return unified

def clean_and_select_columns(df):
    """Select and rename the most important columns"""
    print("\n" + "="*50)
    print("Selecting and renaming key columns...")
    print("="*50)
    
    # Define key columns to keep with new names
    column_mapping = {
        # Product/Drug Information
        'source_product_name': 'drug_name',
        'source_product_id': 'drug_product_id',
        'source': 'drug_source',
        'rxnorm_name': 'drug_rxnorm_name',
        'rxnorm_term_type': 'drug_term_type',
        'rxnorm_name_ingredient': 'active_ingredient',
        'rxnorm_term_type_ingredient': 'ingredient_term_type',
        
        # Adverse Effect Information
        'meddra_name': 'adverse_effect',
        'meddra_term_type': 'adverse_effect_type',
        'label_section': 'label_section',
        'match_method': 'match_method',
        'pred0': 'prediction_score_0',
        'pred1': 'prediction_score_1',
        
        # IDs (for reference)
        'label_id': 'label_id',
        'effect_id': 'effect_id',
        'effect_meddra_id': 'meddra_id',
        'product_label_id': 'product_label_id',
        'ingredient_id': 'ingredient_id'
    }
    
    # Select columns that exist in the dataframe
    available_columns = [col for col in column_mapping.keys() if col in df.columns]
    df_selected = df[available_columns].copy()
    
    # Rename columns
    rename_dict = {k: v for k, v in column_mapping.items() if k in available_columns}
    df_selected = df_selected.rename(columns=rename_dict)
    
    print(f"\nFinal shape: {df_selected.shape}")
    print(f"\nColumns in unified table:")
    for i, col in enumerate(df_selected.columns, 1):
        print(f"  {i}. {col}")
    
    return df_selected

def generate_summary_statistics(df):
    """Generate summary statistics about the unified table"""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\nTotal Records: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    
    if 'drug_name' in df.columns:
        print(f"\nUnique Drugs: {df['drug_name'].nunique():,}")
    
    if 'adverse_effect' in df.columns:
        print(f"Unique Adverse Effects: {df['adverse_effect'].nunique():,}")
        print(f"\nTop 10 Most Common Adverse Effects:")
        top_effects = df['adverse_effect'].value_counts().head(10)
        for effect, count in top_effects.items():
            print(f"  • {effect}: {count:,} occurrences")
    
    if 'active_ingredient' in df.columns:
        print(f"\nUnique Active Ingredients: {df['active_ingredient'].nunique():,}")
    
    # Missing values
    print(f"\nMissing Values per Column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("  No missing values!")
    
    # Data types
    print(f"\nData Types:")
    print(df.dtypes.value_counts())

def main():
    """Main execution function"""
    print("="*50)
    print("DRUG-DRUG INTERACTION DATABASE MERGER")
    print("="*50)
    
    # Load all tables
    tables = load_all_tables()
    
    # Create unified table
    unified = create_unified_table(tables)
    
    # Clean and select key columns
    final_df = clean_and_select_columns(unified)
    
    # Generate summary statistics
    generate_summary_statistics(final_df)
    
    # Save to CSV
    output_file = 'unified_ddi_database.csv'
    print(f"\n" + "="*50)
    print(f"Saving unified table to: {output_file}")
    print("="*50)
    final_df.to_csv(output_file, index=False)
    print(f"✓ Successfully saved!")
    
    # Also save a sample for quick inspection
    sample_file = 'unified_ddi_database_sample.csv'
    final_df.head(1000).to_csv(sample_file, index=False)
    print(f"✓ Sample file saved: {sample_file} (first 1000 rows)")
    
    return final_df

if __name__ == "__main__":
    df = main()
    print("PROCESS COMPLETE!")
    
    print("\nYou can now use 'unified_ddi_database.csv' for your DDI project.")
    print("\nNext steps for your project:")
    print("  1. Load the unified table")
    print("  2. Create drug-pair combinations for interaction prediction")
    print("  3. Extract molecular features for each drug")
    print("  4. Build your prediction models (Random Forest → XGBoost → GNN)")
    