"""
Demonstration of Information Value (IV) improvements to WoE Analysis
This script showcases the new IV features added to the original WoE analysis
"""

import pandas as pd
import numpy as np
from woe_iv_analysis import WoETimeAnalysisWithIV, generate_sample_data
import warnings
warnings.filterwarnings('ignore')


def demonstrate_iv_features():
    """
    Demonstrate all the new IV features
    """
    
    print("="*80)
    print("INFORMATION VALUE (IV) FEATURE DEMONSTRATION")
    print("="*80)
    
    # Generate sample data
    print("\n1. Generating sample dataset...")
    df = generate_sample_data(n_samples=5000, n_months=12)
    print(f"   Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize analyzer
    print("\n2. Initializing WoE and IV analyzer...")
    analyzer = WoETimeAnalysisWithIV(df, target_col='y', date_col='month')
    
    # Calculate WoE and IV
    print("\n3. Calculating WoE bins and Information Value...")
    analyzer.calculate_woe_bins(max_num_bin=5, min_perc_fine_bin=0.05)
    print("   ✓ WoE binning complete")
    print("   ✓ IV calculation complete")
    
    # Display IV Summary
    print("\n4. Information Value Summary:")
    print("-"*80)
    analyzer.print_iv_summary()
    
    # Plot IV visualization
    print("\n5. Generating IV visualization...")
    analyzer.plot_iv_summary()
    
    # Analyze top variable with IV details
    print("\n6. Detailed analysis of top predictive variable...")
    
    # Get the top variable by IV
    if analyzer.iv_summary is not None and len(analyzer.iv_summary) > 0:
        top_variable = analyzer.iv_summary.iloc[0]['Variable']
        top_iv = analyzer.iv_summary.iloc[0]['IV']
        
        print(f"\n   Top Variable: {top_variable}")
        print(f"   IV Value: {top_iv:.4f}")
        print(f"   Predictive Power: {analyzer._interpret_iv(top_iv)}")
        
        # Show enhanced WoE plot with IV contribution
        print(f"\n   Generating enhanced WoE plot with IV contribution for {top_variable}...")
        analyzer.plot_woe_and_bin_distribution_with_iv(top_variable)
        
        # Show IV temporal stability
        print(f"\n   Analyzing IV stability over time for {top_variable}...")
        analyzer.plot_iv_temporal_stability(top_variable, window='M')
    
    # Export results
    print("\n7. Exporting results to Excel...")
    output_file = 'iv_analysis_demo_results.xlsx'
    analyzer.export_results(output_file)
    print(f"   ✓ Results exported to {output_file}")
    
    # Summary of improvements
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS ADDED:")
    print("="*80)
    print("""
    1. INFORMATION VALUE (IV) CALCULATION
       - Automatic IV calculation for each variable
       - IV interpretation (Not Useful/Weak/Medium/Strong)
       - Ranking of variables by predictive power
    
    2. IV VISUALIZATION
       - Summary bar chart with color-coding by predictive power
       - Pie chart showing distribution of variable strength
       - IV contribution by bin visualization
    
    3. IV TEMPORAL STABILITY
       - Track IV changes over time
       - Identify variables with stable predictive power
       - Calculate stability metrics (mean, std, CV)
    
    4. ENHANCED REPORTING
       - Comprehensive IV summary table
       - Export to Excel with multiple sheets
       - IV values included in all plots
    
    5. IMPROVED ANALYSIS WORKFLOW
       - Automatic identification of most predictive variables
       - Quick assessment of model variable importance
       - Better feature selection guidance
    """)
    
    return analyzer


def compare_variables_by_iv():
    """
    Compare different types of variables and their IV values
    """
    print("\n" + "="*80)
    print("VARIABLE COMPARISON BY INFORMATION VALUE")
    print("="*80)
    
    # Create dataset with variables of different predictive powers
    np.random.seed(123)
    n = 5000
    
    # Create features with varying relationships to target
    df = pd.DataFrame({
        'month': pd.date_range('2024-01-01', periods=12, freq='M').repeat(n//12 + 1)[:n],
        
        # Strong predictor (IV > 0.3)
        'strong_predictor': np.random.normal(0, 1, n),
        
        # Medium predictor (0.1 < IV < 0.3)
        'medium_predictor': np.random.uniform(0, 100, n),
        
        # Weak predictor (0.02 < IV < 0.1)
        'weak_predictor': np.random.choice(['A', 'B', 'C'], n),
        
        # Not useful (IV < 0.02)
        'random_noise': np.random.random(n)
    })
    
    # Create target with logical relationship
    df['y'] = (
        (df['strong_predictor'] > 0).astype(int) * 0.6 +
        (df['medium_predictor'] > 50).astype(int) * 0.3 +
        (df['weak_predictor'] == 'A').astype(int) * 0.1 +
        np.random.random(n) * 0.2
    ) > 0.5
    df['y'] = df['y'].astype(int)
    
    # Analyze
    analyzer = WoETimeAnalysisWithIV(df, target_col='y', date_col='month')
    analyzer.calculate_woe_bins()
    
    print("\n" + "-"*80)
    print("EXPECTED VS ACTUAL INFORMATION VALUES:")
    print("-"*80)
    
    for var in ['strong_predictor', 'medium_predictor', 'weak_predictor', 'random_noise']:
        if var in analyzer.iv_dict:
            iv = analyzer.iv_dict[var]
            interpretation = analyzer._interpret_iv(iv)
            print(f"{var:20s}: IV = {iv:.4f} ({interpretation})")
    
    return analyzer


def quick_iv_assessment(df, target_col, date_col=None, top_n=10):
    """
    Quick function to assess IV for all variables in a dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    date_col : str, optional
        Name of date column
    top_n : int
        Number of top variables to display
    
    Returns:
    --------
    pd.DataFrame
        IV summary dataframe
    """
    print("\n" + "="*80)
    print(f"QUICK IV ASSESSMENT - TOP {top_n} VARIABLES")
    print("="*80)
    
    # If no date column, create a dummy one
    if date_col is None:
        df = df.copy()
        df['_dummy_date'] = pd.Timestamp.now()
        date_col = '_dummy_date'
    
    # Run analysis
    analyzer = WoETimeAnalysisWithIV(df, target_col=target_col, date_col=date_col)
    analyzer.calculate_woe_bins()
    
    # Display top variables
    if analyzer.iv_summary is not None:
        top_vars = analyzer.iv_summary.head(top_n)
        print(f"\nTop {min(top_n, len(top_vars))} Variables by Information Value:")
        print(top_vars.to_string(index=False))
        
        # Plot summary
        analyzer.plot_iv_summary(top_n=top_n)
    
    return analyzer.iv_summary


if __name__ == "__main__":
    # Run demonstrations
    print("\n" + "🚀 " * 20)
    print("Starting Information Value (IV) Feature Demonstrations")
    print("🚀 " * 20 + "\n")
    
    # Demo 1: Full feature demonstration
    analyzer = demonstrate_iv_features()
    
    # Demo 2: Compare variables by IV
    print("\n" + "📊 " * 20)
    print("Variable Comparison Demo")
    print("📊 " * 20)
    comparison_analyzer = compare_variables_by_iv()
    
    # Demo 3: Quick assessment
    print("\n" + "⚡ " * 20)
    print("Quick Assessment Demo")
    print("⚡ " * 20)
    
    # Generate a quick test dataset
    test_df = generate_sample_data(n_samples=3000, n_months=6)
    iv_summary = quick_iv_assessment(test_df, target_col='y', date_col='month', top_n=5)
    
    print("\n" + "✅ " * 20)
    print("All demonstrations complete!")
    print("✅ " * 20)
    
    print("""
    
    💡 Usage Tips:
    -------------
    1. Variables with IV > 0.3 are your strongest predictors
    2. Consider removing variables with IV < 0.02
    3. Monitor IV stability over time for model robustness
    4. Use IV ranking for feature selection
    5. Check for suspiciously high IV values (>0.5) - may indicate leakage
    
    📚 Next Steps:
    -------------
    - Use the exported Excel file for detailed analysis
    - Integrate IV calculation into your feature engineering pipeline
    - Monitor IV drift in production models
    - Use IV for variable selection in credit scoring models
    """)
