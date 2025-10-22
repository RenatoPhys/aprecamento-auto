import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scorecardpy as sc
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class WoETimeAnalysisWithIV:
    """
    An enhanced class to perform Weight of Evidence analysis with Information Value (IV)
    calculation and temporal visualization
    """
    
    def __init__(self, df, target_col='y', date_col='month'):
        """
        Initialize the WoE and IV analysis
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_col : str
            Name of the target column (binary: 0 or 1)
        date_col : str
            Name of the date column
        """
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.bins_dict = {}
        self.woe_dict = {}
        self.iv_dict = {}
        self.iv_summary = None
        self.df_binned = None
        
    def calculate_woe_bins(self, max_num_bin=5, min_perc_fine_bin=0.05):
        """
        Calculate WoE, IV and create bins for all numeric and categorical variables
        """
        # Exclude target and date columns
        feature_cols = [col for col in self.df.columns 
                       if col not in [self.target_col, self.date_col]]
        
        # Create bins using scorecardpy
        print("Creating bins and calculating WoE/IV...")
        self.bins_dict = sc.woebin(
            self.df[feature_cols + [self.target_col]], 
            y=self.target_col,
            max_num_bin=max_num_bin,
            min_perc_fine_bin=min_perc_fine_bin,
            print_info=False
        )
        
        # Apply binning to get WoE values
        print("Applying binning...")
        self.df_binned = sc.woebin_ply(self.df, self.bins_dict)
        
        # Store WoE information and calculate IV
        for var in feature_cols:
            if var in self.bins_dict:
                bin_df = self.bins_dict[var]
                self.woe_dict[var] = bin_df[['bin', 'woe', 'count_distr', 
                                            'good', 'bad', 'count']].copy()
                
                # Extract or calculate IV
                if 'total_iv' in bin_df.columns:
                    # IV is already calculated by scorecardpy
                    self.iv_dict[var] = bin_df['total_iv'].iloc[0]
                else:
                    # Calculate IV manually if not present
                    self.iv_dict[var] = self._calculate_iv(bin_df)
        
        # Create IV summary dataframe
        self._create_iv_summary()
    
    def _calculate_iv(self, bin_df):
        """
        Calculate Information Value for a variable
        
        IV = Σ (Good% - Bad%) * WoE
        """
        # Calculate good and bad distributions
        total_good = bin_df['good'].sum()
        total_bad = bin_df['bad'].sum()
        
        if total_good == 0 or total_bad == 0:
            return 0
        
        good_dist = bin_df['good'] / total_good
        bad_dist = bin_df['bad'] / total_bad
        
        # Calculate IV
        iv = ((good_dist - bad_dist) * bin_df['woe']).sum()
        
        return iv
    
    def _create_iv_summary(self):
        """
        Create a summary dataframe of IV values with interpretation
        """
        iv_data = []
        for var, iv_value in self.iv_dict.items():
            interpretation = self._interpret_iv(iv_value)
            iv_data.append({
                'Variable': var,
                'IV': iv_value,
                'Predictive Power': interpretation,
                'Rank': 0  # Will be filled after sorting
            })
        
        self.iv_summary = pd.DataFrame(iv_data)
        self.iv_summary = self.iv_summary.sort_values('IV', ascending=False).reset_index(drop=True)
        self.iv_summary['Rank'] = range(1, len(self.iv_summary) + 1)
    
    def _interpret_iv(self, iv_value):
        """
        Interpret the Information Value
        """
        if iv_value < 0.02:
            return "Not Useful"
        elif iv_value < 0.1:
            return "Weak"
        elif iv_value < 0.3:
            return "Medium"
        elif iv_value < 0.5:
            return "Strong"
        else:
            return "Very Strong (Suspicious)"
    
    def plot_iv_summary(self, top_n=None):
        """
        Plot Information Value summary for all variables
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top variables to display. If None, display all.
        """
        if self.iv_summary is None:
            print("Please run calculate_woe_bins() first")
            return
        
        # Select top_n variables if specified
        plot_data = self.iv_summary.head(top_n) if top_n else self.iv_summary
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Bar chart of IV values
        colors = []
        for iv in plot_data['IV']:
            if iv < 0.02:
                colors.append('red')
            elif iv < 0.1:
                colors.append('orange')
            elif iv < 0.3:
                colors.append('yellow')
            elif iv < 0.5:
                colors.append('lightgreen')
            else:
                colors.append('darkgreen')
        
        bars = ax1.barh(range(len(plot_data)), plot_data['IV'], color=colors)
        
        # Add IV value labels
        for i, (bar, iv_val) in enumerate(zip(bars, plot_data['IV'])):
            ax1.text(iv_val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{iv_val:.4f}', va='center', fontsize=9)
        
        ax1.set_yticks(range(len(plot_data)))
        ax1.set_yticklabels(plot_data['Variable'])
        ax1.set_xlabel('Information Value (IV)', fontsize=12)
        ax1.set_title('Information Value by Variable', fontsize=14, fontweight='bold')
        
        # Add reference lines
        iv_thresholds = [0.02, 0.1, 0.3, 0.5]
        for threshold in iv_thresholds:
            ax1.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
        
        # Add legend for interpretation
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Not Useful (<0.02)'),
            Patch(facecolor='orange', label='Weak (0.02-0.1)'),
            Patch(facecolor='yellow', label='Medium (0.1-0.3)'),
            Patch(facecolor='lightgreen', label='Strong (0.3-0.5)'),
            Patch(facecolor='darkgreen', label='Very Strong (>0.5)')
        ]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # Plot 2: Pie chart of predictive power distribution
        power_counts = plot_data['Predictive Power'].value_counts()
        colors_pie = {
            'Not Useful': 'red',
            'Weak': 'orange',
            'Medium': 'yellow',
            'Strong': 'lightgreen',
            'Very Strong (Suspicious)': 'darkgreen'
        }
        
        wedges, texts, autotexts = ax2.pie(
            power_counts.values,
            labels=[f"{label}\n({count} vars)" for label, count in zip(power_counts.index, power_counts.values)],
            colors=[colors_pie.get(label, 'gray') for label in power_counts.index],
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax2.set_title('Distribution of Variable Predictive Power', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def print_iv_summary(self):
        """
        Print a formatted IV summary table
        """
        if self.iv_summary is None:
            print("Please run calculate_woe_bins() first")
            return
        
        print("\n" + "="*70)
        print("INFORMATION VALUE (IV) SUMMARY")
        print("="*70)
        
        # Format the dataframe for display
        display_df = self.iv_summary.copy()
        display_df['IV'] = display_df['IV'].apply(lambda x: f"{x:.4f}")
        
        print(display_df.to_string(index=False))
        
        print("\n" + "-"*70)
        print("IV Interpretation Guidelines:")
        print("  < 0.02:     Not Useful for Prediction")
        print("  0.02 - 0.1: Weak Predictive Power")
        print("  0.1 - 0.3:  Medium Predictive Power")
        print("  0.3 - 0.5:  Strong Predictive Power")
        print("  > 0.5:      Very Strong (Check for leakage)")
        print("-"*70 + "\n")
    
    def plot_iv_temporal_stability(self, variable, window='M'):
        """
        Plot IV stability over time for a specific variable
        
        Parameters:
        -----------
        variable : str
            Variable name to analyze
        window : str
            Time window for aggregation ('M' for month, 'W' for week, 'D' for day)
        """
        if variable not in self.bins_dict:
            print(f"Variable {variable} not found")
            return
        
        # Create a copy of the data with temporal grouping
        df_temp = self.df.copy()
        df_temp[self.date_col] = pd.to_datetime(df_temp[self.date_col])
        
        # Group by time window
        df_temp['period'] = df_temp[self.date_col].dt.to_period(window)
        
        # Calculate IV for each period
        iv_over_time = []
        periods = df_temp['period'].unique()
        
        for period in sorted(periods):
            period_data = df_temp[df_temp['period'] == period]
            
            if len(period_data) < 30:  # Skip if too few samples
                continue
            
            # Calculate WoE bins for this period
            try:
                bins_period = sc.woebin(
                    period_data[[variable, self.target_col]], 
                    y=self.target_col,
                    max_num_bin=5,
                    print_info=False
                )
                
                if variable in bins_period:
                    iv_value = self._calculate_iv(bins_period[variable])
                    iv_over_time.append({
                        'Period': period.to_timestamp(),
                        'IV': iv_value,
                        'Sample Size': len(period_data)
                    })
            except:
                continue
        
        if not iv_over_time:
            print(f"Could not calculate temporal IV for {variable}")
            return
        
        # Create dataframe and plot
        iv_df = pd.DataFrame(iv_over_time)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot IV over time
        ax1.plot(iv_df['Period'], iv_df['IV'], 'bo-', linewidth=2, markersize=8)
        
        # Add interpretation zones
        ax1.axhline(y=0.02, color='red', linestyle='--', alpha=0.5, label='Not Useful')
        ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Weak')
        ax1.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.5, label='Medium')
        ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Strong')
        
        # Add mean line
        mean_iv = iv_df['IV'].mean()
        ax1.axhline(y=mean_iv, color='black', linestyle='-', alpha=0.7, 
                   label=f'Mean IV: {mean_iv:.4f}')
        
        ax1.set_xlabel('Period', fontsize=12)
        ax1.set_ylabel('Information Value (IV)', fontsize=12)
        ax1.set_title(f'IV Stability Over Time - {variable}', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Add IV values as labels
        for _, row in iv_df.iterrows():
            ax1.text(row['Period'], row['IV'] + 0.01, f'{row["IV"]:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        # Plot sample size
        ax2.bar(iv_df['Period'], iv_df['Sample Size'], alpha=0.6, color='skyblue')
        ax2.set_xlabel('Period', fontsize=12)
        ax2.set_ylabel('Sample Size', fontsize=12)
        ax2.set_title(f'Sample Size Over Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Calculate stability metrics
        iv_std = iv_df['IV'].std()
        iv_cv = (iv_std / mean_iv) * 100 if mean_iv != 0 else 0
        
        print(f"\nIV Stability Metrics for {variable}:")
        print(f"  Mean IV: {mean_iv:.4f}")
        print(f"  Std Dev: {iv_std:.4f}")
        print(f"  Coefficient of Variation: {iv_cv:.2f}%")
        print(f"  Min IV: {iv_df['IV'].min():.4f}")
        print(f"  Max IV: {iv_df['IV'].max():.4f}")
    
    def get_binned_column_name(self, variable):
        """
        Get the actual column name after binning (handles _woe suffix)
        """
        possible_names = [f"{variable}_woe", f"{variable}_bin", variable]
        for name in possible_names:
            if name in self.df_binned.columns:
                return name
        return None
    
    def plot_woe_and_bin_distribution_with_iv(self, variable):
        """
        Enhanced plot: WoE values, bin count percentage, and IV contribution
        """
        if variable not in self.woe_dict:
            print(f"Variable {variable} not found in WoE dictionary")
            return
        
        woe_data = self.woe_dict[variable]
        iv_value = self.iv_dict.get(variable, 0)
        
        # Calculate IV contribution for each bin
        total_good = woe_data['good'].sum()
        total_bad = woe_data['bad'].sum()
        
        if total_good > 0 and total_bad > 0:
            good_dist = woe_data['good'] / total_good
            bad_dist = woe_data['bad'] / total_bad
            iv_contribution = (good_dist - bad_dist) * woe_data['woe']
        else:
            iv_contribution = pd.Series([0] * len(woe_data))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top plot: WoE and bin distribution
        bars = ax1.bar(range(len(woe_data)), woe_data['count_distr'] * 100, 
                       alpha=0.6, color='skyblue', label='Bin Count %')
        
        # Add percentage labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax1.set_ylabel('Bin Count Percentage (%)', fontsize=12)
        ax1.set_xticks(range(len(woe_data)))
        ax1.set_xticklabels(woe_data['bin'], rotation=45, ha='right')
        
        # Create second y-axis for WoE
        ax1_twin = ax1.twinx()
        line = ax1_twin.plot(range(len(woe_data)), woe_data['woe'], 
                            'ro-', linewidth=2, markersize=8, label='WoE')
        
        # Add WoE value labels
        for i, woe_val in enumerate(woe_data['woe']):
            ax1_twin.text(i, woe_val + 0.05, f'{woe_val:.3f}', 
                         ha='center', va='bottom', fontsize=9, color='red')
        
        ax1_twin.set_ylabel('Weight of Evidence (WoE)', fontsize=12, color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.set_title(f'WoE and Bin Distribution for {variable} (Total IV: {iv_value:.4f})', 
                     fontsize=14, fontweight='bold')
        
        # Bottom plot: IV contribution by bin
        colors = ['green' if x > 0 else 'red' for x in iv_contribution]
        bars2 = ax2.bar(range(len(woe_data)), iv_contribution, color=colors, alpha=0.7)
        
        # Add contribution labels
        for i, (bar, contrib) in enumerate(zip(bars2, iv_contribution)):
            ax2.text(bar.get_x() + bar.get_width()/2., contrib,
                    f'{contrib:.4f}', ha='center', 
                    va='bottom' if contrib > 0 else 'top', fontsize=9)
        
        ax2.set_xlabel('Bins', fontsize=12)
        ax2.set_ylabel('IV Contribution', fontsize=12)
        ax2.set_xticks(range(len(woe_data)))
        ax2.set_xticklabels(woe_data['bin'], rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title(f'Information Value Contribution by Bin', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def prepare_temporal_data_simple(self, variable):
        """
        Simpler approach to prepare temporal data
        """
        # Create a dataframe with date, target, and the binned variable
        binned_col = self.get_binned_column_name(variable)
        
        if binned_col is None:
            raise ValueError(f"Could not find binned column for {variable}")
        
        # Create mapping from WoE values to bin labels
        woe_to_bin = dict(zip(self.bins_dict[variable]['woe'], 
                             self.bins_dict[variable]['bin']))
        
        # Create temporal dataframe
        df_temp = pd.DataFrame({
            self.date_col: self.df[self.date_col],
            self.target_col: self.df[self.target_col],
            'woe_value': self.df_binned[binned_col],
            variable: self.df[variable]
        })
        
        # Map WoE values to bin labels
        df_temp['bin'] = df_temp['woe_value'].map(woe_to_bin)
        
        # Ensure month is datetime
        df_temp[self.date_col] = pd.to_datetime(df_temp[self.date_col])
        
        # Remove any unmapped values
        df_temp = df_temp[df_temp['bin'].notna()].copy()
        
        return df_temp
    
    def plot_target_ratio_over_time(self, variable):
        """
        Plot 2: Target ratio (conversion rate) of bins over time
        """
        if variable not in self.bins_dict:
            print(f"Variable {variable} not found in bins dictionary")
            return
        
        try:
            df_temp = self.prepare_temporal_data_simple(variable)
        except Exception as e:
            print(f"Error preparing temporal data: {e}")
            return
        
        # Calculate conversion rate by bin and month
        conversion_by_time = df_temp.groupby([self.date_col, 'bin'])[self.target_col].agg([
            'mean',  # Conversion rate
            'count'  # Volume for reliability
        ]).reset_index()
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Get unique bins ordered by their appearance in the original bins_dict
        bin_order = self.bins_dict[variable]['bin'].tolist()
        unique_bins = [b for b in bin_order if b in conversion_by_time['bin'].unique()]
        
        # Plot each bin's conversion rate over time
        for bin_val in unique_bins:
            bin_data = conversion_by_time[conversion_by_time['bin'] == bin_val]
            plt.plot(bin_data[self.date_col], bin_data['mean'] * 100, 
                    marker='o', linewidth=2, markersize=6, label=f'{bin_val}')
        
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Target Ratio / Conversion Rate (%)', fontsize=12)
        
        # Add IV value to title
        iv_value = self.iv_dict.get(variable, 0)
        plt.title(f'Target Ratio Over Time by Bins - {variable} (IV: {iv_value:.4f})', 
                 fontsize=14, fontweight='bold')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Bins')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_volume_distribution_over_time(self, variable):
        """
        Plot 3: Volume distribution of bins over time
        """
        if variable not in self.bins_dict:
            print(f"Variable {variable} not found in bins dictionary")
            return
        
        try:
            df_temp = self.prepare_temporal_data_simple(variable)
        except Exception as e:
            print(f"Error preparing temporal data: {e}")
            return
        
        # Calculate volume distribution by month
        volume_by_time = df_temp.groupby([self.date_col, 'bin']).size().reset_index(name='count')
        
        # Calculate percentage within each month
        volume_by_time['total_monthly'] = volume_by_time.groupby(self.date_col)['count'].transform('sum')
        volume_by_time['percentage'] = (volume_by_time['count'] / volume_by_time['total_monthly']) * 100
        
        # Get bin order
        bin_order = self.bins_dict[variable]['bin'].tolist()
        available_bins = [b for b in bin_order if b in volume_by_time['bin'].unique()]
        
        # Pivot for stacked area chart
        volume_pivot = volume_by_time.pivot(index=self.date_col, 
                                          columns='bin', 
                                          values='percentage').fillna(0)
        
        # Reorder columns based on bin_order
        volume_pivot = volume_pivot[[col for col in available_bins if col in volume_pivot.columns]]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Stacked area chart
        ax1.stackplot(volume_pivot.index, 
                     [volume_pivot[col] for col in volume_pivot.columns],
                     labels=volume_pivot.columns,
                     alpha=0.7)
        
        ax1.set_ylabel('Volume Distribution (%)', fontsize=12)
        
        # Add IV value to title
        iv_value = self.iv_dict.get(variable, 0)
        ax1.set_title(f'Volume Distribution Over Time (Stacked) - {variable} (IV: {iv_value:.4f})', 
                     fontsize=14, fontweight='bold')
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Bins')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Line chart for absolute volumes
        for bin_val in available_bins:
            bin_data = volume_by_time[volume_by_time['bin'] == bin_val]
            ax2.plot(bin_data[self.date_col], bin_data['count'], 
                    marker='o', linewidth=2, markersize=6, label=f'{bin_val}')
        
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('Absolute Count', fontsize=12)
        ax2.set_title(f'Absolute Volume Over Time - {variable}', 
                     fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Bins')
        ax2.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def analyze_all_variables(self, variables=None, include_iv_analysis=True):
        """
        Perform complete analysis for all or specified variables including IV
        """
        if variables is None:
            variables = [col for col in self.df.columns 
                        if col not in [self.target_col, self.date_col]]
        
        # Print IV summary first if requested
        if include_iv_analysis:
            self.print_iv_summary()
            self.plot_iv_summary()
        
        print(f"\nAnalyzing {len(variables)} variables in detail...\n")
        
        for i, var in enumerate(variables, 1):
            print(f"\n{'='*60}")
            print(f"Variable {i}/{len(variables)}: {var}")
            print(f"{'='*60}\n")
            
            # Check if variable exists in bins
            if var not in self.bins_dict:
                print(f"Skipping {var} - not suitable for binning")
                continue
            
            # Display WoE and IV summary
            print("WoE and IV Summary:")
            woe_summary = self.woe_dict[var][['bin', 'woe', 'count_distr']].copy()
            woe_summary['count_distr'] = (woe_summary['count_distr'] * 100).round(2)
            woe_summary.columns = ['Bin', 'WoE', 'Count %']
            
            iv_value = self.iv_dict.get(var, 0)
            iv_interpretation = self._interpret_iv(iv_value)
            
            print(f"Information Value (IV): {iv_value:.4f} ({iv_interpretation})")
            print("\nBin Details:")
            print(woe_summary.to_string(index=False))
            print("\n")
            
            # Generate all plots
            print("Generating Plot 1: WoE, Bin Distribution, and IV Contribution")
            self.plot_woe_and_bin_distribution_with_iv(var)
            
            print("\nGenerating Plot 2: Target Ratio Over Time")
            self.plot_target_ratio_over_time(var)
            
            print("\nGenerating Plot 3: Volume Distribution Over Time")
            self.plot_volume_distribution_over_time(var)
            
            # Optionally plot IV stability
            if include_iv_analysis:
                print("\nGenerating Plot 4: IV Temporal Stability")
                self.plot_iv_temporal_stability(var)
    
    def export_results(self, output_path='woe_iv_results.xlsx'):
        """
        Export WoE and IV results to Excel file
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Export IV summary
            if self.iv_summary is not None:
                self.iv_summary.to_excel(writer, sheet_name='IV_Summary', index=False)
            
            # Export WoE details for each variable
            for var in self.woe_dict.keys():
                woe_df = self.woe_dict[var].copy()
                woe_df['Variable'] = var
                woe_df['IV'] = self.iv_dict.get(var, 0)
                
                # Truncate sheet name if too long
                sheet_name = f'WoE_{var}'[:31]
                woe_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\nResults exported to {output_path}")


# Example usage with synthetic data
def generate_sample_data(n_samples=10000, n_months=12):
    """
    Generate synthetic data for demonstration
    """
    np.random.seed(42)
    
    # Generate date range
    start_date = datetime.now() - timedelta(days=365)
    dates = pd.date_range(start=start_date, periods=n_months, freq='M')
    
    # Create sample data with varying predictive powers
    data = {
        'month': np.random.choice(dates, n_samples),
        'age': np.random.randint(18, 70, n_samples),  # Medium IV
        'income': np.random.lognormal(10.5, 0.5, n_samples),  # Strong IV
        'credit_score': np.random.randint(300, 850, n_samples),  # Strong IV
        'num_accounts': np.random.poisson(3, n_samples),  # Weak IV
        'employment_years': np.random.exponential(5, n_samples),  # Medium IV
        'debt_ratio': np.random.beta(2, 5, n_samples),  # Strong IV
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.3, 0.3, 0.25, 0.15]),  # Weak IV
        'random_feature': np.random.random(n_samples)  # Not useful (low IV)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with logical relationship to features
    score = (
        (df['credit_score'] - 300) / 550 * 0.35 +  # Strong influence
        (df['income'] / df['income'].max()) * 0.25 +  # Strong influence
        (1 - df['debt_ratio']) * 0.25 +  # Strong influence
        (df['employment_years'] / 20) * 0.10 +  # Medium influence
        (df['age'] / 70) * 0.05 +  # Weak influence
        df['random_feature'] * 0.01  # Almost no influence
    )
    
    # Add some randomness and create binary target
    score += np.random.normal(0, 0.1, n_samples)
    df['y'] = (score > np.percentile(score, 70)).astype(int)
    
    return df


# Main execution
if __name__ == "__main__":
    # Generate or load your data
    print("Generating sample data...")
    df = generate_sample_data(n_samples=10000, n_months=12)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Target distribution:\n{df['y'].value_counts(normalize=True)}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Initialize enhanced WoE and IV analysis
    woe_iv_analyzer = WoETimeAnalysisWithIV(df, target_col='y', date_col='month')
    
    # Calculate WoE bins and IV
    print("\nCalculating WoE bins and Information Value...")
    woe_iv_analyzer.calculate_woe_bins(max_num_bin=5, min_perc_fine_bin=0.05)
    
    # Analyze specific variables with IV
    variables_to_analyze = ['credit_score', 'income', 'debt_ratio', 'age', 'random_feature']
    woe_iv_analyzer.analyze_all_variables(variables=variables_to_analyze, include_iv_analysis=True)
    
    # Export results to Excel
    woe_iv_analyzer.export_results('woe_iv_analysis_results.xlsx')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    # For production use with your own data:
    # 1. Load your data
    # df = pd.read_csv('your_data.csv')
    # 
    # 2. Ensure your date column is in datetime format
    # df['month'] = pd.to_datetime(df['month'])
    # 
    # 3. Initialize and run analysis
    # woe_iv_analyzer = WoETimeAnalysisWithIV(df, target_col='your_target', date_col='your_date')
    # woe_iv_analyzer.calculate_woe_bins()
    # woe_iv_analyzer.analyze_all_variables()
    # woe_iv_analyzer.export_results('your_results.xlsx')
