# Plotting System Documentation

## Directory Structure

```
Plotting/
├── master_plotter.py           # Main runner with all configuration
├── plot_utils.py               # Shared utilities (CSV loading, y-axis scaling)
│
├── plot_1_deterioration.py     # Plot 1: Forecast deterioration over horizons
├── plot_2_variance_box.py      # Plot 2: Model variance boxplots
├── plot_3_variance_lines.py    # Plot 3: Feature sensitivity lines
├── plot_4_bar_by_horizon.py    # Plot 4: Bar charts by horizon (EXAMPLE - COMPLETE)
├── plot_5_bar_by_model.py      # Plot 5: Bar charts by model
├── plot_6_horizon_degradation.py  # Plot 6: MAE vs horizon degradation
├── plot_7_model_comparison.py  # Plot 7: Best MAE per model per horizon
├── plot_8_feature_contribution.py # Plot 8: Progressive feature contribution
├── plot_9_pruning_gain.py      # Plot 9: Pruning improvement percentage
│
└── Plot_X_Name/                # Output subdirectories (auto-created)
    ├── Plot_4_Bar_By_Horizon/
    ├── Plot_5_Bar_By_Model/
    └── ...
```

## Usage

### Run All Enabled Plots
```bash
cd Plotting
python master_plotter.py
```

### Run One Plot Type (for testing)
```bash
cd Plotting
python plot_4_bar_by_horizon.py  # Runs standalone with default settings
```

### Configure Which Plots Run

Edit `master_plotter.py` lines 17-25:
```python
RUN_PLOT_1_DETERIORATION = True   # Set to False to skip
RUN_PLOT_2_VARIANCE_BOX  = True
# ... etc
```

### Adjust Y-Axis Scaling

Edit `master_plotter.py` lines 30-36:

```python
# Separate limits for Price vs Delta? (True = separate, False = shared)
SEPARATE_LIMITS_BY_TARGET = True

# Multiplier for horizon/model comparison plots (4, 5, 6, 7)
CAP_MULTIPLIER_HORIZON_MODEL = 2.5

# Multiplier for variance plots (2, 3)
CAP_MULTIPLIER_VARIANCE = 3.0
```

**What the multipliers do:**
- Uses `median + (IQR × multiplier)` as the y-axis cap
- Higher multiplier = more room before capping (fewer truncated bars)
- Lower multiplier = tighter scale (better for seeing small differences)

### Change Data Source

Edit `master_plotter.py` line 42:
```python
CSV_FILE = "../experiment_results_clean.csv"  # Path relative to Plotting/
```

## Implementation Status

### ✅ Complete
- `master_plotter.py` - Runner with centralized configuration
- `plot_utils.py` - All shared utilities including unified y-axis scaling
- `plot_4_bar_by_horizon.py` - **Reference implementation** with unified scaling

### ⚠️ To Be Extracted
The remaining 8 plot modules need to be extracted from the monolithic plotter.
Each should follow the pattern of `plot_4_bar_by_horizon.py`:

1. Import `plot_utils` functions
2. Define output directory
3. Implement plot function(s)
4. Implement `generate_all_plots()` wrapper
5. Add standalone test block at bottom

**Plot-specific notes:**

**Plot 1** (Deterioration): Reads PKL files directly, doesn't use CSV
**Plots 2-3** (Variance): Use `plot_group='variance'` for y-limits
**Plots 4-7** (Bars/Comparisons): Use `plot_group='horizon_model'` for y-limits
**Plots 8-9** (Analysis): Don't use unified y-limits (each has unique scale)

## Design Principles

1. **Centralized Configuration**: All toggles and settings live in `master_plotter.py`
2. **Modular**: Each plot type is independent; failures don't cascade
3. **Testable**: Each module can run standalone for debugging
4. **Consistent Scaling**: Plots within a group share y-axis limits for visual comparison
5. **No Duplication**: Shared code (CSV loading, scaling logic) lives in `plot_utils.py`

## Y-Axis Scaling System

The unified scaling system ensures plots within the same group have identical y-axis ranges:

**Group A+B (horizon_model):**
- Plot 4: Bar by Horizon
- Plot 5: Bar by Model
- Plot 6: Horizon Degradation
- Plot 7: Model Comparison

**Group C (variance):**
- Plot 2: Variance Boxplots
- Plot 3: Variance Lines

**Ungrouped:**
- Plot 1: Deterioration (time series, unique scale per plot)
- Plot 8: Feature Contribution (cumulative, unique scale)
- Plot 9: Pruning Gain (percentage, symmetric around zero)

### How It Works

1. `master_plotter.py` loads CSV once at startup
2. Calls `compute_unified_ylimits()` to calculate caps for each group
3. Passes `ylimits` dict to every plotting function
4. Each plot calls `get_ylimit_for_plot()` to retrieve its cap
5. Cap is applied with `ax.set_ylim(top=cap)` + visual annotation

### Toggling Scaling Modes

**Mode 1: Separate limits per target (default)**
```python
SEPARATE_LIMITS_BY_TARGET = True
```
- Price plots use Price cap
- Delta plots use Delta cap
- Useful when targets have very different scales

**Mode 2: Global limit across both targets**
```python
SEPARATE_LIMITS_BY_TARGET = False
```
- All plots in a group use one shared cap
- Useful for direct Price vs Delta comparison

## Next Steps

To complete the extraction:

1. Extract remaining 8 plot functions from `18_master_plotter.py`
2. Adapt each to the modular pattern (see `plot_4_bar_by_horizon.py`)
3. Update `master_plotter.py` imports as each module is completed
4. Test each plot standalone, then test the full suite
5. Delete the monolithic `18_master_plotter.py` once migration is complete

## Troubleshooting

**"Module not found" errors:**
- Make sure you're running from inside the `Plotting/` directory
- Check that `plot_utils.py` exists in the same directory

**Y-axis limits seem wrong:**
- Check `SEPARATE_LIMITS_BY_TARGET` setting
- Adjust multipliers (`CAP_MULTIPLIER_HORIZON_MODEL`, `CAP_MULTIPLIER_VARIANCE`)
- Verify CSV contains baseline data (non-Pruned, non-FullWeek)

**Plots missing:**
- Check the corresponding `RUN_PLOT_X` toggle in `master_plotter.py`
- Look for error messages in console output
- Try running the plot module standalone to isolate the issue
