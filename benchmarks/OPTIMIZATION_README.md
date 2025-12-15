# PulseOS Comprehensive Optimization Tools

This directory contains tools for comprehensive optimization and analysis of PulseOS performance across RLHF scenarios.

## Overview

The optimization pipeline consists of multiple phases:

1. **Phase 1: Diagnostic Analysis** - Deep dive into failure cases
2. **Phase 2: Hyperparameter Optimization** - Automated tuning using Optuna
3. **Phase 3: Architectural Improvements** - Enhanced components for complex scenarios
4. **Phase 5: Comprehensive Validation** - Full benchmark suite with visualizations

## Quick Start

### Run Complete Optimization Pipeline

```bash
python benchmarks/comprehensive_optimization.py
```

This will:
- Run diagnostic analysis on failing scenarios
- Optimize hyperparameters for each scenario
- Generate comprehensive visualizations
- Create downloadable data files (CSV, JSON)
- Generate interactive HTML dashboards

### Run Individual Phases

#### Phase 1: Diagnostic Analysis

```bash
python benchmarks/diagnostic_analysis.py
```

Generates:
- Detailed diagnostic reports (`diagnostic_report.md`)
- Visualizations of survival signals, gradients, parameter adaptation
- Root cause analysis for each failing scenario

#### Phase 2: Hyperparameter Optimization

```bash
python benchmarks/hyperparameter_optimization.py
```

Generates:
- Optimal hyperparameter configurations (`optimal_hyperparameters.json`)
- Optimization history plots
- Parameter importance analysis

## Output Structure

```
benchmark_results/
├── charts/                    # High-resolution PNG charts
│   ├── *_learning_curves.png
│   ├── step_reduction_by_scenario.png
│   └── ...
├── data/                      # Downloadable data files
│   ├── *_learning_curves.csv
│   ├── *_learning_curves.json
│   └── ...
├── dashboards/                # Interactive HTML dashboards
│   └── dashboard.html
├── diagnostics/               # Diagnostic analysis
│   ├── multi_objective_normal_th-0.5/
│   ├── linear_bimodal_th-0.5/
│   └── linear_skewed_th-0.3/
├── optimization/              # Hyperparameter optimization results
│   ├── *_study.json
│   ├── optimization_history.png
│   └── param_importances.png
├── diagnostic_report.md       # Phase 1 diagnostic report
├── optimal_hyperparameters.json  # Phase 2 optimal configs
└── FINAL_OPTIMIZATION_REPORT.md  # Comprehensive summary
```

## Downloadable Data Formats

All data is exported in multiple formats for easy analysis:

### CSV Files
- Learning curves: Step-by-step preference values for each trial
- Comparison data: Summary statistics for PPO vs PulseOS
- Step reduction: Percentage improvements by scenario

### JSON Files
- Learning curves: Complete trial data with metadata
- Optimization results: Hyperparameter search history
- Diagnostic data: Detailed internal state tracking

### PNG Charts
- High-resolution (300 DPI) charts suitable for presentations
- Learning curves, comparisons, diagnostic plots
- All charts include proper labels and legends

### HTML Dashboards
- Interactive Plotly visualizations
- Summary statistics cards
- Filterable and zoomable charts

## Key Features

### 1. Diagnostic Analysis (`diagnostic_analysis.py`)

Analyzes why PulseOS fails in certain scenarios:
- Survival signal evolution over time
- Gradient magnitude plots
- Distance to threshold tracking
- Parameter adaptation curves (alpha, epsilon)
- Comparison with PPO behavior

### 2. Hyperparameter Optimization (`hyperparameter_optimization.py`)

Automated tuning using Optuna:
- Bayesian optimization (TPE sampler)
- Scenario-specific configurations
- Validation with multiple trials
- Parameter importance analysis

### 3. Enhanced Components (`pulseos/circuits/enhanced.py`)

Architectural improvements:
- `MultiThresholdPTDC`: Handles bimodal and multi-objective scenarios
- `SkewnessAwareNGCM`: Adapts to skewed distributions
- `MultiObjectiveSurvivalConstraint`: Pareto-based multi-objective optimization

### 4. Visualization Tools (`visualization_tools.py`)

Comprehensive visualization and export:
- Learning curve plots
- Comparison bar charts
- Step reduction visualizations
- Diagnostic heatmaps
- Interactive HTML dashboards

## Usage Examples

### Custom Diagnostic Analysis

```python
from benchmarks.diagnostic_analysis import diagnose_scenario

ppo_diag, pulseos_diag = await diagnose_scenario(
    "my_scenario",
    "linear",
    "normal",
    -0.5,
    num_trials=10
)
```

### Export Custom Visualizations

```python
from benchmarks.visualization_tools import VisualizationExporter

exporter = VisualizationExporter("my_output_dir")
exporter.export_learning_curves(
    "scenario_name",
    ppo_curves,
    pulseos_curves
)
```

### Use Optimal Hyperparameters

```python
from benchmarks.hyperparameter_optimization import AdaptiveConfigSelector

selector = AdaptiveConfigSelector("optimal_hyperparameters.json")
config = selector.select_config("linear_normal")
```

## Dependencies

Required packages (install via `pip install -r requirements.txt`):
- `numpy`
- `matplotlib`
- `pandas`
- `optuna` (for hyperparameter optimization)
- `plotly` (for interactive dashboards)

## Notes

- Phase 3 (Architectural Improvements) requires Runtime modifications to fully integrate
- Hyperparameter optimization can be time-consuming (reduce `n_trials` for faster runs)
- All visualizations are saved in high resolution for presentations
- Data files are in standard formats for easy analysis in Excel, Python, R, etc.

## Next Steps

1. Run diagnostic analysis to understand failure modes
2. Optimize hyperparameters for each scenario type
3. Integrate enhanced components into Runtime
4. Run comprehensive validation with optimized configs
5. Review downloadable charts and data for insights

