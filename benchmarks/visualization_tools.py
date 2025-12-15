"""
Enhanced Visualization Tools with Downloadable Data

Creates comprehensive charts and exports data in multiple formats:
- PNG charts (high resolution)
- CSV data files
- JSON data files
- HTML interactive dashboards
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime


class VisualizationExporter:
    """Creates visualizations and exports data in multiple formats"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "dashboards").mkdir(exist_ok=True)
    
    def export_learning_curves(
        self,
        scenario_name: str,
        ppo_curves: List[List[float]],
        pulseos_curves: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Export learning curves with multiple formats"""
        
        # 1. Create high-resolution PNG chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot PPO curves
        for i, curve in enumerate(ppo_curves):
            ax.plot(curve, alpha=0.3, color='#2E86AB', linewidth=1.5, 
                   label='PPO' if i == 0 else '')
        
        # Plot PulseOS curves
        for i, curve in enumerate(pulseos_curves):
            ax.plot(curve, alpha=0.3, color='#A23B72', linewidth=1.5,
                   label='PulseOS' if i == 0 else '')
        
        # Plot averages
        max_len = max([len(c) for c in ppo_curves + pulseos_curves])
        ppo_avg = self._compute_average_curve(ppo_curves, max_len)
        pulseos_avg = self._compute_average_curve(pulseos_curves, max_len)
        
        ax.plot(ppo_avg, color='#1B4F72', linewidth=3, label='PPO Average', linestyle='--')
        ax.plot(pulseos_avg, color='#6A1B9A', linewidth=3, label='PulseOS Average', linestyle='--')
        
        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Preference Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{scenario_name}: Learning Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.output_dir / "charts" / f"{scenario_name}_learning_curves.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Export CSV data
        csv_path = self.output_dir / "data" / f"{scenario_name}_learning_curves.csv"
        self._export_curves_to_csv(csv_path, ppo_curves, pulseos_curves)
        
        # 3. Export JSON data
        json_path = self.output_dir / "data" / f"{scenario_name}_learning_curves.json"
        self._export_curves_to_json(json_path, ppo_curves, pulseos_curves, metadata)
        
        return chart_path, csv_path, json_path
    
    def export_comparison_chart(
        self,
        title: str,
        scenarios: List[str],
        ppo_values: List[float],
        pulseos_values: List[float],
        ppo_stds: Optional[List[float]] = None,
        pulseos_stds: Optional[List[float]] = None,
        ylabel: str = "Steps to Convergence"
    ):
        """Export comparison bar chart"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        # Create bars
        ppo_bars = ax.bar(x - width/2, ppo_values, width, label='PPO',
                         color='#2E86AB', alpha=0.8, yerr=ppo_stds, capsize=5)
        pulseos_bars = ax.bar(x + width/2, pulseos_values, width, label='PulseOS',
                              color='#A23B72', alpha=0.8, yerr=pulseos_stds, capsize=5)
        
        # Add value labels on bars
        self._add_value_labels(ax, ppo_bars)
        self._add_value_labels(ax, pulseos_bars)
        
        ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        chart_path = self.output_dir / "charts" / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Export CSV
        csv_path = self.output_dir / "data" / f"{title.lower().replace(' ', '_')}.csv"
        self._export_comparison_to_csv(csv_path, scenarios, ppo_values, pulseos_values, 
                                      ppo_stds, pulseos_stds)
        
        return chart_path, csv_path
    
    def export_step_reduction_chart(
        self,
        scenarios: List[str],
        reductions: List[float],
        stds: Optional[List[float]] = None
    ):
        """Export step reduction percentage chart"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#28A745' if r > 50 else '#FFC107' if r > 20 else '#DC3545' 
                 for r in reductions]
        
        bars = ax.barh(scenarios, reductions, color=colors, alpha=0.8, xerr=stds, capsize=5)
        
        # Add value labels
        for i, (bar, reduction) in enumerate(zip(bars, reductions)):
            ax.text(reduction + 2, i, f'{reduction:.1f}%', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Step Reduction (%)', fontsize=12, fontweight='bold')
        ax.set_title('PulseOS Step Reduction by Scenario', fontsize=14, fontweight='bold')
        ax.axvline(x=60, color='green', linestyle='--', alpha=0.5, label='Target (60%)')
        ax.axvline(x=40, color='orange', linestyle='--', alpha=0.5, label='Minimum (40%)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        chart_path = self.output_dir / "charts" / "step_reduction_by_scenario.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Export CSV
        csv_path = self.output_dir / "data" / "step_reduction_by_scenario.csv"
        df = pd.DataFrame({
            'scenario': scenarios,
            'reduction_percent': reductions,
            'std': stds if stds else [0.0] * len(scenarios)
        })
        df.to_csv(csv_path, index=False)
        
        return chart_path, csv_path
    
    def export_diagnostic_heatmap(
        self,
        scenario_name: str,
        metrics: Dict[str, List[List[float]]],
        metric_names: List[str]
    ):
        """Export diagnostic metrics as heatmap"""
        
        # Compute average values for each metric
        avg_metrics = {}
        for metric_name, curves in metrics.items():
            avg_metrics[metric_name] = [np.mean([c[i] if i < len(c) else c[-1] 
                                                 for c in curves]) 
                                        for i in range(max([len(c) for c in curves]))]
        
        # Create heatmap data
        max_len = max([len(v) for v in avg_metrics.values()])
        heatmap_data = []
        for metric_name in metric_names:
            values = avg_metrics.get(metric_name, [0.0] * max_len)
            # Sample every 10th value for readability
            sampled = values[::max(1, len(values) // 50)]
            heatmap_data.append(sampled)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
        
        ax.set_yticks(range(len(metric_names)))
        ax.set_yticklabels(metric_names)
        ax.set_xlabel('Step (sampled)', fontsize=12, fontweight='bold')
        ax.set_title(f'{scenario_name}: Diagnostic Metrics Heatmap', fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        chart_path = self.output_dir / "charts" / f"{scenario_name}_diagnostic_heatmap.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def create_interactive_dashboard(
        self,
        scenario_results: Dict[str, Dict[str, Any]],
        output_filename: str = "dashboard.html"
    ):
        """Create interactive HTML dashboard"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PulseOS Benchmark Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #A23B72;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .summary-card .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .chart-container {{
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PulseOS Benchmark Dashboard</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
"""
        
        # Calculate summary statistics
        all_reductions = []
        for scenario, results in scenario_results.items():
            if 'reduction' in results:
                all_reductions.append(results['reduction'])
        
        if all_reductions:
            avg_reduction = np.mean(all_reductions)
            min_reduction = min(all_reductions)
            max_reduction = max(all_reductions)
            
            html_content += f"""
            <div class="summary-card">
                <h3>Average Reduction</h3>
                <div class="value">{avg_reduction:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Best Scenario</h3>
                <div class="value">{max_reduction:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Worst Scenario</h3>
                <div class="value">{min_reduction:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Scenarios Tested</h3>
                <div class="value">{len(scenario_results)}</div>
            </div>
"""
        
        html_content += """
        </div>
        
        <div class="chart-container">
            <h2>Step Reduction by Scenario</h2>
            <div id="reductionChart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Convergence Comparison</h2>
            <div id="convergenceChart"></div>
        </div>
    </div>
    
    <script>
"""
        
        # Add JavaScript for Plotly charts
        scenarios = list(scenario_results.keys())
        reductions = [scenario_results[s].get('reduction', 0) for s in scenarios]
        ppo_steps = [scenario_results[s].get('ppo_steps', 0) for s in scenarios]
        pulseos_steps = [scenario_results[s].get('pulseos_steps', 0) for s in scenarios]
        
        html_content += f"""
        // Step Reduction Chart
        var reductionData = [{{
            x: {json.dumps(scenarios)},
            y: {json.dumps(reductions)},
            type: 'bar',
            marker: {{
                color: {json.dumps(['#28A745' if r > 50 else '#FFC107' if r > 20 else '#DC3545' for r in reductions])}
            }}
        }}];
        
        var reductionLayout = {{
            title: 'Step Reduction by Scenario',
            xaxis: {{ title: 'Scenario' }},
            yaxis: {{ title: 'Reduction (%)' }},
            shapes: [
                {{type: 'line', x0: -0.5, x1: {len(scenarios)-0.5}, y0: 60, y1: 60, 
                 line: {{color: 'green', width: 2, dash: 'dash'}}}},
                {{type: 'line', x0: -0.5, x1: {len(scenarios)-0.5}, y0: 40, y1: 40, 
                 line: {{color: 'orange', width: 2, dash: 'dash'}}}}
            ]
        }};
        
        Plotly.newPlot('reductionChart', reductionData, reductionLayout);
        
        // Convergence Comparison Chart
        var convergenceData = [
            {{
                name: 'PPO',
                x: {json.dumps(scenarios)},
                y: {json.dumps(ppo_steps)},
                type: 'bar',
                marker: {{color: '#2E86AB'}}
            }},
            {{
                name: 'PulseOS',
                x: {json.dumps(scenarios)},
                y: {json.dumps(pulseos_steps)},
                type: 'bar',
                marker: {{color: '#A23B72'}}
            }}
        ];
        
        var convergenceLayout = {{
            title: 'Steps to Convergence Comparison',
            xaxis: {{ title: 'Scenario' }},
            yaxis: {{ title: 'Steps' }},
            barmode: 'group'
        }};
        
        Plotly.newPlot('convergenceChart', convergenceData, convergenceLayout);
    </script>
</body>
</html>
"""
        
        dashboard_path = self.output_dir / "dashboards" / output_filename
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        return dashboard_path
    
    def _compute_average_curve(self, curves: List[List[float]], max_len: int) -> List[float]:
        """Compute average curve across multiple trials"""
        avg = []
        for i in range(max_len):
            values = [c[i] if i < len(c) else c[-1] for c in curves]
            avg.append(np.mean(values))
        return avg
    
    def _add_value_labels(self, ax, bars):
        """Add value labels on bars"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    def _export_curves_to_csv(self, path: Path, ppo_curves: List[List[float]], 
                              pulseos_curves: List[List[float]]):
        """Export learning curves to CSV"""
        max_len = max([len(c) for c in ppo_curves + pulseos_curves])
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'PPO_Trial_1', 'PPO_Trial_2', 'PPO_Trial_3', 
                           'PPO_Trial_4', 'PPO_Trial_5', 'PPO_Average',
                           'PulseOS_Trial_1', 'PulseOS_Trial_2', 'PulseOS_Trial_3',
                           'PulseOS_Trial_4', 'PulseOS_Trial_5', 'PulseOS_Average'])
            
            ppo_avg = self._compute_average_curve(ppo_curves, max_len)
            pulseos_avg = self._compute_average_curve(pulseos_curves, max_len)
            
            for i in range(max_len):
                row = [i]
                # PPO trials
                for curve in ppo_curves:
                    row.append(curve[i] if i < len(curve) else curve[-1])
                row.append(ppo_avg[i] if i < len(ppo_avg) else ppo_avg[-1])
                # PulseOS trials
                for curve in pulseos_curves:
                    row.append(curve[i] if i < len(curve) else curve[-1])
                row.append(pulseos_avg[i] if i < len(pulseos_avg) else pulseos_avg[-1])
                writer.writerow(row)
    
    def _export_curves_to_json(self, path: Path, ppo_curves: List[List[float]], 
                               pulseos_curves: List[List[float]], 
                               metadata: Optional[Dict[str, Any]]):
        """Export learning curves to JSON"""
        data = {
            'ppo_curves': ppo_curves,
            'pulseos_curves': pulseos_curves,
            'ppo_average': self._compute_average_curve(ppo_curves, 
                max([len(c) for c in ppo_curves])),
            'pulseos_average': self._compute_average_curve(pulseos_curves,
                max([len(c) for c in pulseos_curves])),
            'metadata': metadata or {}
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_comparison_to_csv(self, path: Path, scenarios: List[str],
                                  ppo_values: List[float], pulseos_values: List[float],
                                  ppo_stds: Optional[List[float]], 
                                  pulseos_stds: Optional[List[float]]):
        """Export comparison data to CSV"""
        df = pd.DataFrame({
            'scenario': scenarios,
            'ppo_mean': ppo_values,
            'ppo_std': ppo_stds if ppo_stds else [0.0] * len(scenarios),
            'pulseos_mean': pulseos_values,
            'pulseos_std': pulseos_stds if pulseos_stds else [0.0] * len(scenarios),
            'reduction_percent': [((ppo - pulseos) / ppo * 100) if ppo > 0 else 0.0
                                 for ppo, pulseos in zip(ppo_values, pulseos_values)]
        })
        df.to_csv(path, index=False)

