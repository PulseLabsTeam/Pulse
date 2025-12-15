"""
Comprehensive Optimization Pipeline

Integrates all phases:
- Phase 1: Diagnostic Analysis
- Phase 2: Hyperparameter Optimization
- Phase 3: Architectural Improvements
- Phase 4: Ablation Studies
- Phase 5: Comprehensive Re-validation

Generates downloadable charts and data files.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

from benchmarks.diagnostic_analysis import (
    diagnose_scenario, create_diagnostic_visualizations, generate_diagnostic_report
)
from benchmarks.hyperparameter_optimization import (
    optimize_all_scenarios, AdaptiveConfigSelector
)
from benchmarks.visualization_tools import VisualizationExporter
from benchmarks.strategic_benchmark_suite import (
    test1_rlhf_variants, TrialResult, BenchmarkResult
)


class ComprehensiveOptimizer:
    """Orchestrates the complete optimization pipeline"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.exporter = VisualizationExporter(self.output_dir)
        self.results: Dict[str, Any] = {}
    
    async def run_phase_1_diagnostics(self):
        """Phase 1: Diagnostic Analysis"""
        print("\n" + "="*70)
        print("PHASE 1: DIAGNOSTIC ANALYSIS")
        print("="*70)
        
        failing_scenarios = [
            ("multi_objective_normal_th-0.5", "multi_objective", "normal", -0.5),
            ("linear_bimodal_th-0.5", "linear", "bimodal", -0.5),
            ("linear_skewed_th-0.3", "linear", "skewed", -0.3),
        ]
        
        all_diagnostics = {}
        
        for scenario_name, reward_model, pref_dist, threshold in failing_scenarios:
            print(f"\nDiagnosing: {scenario_name}")
            ppo_diag, pulseos_diag = await diagnose_scenario(
                scenario_name,
                reward_model,
                pref_dist,
                threshold,
                num_trials=5
            )
            
            all_diagnostics[scenario_name] = (ppo_diag, pulseos_diag)
            create_diagnostic_visualizations(ppo_diag, pulseos_diag, self.output_dir)
        
        generate_diagnostic_report(all_diagnostics, self.output_dir)
        self.results['phase1'] = all_diagnostics
        
        return all_diagnostics
    
    async def run_phase_2_hyperparameter_optimization(self, n_trials: int = 30):
        """Phase 2: Hyperparameter Optimization"""
        print("\n" + "="*70)
        print("PHASE 2: HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        
        from benchmarks.hyperparameter_optimization import HyperparameterOptimizer
        
        optimizer = HyperparameterOptimizer(self.output_dir)
        
        scenarios = [
            ("multi_objective_normal_th-0.5", "multi_objective", "normal", -0.5),
            ("linear_bimodal_th-0.5", "linear", "bimodal", -0.5),
            ("linear_skewed_th-0.3", "linear", "skewed", -0.3),
        ]
        
        best_configs = {}
        
        for scenario_name, reward_model, pref_dist, threshold in scenarios:
            try:
                best_config, best_score = await optimizer.optimize_scenario(
                    scenario_name,
                    reward_model,
                    pref_dist,
                    threshold,
                    n_trials=n_trials,
                    n_validation_trials=3
                )
                best_configs[scenario_name] = {
                    'config': best_config.to_dict(),
                    'score': best_score
                }
            except Exception as e:
                print(f"Warning: Optimization failed for {scenario_name}: {e}")
                continue
        
        # Store best configs properly
        for name, result in best_configs.items():
            if name not in optimizer.best_configs:
                # Create HyperparameterConfig from dict
                from benchmarks.hyperparameter_optimization import HyperparameterConfig
                config = HyperparameterConfig(**result['config'])
                optimizer.best_configs[name] = config
        optimizer.save_best_configs()
        
        self.results['phase2'] = best_configs
        return best_configs
    
    async def run_phase_3_architectural_improvements(self):
        """Phase 3: Test Architectural Improvements"""
        print("\n" + "="*70)
        print("PHASE 3: ARCHITECTURAL IMPROVEMENTS")
        print("="*70)
        
        print("Note: Architectural improvements require Runtime modifications.")
        print("Testing framework ready - implement Runtime integration to test.")
        
        # Placeholder for architectural improvement tests
        # In practice, this would:
        # 1. Test MultiThresholdPTDC on bimodal scenarios
        # 2. Test SkewnessAwareNGCM on skewed scenarios
        # 3. Test MultiObjectiveSurvivalConstraint on multi-objective scenarios
        
        self.results['phase3'] = {'status': 'framework_ready'}
        return self.results['phase3']
    
    async def run_phase_5_comprehensive_validation(self, use_optimized_configs: bool = True):
        """Phase 5: Comprehensive Re-validation"""
        print("\n" + "="*70)
        print("PHASE 5: COMPREHENSIVE RE-VALIDATION")
        print("="*70)
        
        # Run full benchmark suite
        test1_results = await test1_rlhf_variants(num_trials=10)
        
        # Extract results for visualization
        scenario_results = {}
        scenarios = []
        ppo_steps = []
        pulseos_steps = []
        reductions = []
        
        for variant_name, benchmark_result in test1_results.items():
            # BenchmarkResult has ppo_results and pulseos_results directly
            ppo_trials = benchmark_result.ppo_results if hasattr(benchmark_result, 'ppo_results') else []
            pulseos_trials = benchmark_result.pulseos_results if hasattr(benchmark_result, 'pulseos_results') else []
            
            # Also check variant_results if available
            if not ppo_trials and benchmark_result.variant_results:
                variant_data = benchmark_result.variant_results.get(variant_name, {})
                ppo_trials = variant_data.get("PPO", [])
                pulseos_trials = variant_data.get("PulseOS", [])
            
            if ppo_trials and pulseos_trials:
                ppo_avg = np.mean([t.steps_to_convergence for t in ppo_trials])
                pulseos_avg = np.mean([t.steps_to_convergence for t in pulseos_trials])
                reduction = ((ppo_avg - pulseos_avg) / ppo_avg * 100) if ppo_avg > 0 else 0.0
                
                scenarios.append(variant_name)
                ppo_steps.append(ppo_avg)
                pulseos_steps.append(pulseos_avg)
                reductions.append(reduction)
                
                scenario_results[variant_name] = {
                    'ppo_steps': ppo_avg,
                    'pulseos_steps': pulseos_avg,
                    'reduction': reduction,
                    'ppo_curves': [t.learning_curve for t in ppo_trials],
                    'pulseos_curves': [t.learning_curve for t in pulseos_trials]
                }
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # Learning curves
        for scenario_name, results in scenario_results.items():
            self.exporter.export_learning_curves(
                scenario_name,
                results['ppo_curves'],
                results['pulseos_curves'],
                metadata={'reduction': results['reduction']}
            )
        
        # Comparison charts
        ppo_stds = []
        pulseos_stds = []
        for scenario in scenarios:
            if scenario in test1_results:
                benchmark_result = test1_results[scenario]
                ppo_trials = benchmark_result.ppo_results if hasattr(benchmark_result, 'ppo_results') else []
                pulseos_trials = benchmark_result.pulseos_results if hasattr(benchmark_result, 'pulseos_results') else []
                
                if not ppo_trials and benchmark_result.variant_results:
                    variant_data = benchmark_result.variant_results.get(scenario, {})
                    ppo_trials = variant_data.get("PPO", [])
                    pulseos_trials = variant_data.get("PulseOS", [])
                
                ppo_stds.append(np.std([t.steps_to_convergence for t in ppo_trials]) if ppo_trials else 0.0)
                pulseos_stds.append(np.std([t.steps_to_convergence for t in pulseos_trials]) if pulseos_trials else 0.0)
            else:
                ppo_stds.append(0.0)
                pulseos_stds.append(0.0)
        
        self.exporter.export_comparison_chart(
            "Steps to Convergence Comparison",
            scenarios,
            ppo_steps,
            pulseos_steps,
            ppo_stds,
            pulseos_stds
        )
        
        self.exporter.export_step_reduction_chart(
            scenarios,
            reductions,
            [0.0] * len(scenarios)  # TODO: compute stds
        )
        
        # Create interactive dashboard
        dashboard_path = self.exporter.create_interactive_dashboard(scenario_results)
        print(f"Interactive dashboard: {dashboard_path}")
        
        # Generate summary report
        self._generate_final_report(scenario_results)
        
        self.results['phase5'] = scenario_results
        return scenario_results
    
    def _generate_final_report(self, scenario_results: Dict[str, Any]):
        """Generate final comprehensive report"""
        report_path = self.output_dir / "FINAL_OPTIMIZATION_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# PulseOS Comprehensive Optimization Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            reductions = [r['reduction'] for r in scenario_results.values()]
            avg_reduction = np.mean(reductions)
            min_reduction = min(reductions)
            max_reduction = max(reductions)
            
            f.write(f"- **Average Step Reduction:** {avg_reduction:.1f}%\n")
            f.write(f"- **Best Scenario:** {max_reduction:.1f}%\n")
            f.write(f"- **Worst Scenario:** {min_reduction:.1f}%\n")
            f.write(f"- **Scenarios Tested:** {len(scenario_results)}\n\n")
            
            f.write("## Results by Scenario\n\n")
            f.write("| Scenario | PPO Steps | PulseOS Steps | Reduction |\n")
            f.write("|----------|-----------|---------------|----------|\n")
            
            for scenario_name, results in scenario_results.items():
                f.write(f"| {scenario_name} | {results['ppo_steps']:.1f} | "
                       f"{results['pulseos_steps']:.1f} | {results['reduction']:.1f}% |\n")
            
            f.write("\n## Visualizations\n\n")
            f.write("All charts and data files are available in:\n")
            f.write("- `charts/` - High-resolution PNG charts\n")
            f.write("- `data/` - CSV and JSON data files\n")
            f.write("- `dashboards/` - Interactive HTML dashboards\n")
            f.write("- `diagnostics/` - Diagnostic analysis plots\n\n")
            
            f.write("## Recommendations\n\n")
            
            if reductions:
                avg_reduction = np.mean(reductions)
                if avg_reduction >= 60:
                    f.write("✅ **SUCCESS:** PulseOS achieves target 60%+ average reduction.\n\n")
                elif avg_reduction >= 40:
                    f.write("⚠️ **PARTIAL SUCCESS:** PulseOS achieves 40-60% reduction. "
                           "Consider further optimization.\n\n")
                else:
                    f.write("❌ **NEEDS IMPROVEMENT:** Average reduction below 40%. "
                           "Review diagnostic analysis and consider architectural changes.\n\n")
            else:
                f.write("⚠️ **No results available** - Check benchmark execution.\n\n")
        
        print(f"\nFinal report saved to: {report_path}")


async def main():
    """Run comprehensive optimization pipeline"""
    
    print("="*70)
    print("PULSEOS COMPREHENSIVE OPTIMIZATION PIPELINE")
    print("="*70)
    
    output_dir = Path("benchmark_results")
    optimizer = ComprehensiveOptimizer(output_dir)
    
    # Phase 1: Diagnostics
    await optimizer.run_phase_1_diagnostics()
    
    # Phase 2: Hyperparameter Optimization (reduced trials for faster execution)
    await optimizer.run_phase_2_hyperparameter_optimization(n_trials=20)
    
    # Phase 3: Architectural Improvements
    await optimizer.run_phase_3_architectural_improvements()
    
    # Phase 5: Comprehensive Validation
    await optimizer.run_phase_5_comprehensive_validation()
    
    # Save all results
    results_path = output_dir / "optimization_results.json"
    with open(results_path, 'w') as f:
        json.dump(optimizer.results, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("OPTIMIZATION PIPELINE COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir.absolute()}")
    print(f"\nDownloadable files:")
    print(f"  - Charts: {output_dir / 'charts'}")
    print(f"  - Data: {output_dir / 'data'}")
    print(f"  - Dashboards: {output_dir / 'dashboards'}")
    print(f"  - Diagnostics: {output_dir / 'diagnostics'}")


if __name__ == "__main__":
    asyncio.run(main())

