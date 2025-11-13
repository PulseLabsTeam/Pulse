"""
Week 4 Day 22-25: Report Generator

Generates comprehensive report with results, analysis, and valuation assessment.
"""

import json
from pathlib import Path
from datetime import datetime

def load_results():
    """Load evaluation results."""
    results_path = Path("pulseos_rlhf/evaluation_results.json")
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data['all_results'], data['analysis']

def generate_report():
    """Generate comprehensive report."""
    all_results, analysis = load_results()
    
    report = []
    report.append("=" * 80)
    report.append("PULSEOS RLHF TRAINING - COMPREHENSIVE RESULTS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 80)
    
    baseline_mean = analysis['baseline']['mean']
    best_method = None
    best_improvement = 0
    
    for method in analysis.keys():
        if method != 'baseline':
            improvement = analysis[method]['improvement_percent']
            if improvement > best_improvement:
                best_improvement = improvement
                best_method = method
    
    report.append(f"Baseline PPO: {baseline_mean:.1f} ± {analysis['baseline']['std']:.1f} samples")
    if best_method:
        report.append(f"Best Method: {best_method.replace('_', ' ').title()}")
        report.append(f"Improvement: {best_improvement:.1f}%")
        report.append(f"Statistical Significance: {'Yes' if analysis[best_method]['significant'] else 'No'}")
    report.append("")
    
    # Detailed Results
    report.append("DETAILED RESULTS")
    report.append("-" * 80)
    
    for method in analysis.keys():
        method_data = analysis[method]
        report.append(f"\n{method.replace('_', ' ').title()}:")
        report.append(f"  Mean samples: {method_data['mean']:.1f} ± {method_data['std']:.1f}")
        
        if method != 'baseline':
            report.append(f"  Improvement: {method_data['improvement_percent']:.1f}%")
            report.append(f"  p-value: {method_data['p_value']:.4f}")
            report.append(f"  Significant: {'Yes' if method_data['significant'] else 'No'}")
            report.append(f"  Cohen's d: {method_data['cohens_d']:.3f}")
        
        report.append(f"  Individual trials: {method_data['samples']}")
    
    report.append("")
    
    # Statistical Analysis
    report.append("STATISTICAL ANALYSIS")
    report.append("-" * 80)
    
    report.append("\nT-tests (vs Baseline):")
    for method in analysis.keys():
        if method != 'baseline':
            method_data = analysis[method]
            report.append(f"  {method.replace('_', ' ').title()}:")
            report.append(f"    t-statistic: {method_data.get('t_stat', 'N/A')}")
            report.append(f"    p-value: {method_data['p_value']:.4f}")
            report.append(f"    Effect size (Cohen's d): {method_data['cohens_d']:.3f}")
            if method_data['cohens_d'] < 0.2:
                effect = "negligible"
            elif method_data['cohens_d'] < 0.5:
                effect = "small"
            elif method_data['cohens_d'] < 0.8:
                effect = "medium"
            else:
                effect = "large"
            report.append(f"    Effect size interpretation: {effect}")
    
    report.append("")
    
    # Valuation Assessment
    report.append("VALUATION ASSESSMENT")
    report.append("-" * 80)
    
    if best_method:
        improvement = best_improvement
        significant = analysis[best_method]['significant']
        
        report.append(f"\nBest Improvement: {improvement:.1f}%")
        report.append(f"Statistical Significance: {'Yes' if significant else 'No'}")
        report.append("")
        
        if improvement >= 40 and significant:
            valuation = "$15M-$30M"
            buyers = "Anthropic, OpenAI, Google, Meta"
            assessment = "EXCELLENT"
        elif improvement >= 30 and significant:
            valuation = "$10M-$20M"
            buyers = "Mid-tier AI labs, research institutions"
            assessment = "STRONG"
        elif improvement >= 20 and significant:
            valuation = "$8M-$15M"
            buyers = "Research-focused buyers"
            assessment = "GOOD"
        elif improvement >= 10 and significant:
            valuation = "$6M-$10M"
            buyers = "Research-focused buyers"
            assessment = "MODEST"
        else:
            valuation = "$3M-$8M"
            buyers = "Patent + research IP only"
            assessment = "LOW"
        
        report.append(f"Assessment: {assessment}")
        report.append(f"Valuation Range: {valuation}")
        report.append(f"Potential Buyers: {buyers}")
    else:
        report.append("\nNo improvement detected. Valuation: $3M-$8M (simulation results only)")
    
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 80)
    
    if best_method and best_improvement >= 20:
        report.append("\n1. Proceed with full validation on larger models (GPT-2 medium/large)")
        report.append("2. Prepare investor materials with validated results")
        report.append("3. Start outreach to potential buyers")
        report.append("4. Consider publication of results")
    elif best_method and best_improvement >= 10:
        report.append("\n1. Continue optimization of hyperparameters")
        report.append("2. Test on different model sizes")
        report.append("3. Document findings for future development")
    else:
        report.append("\n1. Review implementation for bugs")
        report.append("2. Try different hyperparameters")
        report.append("3. Consider focusing on simulation results")
    
    report.append("")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    output_path = Path("pulseos_rlhf/RESULTS_REPORT.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print("=" * 80)
    print("REPORT GENERATED")
    print("=" * 80)
    print(f"Saved to: {output_path}")
    print()
    print(report_text)

if __name__ == "__main__":
    generate_report()


