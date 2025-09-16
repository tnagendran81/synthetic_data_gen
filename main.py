#!/usr/bin/env python3
"""
Enterprise Synthetic Data Generator - Main Entry Point
=====================================================

Usage:
    python main.py              # Launch Gradio UI
    python main.py --cli         # Command line interface
    python main.py --help        # Show help
"""

import argparse
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.gradio_ui import launch_app
from src.data_generator import SyntheticDataGenerator

def cli_interface():
    """Command line interface for batch generation"""
    print("🏭 Synthetic Data Generator - CLI Mode")
    print("=" * 50)

    # Simple CLI example
    generator = SyntheticDataGenerator()

    # Generate sample data
    dataset = generator.generate_full_dataset(
        selected_industries=["Technology", "Finance"],
        selected_regions=["Americas"],
        selected_countries={"Americas": ["USA"]},
        num_companies_per_region=2,
        company_sizes=["startup", "medium"]
    )

    # Save to CSV
    csv_files = generator.save_to_csv(
        dataset['companies'], 
        dataset['employees'], 
        dataset['products']
    )

    print(f"✅ Generated {dataset['summary']['total_companies']} companies")
    print(f"✅ Generated {dataset['summary']['total_employees']} employees") 
    print(f"✅ Generated {dataset['summary']['total_products']} products")
    print(f"📁 Files saved: {', '.join(csv_files)}")

def main():
    parser = argparse.ArgumentParser(description='Enterprise Synthetic Data Generator')
    parser.add_argument('--cli', action='store_true', help='Run in command line mode')
    parser.add_argument('--port', type=int, default=7860, help='Port for Gradio interface')

    args = parser.parse_args()

    if args.cli:
        cli_interface()
    else:
        print("🚀 Launching Gradio Web Interface...")
        launch_app()

if __name__ == "__main__":
    main()
