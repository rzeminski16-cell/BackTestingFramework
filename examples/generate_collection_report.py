"""
Example: Generate Data Collection Report

Demonstrates how to use the CollectionReportGenerator to analyze
collected data and generate comprehensive reports.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Classes.DataCollection.collection_report_generator import CollectionReportGenerator
from Classes.DataCollection.validation_engine import ValidationEngine
from Classes.DataCollection.config import ValidationConfig


def main():
    """Generate comprehensive data collection report."""

    # Configuration
    project_root = Path(__file__).parent.parent
    data_dir = project_root  # Assumes raw_data/ is in project root

    print("=" * 80)
    print("DATA COLLECTION REPORT GENERATOR")
    print("=" * 80)
    print(f"Data Directory: {data_dir}")
    print()

    # Initialize validation engine
    print("Initializing validation engine...")
    validation_config = ValidationConfig()
    validation_engine = ValidationEngine(validation_config)

    # Initialize report generator
    print("Initializing report generator...")
    generator = CollectionReportGenerator(
        data_dir=data_dir,
        validation_engine=validation_engine
    )

    # Generate report
    print("Analyzing collected data...")
    print("(This may take a few minutes for large datasets)")
    print()

    summary, file_stats = generator.generate_report(
        session_id=None,  # Auto-generates timestamp
        validate_data=True  # Run validation on all files
    )

    # Print summary to console
    generator.print_summary(summary)

    # Save detailed reports
    print("Saving detailed reports...")
    report_path = generator.save_report(summary, file_stats)

    print(f"\n✓ Text report saved to: {report_path}")
    print(f"✓ JSON report saved to: {report_path.with_suffix('.json')}")
    print()

    # Print some interesting statistics
    if file_stats:
        print("HIGHLIGHTS")
        print("=" * 80)

        # Files with highest quality scores
        validated_files = [f for f in file_stats if f.validation_score is not None]
        if validated_files:
            top_quality = sorted(validated_files, key=lambda x: x.validation_score, reverse=True)[:5]
            print("\nTop 5 Highest Quality Files:")
            for i, f in enumerate(top_quality, 1):
                print(f"  {i}. {f.symbol} ({f.data_type}): {f.validation_score:.1f}/100")

        # Files with most missing data
        files_with_missing = [f for f in file_stats if f.missing_percentage > 0]
        if files_with_missing:
            most_missing = sorted(files_with_missing, key=lambda x: x.missing_percentage, reverse=True)[:5]
            print("\nTop 5 Files with Most Missing Data:")
            for i, f in enumerate(most_missing, 1):
                print(f"  {i}. {f.symbol} ({f.data_type}): {f.missing_percentage:.2f}% missing")

        # Largest files
        largest_files = sorted(file_stats, key=lambda x: x.file_size_kb, reverse=True)[:5]
        print("\nTop 5 Largest Files:")
        for i, f in enumerate(largest_files, 1):
            size_mb = f.file_size_kb / 1024
            print(f"  {i}. {f.symbol} ({f.data_type}): {size_mb:.2f} MB ({f.num_rows:,} rows)")

        print("\n" + "=" * 80)

    print("\nReport generation complete!")


if __name__ == "__main__":
    main()
